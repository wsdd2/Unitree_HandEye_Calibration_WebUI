"""Strict camera-intrinsics and hand-eye calibration quality gates.

This module deliberately separates *solving* from *deployment approval*.
OpenCV can return a numerically self-consistent transform even when every
sample shares the same systematic error.  A result is therefore deployable
only after capture preflight, multi-method agreement, subset stability, and
optional cross-session agreement all pass.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .calibration_target import build_object_points
from .intrinsics import write_intrinsics_manifest
from .solver import (
    MODE_EYE_IN_HAND,
    MODE_EYE_TO_HAND,
    normalize_mode,
    record_transform,
    validate_capture_records,
)
from .transforms import average_transforms, invert_transform, transform_to_record


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

RIGHT_ARM_JOINT_NAMES = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}

LEFT_ARM_JOINT_NAMES = {
    name.replace("right_", "left_") for name in RIGHT_ARM_JOINT_NAMES
}


@dataclass(frozen=True)
class IntrinsicsThresholds:
    min_images: int = 24
    max_rms_px: float = 0.35
    max_view_rms_px: float = 0.50
    min_edge_coverage_ratio: float = 0.08
    max_edge_coverage_ratio: float = 0.92
    subset_trials: int = 30
    subset_fraction: float = 0.70
    subset_focal_p95_ratio: float = 0.005
    subset_principal_p95_px: float = 3.0


@dataclass(frozen=True)
class CaptureThresholds:
    max_reprojection_rms_px: float = 0.30
    max_sync_delta_sec: float = 0.03
    max_state_age_sec: float = 0.10
    max_joint_velocity_rad_s: float = 0.01
    require_intrinsics_manifest: bool = True


@dataclass(frozen=True)
class HandEyeThresholds:
    min_samples: int = 24
    closure_translation_max_m: float = 0.010
    closure_rotation_max_deg: float = 1.0
    method_translation_max_m: float = 0.005
    method_rotation_max_deg: float = 0.50
    split_translation_max_m: float = 0.005
    split_rotation_max_deg: float = 0.50
    subset_translation_p95_m: float = 0.005
    subset_rotation_p95_deg: float = 0.50
    subset_translation_max_m: float = 0.010
    subset_rotation_max_deg: float = 1.0
    baseline_translation_max_m: float = 0.005
    baseline_rotation_max_deg: float = 0.50
    subset_trials: int = 100
    subset_fraction: float = 0.65
    random_seed: int = 20260813


def validate_live_fk_snapshot(
    snapshot: dict[str, Any],
    *,
    arm_side: str,
    max_joint_velocity_rad_s: float,
) -> None:
    """Reject incomplete or moving FK snapshots before a capture is written."""
    required = LEFT_ARM_JOINT_NAMES if arm_side == "left" else RIGHT_ARM_JOINT_NAMES
    positions = snapshot.get("arm_joints", {})
    velocities = snapshot.get("arm_joint_velocities_rad_s", {})
    missing_positions = sorted(required.difference(positions))
    missing_velocities = sorted(required.difference(velocities))
    if missing_positions:
        raise RuntimeError(f"strict capture missing joint positions: {missing_positions}")
    if missing_velocities:
        raise RuntimeError(f"strict capture missing joint velocities: {missing_velocities}")
    maximum = max(abs(float(velocities[name])) for name in required)
    if (
        max_joint_velocity_rad_s > 0.0
        and maximum > max_joint_velocity_rad_s
    ):
        raise RuntimeError(
            f"strict capture arm still moving: {maximum:.6f}rad/s > "
            f"{max_joint_velocity_rad_s:.6f}rad/s"
        )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def transform_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Return candidate-reference SE(3) difference metrics."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    relative_rotation = reference[:3, :3].T @ candidate[:3, :3]
    cosine = np.clip((float(np.trace(relative_rotation)) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    translation_xyz = candidate[:3, 3] - reference[:3, 3]
    return {
        "translation_xyz_m": translation_xyz.astype(float).tolist(),
        "translation_norm_m": float(np.linalg.norm(translation_xyz)),
        "rotation_deg": rotation_deg,
    }


def solve_external(
    records: Sequence[dict[str, Any]],
    mode: str,
    method: int,
) -> np.ndarray:
    """Solve one hand-eye transform with explicit OpenCV frame conventions."""
    normalized_mode = normalize_mode(mode)
    validate_capture_records(list(records), expected_mode=normalized_mode)
    hand2base = [record_transform(record, "hand2base") for record in records]
    base2hand = [record_transform(record, "base2hand") for record in records]
    target2cam = [record_transform(record, "target2cam") for record in records]
    gripper2base = (
        hand2base if normalized_mode == MODE_EYE_IN_HAND else base2hand
    )
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in gripper2base],
        [transform[:3, 3].reshape(3, 1) for transform in gripper2base],
        [transform[:3, :3] for transform in target2cam],
        [transform[:3, 3].reshape(3, 1) for transform in target2cam],
        method=method,
    )
    external = np.eye(4, dtype=np.float64)
    external[:3, :3] = rotation
    external[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.isfinite(external).all():
        raise ValueError("calibrateHandEye returned non-finite values")
    return external


def _fixed_transforms(
    records: Sequence[dict[str, Any]],
    mode: str,
    external: np.ndarray,
) -> list[np.ndarray]:
    normalized_mode = normalize_mode(mode)
    hand2base = [record_transform(record, "hand2base") for record in records]
    base2hand = [record_transform(record, "base2hand") for record in records]
    target2cam = [record_transform(record, "target2cam") for record in records]
    if normalized_mode == MODE_EYE_IN_HAND:
        return [
            hand2base[index] @ external @ target2cam[index]
            for index in range(len(records))
        ]
    return [
        base2hand[index] @ external @ target2cam[index]
        for index in range(len(records))
    ]


def closure_report(
    records: Sequence[dict[str, Any]],
    mode: str,
    external: np.ndarray,
) -> dict[str, Any]:
    fixed = _fixed_transforms(records, mode, external)
    mean = average_transforms(fixed)
    translation = [
        float(np.linalg.norm(transform[:3, 3] - mean[:3, 3]))
        for transform in fixed
    ]
    rotation = [
        transform_delta(mean, transform)["rotation_deg"] for transform in fixed
    ]
    pair_translation: list[float] = []
    pair_rotation: list[float] = []
    for left in range(len(fixed)):
        for right in range(left + 1, len(fixed)):
            delta = transform_delta(fixed[left], fixed[right])
            pair_translation.append(float(delta["translation_norm_m"]))
            pair_rotation.append(float(delta["rotation_deg"]))
    return {
        "fixed_transform_mean": transform_to_record(mean),
        "translation_residual_m": {
            "mean": float(np.mean(translation)),
            "p95": _percentile(translation, 95.0),
            "max": float(np.max(translation)),
        },
        "rotation_residual_deg": {
            "mean": float(np.mean(rotation)),
            "p95": _percentile(rotation, 95.0),
            "max": float(np.max(rotation)),
        },
        "pairwise_translation_m": {
            "p95": _percentile(pair_translation, 95.0),
            "max": max(pair_translation, default=0.0),
        },
        "pairwise_rotation_deg": {
            "p95": _percentile(pair_rotation, 95.0),
            "max": max(pair_rotation, default=0.0),
        },
    }


def _camera_serial(record: dict[str, Any]) -> str:
    info = record.get("camera_info", {})
    for key in ("selected_serial", "serial", "requested_serial"):
        value = info.get(key) if isinstance(info, dict) else None
        if value not in (None, ""):
            return str(value)
    return ""


def validate_capture_preflight(
    records: Sequence[dict[str, Any]],
    *,
    thresholds: CaptureThresholds,
    expected_serial: str,
    expected_size: tuple[int, int],
    expected_base_frame: str,
    expected_hand_frame: str,
    expected_urdf_sha256: str = "",
) -> dict[str, Any]:
    """Validate capture context and every synchronized FK/PnP sample."""
    failures: list[str] = []
    warnings: list[str] = []
    width, height = expected_size
    all_pixels: list[np.ndarray] = []
    urdf_hashes: set[str] = set()
    required_joints = (
        LEFT_ARM_JOINT_NAMES
        if expected_hand_frame.startswith("left_")
        else RIGHT_ARM_JOINT_NAMES
    )

    for index, record in enumerate(records, start=1):
        label = Path(str(record.get("_json_path", f"sample_{index}"))).name
        camera_info = record.get("camera_info", {})
        intrinsics = record.get("camera_intrinsics", {})
        timing = record.get("capture_timing", {})
        fk_timing = timing.get("fk", {}) if isinstance(timing, dict) else {}
        fk_metadata = record.get("fk_metadata", {})
        arm_joints = fk_metadata.get("arm_joints_rad", {})
        arm_velocities = fk_metadata.get("arm_joint_velocities_rad_s", {})
        model = fk_metadata.get("model", {})

        if expected_serial and _camera_serial(record) != expected_serial:
            failures.append(
                f"{label}: camera serial {_camera_serial(record)!r} != "
                f"{expected_serial!r}"
            )
        actual_size = (
            int(camera_info.get("width", 0)),
            int(camera_info.get("height", 0)),
        )
        if actual_size != expected_size:
            failures.append(f"{label}: image size {actual_size} != {expected_size}")
        if thresholds.require_intrinsics_manifest and (
            intrinsics.get("manifest_status") != "validated"
        ):
            failures.append(f"{label}: intrinsics manifest is not validated")

        pose_input = record.get("hand_pose_input", {})
        if pose_input.get("base_frame_name") != expected_base_frame:
            failures.append(
                f"{label}: base frame {pose_input.get('base_frame_name')!r} != "
                f"{expected_base_frame!r}"
            )
        if pose_input.get("frame_name") != expected_hand_frame:
            failures.append(
                f"{label}: hand frame {pose_input.get('frame_name')!r} != "
                f"{expected_hand_frame!r}"
            )

        missing_positions = sorted(required_joints.difference(arm_joints))
        missing_velocities = sorted(required_joints.difference(arm_velocities))
        if missing_positions:
            failures.append(f"{label}: missing arm positions {missing_positions}")
        if missing_velocities:
            failures.append(f"{label}: missing arm velocities {missing_velocities}")

        max_velocity = max(
            (abs(float(arm_velocities[name])) for name in required_joints if name in arm_velocities),
            default=float("inf"),
        )
        if (
            thresholds.max_joint_velocity_rad_s > 0.0
            and max_velocity > thresholds.max_joint_velocity_rad_s
        ):
            failures.append(
                f"{label}: arm velocity {max_velocity:.6f}rad/s > "
                f"{thresholds.max_joint_velocity_rad_s:.6f}"
            )

        sync_delta = fk_timing.get("sync_delta_sec")
        if sync_delta is None or abs(float(sync_delta)) > thresholds.max_sync_delta_sec:
            failures.append(
                f"{label}: sync delta {sync_delta!r}s exceeds "
                f"{thresholds.max_sync_delta_sec:.3f}s"
            )
        state_age = fk_timing.get("state_age_sec_at_snapshot")
        if state_age is None or float(state_age) > thresholds.max_state_age_sec:
            failures.append(
                f"{label}: state age {state_age!r}s exceeds "
                f"{thresholds.max_state_age_sec:.3f}s"
            )

        reprojection = record.get("target_reprojection_rms_px")
        if (
            reprojection is None
            or not np.isfinite(float(reprojection))
            or float(reprojection) > thresholds.max_reprojection_rms_px
        ):
            failures.append(
                f"{label}: reprojection RMS {reprojection!r}px exceeds "
                f"{thresholds.max_reprojection_rms_px:.3f}px"
            )

        current_hash = str(model.get("urdf_sha256", "") or "")
        if current_hash:
            urdf_hashes.add(current_hash)
        if expected_urdf_sha256 and current_hash != expected_urdf_sha256:
            failures.append(
                f"{label}: URDF hash {current_hash!r} != "
                f"{expected_urdf_sha256!r}"
            )

        pixels = np.asarray(record.get("image_points_px", []), dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1:] != (2,):
            failures.append(f"{label}: invalid image_points_px")
        else:
            all_pixels.append(pixels)

    if len(urdf_hashes) > 1:
        failures.append(f"mixed URDF hashes: {sorted(urdf_hashes)}")

    coverage: dict[str, float] = {}
    if all_pixels:
        pixels = np.concatenate(all_pixels, axis=0)
        coverage = {
            "min_x_ratio": float(np.min(pixels[:, 0]) / width),
            "max_x_ratio": float(np.max(pixels[:, 0]) / width),
            "min_y_ratio": float(np.min(pixels[:, 1]) / height),
            "max_y_ratio": float(np.max(pixels[:, 1]) / height),
        }
        if (
            coverage["min_x_ratio"] > 0.15
            or coverage["max_x_ratio"] < 0.85
            or coverage["min_y_ratio"] > 0.15
            or coverage["max_y_ratio"] < 0.85
        ):
            warnings.append(
                "chessboard observations do not cover enough of the image edges"
            )

    # No sensor in the current capture stack observes torso pose in world.
    warnings.append(
        "torso_link-to-board rigidity cannot be verified from arm JointState alone"
    )
    return {
        "sample_count": len(records),
        "expected": {
            "camera_serial": expected_serial,
            "image_size": [width, height],
            "base_frame": expected_base_frame,
            "hand_frame": expected_hand_frame,
            "urdf_sha256": expected_urdf_sha256,
        },
        "thresholds": asdict(thresholds),
        "image_coverage": coverage,
        "urdf_hashes": sorted(urdf_hashes),
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _stability_stats(deltas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    translations = [float(delta["translation_norm_m"]) for delta in deltas]
    rotations = [float(delta["rotation_deg"]) for delta in deltas]
    return {
        "count": len(deltas),
        "translation_m": {
            "mean": float(np.mean(translations)) if translations else None,
            "p95": _percentile(translations, 95.0),
            "max": max(translations, default=None),
        },
        "rotation_deg": {
            "mean": float(np.mean(rotations)) if rotations else None,
            "p95": _percentile(rotations, 95.0),
            "max": max(rotations, default=None),
        },
        "deltas": list(deltas),
    }


def subset_stability_report(
    records: Sequence[dict[str, Any]],
    *,
    mode: str,
    method: int,
    reference: np.ndarray,
    thresholds: HandEyeThresholds,
) -> dict[str, Any]:
    count = len(records)
    split_deltas: list[dict[str, Any]] = []
    midpoint = count // 2
    split_sets = (
        ("chronological_first", range(0, midpoint)),
        ("chronological_second", range(midpoint, count)),
        ("interleaved_odd", range(0, count, 2)),
        ("interleaved_even", range(1, count, 2)),
    )
    for label, indices in split_sets:
        subset = [records[index] for index in indices]
        if len(subset) >= 3:
            candidate = solve_external(subset, mode, method)
            delta = transform_delta(reference, candidate)
            delta["subset"] = label
            split_deltas.append(delta)

    subset_size = max(8, int(math.ceil(count * thresholds.subset_fraction)))
    subset_size = min(count, subset_size)
    rng = random.Random(thresholds.random_seed)
    random_deltas: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while (
        len(random_deltas) < thresholds.subset_trials
        and attempts < thresholds.subset_trials * 20
    ):
        attempts += 1
        indices = tuple(sorted(rng.sample(range(count), subset_size)))
        if indices in seen or len(indices) == count:
            continue
        seen.add(indices)
        subset = [records[index] for index in indices]
        candidate = solve_external(subset, mode, method)
        delta = transform_delta(reference, candidate)
        delta["sample_indices"] = [index + 1 for index in indices]
        random_deltas.append(delta)
    return {
        "split_half": _stability_stats(split_deltas),
        "random_subsets": {
            **_stability_stats(random_deltas),
            "subset_size": subset_size,
            "requested_trials": thresholds.subset_trials,
            "random_seed": thresholds.random_seed,
        },
    }


def strict_handeye_report(
    records: Sequence[dict[str, Any]],
    *,
    mode: str,
    primary_method: str,
    capture_thresholds: CaptureThresholds,
    handeye_thresholds: HandEyeThresholds,
    expected_serial: str,
    expected_size: tuple[int, int],
    expected_base_frame: str,
    expected_hand_frame: str,
    expected_urdf_sha256: str = "",
    baseline_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve and return a strict deployment report."""
    if len(records) < handeye_thresholds.min_samples:
        raise ValueError(
            f"strict hand-eye needs {handeye_thresholds.min_samples} samples, "
            f"got {len(records)}"
        )
    if primary_method not in HAND_EYE_METHODS:
        raise ValueError(f"unknown hand-eye method: {primary_method}")

    preflight = validate_capture_preflight(
        records,
        thresholds=capture_thresholds,
        expected_serial=expected_serial,
        expected_size=expected_size,
        expected_base_frame=expected_base_frame,
        expected_hand_frame=expected_hand_frame,
        expected_urdf_sha256=expected_urdf_sha256,
    )
    primary = solve_external(
        records,
        mode,
        HAND_EYE_METHODS[primary_method],
    )
    closure = closure_report(records, mode, primary)

    methods: dict[str, Any] = {}
    for name in ("park", "horaud", "tsai"):
        candidate = solve_external(records, mode, HAND_EYE_METHODS[name])
        methods[name] = {
            "transform": transform_to_record(candidate),
            "delta_from_primary": transform_delta(primary, candidate),
        }

    stability = subset_stability_report(
        records,
        mode=mode,
        method=HAND_EYE_METHODS[primary_method],
        reference=primary,
        thresholds=handeye_thresholds,
    )
    baseline = (
        None
        if baseline_transform is None
        else {
            "transform": transform_to_record(baseline_transform),
            "delta_from_baseline": transform_delta(baseline_transform, primary),
        }
    )

    failures = list(preflight["failures"])
    pair_translation = float(closure["pairwise_translation_m"]["max"])
    pair_rotation = float(closure["pairwise_rotation_deg"]["max"])
    if pair_translation > handeye_thresholds.closure_translation_max_m:
        failures.append(
            f"pairwise closure translation {pair_translation:.6f}m > "
            f"{handeye_thresholds.closure_translation_max_m:.6f}m"
        )
    if pair_rotation > handeye_thresholds.closure_rotation_max_deg:
        failures.append(
            f"pairwise closure rotation {pair_rotation:.6f}deg > "
            f"{handeye_thresholds.closure_rotation_max_deg:.6f}deg"
        )

    for name, entry in methods.items():
        if name == primary_method:
            continue
        delta = entry["delta_from_primary"]
        if (
            float(delta["translation_norm_m"])
            > handeye_thresholds.method_translation_max_m
        ):
            failures.append(f"{name} translation disagrees with {primary_method}")
        if float(delta["rotation_deg"]) > handeye_thresholds.method_rotation_max_deg:
            failures.append(f"{name} rotation disagrees with {primary_method}")

    split = stability["split_half"]
    if (
        split["translation_m"]["max"] is None
        or split["translation_m"]["max"]
        > handeye_thresholds.split_translation_max_m
    ):
        failures.append("split-half translation stability failed")
    if (
        split["rotation_deg"]["max"] is None
        or split["rotation_deg"]["max"]
        > handeye_thresholds.split_rotation_max_deg
    ):
        failures.append("split-half rotation stability failed")

    subsets = stability["random_subsets"]
    if (
        subsets["translation_m"]["p95"] is None
        or subsets["translation_m"]["p95"]
        > handeye_thresholds.subset_translation_p95_m
        or subsets["translation_m"]["max"]
        > handeye_thresholds.subset_translation_max_m
    ):
        failures.append("random-subset translation stability failed")
    if (
        subsets["rotation_deg"]["p95"] is None
        or subsets["rotation_deg"]["p95"]
        > handeye_thresholds.subset_rotation_p95_deg
        or subsets["rotation_deg"]["max"]
        > handeye_thresholds.subset_rotation_max_deg
    ):
        failures.append("random-subset rotation stability failed")

    if baseline is not None:
        baseline_delta = baseline["delta_from_baseline"]
        if (
            float(baseline_delta["translation_norm_m"])
            > handeye_thresholds.baseline_translation_max_m
        ):
            failures.append("cross-session baseline translation failed")
        if (
            float(baseline_delta["rotation_deg"])
            > handeye_thresholds.baseline_rotation_max_deg
        ):
            failures.append("cross-session baseline rotation failed")

    report = {
        "schema": "handeye_calib.strict_handeye_report/v1",
        "primary_method": primary_method,
        "primary_transform": transform_to_record(primary),
        "capture_preflight": preflight,
        "closure": closure,
        "method_consensus": methods,
        "subset_stability": stability,
        "cross_session_baseline": baseline,
        "thresholds": asdict(handeye_thresholds),
        "deployable": not failures,
        "failures": failures,
        "warnings": list(preflight["warnings"]),
    }
    return primary, report


def _calibrate_camera(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    fix_k3: bool,
) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    return cv2.calibrateCamera(
        list(object_points),
        list(image_points),
        image_size,
        None,
        None,
        flags=flags,
    )


def calibrate_intrinsics_strict(
    *,
    image_dir: Path,
    output_root: Path,
    cols: int,
    rows: int,
    square_mm: float,
    gamma: float,
    camera_serial: str,
    camera_model: str,
    stream_name: str,
    thresholds: IntrinsicsThresholds,
    fix_k3: bool,
    random_seed: int = 20260813,
) -> tuple[Path, dict[str, Any]]:
    """Calibrate intrinsics and only emit a manifest when strict gates pass."""
    # Import lazily so hand-eye-only solving does not require Flask, which is
    # only needed by the interactive camera preview.
    from .chessboard import find_chessboard_corners

    images = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for path in image_dir.glob(suffix)
    )
    if not images:
        raise ValueError(f"no calibration images in {image_dir}")
    objp = build_object_points(cols, rows, square_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used: list[Path] = []
    rejected: list[dict[str, str]] = []
    image_size: tuple[int, int] | None = None
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(path), "reason": "read_failed"})
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected.append({"image": str(path), "reason": "size_mismatch"})
            continue
        corners, method = find_chessboard_corners(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            (cols, rows),
            gamma,
        )
        if corners is None:
            rejected.append({"image": str(path), "reason": "corners_not_found"})
            continue
        object_points.append(objp.copy())
        image_points.append(corners.astype(np.float32))
        used.append(path)
        rejected.append({"image": str(path), "reason": f"used:{method}"})
    if image_size is None:
        raise ValueError("no readable calibration image")
    if len(used) < thresholds.min_images:
        raise ValueError(
            f"strict intrinsics needs {thresholds.min_images} images, got {len(used)}"
        )

    rms, matrix, distortion, rvecs, tvecs = _calibrate_camera(
        object_points,
        image_points,
        image_size,
        fix_k3=fix_k3,
    )
    view_rms: list[float] = []
    for obj, corners, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, matrix, distortion
        )
        residual = (
            np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            - np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        )
        view_rms.append(
            float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        )

    pixels = np.concatenate(
        [np.asarray(corners).reshape(-1, 2) for corners in image_points],
        axis=0,
    )
    width, height = image_size
    coverage = {
        "min_x_ratio": float(np.min(pixels[:, 0]) / width),
        "max_x_ratio": float(np.max(pixels[:, 0]) / width),
        "min_y_ratio": float(np.min(pixels[:, 1]) / height),
        "max_y_ratio": float(np.max(pixels[:, 1]) / height),
    }

    rng = random.Random(random_seed)
    subset_size = min(
        len(used),
        max(10, int(math.ceil(len(used) * thresholds.subset_fraction))),
    )
    focal_delta: list[float] = []
    principal_delta: list[float] = []
    subset_failures: list[str] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while (
        len(focal_delta) < thresholds.subset_trials
        and attempts < thresholds.subset_trials * 20
    ):
        attempts += 1
        indices = tuple(sorted(rng.sample(range(len(used)), subset_size)))
        if indices in seen or len(indices) == len(used):
            continue
        seen.add(indices)
        try:
            _, candidate, _, _, _ = _calibrate_camera(
                [object_points[index] for index in indices],
                [image_points[index] for index in indices],
                image_size,
                fix_k3=fix_k3,
            )
        except cv2.error as exc:
            subset_failures.append(str(exc))
            continue
        focal_delta.append(
            max(
                abs(float(candidate[0, 0] - matrix[0, 0])) / matrix[0, 0],
                abs(float(candidate[1, 1] - matrix[1, 1])) / matrix[1, 1],
            )
        )
        principal_delta.append(
            float(
                np.linalg.norm(
                    candidate[[0, 1], [2, 2]] - matrix[[0, 1], [2, 2]]
                )
            )
        )

    failures: list[str] = []
    if float(rms) > thresholds.max_rms_px:
        failures.append(
            f"global RMS {float(rms):.6f}px > {thresholds.max_rms_px:.6f}px"
        )
    if max(view_rms) > thresholds.max_view_rms_px:
        failures.append(
            f"view RMS max {max(view_rms):.6f}px > "
            f"{thresholds.max_view_rms_px:.6f}px"
        )
    if (
        coverage["min_x_ratio"] > thresholds.min_edge_coverage_ratio
        or coverage["min_y_ratio"] > thresholds.min_edge_coverage_ratio
        or coverage["max_x_ratio"] < thresholds.max_edge_coverage_ratio
        or coverage["max_y_ratio"] < thresholds.max_edge_coverage_ratio
    ):
        failures.append("calibration corners do not cover all image edges")
    focal_p95 = _percentile(focal_delta, 95.0)
    principal_p95 = _percentile(principal_delta, 95.0)
    if (
        focal_p95 is None
        or focal_p95 > thresholds.subset_focal_p95_ratio
    ):
        failures.append("intrinsics focal-length subset stability failed")
    if (
        principal_p95 is None
        or principal_p95 > thresholds.subset_principal_p95_px
    ):
        failures.append("intrinsics principal-point subset stability failed")
    if subset_failures:
        failures.append("one or more intrinsics subset solves failed")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"camera_intrinsics_strict_{stamp}_npy"
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "camera_matrix.npy", matrix)
    np.save(output_dir / "dist_coeffs.npy", distortion)
    report = {
        "schema": "handeye_calib.strict_intrinsics_report/v1",
        "image_dir": str(image_dir.resolve()),
        "image_size": [width, height],
        "pattern": {"cols": cols, "rows": rows, "square_mm": square_mm},
        "camera": {
            "serial": camera_serial,
            "model": camera_model,
            "stream": stream_name,
        },
        "used_images": [str(path) for path in used],
        "rejected": [
            entry for entry in rejected if not entry["reason"].startswith("used:")
        ],
        "global_rms_px": float(rms),
        "view_rms_px": {
            "mean": float(np.mean(view_rms)),
            "p95": _percentile(view_rms, 95.0),
            "max": max(view_rms),
        },
        "coverage": coverage,
        "camera_matrix": matrix.astype(float).tolist(),
        "dist_coeffs": distortion.reshape(-1).astype(float).tolist(),
        "subset_stability": {
            "trials": len(focal_delta),
            "subset_size": subset_size,
            "focal_delta_ratio_p95": focal_p95,
            "focal_delta_ratio_max": max(focal_delta, default=None),
            "principal_delta_px_p95": principal_p95,
            "principal_delta_px_max": max(principal_delta, default=None),
            "failures": subset_failures,
        },
        "thresholds": asdict(thresholds),
        "deployable": not failures,
        "failures": failures,
    }
    report_path = output_dir / "strict_intrinsics_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not failures:
        manifest_path = write_intrinsics_manifest(
            output_dir,
            matrix,
            distortion,
            image_size,
            camera_serial=camera_serial,
            camera_model=camera_model,
            stream_name=stream_name,
        )
        report["manifest_path"] = str(manifest_path.resolve())
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return output_dir, report


def load_handeye_transform(path: Path, mode: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "cam2hand" if normalize_mode(mode) == MODE_EYE_IN_HAND else "cam2base"
    try:
        transform = np.asarray(payload[key]["transform"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} does not contain {key}.transform") from exc
    if transform.shape != (4, 4):
        raise ValueError(f"{path} {key}.transform is not 4x4")
    return transform
