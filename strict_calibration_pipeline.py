#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict end-to-end calibration solver and deployment gate.

Capture remains interactive and is intentionally handled by the existing
``chessboard.py`` and ``capture_handeye.py`` tools.  This command owns the
high-risk steps: solving intrinsics, solving hand-eye, comparing independent
sessions, and deciding whether an output is deployable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from handeye_calib.solver import (
    MODE_EYE_IN_HAND,
    load_capture_records,
    normalize_mode,
)
from handeye_calib.strict_pipeline import (
    CaptureThresholds,
    HandEyeThresholds,
    IntrinsicsThresholds,
    calibrate_intrinsics_strict,
    load_handeye_transform,
    strict_handeye_report,
    transform_delta,
)
from handeye_calib.transforms import invert_transform, transform_to_record


PROJECT_ROOT = Path(__file__).resolve().parent


def _intrinsics_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "intrinsics",
        help="Solve camera intrinsics with coverage and subset-stability gates",
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--camera-model", default="RealSense D435")
    parser.add_argument("--stream-name", default="color")
    parser.add_argument("--fix-k3", action="store_true")
    parser.add_argument("--min-images", type=int, default=24)
    parser.add_argument("--max-rms-px", type=float, default=0.35)
    parser.add_argument("--max-view-rms-px", type=float, default=0.50)
    parser.add_argument("--coverage-min-ratio", type=float, default=0.08)
    parser.add_argument("--coverage-max-ratio", type=float, default=0.92)
    parser.add_argument("--subset-trials", type=int, default=30)
    parser.add_argument("--subset-focal-p95-ratio", type=float, default=0.005)
    parser.add_argument("--subset-principal-p95-px", type=float, default=3.0)


def _handeye_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "handeye",
        help="Solve hand-eye with strict capture and stability gates",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--mode", default="eye-in-hand")
    parser.add_argument("--method", choices=("park", "horaud", "tsai"), default="park")
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--base-frame", default="torso_link")
    parser.add_argument("--hand-frame", default="right_wrist_yaw_link")
    parser.add_argument("--urdf-sha256", default="")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--allow-legacy-intrinsics", action="store_true")
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=0.30)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.03)
    parser.add_argument("--max-state-age-sec", type=float, default=0.10)
    parser.add_argument(
        "--max-joint-velocity-rad-s",
        type=float,
        default=0.01,
        help="Maximum arm velocity; 0 disables this gate.",
    )
    parser.add_argument("--closure-translation-max-mm", type=float, default=10.0)
    parser.add_argument("--closure-rotation-max-deg", type=float, default=1.0)
    parser.add_argument("--method-translation-max-mm", type=float, default=5.0)
    parser.add_argument("--method-rotation-max-deg", type=float, default=0.50)
    parser.add_argument("--split-translation-max-mm", type=float, default=5.0)
    parser.add_argument("--split-rotation-max-deg", type=float, default=0.50)
    parser.add_argument("--subset-translation-p95-mm", type=float, default=5.0)
    parser.add_argument("--subset-rotation-p95-deg", type=float, default=0.50)
    parser.add_argument("--subset-translation-max-mm", type=float, default=10.0)
    parser.add_argument("--subset-rotation-max-deg", type=float, default=1.0)
    parser.add_argument("--baseline-translation-max-mm", type=float, default=5.0)
    parser.add_argument("--baseline-rotation-max-deg", type=float, default=0.50)
    parser.add_argument("--subset-trials", type=int, default=100)
    parser.add_argument("--subset-fraction", type=float, default=0.65)
    parser.add_argument("--random-seed", type=int, default=20260813)


def _compare_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Compare two independently captured hand-eye results",
    )
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--mode", default="eye-in-hand")
    parser.add_argument("--max-translation-mm", type=float, default=5.0)
    parser.add_argument("--max-rotation-deg", type=float, default=0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict RealSense + H2 calibration pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _intrinsics_parser(subparsers)
    _handeye_parser(subparsers)
    _compare_parser(subparsers)
    args = parser.parse_args()
    if getattr(args, "cols", 2) < 2 or getattr(args, "rows", 2) < 2:
        parser.error("--cols/--rows must be >= 2")
    for name, value in vars(args).items():
        if name.startswith("max_") and isinstance(value, (int, float)) and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def run_intrinsics(args: argparse.Namespace) -> int:
    thresholds = IntrinsicsThresholds(
        min_images=args.min_images,
        max_rms_px=args.max_rms_px,
        max_view_rms_px=args.max_view_rms_px,
        min_edge_coverage_ratio=args.coverage_min_ratio,
        max_edge_coverage_ratio=args.coverage_max_ratio,
        subset_trials=args.subset_trials,
        subset_focal_p95_ratio=args.subset_focal_p95_ratio,
        subset_principal_p95_px=args.subset_principal_p95_px,
    )
    output_dir, report = calibrate_intrinsics_strict(
        image_dir=Path(args.image_dir),
        output_root=Path(args.output_root),
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        gamma=args.gamma,
        camera_serial=args.camera_serial,
        camera_model=args.camera_model,
        stream_name=args.stream_name,
        thresholds=thresholds,
        fix_k3=args.fix_k3,
    )
    status = "PASS" if report["deployable"] else "FAIL"
    print(f"[INTRINSICS][{status}] {output_dir.resolve()}")
    for failure in report["failures"]:
        print(f"[INTRINSICS][FAIL] {failure}", file=sys.stderr)
    return 0 if report["deployable"] else 2


def _handeye_thresholds(args: argparse.Namespace) -> HandEyeThresholds:
    millimetres = 0.001
    return HandEyeThresholds(
        min_samples=args.min_samples,
        closure_translation_max_m=args.closure_translation_max_mm * millimetres,
        closure_rotation_max_deg=args.closure_rotation_max_deg,
        method_translation_max_m=args.method_translation_max_mm * millimetres,
        method_rotation_max_deg=args.method_rotation_max_deg,
        split_translation_max_m=args.split_translation_max_mm * millimetres,
        split_rotation_max_deg=args.split_rotation_max_deg,
        subset_translation_p95_m=args.subset_translation_p95_mm * millimetres,
        subset_rotation_p95_deg=args.subset_rotation_p95_deg,
        subset_translation_max_m=args.subset_translation_max_mm * millimetres,
        subset_rotation_max_deg=args.subset_rotation_max_deg,
        baseline_translation_max_m=args.baseline_translation_max_mm * millimetres,
        baseline_rotation_max_deg=args.baseline_rotation_max_deg,
        subset_trials=args.subset_trials,
        subset_fraction=args.subset_fraction,
        random_seed=args.random_seed,
    )


def run_handeye(args: argparse.Namespace) -> int:
    mode = normalize_mode(args.mode)
    records = load_capture_records(Path(args.data_dir))
    capture_thresholds = CaptureThresholds(
        max_reprojection_rms_px=args.max_reprojection_rms_px,
        max_sync_delta_sec=args.max_sync_delta_sec,
        max_state_age_sec=args.max_state_age_sec,
        max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
        require_intrinsics_manifest=not args.allow_legacy_intrinsics,
    )
    baseline = (
        None
        if not args.baseline_json
        else load_handeye_transform(Path(args.baseline_json), mode)
    )
    primary, report = strict_handeye_report(
        records,
        mode=mode,
        primary_method=args.method,
        capture_thresholds=capture_thresholds,
        handeye_thresholds=_handeye_thresholds(args),
        expected_serial=args.camera_serial,
        expected_size=(args.width, args.height),
        expected_base_frame=args.base_frame,
        expected_hand_frame=args.hand_frame,
        expected_urdf_sha256=args.urdf_sha256,
        baseline_transform=baseline,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{mode}_strict_{stamp}.json"
    payload: dict = {
        "schema": "handeye_calib.strict_result/v1",
        "calibration_method": mode,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "deployable": bool(report["deployable"]),
        "strict_quality": report,
        "context": {
            "camera_serial": args.camera_serial,
            "image_size": [args.width, args.height],
            "base_frame": args.base_frame,
            "hand_frame": args.hand_frame,
            "urdf_sha256": args.urdf_sha256,
            "data_dir": str(Path(args.data_dir).resolve()),
        },
    }
    if mode == MODE_EYE_IN_HAND:
        payload["cam2hand"] = transform_to_record(primary)
        payload["hand2cam"] = transform_to_record(invert_transform(primary))
    else:
        payload["cam2base"] = transform_to_record(primary)
        payload["base2cam"] = transform_to_record(invert_transform(primary))
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = "PASS" if report["deployable"] else "FAIL"
    print(f"[HANDEYE][{status}] {output_path.resolve()}")
    for failure in report["failures"]:
        print(f"[HANDEYE][FAIL] {failure}", file=sys.stderr)
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"[HANDEYE][WARN] {warning}", file=sys.stderr)
    return 0 if report["deployable"] else 2


def run_compare(args: argparse.Namespace) -> int:
    mode = normalize_mode(args.mode)
    first = load_handeye_transform(Path(args.first), mode)
    second = load_handeye_transform(Path(args.second), mode)
    delta = transform_delta(first, second)
    payload = {
        "first": str(Path(args.first).resolve()),
        "second": str(Path(args.second).resolve()),
        "delta": delta,
        "thresholds": {
            "translation_m": args.max_translation_mm * 0.001,
            "rotation_deg": args.max_rotation_deg,
        },
    }
    payload["pass"] = (
        float(delta["translation_norm_m"]) <= args.max_translation_mm * 0.001
        and float(delta["rotation_deg"]) <= args.max_rotation_deg
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["pass"] else 2


def main() -> int:
    args = parse_args()
    if args.command == "intrinsics":
        return run_intrinsics(args)
    if args.command == "handeye":
        return run_handeye(args)
    if args.command == "compare":
        return run_compare(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
