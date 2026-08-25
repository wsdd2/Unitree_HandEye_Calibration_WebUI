# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .transforms import average_transforms, invert_transform, transform_to_record


MODE_EYE_IN_HAND = "eye_in_hand"
MODE_EYE_TO_HAND = "eye_to_hand"
VALID_MODES = (MODE_EYE_IN_HAND, MODE_EYE_TO_HAND)


def normalize_mode(mode: str) -> str:
    value = mode.strip().lower().replace("-", "_")
    if value in ("hand_in_eye", "eye_in_hand", "in_hand"):
        return MODE_EYE_IN_HAND
    if value in ("hand_to_eye", "eye_to_hand", "to_hand", "fixed_camera"):
        return MODE_EYE_TO_HAND
    raise ValueError(f"不支持的手眼模式: {mode}")


def opencv_method_from_name(name: str) -> int:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    key = name.strip().lower()
    if key not in methods:
        raise ValueError(f"不支持的 OpenCV 手眼方法: {name}")
    return int(methods[key])


def load_capture_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        if not _is_capture_record(record):
            continue
        record["_json_path"] = str(path)
        records.append(record)
    if records:
        validate_capture_records(records)
    return records


def record_transform(record: dict[str, Any], key: str) -> np.ndarray:
    try:
        transform = np.asarray(record[key]["transform"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{_record_name(record)} 的 {key} 缺少有效 transform") from exc
    _validate_se3(transform, f"{_record_name(record)}.{key}")
    return transform


def _is_capture_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    required = ("mode", "hand_pose_input", "hand2base", "base2hand", "target2cam")
    return all(key in record for key in required)


def _record_name(record: dict[str, Any]) -> str:
    return str(record.get("_json_path") or record.get("image") or "<memory sample>")


def _validate_se3(transform: np.ndarray, label: str, atol: float = 1e-6) -> None:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} 必须是有限数值的 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError(f"{label} 的齐次矩阵末行无效")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{label} 的旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{label} 的旋转矩阵行列式不为 +1")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _camera_serial(record: dict[str, Any]) -> str:
    info = record.get("camera_info", {})
    if not isinstance(info, dict):
        raise ValueError(f"{_record_name(record)} 的 camera_info 必须是对象")
    for key in ("selected_serial", "serial", "requested_serial"):
        value = info.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _intrinsics_identifier(record: dict[str, Any]) -> str:
    intrinsics = record.get("camera_intrinsics")
    if intrinsics is None:
        # 旧采集文件可能把内参嵌在 camera_info 中。
        camera_info = record.get("camera_info", {})
        intrinsics = camera_info.get("intrinsics", {}) if isinstance(camera_info, dict) else {}
    return _canonical(intrinsics)


def _consistency_signature(record: dict[str, Any]) -> dict[str, str]:
    pose = record.get("hand_pose_input", {})
    board = record.get("board", {})
    if not isinstance(pose, dict) or not isinstance(board, dict):
        raise ValueError(
            f"{_record_name(record)} 的 hand_pose_input 和 board 必须是对象"
        )
    board_id = {
        "inner_corners_cols": board.get("inner_corners_cols"),
        "inner_corners_rows": board.get("inner_corners_rows"),
        "square_m": board.get(
            "square_m",
            None if board.get("square_mm") is None else float(board["square_mm"]) / 1000.0,
        ),
    }
    return {
        "mode": normalize_mode(str(record.get("mode", ""))),
        "base_frame": str(pose.get("base_frame_name", "base")),
        "hand_frame": str(pose.get("frame_name", "")),
        "camera_serial": _camera_serial(record),
        "board": _canonical(board_id),
        "intrinsics": _intrinsics_identifier(record),
    }


def validate_capture_records(
    records: list[dict[str, Any]],
    expected_mode: str | None = None,
) -> None:
    if not records:
        return
    if not _is_capture_record(records[0]):
        raise ValueError("第 1 个输入不是采集 JSON schema")
    reference = _consistency_signature(records[0])
    if expected_mode is not None and reference["mode"] != normalize_mode(expected_mode):
        raise ValueError(
            f"样本模式 {reference['mode']} 与请求模式 {normalize_mode(expected_mode)} 不一致"
        )
    for index, record in enumerate(records):
        if not _is_capture_record(record):
            raise ValueError(f"第 {index + 1} 个输入不是采集 JSON schema")
        signature = _consistency_signature(record)
        for field, expected in reference.items():
            if signature[field] != expected:
                raise ValueError(
                    f"样本上下文不一致: {_record_name(record)} 的 {field}="
                    f"{signature[field]!r}，预期 {expected!r}"
                )
        hand2base = record_transform(record, "hand2base")
        base2hand = record_transform(record, "base2hand")
        record_transform(record, "target2cam")
        if not np.allclose(base2hand, invert_transform(hand2base), atol=1e-6, rtol=0.0):
            raise ValueError(
                f"{_record_name(record)} 的 base2hand 不是 hand2base 的逆矩阵"
            )


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    value = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(value))


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(_rotation_angle_rad(left[:3, :3].T @ right[:3, :3])))


def _check_motion_excitation(transforms: list[np.ndarray]) -> dict[str, Any]:
    reference_inv = invert_transform(transforms[0])
    rotation_vectors = []
    for transform in transforms[1:]:
        delta = reference_inv @ transform
        rvec, _ = cv2.Rodrigues(delta[:3, :3])
        rotation_vectors.append(rvec.reshape(3))
    matrix = np.asarray(rotation_vectors, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size < 2:
        singular_values = np.pad(singular_values, (0, 2 - singular_values.size))
    max_relative_rotation_deg = max(
        (float(np.degrees(np.linalg.norm(value))) for value in rotation_vectors),
        default=0.0,
    )
    translation_extent_m = float(
        np.linalg.norm(
            np.ptp(
                np.asarray([transform[:3, 3] for transform in transforms], dtype=np.float64),
                axis=0,
            )
        )
    )
    report = {
        "max_relative_rotation_deg": max_relative_rotation_deg,
        "rotation_excitation_singular_values_rad": singular_values.astype(float).tolist(),
        "translation_extent_m": translation_extent_m,
    }
    if max_relative_rotation_deg < 5.0:
        raise ValueError(
            f"运动激励退化：最大相对旋转仅 {max_relative_rotation_deg:.3f}°，至少需要 5°"
        )
    if singular_values[1] < np.deg2rad(0.5) or singular_values[1] / max(
        singular_values[0], 1e-12
    ) < 0.01:
        raise ValueError("运动激励退化：相对旋转轴近似共线，请增加不同轴向的转动")
    return report


def _record_context(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    first = records[0]
    return {
        "camera_info": first.get("camera_info", {}),
        "robot_context": first.get("robot_context", {}),
        "board": first.get("board", {}),
        "base_frame": first.get("hand_pose_input", {}).get("base_frame_name", "base"),
        "hand_frame": first.get("hand_pose_input", {}).get("frame_name", ""),
        "pose_units": {
            "translation": first.get("hand_pose_input", {}).get("translation_unit", ""),
            "rotation": first.get("hand_pose_input", {}).get("rotation_unit", ""),
            "euler_order": first.get("hand_pose_input", {}).get("euler_order", ""),
        },
    }


def _translation_spread(transforms: list[np.ndarray]) -> list[float]:
    if not transforms:
        return [0.0, 0.0, 0.0]
    t = np.asarray([x[:3, 3] for x in transforms], dtype=np.float64)
    return np.std(t, axis=0).astype(float).tolist()


def _rotation_spread_deg(transforms: list[np.ndarray], mean_transform: np.ndarray) -> list[float]:
    out = []
    mean_inv = invert_transform(mean_transform)
    for transform in transforms:
        delta = mean_inv @ transform
        rvec, _ = cv2.Rodrigues(delta[:3, :3])
        out.append(float(np.linalg.norm(rvec) * 180.0 / np.pi))
    return out


def _save_common_arrays(
    npy_dir: Path,
    records: list[dict[str, Any]],
    T_hand2base: list[np.ndarray],
    T_base2hand: list[np.ndarray],
    T_target2cam: list[np.ndarray],
) -> None:
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / "T_hand2base.npy", np.asarray(T_hand2base, dtype=np.float64))
    np.save(npy_dir / "T_base2hand.npy", np.asarray(T_base2hand, dtype=np.float64))
    np.save(npy_dir / "T_target2cam.npy", np.asarray(T_target2cam, dtype=np.float64))
    np.save(npy_dir / "R_hand2base.npy", np.asarray([t[:3, :3] for t in T_hand2base], dtype=np.float64))
    np.save(npy_dir / "t_hand2base_m.npy", np.asarray([t[:3, 3] for t in T_hand2base], dtype=np.float64).reshape(-1, 3, 1))
    np.save(npy_dir / "R_target2cam.npy", np.asarray([t[:3, :3] for t in T_target2cam], dtype=np.float64))
    np.save(npy_dir / "t_target2cam_m.npy", np.asarray([t[:3, 3] for t in T_target2cam], dtype=np.float64).reshape(-1, 3, 1))
    np.save(npy_dir / "image_names.npy", np.asarray([r.get("image", "") for r in records]))
    np.save(npy_dir / "json_paths.npy", np.asarray([r.get("_json_path", "") for r in records]))


def _calibrate_external(
    records: list[dict[str, Any]],
    mode: str,
    method: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    T_hand2base = [record_transform(record, "hand2base") for record in records]
    T_base2hand = [record_transform(record, "base2hand") for record in records]
    T_target2cam = [record_transform(record, "target2cam") for record in records]
    R_target2cam = [transform[:3, :3] for transform in T_target2cam]
    t_target2cam = [transform[:3, 3].reshape(3, 1) for transform in T_target2cam]
    gripper = T_hand2base if mode == MODE_EYE_IN_HAND else T_base2hand
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in gripper],
        [transform[:3, 3].reshape(3, 1) for transform in gripper],
        R_target2cam,
        t_target2cam,
        method=method,
    )
    external = np.eye(4, dtype=np.float64)
    external[:3, :3] = rotation
    external[:3, 3] = translation.reshape(3)
    _validate_se3(external, "calibrateHandEye result", atol=1e-5)
    if mode == MODE_EYE_IN_HAND:
        fixed_each = [
            T_hand2base[index] @ external @ T_target2cam[index]
            for index in range(len(records))
        ]
    else:
        fixed_each = [
            T_base2hand[index] @ external @ T_target2cam[index]
            for index in range(len(records))
        ]
    return external, fixed_each


def _quality_report(
    records: list[dict[str, Any]],
    *,
    mode: str,
    method: int,
    external: np.ndarray,
    fixed_each: list[np.ndarray],
    fixed_name: str,
    excitation: dict[str, Any],
    include_leave_one_out: bool,
) -> dict[str, Any]:
    fixed_mean = average_transforms(fixed_each)
    translation_residuals = [
        float(np.linalg.norm(transform[:3, 3] - fixed_mean[:3, 3]))
        for transform in fixed_each
    ]
    rotation_errors = _rotation_spread_deg(fixed_each, fixed_mean)
    pair_translation_max = 0.0
    pair_rotation_max = 0.0
    for left in range(len(fixed_each)):
        for right in range(left + 1, len(fixed_each)):
            pair_translation_max = max(
                pair_translation_max,
                float(
                    np.linalg.norm(
                        fixed_each[left][:3, 3] - fixed_each[right][:3, 3]
                    )
                ),
            )
            pair_rotation_max = max(
                pair_rotation_max,
                _rotation_delta_deg(fixed_each[left], fixed_each[right]),
            )

    loo_translation: list[float] = []
    loo_rotation: list[float] = []
    loo_failures: list[str] = []
    if include_leave_one_out and len(records) >= 4:
        for index in range(len(records)):
            subset = records[:index] + records[index + 1 :]
            try:
                candidate, _ = _calibrate_external(subset, mode, method)
                loo_translation.append(
                    float(np.linalg.norm(candidate[:3, 3] - external[:3, 3]))
                )
                loo_rotation.append(_rotation_delta_deg(candidate, external))
            except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
                loo_failures.append(f"leave_out_{index + 1}: {exc}")

    translation_mean = float(np.mean(translation_residuals))
    translation_max = float(np.max(translation_residuals))
    rotation_mean = float(np.mean(rotation_errors))
    rotation_max = float(np.max(rotation_errors))
    loo_translation_mean = float(np.mean(loo_translation)) if loo_translation else None
    loo_translation_max = float(np.max(loo_translation)) if loo_translation else None
    loo_rotation_mean = float(np.mean(loo_rotation)) if loo_rotation else None
    loo_rotation_max = float(np.max(loo_rotation)) if loo_rotation else None

    consensus: dict[str, Any] = {
        "reference_method": int(cv2.CALIB_HAND_EYE_PARK),
        "translation_delta_m": None,
        "rotation_delta_deg": None,
        "error": None,
    }
    if method != cv2.CALIB_HAND_EYE_PARK:
        try:
            park_external, _ = _calibrate_external(
                records,
                mode,
                cv2.CALIB_HAND_EYE_PARK,
            )
            consensus["translation_delta_m"] = float(
                np.linalg.norm(park_external[:3, 3] - external[:3, 3])
            )
            consensus["rotation_delta_deg"] = _rotation_delta_deg(
                park_external,
                external,
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            consensus["error"] = str(exc)

    warnings = []
    if translation_mean > 0.02 or translation_max > 0.05:
        warnings.append("固定变换平移闭环残差偏大")
    if rotation_mean > 2.0 or rotation_max > 5.0:
        warnings.append("固定变换旋转闭环残差偏大")
    if loo_translation_max is not None and loo_translation_max > 0.03:
        warnings.append("留一法外参平移稳定性偏低")
    if loo_rotation_max is not None and loo_rotation_max > 3.0:
        warnings.append("留一法外参旋转稳定性偏低")
    if loo_failures:
        warnings.append("部分留一法求解失败")
    if consensus["error"]:
        warnings.append("无法完成与 Park 方法的交叉校验")
    if (
        consensus["translation_delta_m"] is not None
        and consensus["translation_delta_m"] > 0.01
    ):
        warnings.append("所选方法与 Park 的外参平移差异超过 1 cm")
    if (
        consensus["rotation_delta_deg"] is not None
        and consensus["rotation_delta_deg"] > 1.0
    ):
        warnings.append("所选方法与 Park 的外参旋转差异超过 1°")

    # 保留旧字段，并增加可直接用于自动验收的汇总字段。
    return {
        "fixed_transform": fixed_name,
        "translation_std_m_xyz": _translation_spread(fixed_each),
        "rotation_error_deg_each": rotation_errors,
        "translation_residual_norm_m_each": translation_residuals,
        "translation_residual_norm_mean_m": translation_mean,
        "translation_residual_norm_max_m": translation_max,
        "rotation_error_deg_mean": rotation_mean,
        "rotation_error_deg_max": rotation_max,
        "pairwise_closure_translation_max_m": pair_translation_max,
        "pairwise_closure_rotation_max_deg": pair_rotation_max,
        "pairwise_closure_max": {
            "translation_m": pair_translation_max,
            "rotation_deg": pair_rotation_max,
        },
        "leave_one_out_external_stability": {
            "translation_delta_m_each": loo_translation,
            "translation_delta_mean_m": loo_translation_mean,
            "translation_delta_max_m": loo_translation_max,
            "rotation_delta_deg_each": loo_rotation,
            "rotation_delta_mean_deg": loo_rotation_mean,
            "rotation_delta_max_deg": loo_rotation_max,
            "failures": loo_failures,
        },
        "method_consensus_with_park": consensus,
        "motion_excitation": excitation,
        "pass": not warnings,
        "status": "pass" if not warnings else "warn",
        "warnings": warnings,
    }


def solve_handeye(
    records: list[dict[str, Any]],
    *,
    mode: str,
    output_dir: Path,
    min_samples: int = 8,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
    filtering: dict[str, Any] | None = None,
    include_leave_one_out: bool = True,
) -> Path:
    mode = normalize_mode(mode)
    if len(records) < min_samples:
        raise ValueError(f"样本不足：{len(records)}/{min_samples}")
    validate_capture_records(records, expected_mode=mode)

    T_hand2base = [record_transform(r, "hand2base") for r in records]
    T_base2hand = [record_transform(r, "base2hand") for r in records]
    T_target2cam = [record_transform(r, "target2cam") for r in records]
    excitation = _check_motion_excitation(T_hand2base)
    base_frame = records[0].get("hand_pose_input", {}).get("base_frame_name", "base")

    R_target2cam = [t[:3, :3] for t in T_target2cam]
    t_target2cam = [t[:3, 3].reshape(3, 1) for t in T_target2cam]

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{mode}_{stamp}.json"
    npy_dir = output_dir / f"{mode}_{stamp}_npy"
    _save_common_arrays(npy_dir, records, T_hand2base, T_base2hand, T_target2cam)

    if mode == MODE_EYE_IN_HAND:
        # OpenCV 原生 eye-in-hand：输入 ^baseT_hand 与 ^camT_target，输出 ^handT_cam。
        R_gripper2base = [t[:3, :3] for t in T_hand2base]
        t_gripper2base = [t[:3, 3].reshape(3, 1) for t in T_hand2base]
        R_cam2hand, t_cam2hand = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method,
        )
        T_cam2hand = np.eye(4, dtype=np.float64)
        T_cam2hand[:3, :3] = R_cam2hand
        T_cam2hand[:3, 3] = t_cam2hand.reshape(3)
        T_hand2cam = invert_transform(T_cam2hand)
        fixed_target2base_each = [T_hand2base[i] @ T_cam2hand @ T_target2cam[i] for i in range(len(records))]
        fixed_target2base = average_transforms(fixed_target2base_each)

        np.save(npy_dir / "T_cam2hand.npy", T_cam2hand)
        np.save(npy_dir / "T_hand2cam.npy", T_hand2cam)
        np.save(npy_dir / "T_target2base_each.npy", np.asarray(fixed_target2base_each, dtype=np.float64))
        np.save(npy_dir / "T_target2base_mean.npy", fixed_target2base)

        result_payload = {
            "cam2hand": transform_to_record(T_cam2hand),
            "hand2cam": transform_to_record(T_hand2cam),
            "target2base_mean": transform_to_record(fixed_target2base),
            "quality": _quality_report(
                records,
                mode=mode,
                method=method,
                external=T_cam2hand,
                fixed_each=fixed_target2base_each,
                fixed_name="target2base",
                excitation=excitation,
                include_leave_one_out=include_leave_one_out,
            ),
        }
        convention = {
            "hand2base": "机器人输入位姿解析为 T_base_hand，即点从灵巧手/末端坐标变换到机器人基座坐标。",
            "target2cam": "棋盘格坐标到相机坐标，来自 solvePnP。",
            "cam2hand": "求解结果：相机坐标到灵巧手/末端坐标，适用于 eye-in-hand。",
            "cam2base_runtime": (
                "眼在手上相机相对基准不是固定外参；运行时逐帧计算 "
                "T_base_cam(q)=T_base_hand(q)*T_hand_cam，其中 T_hand_cam 即 cam2hand。"
            ),
        }
    else:
        # 固定相机、标定板在手上：把 ^handT_base 作为 OpenCV 的 gripper2base，输出等价 ^baseT_cam。
        R_gripper2base = [t[:3, :3] for t in T_base2hand]
        t_gripper2base = [t[:3, 3].reshape(3, 1) for t in T_base2hand]
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method,
        )
        T_cam2base = np.eye(4, dtype=np.float64)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.reshape(3)
        T_base2cam = invert_transform(T_cam2base)
        fixed_target2hand_each = [T_base2hand[i] @ T_cam2base @ T_target2cam[i] for i in range(len(records))]
        fixed_target2hand = average_transforms(fixed_target2hand_each)

        np.save(npy_dir / "T_cam2base.npy", T_cam2base)
        np.save(npy_dir / "T_base2cam.npy", T_base2cam)
        np.save(npy_dir / "T_cam2reference.npy", T_cam2base)
        np.save(npy_dir / "T_reference2cam.npy", T_base2cam)
        np.save(npy_dir / "T_target2hand_each.npy", np.asarray(fixed_target2hand_each, dtype=np.float64))
        np.save(npy_dir / "T_target2hand_mean.npy", fixed_target2hand)

        result_payload = {
            "cam2base": transform_to_record(T_cam2base),
            "base2cam": transform_to_record(T_base2cam),
            "cam2reference": transform_to_record(T_cam2base),
            "reference2cam": transform_to_record(T_base2cam),
            "reference_frame": base_frame,
            "target2hand_mean": transform_to_record(fixed_target2hand),
            "quality": _quality_report(
                records,
                mode=mode,
                method=method,
                external=T_cam2base,
                fixed_each=fixed_target2hand_each,
                fixed_name="target2hand",
                excitation=excitation,
                include_leave_one_out=include_leave_one_out,
            ),
        }
        convention = {
            "hand2base": (
                f"机器人输入位姿解析为 T_{base_frame}_hand，"
                f"即点从灵巧手/末端坐标变换到 {base_frame} 坐标。"
            ),
            "target2cam": "棋盘格坐标到相机坐标，来自 solvePnP。",
            "cam2base": f"兼容字段名；实际为相机坐标到 {base_frame} 坐标。",
            "cam2reference": f"求解结果：相机坐标到 {base_frame} 坐标，适用于 eye-to-hand。",
        }

    payload: dict[str, Any] = {
        "calibration_method": mode,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "opencv_calibrateHandEye_method": int(method),
        "convention": convention,
        "context": _record_context(records),
        "npy_dir": str(npy_dir.resolve()),
        "filtering": filtering
        or {
            "rules": [],
            "dropped_samples": [],
            "retained_sample_count": len(records),
        },
        "samples": [
            {
                "image": r.get("image"),
                "json": r.get("_json_path"),
                "target_reprojection_rms_px": r.get("target_reprojection_rms_px"),
                "capture_mode": r.get("mode"),
                "camera_name": r.get("camera_info", {}).get("camera_name", ""),
                "camera_mount": r.get("camera_info", {}).get("mount", ""),
            }
            for r in records
        ],
    }
    payload.update(result_payload)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path
