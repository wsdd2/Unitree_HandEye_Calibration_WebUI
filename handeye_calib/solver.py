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
        record["_json_path"] = str(path)
        records.append(record)
    return records


def record_transform(record: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(record[key]["transform"], dtype=np.float64)


def _record_context(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    first = records[0]
    return {
        "camera_info": first.get("camera_info", {}),
        "robot_context": first.get("robot_context", {}),
        "board": first.get("board", {}),
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


def solve_handeye(
    records: list[dict[str, Any]],
    *,
    mode: str,
    output_dir: Path,
    min_samples: int = 8,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> Path:
    mode = normalize_mode(mode)
    if len(records) < min_samples:
        raise ValueError(f"样本不足：{len(records)}/{min_samples}")

    T_hand2base = [record_transform(r, "hand2base") for r in records]
    T_base2hand = [record_transform(r, "base2hand") for r in records]
    T_target2cam = [record_transform(r, "target2cam") for r in records]

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
            "quality": {
                "fixed_transform": "target2base",
                "translation_std_m_xyz": _translation_spread(fixed_target2base_each),
                "rotation_error_deg_each": _rotation_spread_deg(fixed_target2base_each, fixed_target2base),
            },
        }
        convention = {
            "hand2base": "机器人输入位姿解析为 T_base_hand，即点从灵巧手/末端坐标变换到机器人基座坐标。",
            "target2cam": "棋盘格坐标到相机坐标，来自 solvePnP。",
            "cam2hand": "求解结果：相机坐标到灵巧手/末端坐标，适用于 eye-in-hand。",
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
        np.save(npy_dir / "T_target2hand_each.npy", np.asarray(fixed_target2hand_each, dtype=np.float64))
        np.save(npy_dir / "T_target2hand_mean.npy", fixed_target2hand)

        result_payload = {
            "cam2base": transform_to_record(T_cam2base),
            "base2cam": transform_to_record(T_base2cam),
            "target2hand_mean": transform_to_record(fixed_target2hand),
            "quality": {
                "fixed_transform": "target2hand",
                "translation_std_m_xyz": _translation_spread(fixed_target2hand_each),
                "rotation_error_deg_each": _rotation_spread_deg(fixed_target2hand_each, fixed_target2hand),
            },
        }
        convention = {
            "hand2base": "机器人输入位姿解析为 T_base_hand，即点从灵巧手/末端坐标变换到机器人基座坐标。",
            "target2cam": "棋盘格坐标到相机坐标，来自 solvePnP。",
            "cam2base": "求解结果：相机坐标到机器人基座坐标，适用于 eye-to-hand。",
        }

    payload: dict[str, Any] = {
        "calibration_method": mode,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "opencv_calibrateHandEye_method": int(method),
        "convention": convention,
        "context": _record_context(records),
        "npy_dir": str(npy_dir.resolve()),
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
