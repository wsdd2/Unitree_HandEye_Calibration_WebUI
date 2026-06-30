# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .transforms import invert_transform, transform_from_rvec_tvec, transform_to_record


def load_camera_params(
    *,
    camera_matrix_npy: str = "",
    dist_coeffs_npy: str = "",
    camera_json: str = "",
) -> Optional[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    if camera_matrix_npy or dist_coeffs_npy:
        if not camera_matrix_npy or not dist_coeffs_npy:
            raise ValueError("--camera-matrix-npy 和 --dist-coeffs-npy 必须同时提供")
        camera_matrix = np.load(camera_matrix_npy).astype(np.float64)
        dist_coeffs = np.load(dist_coeffs_npy).reshape(-1, 1).astype(np.float64)
        return camera_matrix, dist_coeffs, {
            "source": "npy",
            "camera_matrix_npy": str(Path(camera_matrix_npy).resolve()),
            "dist_coeffs_npy": str(Path(dist_coeffs_npy).resolve()),
        }

    if camera_json:
        path = Path(camera_json)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        color = payload.get("color", payload)
        if "camera_matrix" in color:
            camera_matrix = np.asarray(color["camera_matrix"], dtype=np.float64)
        else:
            camera_matrix = np.array(
                [
                    [float(color["fx"]), 0.0, float(color["ppx"])],
                    [0.0, float(color["fy"]), float(color["ppy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        coeffs = color.get("coeffs") or color.get("distortion_coefficients") or payload.get("coeffs") or [0, 0, 0, 0, 0]
        if isinstance(coeffs, dict):
            coeffs = [coeffs.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")]
        dist_coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)
        return camera_matrix, dist_coeffs, {"source": "json", "camera_json": str(path.resolve())}

    return None


def save_capture_record(
    *,
    out_dir: Path,
    image_bgr: np.ndarray,
    mode: str,
    camera_index: int,
    camera_info: dict[str, Any],
    camera_intrinsics: dict[str, Any],
    robot_context: dict[str, Any],
    pattern_size: tuple[int, int],
    square_mm: float,
    detection_method: str,
    corners: np.ndarray,
    target_rvec: np.ndarray,
    target_tvec: np.ndarray,
    target_reproj_rms: float,
    pose_text: str,
    pose_values: list[float],
    pose_units: dict[str, str],
    T_hand2base: np.ndarray,
    hand_frame_name: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"{stamp}_cam{camera_index}_{mode}"
    image_path = out_dir / f"{stem}.jpg"
    json_path = out_dir / f"{stem}.json"

    ok = cv2.imwrite(str(image_path), image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {image_path}")

    T_target2cam = transform_from_rvec_tvec(target_rvec, target_tvec)
    T_base2hand = invert_transform(T_hand2base)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "image": image_path.name,
        "camera_index": camera_index,
        "camera_info": camera_info,
        "camera_intrinsics": camera_intrinsics,
        "robot_context": robot_context,
        "board": {
            "inner_corners_cols": int(pattern_size[0]),
            "inner_corners_rows": int(pattern_size[1]),
            "square_mm": float(square_mm),
            "square_m": float(square_mm) / 1000.0,
        },
        "detection_method": detection_method,
        "image_points_px": corners.reshape(-1, 2).astype(float).tolist(),
        "target2cam": transform_to_record(T_target2cam),
        "target_reprojection_rms_px": float(target_reproj_rms),
        "hand_pose_input": {
            "frame_name": hand_frame_name,
            "raw": pose_text,
            "values": pose_values,
            "translation_unit": pose_units["translation"],
            "rotation_unit": pose_units["rotation"],
            "euler_order": pose_units["euler_order"],
            "meaning": "T_base_hand: 点从灵巧手/末端坐标变换到机器人基座坐标",
        },
        "hand2base": transform_to_record(T_hand2base),
        "base2hand": transform_to_record(T_base2hand),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return image_path
