# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np


def parse_pose_text(text: str) -> list[float]:
    cleaned = text.strip().replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
    parts = [p for p in re.split(r"[,\s]+", cleaned) if p]
    if len(parts) != 6:
        raise ValueError("请输入 6 个数：x y z rx ry rz")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("位姿中包含无法解析的数字") from exc


def axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"不支持的欧拉轴: {axis}")


def euler_to_rotation(angles: list[float], order: str, unit: str) -> np.ndarray:
    vals = np.asarray(angles, dtype=np.float64)
    if unit == "deg":
        vals = np.deg2rad(vals)
    rotation = np.eye(3, dtype=np.float64)
    for axis, angle in zip(order.lower(), vals):
        rotation = rotation @ axis_rotation(axis, float(angle))
    return rotation


def pose_to_transform(
    pose: list[float],
    translation_unit: str,
    rotation_unit: str,
    euler_order: str,
) -> np.ndarray:
    scale = 0.001 if translation_unit == "mm" else 1.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_to_rotation(pose[3:], euler_order, rotation_unit)
    transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64) * scale
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return transform


def transform_to_record(transform: np.ndarray) -> dict[str, Any]:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return {
        "rvec": rvec.reshape(3).astype(float).tolist(),
        "tvec_m": transform[:3, 3].astype(float).tolist(),
        "rotation_matrix": transform[:3, :3].astype(float).tolist(),
        "transform": transform.astype(float).tolist(),
    }


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4, dtype=np.float64)
    translations = np.asarray([t[:3, 3] for t in transforms], dtype=np.float64).mean(axis=0)
    rvecs = []
    for transform in transforms:
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        rvecs.append(rvec.reshape(3))
    mean_rvec = np.asarray(rvecs, dtype=np.float64).mean(axis=0)
    rotation, _ = cv2.Rodrigues(mean_rvec.reshape(3, 1))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translations
    return out
