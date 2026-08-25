# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np


EULER_MODE_ROS_RPY = "ros_rpy_rzryrx"
EULER_MODE_LEGACY_XYZ = "legacy_rxryrz"


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
    if vals.shape != (3,):
        raise ValueError("欧拉角必须恰好包含 3 个值")
    if unit == "deg":
        vals = np.deg2rad(vals)
    elif unit != "rad":
        raise ValueError(f"不支持的旋转单位: {unit}")
    mode = order.strip().lower().replace("-", "_")
    if mode in (EULER_MODE_ROS_RPY, "ros_rpy", "rpy"):
        roll, pitch, yaw = vals
        return (
            axis_rotation("z", float(yaw))
            @ axis_rotation("y", float(pitch))
            @ axis_rotation("x", float(roll))
        )
    if mode in (EULER_MODE_LEGACY_XYZ, "legacy_xyz"):
        mode = "xyz"
    if len(mode) != 3 or set(mode) != {"x", "y", "z"}:
        raise ValueError(f"不支持的欧拉角模式/顺序: {order}")
    rotation = np.eye(3, dtype=np.float64)
    for axis, angle in zip(mode, vals):
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
    # Markley 四元数平均不会在 Rodrigues 的 +/-pi 分支切面上相互抵消。
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for transform in transforms:
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        trace = float(np.trace(rotation))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                ]
            )
        else:
            index = int(np.argmax(np.diag(rotation)))
            j, k = (index + 1) % 3, (index + 2) % 3
            s = np.sqrt(max(0.0, 1.0 + rotation[index, index] - rotation[j, j] - rotation[k, k])) * 2.0
            xyz = np.zeros(3, dtype=np.float64)
            xyz[index] = 0.25 * s
            xyz[j] = (rotation[j, index] + rotation[index, j]) / s
            xyz[k] = (rotation[k, index] + rotation[index, k]) / s
            quat = np.array(
                [
                    (rotation[k, j] - rotation[j, k]) / s,
                    xyz[0],
                    xyz[1],
                    xyz[2],
                ]
            )
        quat /= np.linalg.norm(quat)
        accumulator += np.outer(quat, quat)
    quat = np.linalg.eigh(accumulator)[1][:, -1]
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translations
    return out
