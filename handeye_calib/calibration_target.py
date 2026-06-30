# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np


def build_object_points(inner_cols: int, inner_rows: int, square_mm: float) -> np.ndarray:
    square_m = float(square_mm) / 1000.0
    objp = np.zeros((inner_rows * inner_cols, 3), dtype=np.float32)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * square_m
    return objp


def solve_target_pose(
    objp: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    err = cv2.norm(corners, projected, cv2.NORM_L2) ** 2
    rms = float(np.sqrt(err / len(projected)))
    return rvec.reshape(3), tvec.reshape(3), rms
