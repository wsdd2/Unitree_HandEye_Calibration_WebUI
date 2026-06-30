# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handeye_calib.calibration_target import build_object_points
from handeye_calib.chessboard import find_chessboard_corners


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "camera_calib_image"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_images(image_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(image_dir.glob(pattern)))
    return images


def per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_paths: list[Path],
) -> list[dict]:
    errors: list[dict] = []
    for objp, corners, rvec, tvec, path in zip(object_points, image_points, rvecs, tvecs, image_paths):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
        errors.append({"image": str(path), "mean_reprojection_error_px": float(error)})
    return errors


def calibrate(args: argparse.Namespace) -> int:
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {image_dir}")

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[Path] = []
    rejected: list[dict] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(path), "reason": "failed_to_read"})
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif image_size != (gray.shape[1], gray.shape[0]):
            rejected.append({"image": str(path), "reason": "image_size_mismatch"})
            continue

        corners, method = find_chessboard_corners(gray, pattern_size, args.gamma)
        if corners is None:
            rejected.append({"image": str(path), "reason": "chessboard_not_found"})
            continue

        object_points.append(objp.copy())
        image_points.append(corners.astype(np.float32))
        used_images.append(path)
        print(f"[USE] {path} method={method}")

    if len(used_images) < args.min_images:
        raise RuntimeError(
            f"Only {len(used_images)} valid images, need at least {args.min_images}. "
            f"Rejected {len(rejected)} images."
        )
    if image_size is None:
        raise RuntimeError("No readable calibration images.")

    flags = 0
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    errors = per_view_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
        used_images,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_dir = output_dir / f"camera_intrinsics_{stamp}_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    camera_matrix_path = npy_dir / "camera_matrix.npy"
    dist_coeffs_path = npy_dir / "dist_coeffs.npy"
    np.save(camera_matrix_path, camera_matrix)
    np.save(dist_coeffs_path, dist_coeffs)

    report = {
        "image_dir": str(image_dir),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "pattern": {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm},
        "used_image_count": len(used_images),
        "rejected_image_count": len(rejected),
        "rms_reprojection_error_px": float(rms),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).astype(float).tolist(),
        "camera_matrix_npy": str(camera_matrix_path),
        "dist_coeffs_npy": str(dist_coeffs_path),
        "per_view_errors": errors,
        "rejected": rejected,
    }
    report_path = output_dir / f"camera_intrinsics_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] rms={rms:.4f}px used={len(used_images)} rejected={len(rejected)}")
    print(f"[JSON] {report_path}")
    print(f"[NPY]  {camera_matrix_path}")
    print(f"[NPY]  {dist_coeffs_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cols", type=int, default=11, help="棋盘横向内角点数")
    parser.add_argument("--rows", type=int, default=8, help="棋盘纵向内角点数")
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--min-images", type=int, default=8)
    parser.add_argument("--fix-k3", action="store_true", help="固定 k3 畸变项，样本较少时可尝试")
    args = parser.parse_args()
    if args.cols < 2 or args.rows < 2:
        parser.error("--cols/--rows must be >= 2")
    if args.square_mm <= 0:
        parser.error("--square-mm must be > 0")
    if args.min_images < 3:
        parser.error("--min-images must be >= 3")
    return args


if __name__ == "__main__":
    try:
        sys.exit(calibrate(parse_args()))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
