# -*- coding: utf-8 -*-
"""Recompute capture target poses after correcting camera intrinsics."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from handeye_calib.calibration_target import build_object_points, solve_target_pose
from handeye_calib.camera import RealSenseD435i
from handeye_calib.io_utils import load_camera_params
from handeye_calib.transforms import transform_from_rvec_tvec, transform_to_record


def resolve_intrinsics(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    loaded = load_camera_params(
        camera_matrix_npy=args.camera_matrix_npy,
        dist_coeffs_npy=args.dist_coeffs_npy,
        camera_json=args.camera_json,
    )
    if loaded is not None:
        return loaded

    cam = RealSenseD435i(
        serial=args.cam_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_name=args.camera_name,
        mount=args.camera_mount,
        color_only=True,
    )
    try:
        cam.open()
        cam.start()
        camera_matrix, dist_coeffs, info = cam.color_intrinsics()
        info["source"] = "realsense_profile"
        info["serial"] = args.cam_serial
        return camera_matrix, dist_coeffs, info
    finally:
        cam.close()


def corrected_camera_info(
    original: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    info = dict(original)
    info.update(
        {
            "serial": args.cam_serial,
            "requested_serial": args.cam_serial,
            "selected_serial": args.cam_serial,
            "camera_name": args.camera_name,
            "mount": args.camera_mount,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "color_only": True,
        }
    )
    return info


def recompute_record(
    record: dict[str, Any],
    args: argparse.Namespace,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_intrinsics: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    board = record["board"]
    cols = int(board["inner_corners_cols"])
    rows = int(board["inner_corners_rows"])
    square_mm = float(board["square_mm"])
    corners = np.asarray(record["image_points_px"], dtype=np.float32).reshape(-1, 1, 2)
    expected_count = cols * rows
    if len(corners) != expected_count:
        raise ValueError(
            f"角点数量错误: {len(corners)}，棋盘 {cols}x{rows} 应为 {expected_count}"
        )

    objp = build_object_points(cols, rows, square_mm)
    rvec, tvec, rms = solve_target_pose(
        objp,
        corners,
        camera_matrix,
        dist_coeffs,
    )
    corrected = dict(record)
    corrected["camera_info"] = corrected_camera_info(
        record.get("camera_info", {}),
        args,
    )
    corrected["camera_intrinsics"] = dict(camera_intrinsics)
    corrected["target2cam"] = transform_to_record(
        transform_from_rvec_tvec(rvec, tvec)
    )
    corrected["target_reprojection_rms_px"] = float(rms)
    corrected["pnp_recomputed"] = {
        "reason": "corrected_camera_intrinsics",
        "source_json": record.get("_json_path", ""),
    }
    return corrected, rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用正确相机内参重新计算已采集手眼样本的 target2cam"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cam-serial", required=True)
    parser.add_argument("--camera-name", default="right_wrist_d435")
    parser.add_argument("--camera-mount", default="wrist")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-matrix-npy", default="")
    parser.add_argument("--dist-coeffs-npy", default="")
    parser.add_argument("--camera-json", default="")
    args = parser.parse_args()
    if Path(args.input_dir).resolve() == Path(args.output_dir).resolve():
        parser.error("--output-dir 必须与 --input-dir 不同，避免覆盖原始数据")
    return args


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"没有采集 JSON: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，请换一个新目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix, dist_coeffs, intrinsics_info = resolve_intrinsics(args)
    print(f"[INTRINSICS] {intrinsics_info}")
    rms_values: list[float] = []
    for json_path in json_paths:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        record["_json_path"] = str(json_path.resolve())
        corrected, rms = recompute_record(
            record,
            args,
            camera_matrix,
            dist_coeffs,
            intrinsics_info,
        )
        corrected.pop("_json_path", None)
        output_json = output_dir / json_path.name
        output_json.write_text(
            json.dumps(corrected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_name = str(record.get("image", ""))
        if image_name:
            source_image = input_dir / image_name
            if source_image.exists():
                shutil.copy2(source_image, output_dir / image_name)
        rms_values.append(rms)
        print(f"[RECOMPUTE] rms={rms:.6f}px json={json_path.name}")

    print(
        f"[DONE] samples={len(rms_values)} "
        f"rms_mean={float(np.mean(rms_values)):.6f}px "
        f"rms_max={float(np.max(rms_values)):.6f}px "
        f"output={output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
