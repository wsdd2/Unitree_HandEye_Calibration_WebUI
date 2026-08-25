# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from handeye_calib.solver import load_capture_records, normalize_mode, opencv_method_from_name, solve_handeye


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线求解 RealSense D435i 灵巧手手眼标定结果")
    parser.add_argument("--data-dir", required=True, help="包含采集 JSON 的目录，如 data/eye_in_hand_xxx/camera_0")
    parser.add_argument("--mode", required=True, help="eye-in-hand/hand-in-eye 或 eye-to-hand/hand-to-eye")
    parser.add_argument("--output-dir", default="", help="结果输出目录，默认 outputs")
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument(
        "--drop-highest-rms",
        type=int,
        default=0,
        help="按 target_reprojection_rms_px 排除误差最大的 N 个样本",
    )
    parser.add_argument(
        "--drop-sample-index",
        type=int,
        action="append",
        default=[],
        help="在 RMS 过滤后，排除剩余序列中的第 N 个样本（从 1 开始）；可重复指定",
    )
    parser.add_argument(
        "--drop-rotation-error-above-deg",
        type=float,
        default=0.0,
        help="临时求解后，自动排除手眼闭环旋转误差超过该阈值的样本；0 表示关闭",
    )
    parser.add_argument("--handeye-method", choices=("tsai", "park", "horaud", "andreff", "daniilidis"), default="tsai")
    args = parser.parse_args()
    normalize_mode(args.mode)
    if args.min_samples < 3:
        parser.error("--min-samples 必须 >= 3")
    if args.drop_highest_rms < 0:
        parser.error("--drop-highest-rms 必须 >= 0")
    if any(index < 1 for index in args.drop_sample_index):
        parser.error("--drop-sample-index 必须 >= 1")
    if args.drop_rotation_error_above_deg < 0:
        parser.error("--drop-rotation-error-above-deg 必须 >= 0")
    return args


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    records = load_capture_records(data_dir)
    initial_sample_count = len(records)
    filtering = {
        "rules": [
            {
                "name": "drop_highest_rms",
                "count": args.drop_highest_rms,
                "enabled": args.drop_highest_rms > 0,
            },
            {
                "name": "drop_sample_index",
                "indices_after_rms_filter": sorted(set(args.drop_sample_index)),
                "enabled": bool(args.drop_sample_index),
            },
            {
                "name": "drop_rotation_error_above_deg",
                "threshold_deg": args.drop_rotation_error_above_deg,
                "enabled": args.drop_rotation_error_above_deg > 0,
            },
        ],
        "initial_sample_count": initial_sample_count,
        "dropped_samples": [],
    }
    print(f"[LOAD] data_dir={data_dir.resolve()} samples={len(records)}")
    if args.drop_highest_rms:
        if args.drop_highest_rms >= len(records):
            raise ValueError(
                f"无法排除 {args.drop_highest_rms} 个样本：当前总样本数只有 {len(records)}"
            )
        ranked = sorted(
            records,
            key=lambda record: float(
                record.get("target_reprojection_rms_px")
                if record.get("target_reprojection_rms_px") is not None
                else float("-inf")
            ),
            reverse=True,
        )
        dropped = ranked[: args.drop_highest_rms]
        dropped_paths = {record["_json_path"] for record in dropped}
        records = [record for record in records if record["_json_path"] not in dropped_paths]
        for record in dropped:
            filtering["dropped_samples"].append(
                {
                    "rule": "drop_highest_rms",
                    "json": record.get("_json_path"),
                    "image": record.get("image"),
                    "target_reprojection_rms_px": record.get(
                        "target_reprojection_rms_px"
                    ),
                }
            )
            print(
                "[DROP] "
                f"rms={float(record['target_reprojection_rms_px']):.6f}px "
                f"image={record.get('image', '')}"
            )
        print(f"[FILTER] dropped={len(dropped)} retained={len(records)}")
    if args.drop_sample_index:
        requested_indices = set(args.drop_sample_index)
        invalid_indices = sorted(index for index in requested_indices if index > len(records))
        if invalid_indices:
            raise ValueError(
                f"RMS 过滤后样本序号超出范围: {invalid_indices}，"
                f"当前剩余样本数为 {len(records)}"
            )
        retained_records = []
        for index, record in enumerate(records, start=1):
            if index not in requested_indices:
                retained_records.append(record)
                continue
            rms = record.get("target_reprojection_rms_px")
            filtering["dropped_samples"].append(
                {
                    "rule": "drop_sample_index",
                    "index_after_rms_filter": index,
                    "json": record.get("_json_path"),
                    "image": record.get("image"),
                    "target_reprojection_rms_px": rms,
                }
            )
            rms_text = "unknown" if rms is None else f"{float(rms):.6f}px"
            print(
                "[DROP-INDEX] "
                f"post_rms_index={index} rms={rms_text} "
                f"image={record.get('image', '')}"
            )
        records = retained_records
        print(
            f"[FILTER-INDEX] dropped={len(requested_indices)} retained={len(records)}"
        )
    method = opencv_method_from_name(args.handeye_method)
    if args.drop_rotation_error_above_deg > 0:
        with tempfile.TemporaryDirectory(prefix="handeye_filter_") as temp_dir:
            pilot_path = solve_handeye(
                records,
                mode=args.mode,
                output_dir=Path(temp_dir),
                min_samples=args.min_samples,
                method=method,
                include_leave_one_out=False,
            )
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        rotation_errors = pilot.get("quality", {}).get("rotation_error_deg_each", [])
        if len(rotation_errors) != len(records):
            raise RuntimeError(
                "临时求解返回的旋转误差数量与样本数量不一致: "
                f"{len(rotation_errors)} != {len(records)}"
            )
        retained_records = []
        dropped_rotation_count = 0
        for index, (record, error_deg) in enumerate(
            zip(records, rotation_errors),
            start=1,
        ):
            error_deg = float(error_deg)
            if error_deg <= args.drop_rotation_error_above_deg:
                retained_records.append(record)
                continue
            dropped_rotation_count += 1
            filtering["dropped_samples"].append(
                {
                    "rule": "drop_rotation_error_above_deg",
                    "index_after_previous_filters": index,
                    "threshold_deg": args.drop_rotation_error_above_deg,
                    "rotation_error_deg": error_deg,
                    "json": record.get("_json_path"),
                    "image": record.get("image"),
                }
            )
            print(
                "[DROP-ROTATION] "
                f"post_filter_index={index} error={error_deg:.6f}deg "
                f"image={record.get('image', '')}"
            )
        records = retained_records
        print(
            f"[FILTER-ROTATION] threshold={args.drop_rotation_error_above_deg:.6f}deg "
            f"dropped={dropped_rotation_count} retained={len(records)}"
        )
    filtering["retained_sample_count"] = len(records)
    path = solve_handeye(
        records,
        mode=args.mode,
        output_dir=output_dir,
        min_samples=args.min_samples,
        method=method,
        filtering=filtering,
    )
    print(f"[DONE] saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
