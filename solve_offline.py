# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
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
    parser.add_argument("--handeye-method", choices=("tsai", "park", "horaud", "andreff", "daniilidis"), default="tsai")
    args = parser.parse_args()
    normalize_mode(args.mode)
    if args.min_samples < 3:
        parser.error("--min-samples 必须 >= 3")
    return args


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    records = load_capture_records(data_dir)
    print(f"[LOAD] data_dir={data_dir.resolve()} samples={len(records)}")
    path = solve_handeye(
        records,
        mode=args.mode,
        output_dir=output_dir,
        min_samples=args.min_samples,
        method=opencv_method_from_name(args.handeye_method),
    )
    print(f"[DONE] saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
