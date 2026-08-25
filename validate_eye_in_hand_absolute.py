# -*- coding: utf-8 -*-
"""用独立新姿态验证 eye-in-hand 外参及 torso_link 下的绝对位置精度。

验证帧仍使用 capture_handeye.py 产生的 JSON，但不得参与原手眼求解。
若提供由独立测量系统获得的参考点真值，则同时计算绝对误差；否则只报告
多姿态闭环一致性，不能据此声明 FK/torso_link 的绝对精度。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from handeye_calib.solver import (
    MODE_EYE_IN_HAND,
    load_capture_records,
    normalize_mode,
    record_transform,
)
from handeye_calib.transforms import transform_to_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用未参与求解的新姿态验证 eye-in-hand 外参和 torso 绝对坐标"
    )
    parser.add_argument(
        "--validation-data-dir",
        required=True,
        help="capture_handeye.py 采集的新验证 JSON 所在目录",
    )
    parser.add_argument(
        "--handeye-json",
        required=True,
        help="待验证的 eye_in_hand_*.json",
    )
    parser.add_argument(
        "--truth-xyz-m",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="参考点在 base/torso 坐标系下的独立实测真值，单位 m",
    )
    parser.add_argument(
        "--reference-point",
        choices=("first-inner-corner", "inner-grid-center"),
        default="first-inner-corner",
        help="真值对应棋盘第一个内角点或内角点网格中心",
    )
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument(
        "--max-reprojection-rms-px",
        type=float,
        default=0.5,
        help="任一验证帧超过此重投影 RMS 则验证失败；0 表示关闭",
    )
    parser.add_argument(
        "--pass-rmse-mm",
        type=float,
        default=10.0,
        help="绝对误差向量 RMSE 通过阈值",
    )
    parser.add_argument(
        "--pass-p95-mm",
        type=float,
        default=15.0,
        help="绝对误差向量 P95 通过阈值",
    )
    parser.add_argument(
        "--pass-max-mm",
        type=float,
        default=20.0,
        help="绝对误差向量最大值通过阈值",
    )
    parser.add_argument(
        "--allow-calibration-samples",
        action="store_true",
        help="允许验证目录与原求解样本重名，仅供复算/测试；现场绝对验证不要使用",
    )
    parser.add_argument("--output-json", default="", help="报告 JSON 路径")
    parser.add_argument("--output-csv", default="", help="逐帧结果 CSV 路径")
    parser.add_argument(
        "--annotated-dir",
        default="",
        help="可选：输出标出被测棋盘参考点的图像目录",
    )
    args = parser.parse_args()

    if args.min_samples < 3:
        parser.error("--min-samples 必须 >= 3")
    for name in (
        "max_reprojection_rms_px",
        "pass_rmse_mm",
        "pass_p95_mm",
        "pass_max_mm",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须 >= 0")
    return args


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return payload


def _se3_from_result(result: dict[str, Any], key: str) -> np.ndarray:
    try:
        transform = np.asarray(result[key]["transform"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"手眼结果缺少 {key}.transform") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{key}.transform 必须是有限数值的 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{key}.transform 齐次矩阵末行无效")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{key}.transform 旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{key}.transform 旋转矩阵行列式不为 +1")
    return transform


def _camera_serial(container: dict[str, Any]) -> str:
    info = container.get("camera_info", {})
    if not isinstance(info, dict):
        return ""
    for key in ("selected_serial", "serial", "requested_serial"):
        value = info.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _capture_names_from_result(result: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for sample in result.get("samples", []):
        if not isinstance(sample, dict):
            continue
        for key in ("json", "image"):
            value = sample.get(key)
            if value:
                names.add(Path(str(value)).stem)
    for sample in result.get("filtering", {}).get("dropped_samples", []):
        if not isinstance(sample, dict):
            continue
        for key in ("json", "image"):
            value = sample.get(key)
            if value:
                names.add(Path(str(value)).stem)
    return names


def _validate_context(
    records: list[dict[str, Any]],
    result: dict[str, Any],
    allow_calibration_samples: bool,
) -> list[str]:
    if normalize_mode(str(result.get("calibration_method", ""))) != MODE_EYE_IN_HAND:
        raise ValueError("手眼结果不是 eye-in-hand")

    first = records[0]
    if normalize_mode(str(first.get("mode", ""))) != MODE_EYE_IN_HAND:
        raise ValueError("验证样本不是 eye-in-hand")

    expected = result.get("context", {})
    robot_context = expected.get("robot_context", {})
    capture_pose = first.get("hand_pose_input", {})
    checks = (
        (
            "camera serial",
            _camera_serial(expected),
            _camera_serial(first),
        ),
        (
            "base frame",
            str(expected.get("base_frame") or robot_context.get("base_frame") or ""),
            str(capture_pose.get("base_frame_name") or ""),
        ),
        (
            "hand frame",
            str(expected.get("hand_frame") or robot_context.get("hand_frame") or ""),
            str(capture_pose.get("frame_name") or ""),
        ),
    )
    for label, wanted, actual in checks:
        if wanted and actual and wanted != actual:
            raise ValueError(f"{label} 不一致：手眼结果={wanted!r}，验证样本={actual!r}")

    expected_board = expected.get("board", {})
    actual_board = first.get("board", {})
    for key in ("inner_corners_cols", "inner_corners_rows"):
        if expected_board.get(key) != actual_board.get(key):
            raise ValueError(
                f"棋盘参数 {key} 不一致："
                f"{expected_board.get(key)!r} != {actual_board.get(key)!r}"
            )
    expected_square = float(
        expected_board.get(
            "square_m",
            float(expected_board.get("square_mm", 0.0)) / 1000.0,
        )
    )
    actual_square = float(
        actual_board.get(
            "square_m",
            float(actual_board.get("square_mm", 0.0)) / 1000.0,
        )
    )
    if not np.isclose(expected_square, actual_square, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"棋盘方格尺寸不一致：{expected_square!r} != {actual_square!r}"
        )

    original_names = _capture_names_from_result(result)
    overlapping = sorted(
        {
            Path(str(record.get("_json_path", ""))).stem
            for record in records
            if Path(str(record.get("_json_path", ""))).stem in original_names
        }
    )
    if overlapping and not allow_calibration_samples:
        preview = ", ".join(overlapping[:5])
        raise ValueError(
            "验证数据包含参与原求解的样本，不能作为独立验证："
            f"{preview}{' ...' if len(overlapping) > 5 else ''}；"
            "仅做复算时可添加 --allow-calibration-samples"
        )
    return overlapping


def _reference_point_in_board(
    record: dict[str, Any],
    reference_point: str,
) -> np.ndarray:
    point = np.ones(4, dtype=np.float64)
    point[:3] = 0.0
    if reference_point == "first-inner-corner":
        return point

    board = record["board"]
    cols = int(board["inner_corners_cols"])
    rows = int(board["inner_corners_rows"])
    square_m = float(
        board.get(
            "square_m",
            float(board["square_mm"]) / 1000.0,
        )
    )
    point[0] = (cols - 1) * square_m * 0.5
    point[1] = (rows - 1) * square_m * 0.5
    return point


def _reference_pixel(record: dict[str, Any], reference_point: str) -> np.ndarray:
    points = np.asarray(record.get("image_points_px"), dtype=np.float64).reshape(-1, 2)
    if len(points) < 1 or not np.all(np.isfinite(points)):
        raise ValueError(
            f"{record.get('_json_path', record.get('image'))} 缺少有效 image_points_px"
        )
    if reference_point == "first-inner-corner":
        return points[0]
    return points.mean(axis=0)


def _annotate_reference_images(
    records: list[dict[str, Any]],
    validation_dir: Path,
    annotated_dir: Path,
    reference_point: str,
) -> list[str]:
    annotated_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for record in records:
        image_name = str(record.get("image") or "")
        image_path = validation_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取验证图像：{image_path}")
        pixel = _reference_pixel(record, reference_point)
        center = tuple(int(round(value)) for value in pixel)
        cv2.circle(image, center, 14, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(
            image,
            center,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = (
            "MEASURE THIS: FIRST INNER CORNER"
            if reference_point == "first-inner-corner"
            else "MEASURE THIS: INNER-GRID CENTER"
        )
        text_origin = (max(5, center[0] - 220), max(35, center[1] - 24))
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_path = annotated_dir / image_name
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"写入标注图像失败：{output_path}")
        outputs.append(str(output_path.resolve()))
    return outputs


def _average_rotation(transforms: list[np.ndarray]) -> np.ndarray:
    matrix = np.mean([transform[:3, :3] for transform in transforms], axis=0)
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _xyz_stats(values_m: np.ndarray) -> dict[str, Any]:
    return {
        "mean_m": values_m.mean(axis=0).tolist(),
        "std_mm": (values_m.std(axis=0) * 1000.0).tolist(),
        "min_m": values_m.min(axis=0).tolist(),
        "max_m": values_m.max(axis=0).tolist(),
        "range_mm": (np.ptp(values_m, axis=0) * 1000.0).tolist(),
    }


def _absolute_stats(
    predictions_m: np.ndarray,
    truth_m: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    errors_m = predictions_m - truth_m.reshape(1, 3)
    abs_errors_m = np.abs(errors_m)
    norm_errors_m = np.linalg.norm(errors_m, axis=1)
    return (
        {
            "truth_xyz_m": truth_m.tolist(),
            "error_definition": "prediction_minus_truth",
            "bias_xyz_mm": (errors_m.mean(axis=0) * 1000.0).tolist(),
            "rmse_xyz_mm": (
                np.sqrt(np.mean(np.square(errors_m), axis=0)) * 1000.0
            ).tolist(),
            "mae_xyz_mm": (abs_errors_m.mean(axis=0) * 1000.0).tolist(),
            "p95_abs_xyz_mm": (
                np.percentile(abs_errors_m, 95, axis=0) * 1000.0
            ).tolist(),
            "max_abs_xyz_mm": (abs_errors_m.max(axis=0) * 1000.0).tolist(),
            "norm_error_mm": {
                "mean": float(norm_errors_m.mean() * 1000.0),
                "rmse": float(
                    math.sqrt(float(np.mean(np.square(norm_errors_m)))) * 1000.0
                ),
                "p95": float(np.percentile(norm_errors_m, 95) * 1000.0),
                "max": float(norm_errors_m.max() * 1000.0),
            },
        },
        errors_m,
        norm_errors_m,
    )


def _default_output_path(validation_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return validation_dir / f"eye_in_hand_absolute_validation_{timestamp}.json"


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    has_truth: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "json",
        "image",
        "reprojection_rms_px",
        "predicted_x_m",
        "predicted_y_m",
        "predicted_z_m",
        "rotation_delta_deg",
    ]
    if has_truth:
        fields += ["error_x_mm", "error_y_mm", "error_z_mm", "error_norm_mm"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validation_dir = Path(args.validation_data_dir).expanduser()
    handeye_path = Path(args.handeye_json).expanduser()
    records = load_capture_records(validation_dir)
    if len(records) < args.min_samples:
        raise ValueError(
            f"验证样本不足：{len(records)} < --min-samples {args.min_samples}"
        )
    result = _load_json(handeye_path)
    overlap = _validate_context(
        records,
        result,
        allow_calibration_samples=args.allow_calibration_samples,
    )
    camera_to_hand = _se3_from_result(result, "cam2hand")

    board_transforms: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rms_values: list[float] = []
    for record in records:
        base_to_hand = record_transform(record, "hand2base")
        camera_to_board = record_transform(record, "target2cam")
        base_to_board = base_to_hand @ camera_to_hand @ camera_to_board
        board_transforms.append(base_to_board)
        reference_in_board = _reference_point_in_board(
            record,
            args.reference_point,
        )
        predictions.append((base_to_board @ reference_in_board)[:3])
        rms_values.append(float(record.get("target_reprojection_rms_px", float("nan"))))

    predicted_m = np.asarray(predictions, dtype=np.float64)
    mean_rotation = _average_rotation(board_transforms)
    rotation_delta_deg = np.asarray(
        [
            _rotation_delta_deg(mean_rotation, transform[:3, :3])
            for transform in board_transforms
        ],
        dtype=np.float64,
    )
    rms_array = np.asarray(rms_values, dtype=np.float64)
    warnings: list[str] = []
    failures: list[str] = []

    if not np.all(np.isfinite(rms_array)):
        failures.append("部分验证样本缺少有限的 target_reprojection_rms_px")
    if (
        args.max_reprojection_rms_px > 0
        and np.any(rms_array > args.max_reprojection_rms_px)
    ):
        failures.append(
            "存在重投影 RMS 超过阈值的验证帧："
            f"max={np.nanmax(rms_array):.6f}px > "
            f"{args.max_reprojection_rms_px:.6f}px"
        )

    absolute: dict[str, Any] | None = None
    errors_m: np.ndarray | None = None
    norm_errors_m: np.ndarray | None = None
    if args.truth_xyz_m is not None:
        truth_m = np.asarray(args.truth_xyz_m, dtype=np.float64)
        absolute, errors_m, norm_errors_m = _absolute_stats(predicted_m, truth_m)
        norm_stats = absolute["norm_error_mm"]
        threshold_checks = (
            ("RMSE", norm_stats["rmse"], args.pass_rmse_mm),
            ("P95", norm_stats["p95"], args.pass_p95_mm),
            ("MAX", norm_stats["max"], args.pass_max_mm),
        )
        for label, value, threshold in threshold_checks:
            if threshold > 0 and value > threshold:
                failures.append(
                    f"绝对误差向量 {label}={value:.3f}mm > {threshold:.3f}mm"
                )
    else:
        warnings.append(
            "未提供 --truth-xyz-m：本报告只能验证多姿态闭环一致性，"
            "不能验证 torso/FK 绝对精度"
        )

    sample_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for index, (record, prediction) in enumerate(
        zip(records, predicted_m),
        start=1,
    ):
        sample = {
            "index": index,
            "json": record.get("_json_path"),
            "image": record.get("image"),
            "target_reprojection_rms_px": rms_values[index - 1],
            "predicted_xyz_m": prediction.tolist(),
            "reference_pixel_xy": _reference_pixel(
                record,
                args.reference_point,
            ).tolist(),
            "rotation_delta_deg": float(rotation_delta_deg[index - 1]),
        }
        csv_row: dict[str, Any] = {
            "index": index,
            "json": record.get("_json_path"),
            "image": record.get("image"),
            "reprojection_rms_px": rms_values[index - 1],
            "predicted_x_m": prediction[0],
            "predicted_y_m": prediction[1],
            "predicted_z_m": prediction[2],
            "rotation_delta_deg": rotation_delta_deg[index - 1],
        }
        if errors_m is not None and norm_errors_m is not None:
            error_mm = errors_m[index - 1] * 1000.0
            sample["error_xyz_mm"] = error_mm.tolist()
            sample["error_norm_mm"] = float(norm_errors_m[index - 1] * 1000.0)
            csv_row.update(
                {
                    "error_x_mm": error_mm[0],
                    "error_y_mm": error_mm[1],
                    "error_z_mm": error_mm[2],
                    "error_norm_mm": norm_errors_m[index - 1] * 1000.0,
                }
            )
        sample_rows.append(sample)
        csv_rows.append(csv_row)

    status = "fail" if failures else ("warn" if warnings else "pass")
    annotated_images: list[str] = []
    if args.annotated_dir:
        annotated_images = _annotate_reference_images(
            records,
            validation_dir,
            Path(args.annotated_dir).expanduser(),
            args.reference_point,
        )
    payload: dict[str, Any] = {
        "schema": "eye_in_hand_absolute_validation",
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pass": not failures,
        "validation_mode": "absolute" if absolute is not None else "closure_only",
        "handeye_json": str(handeye_path.resolve()),
        "validation_data_dir": str(validation_dir.resolve()),
        "sample_count": len(records),
        "reference_point": args.reference_point,
        "transform_chain": "T_base_board = T_base_hand * T_hand_camera * T_camera_board",
        "overlap_with_calibration_samples": overlap,
        "context": {
            "base_frame": records[0]["hand_pose_input"].get("base_frame_name"),
            "hand_frame": records[0]["hand_pose_input"].get("frame_name"),
            "camera_serial": _camera_serial(records[0]),
            "board": records[0].get("board"),
        },
        "gates": {
            "max_reprojection_rms_px": args.max_reprojection_rms_px,
            "pass_rmse_mm": args.pass_rmse_mm,
            "pass_p95_mm": args.pass_p95_mm,
            "pass_max_mm": args.pass_max_mm,
        },
        "reprojection_rms_px": {
            "mean": float(np.nanmean(rms_array)),
            "max": float(np.nanmax(rms_array)),
        },
        "predicted_position": _xyz_stats(predicted_m),
        "predicted_board_orientation": {
            "mean": transform_to_record(
                np.block(
                    [
                        [mean_rotation, np.zeros((3, 1), dtype=np.float64)],
                        [np.zeros((1, 3), dtype=np.float64), np.ones((1, 1))],
                    ]
                )
            ),
            "rotation_delta_deg_mean": float(rotation_delta_deg.mean()),
            "rotation_delta_deg_max": float(rotation_delta_deg.max()),
        },
        "absolute_accuracy": absolute,
        "annotated_images": annotated_images,
        "warnings": warnings,
        "failures": failures,
        "samples": sample_rows,
    }

    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else _default_output_path(validation_dir)
    )
    output_csv = (
        Path(args.output_csv).expanduser()
        if args.output_csv
        else output_json.with_suffix(".csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    _write_csv(output_csv, csv_rows, has_truth=absolute is not None)

    print(f"[VALIDATION] mode={payload['validation_mode']} samples={len(records)}")
    print(
        "[CLOSURE] "
        f"std_xyz_mm={np.round(predicted_m.std(axis=0) * 1000.0, 3).tolist()} "
        f"range_xyz_mm={np.round(np.ptp(predicted_m, axis=0) * 1000.0, 3).tolist()}"
    )
    if absolute is not None:
        print(
            "[ABSOLUTE] "
            f"bias_xyz_mm={np.round(absolute['bias_xyz_mm'], 3).tolist()} "
            f"rmse_xyz_mm={np.round(absolute['rmse_xyz_mm'], 3).tolist()}"
        )
        print(
            "[ABSOLUTE-NORM] "
            f"rmse={absolute['norm_error_mm']['rmse']:.3f}mm "
            f"p95={absolute['norm_error_mm']['p95']:.3f}mm "
            f"max={absolute['norm_error_mm']['max']:.3f}mm"
        )
    for warning in warnings:
        print(f"[WARN] {warning}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    print(f"[REPORT] json={output_json.resolve()}")
    print(f"[REPORT] csv={output_csv.resolve()}")
    print(f"[STATUS] {status}")
    return payload, 2 if failures else 0


def main() -> int:
    args = parse_args()
    try:
        _, exit_code = run(args)
        return exit_code
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
