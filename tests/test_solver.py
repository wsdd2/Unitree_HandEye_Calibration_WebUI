# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.solver import (  # noqa: E402
    MODE_EYE_IN_HAND,
    MODE_EYE_TO_HAND,
    load_capture_records,
    solve_handeye,
    validate_capture_records,
)
from handeye_calib.transforms import (  # noqa: E402
    average_transforms,
    axis_rotation,
    euler_to_rotation,
    invert_transform,
    transform_to_record,
)


def make_transform(rvec: tuple[float, float, float], xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transform[:3, 3] = xyz
    return transform


def synthetic_records(mode: str, count: int = 10) -> tuple[list[dict], np.ndarray]:
    external = make_transform((0.18, -0.12, 0.09), (0.08, -0.03, 0.12))
    fixed = make_transform((-0.15, 0.22, 0.11), (0.42, 0.06, 0.25))
    records = []
    for index in range(count):
        factor = index + 1
        hand2base = make_transform(
            (
                0.07 * np.sin(factor * 0.8),
                0.09 * np.cos(factor * 0.55),
                0.055 * np.sin(factor * 0.37 + 0.2),
            ),
            (
                0.28 + 0.018 * index,
                -0.12 + 0.025 * np.sin(factor),
                0.34 + 0.012 * np.cos(factor * 0.7),
            ),
        )
        base2hand = invert_transform(hand2base)
        if mode == MODE_EYE_IN_HAND:
            target2cam = invert_transform(external) @ base2hand @ fixed
        else:
            target2cam = invert_transform(external) @ hand2base @ fixed
        records.append(
            {
                "mode": mode,
                "image": f"{index:03d}.jpg",
                "camera_info": {"serial": "SYNTHETIC-001", "camera_name": "test"},
                "camera_intrinsics": {
                    "identifier": "intrinsics-v1",
                    "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
                },
                "robot_context": {"robot_model": "synthetic"},
                "board": {
                    "inner_corners_cols": 8,
                    "inner_corners_rows": 5,
                    "square_mm": 20.0,
                    "square_m": 0.02,
                },
                "hand_pose_input": {
                    "base_frame_name": "base",
                    "frame_name": "tool",
                    "translation_unit": "m",
                    "rotation_unit": "rad",
                    "euler_order": "ros_rpy_rzryrx",
                },
                "hand2base": transform_to_record(hand2base),
                "base2hand": transform_to_record(base2hand),
                "target2cam": transform_to_record(target2cam),
                "target_reprojection_rms_px": 0.1,
            }
        )
    return records, external


class TransformTests(unittest.TestCase):
    def test_ros_rpy_and_legacy_modes_are_explicit(self) -> None:
        angles = [0.2, -0.3, 0.4]
        ros = euler_to_rotation(angles, "ros_rpy_rzryrx", "rad")
        expected = axis_rotation("z", 0.4) @ axis_rotation("y", -0.3) @ axis_rotation("x", 0.2)
        legacy = euler_to_rotation(angles, "legacy_rxryrz", "rad")
        self.assertTrue(np.allclose(ros, expected))
        self.assertTrue(
            np.allclose(
                legacy,
                axis_rotation("x", 0.2) @ axis_rotation("y", -0.3) @ axis_rotation("z", 0.4),
            )
        )
        self.assertFalse(np.allclose(ros, legacy))
        legacy_zyx = euler_to_rotation(angles, "zyx", "rad")
        self.assertTrue(
            np.allclose(
                legacy_zyx,
                axis_rotation("z", 0.2) @ axis_rotation("y", -0.3) @ axis_rotation("x", 0.4),
            )
        )
        self.assertFalse(np.allclose(ros, legacy_zyx))

    def test_so3_average_handles_plus_minus_pi(self) -> None:
        transforms = []
        for angle in (np.pi - 0.01, -np.pi + 0.01):
            transform = np.eye(4)
            transform[:3, :3] = axis_rotation("z", angle)
            transforms.append(transform)
        mean = average_transforms(transforms)
        self.assertLess(mean[0, 0], -0.999)
        self.assertLess(mean[1, 1], -0.999)


class SolverTests(unittest.TestCase):
    def _solve_and_assert(
        self,
        mode: str,
        result_key: str,
        method: int = cv2.CALIB_HAND_EYE_PARK,
    ) -> None:
        records, expected = synthetic_records(mode)
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = solve_handeye(
                records,
                mode=mode,
                output_dir=Path(temp_dir),
                min_samples=8,
                method=method,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
        actual = np.asarray(result[result_key]["transform"])
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))
        self.assertEqual(result["quality"]["status"], "pass")
        self.assertIn("translation_residual_norm_mean_m", result["quality"])
        self.assertIn("pairwise_closure_max", result["quality"])
        self.assertEqual(
            len(result["quality"]["leave_one_out_external_stability"]["translation_delta_m_each"]),
            len(records),
        )

    def test_eye_in_hand_synthetic(self) -> None:
        self._solve_and_assert(MODE_EYE_IN_HAND, "cam2hand")

    def test_eye_in_hand_tsai_warns_for_weak_motion(self) -> None:
        records, _ = synthetic_records(MODE_EYE_IN_HAND)
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = solve_handeye(
                records,
                mode=MODE_EYE_IN_HAND,
                output_dir=Path(temp_dir),
                min_samples=8,
                method=cv2.CALIB_HAND_EYE_TSAI,
                include_leave_one_out=False,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["quality"]["status"], "warn")

    def test_eye_to_hand_reverse_direction_synthetic(self) -> None:
        self._solve_and_assert(MODE_EYE_TO_HAND, "cam2base")

    def test_mixed_samples_and_wrong_inverse_are_rejected(self) -> None:
        records, _ = synthetic_records(MODE_EYE_IN_HAND)
        records[-1]["mode"] = MODE_EYE_TO_HAND
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_capture_records(records)

        records, _ = synthetic_records(MODE_EYE_IN_HAND)
        records[-1]["base2hand"]["transform"][0][3] += 0.01
        with self.assertRaisesRegex(ValueError, "不是 hand2base 的逆矩阵"):
            validate_capture_records(records)

    def test_loader_ignores_non_capture_json(self) -> None:
        records, _ = synthetic_records(MODE_EYE_IN_HAND, count=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "capture.json").write_text(
                json.dumps(records[0]), encoding="utf-8"
            )
            (directory / "result.json").write_text(
                json.dumps({"calibration_method": MODE_EYE_IN_HAND}),
                encoding="utf-8",
            )
            loaded = load_capture_records(directory)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0]["_json_path"].endswith("capture.json"))

    def test_20260810_eye_to_hand_regression_when_handoff_is_available(self) -> None:
        workspace = PROJECT_ROOT.parent
        data_dir = (
            workspace
            / "_calib_handoff"
            / "abdomen_eye_to_hand_20260810"
            / "abdomen_right_20260810_105223"
            / "h2_abdomen_d435"
        )
        expected_path = (
            workspace
            / "_calib_handoff"
            / "abdomen_eye_to_hand_20260810"
            / "eye_to_hand_20260810_111551.json"
        )
        if not data_dir.is_dir() or not expected_path.is_file():
            self.skipTest("20260810 handoff dataset is not available")
        records = load_capture_records(data_dir)
        expected = np.asarray(
            json.loads(expected_path.read_text(encoding="utf-8"))["cam2reference"][
                "transform"
            ],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = solve_handeye(
                records,
                mode=MODE_EYE_TO_HAND,
                output_dir=Path(temp_dir),
                min_samples=12,
                method=cv2.CALIB_HAND_EYE_PARK,
            )
            actual = np.asarray(
                json.loads(result_path.read_text(encoding="utf-8"))["cam2reference"][
                    "transform"
                ],
                dtype=np.float64,
            )
        self.assertTrue(np.allclose(actual, expected, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
