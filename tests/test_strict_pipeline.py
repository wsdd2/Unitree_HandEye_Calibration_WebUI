from __future__ import annotations

import unittest

import cv2
import numpy as np

from handeye_calib.solver import MODE_EYE_IN_HAND
from handeye_calib.strict_pipeline import (
    CaptureThresholds,
    HandEyeThresholds,
    strict_handeye_report,
    transform_delta,
    validate_capture_preflight,
    validate_live_fk_snapshot,
)
from handeye_calib.transforms import invert_transform, transform_to_record


RIGHT_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def make_transform(
    rvec: tuple[float, float, float],
    xyz: tuple[float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transform[:3, 3] = xyz
    return transform


def strict_records(count: int = 30) -> tuple[list[dict], np.ndarray]:
    external = make_transform((0.16, -0.10, 0.08), (0.09, -0.04, 0.11))
    fixed = make_transform((-0.11, 0.18, 0.07), (0.48, 0.02, 0.28))
    records = []
    for index in range(count):
        factor = index + 1
        hand2base = make_transform(
            (
                0.23 * np.sin(factor * 0.47),
                0.19 * np.cos(factor * 0.31),
                0.17 * np.sin(factor * 0.73 + 0.2),
            ),
            (
                0.24 + 0.006 * index,
                -0.17 + 0.06 * np.sin(factor * 0.4),
                0.26 + 0.05 * np.cos(factor * 0.29),
            ),
        )
        target2cam = invert_transform(external) @ invert_transform(hand2base) @ fixed
        corner_x = 20.0 if index % 2 == 0 else 1260.0
        corner_y = 20.0 if (index // 2) % 2 == 0 else 700.0
        records.append(
            {
                "mode": MODE_EYE_IN_HAND,
                "image": f"{index:03d}.jpg",
                "_json_path": f"/tmp/{index:03d}.json",
                "camera_info": {
                    "serial": "346522074739",
                    "selected_serial": "346522074739",
                    "width": 1280,
                    "height": 720,
                },
                "camera_intrinsics": {
                    "manifest_status": "validated",
                    "manifest_path": "/tmp/intrinsics_manifest.json",
                    "identifier": "strict-intrinsics",
                },
                "board": {
                    "inner_corners_cols": 8,
                    "inner_corners_rows": 5,
                    "square_mm": 20.0,
                },
                "hand_pose_input": {
                    "base_frame_name": "torso_link",
                    "frame_name": "right_wrist_yaw_link",
                },
                "hand2base": transform_to_record(hand2base),
                "base2hand": transform_to_record(invert_transform(hand2base)),
                "target2cam": transform_to_record(target2cam),
                "target_reprojection_rms_px": 0.1,
                "capture_timing": {
                    "fk": {
                        "sync_delta_sec": 0.005,
                        "state_age_sec_at_snapshot": 0.01,
                    }
                },
                "fk_metadata": {
                    "model": {"urdf_sha256": "strict-urdf"},
                    "arm_joints_rad": {
                        name: float(index) * 0.001 for name in RIGHT_JOINTS
                    },
                    "arm_joint_velocities_rad_s": {
                        name: 0.0 for name in RIGHT_JOINTS
                    },
                },
                "image_points_px": [
                    [corner_x, corner_y],
                    [640.0, 360.0],
                ],
            }
        )
    return records, external


class StrictPipelineTests(unittest.TestCase):
    def test_exact_synthetic_handeye_is_deployable(self) -> None:
        records, expected = strict_records()
        actual, report = strict_handeye_report(
            records,
            mode=MODE_EYE_IN_HAND,
            primary_method="park",
            capture_thresholds=CaptureThresholds(),
            handeye_thresholds=HandEyeThresholds(
                subset_trials=20,
                subset_fraction=0.70,
            ),
            expected_serial="346522074739",
            expected_size=(1280, 720),
            expected_base_frame="torso_link",
            expected_hand_frame="right_wrist_yaw_link",
            expected_urdf_sha256="strict-urdf",
        )
        self.assertTrue(report["deployable"], report["failures"])
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))

    def test_legacy_intrinsics_are_rejected(self) -> None:
        records, _ = strict_records()
        records[0]["camera_intrinsics"]["manifest_status"] = "legacy_missing"
        report = validate_capture_preflight(
            records,
            thresholds=CaptureThresholds(),
            expected_serial="346522074739",
            expected_size=(1280, 720),
            expected_base_frame="torso_link",
            expected_hand_frame="right_wrist_yaw_link",
            expected_urdf_sha256="strict-urdf",
        )
        self.assertFalse(report["pass"])
        self.assertTrue(
            any("manifest" in failure for failure in report["failures"])
        )

    def test_missing_joint_velocity_is_rejected(self) -> None:
        records, _ = strict_records()
        del records[3]["fk_metadata"]["arm_joint_velocities_rad_s"][
            "right_wrist_yaw_joint"
        ]
        report = validate_capture_preflight(
            records,
            thresholds=CaptureThresholds(),
            expected_serial="346522074739",
            expected_size=(1280, 720),
            expected_base_frame="torso_link",
            expected_hand_frame="right_wrist_yaw_link",
        )
        self.assertFalse(report["pass"])
        self.assertTrue(
            any("missing arm velocities" in failure for failure in report["failures"])
        )

    def test_live_snapshot_requires_complete_stationary_arm(self) -> None:
        snapshot = {
            "arm_joints": {name: 0.0 for name in RIGHT_JOINTS},
            "arm_joint_velocities_rad_s": {
                name: 0.0 for name in RIGHT_JOINTS
            },
        }
        validate_live_fk_snapshot(
            snapshot,
            arm_side="right",
            max_joint_velocity_rad_s=0.01,
        )
        del snapshot["arm_joint_velocities_rad_s"]["right_wrist_yaw_joint"]
        with self.assertRaisesRegex(RuntimeError, "missing joint velocities"):
            validate_live_fk_snapshot(
                snapshot,
                arm_side="right",
                max_joint_velocity_rad_s=0.01,
            )
        snapshot["arm_joint_velocities_rad_s"]["right_wrist_yaw_joint"] = 1.0
        validate_live_fk_snapshot(
            snapshot,
            arm_side="right",
            max_joint_velocity_rad_s=0.0,
        )

    def test_transform_delta_detects_rotation_and_translation(self) -> None:
        reference = np.eye(4, dtype=np.float64)
        candidate = make_transform(
            (0.0, 0.0, np.deg2rad(10.0)),
            (0.01, 0.0, 0.0),
        )
        delta = transform_delta(reference, candidate)
        self.assertAlmostEqual(delta["translation_norm_m"], 0.01)
        self.assertAlmostEqual(delta["rotation_deg"], 10.0)


if __name__ == "__main__":
    unittest.main()
