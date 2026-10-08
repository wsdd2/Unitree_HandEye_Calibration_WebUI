# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.h2_arm_sdk import RIGHT_ARM_JOINTS
from handeye_calib.h2_ee_ik import resolve_urdf
from handeye_calib.h2_gravity import H2GravityCompensator, JOINT_TO_MOTOR
from handeye_calib.upper_arm_sweep import RAISE_POSE_DEG


class H2GravityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.urdf = resolve_urdf(
                str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf")
            )
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.comp = H2GravityCompensator(str(cls.urdf), wrist_payload_kg=0.20, payload_arm="right")

    def _right_command(self, deg: tuple[float, ...]) -> dict[int, float]:
        return {
            motor: math.radians(value)
            for motor, value in zip(RIGHT_ARM_JOINTS, deg)
        }

    def test_raise_pose_needs_nonzero_shoulder_pitch_tau(self) -> None:
        tau = self.comp.motor_tau(self._right_command(RAISE_POSE_DEG["right"]))
        shoulder = tau[JOINT_TO_MOTOR["right_shoulder_pitch_joint"]]
        elbow = tau[JOINT_TO_MOTOR["right_elbow_joint"]]
        self.assertGreater(abs(shoulder), 2.0)
        self.assertGreater(abs(elbow), 0.3)
        self.assertTrue(all(motor in tau for motor in RIGHT_ARM_JOINTS))

    def test_payload_increases_shoulder_load(self) -> None:
        bare = H2GravityCompensator(
            str(self.urdf),
            wrist_payload_kg=0.0,
            payload_arm="right",
        )
        heavy = H2GravityCompensator(
            str(self.urdf),
            wrist_payload_kg=0.8,
            payload_arm="right",
        )
        q = self._right_command(RAISE_POSE_DEG["right"])
        motor = JOINT_TO_MOTOR["right_shoulder_pitch_joint"]
        self.assertGreater(abs(heavy.motor_tau(q)[motor]), abs(bare.motor_tau(q)[motor]))

    def test_tau_scale_zero_is_off(self) -> None:
        off = H2GravityCompensator(str(self.urdf), tau_scale=0.0)
        tau = off.motor_tau(self._right_command(RAISE_POSE_DEG["right"]))
        self.assertTrue(all(abs(value) < 1e-9 for value in tau.values()))


if __name__ == "__main__":
    unittest.main()
