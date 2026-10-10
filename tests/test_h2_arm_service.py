from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.h2_arm_service import (  # noqa: E402
    H2ArmServiceController,
    write_joint_yaml,
)


class H2ArmServiceTests(unittest.TestCase):
    def test_joint_yaml_uses_service_gains_and_degrees(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "joint_point.yaml"
            write_joint_yaml(
                path,
                arm="right",
                target_deg=[-55, -14, 0, 40, -90, 0, 0],
                seconds=0.4,
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn('control_mode: "right"', text)
        self.assertIn("move_duration: 0.400", text)
        self.assertIn("arm_kp: 140.0", text)
        self.assertIn("right_target: [-55.000000, -14.000000", text)

    def test_controller_records_only_successful_target(self) -> None:
        calls = []

        def caller(service, timeout):
            calls.append((service, timeout))
            return True, "ok"

        with tempfile.TemporaryDirectory() as folder:
            controller = H2ArmServiceController(
                Path(folder),
                caller=caller,
            )
            target = [math.radians(value) for value in (-55, -14, 0, 40, -90, 0, 0)]
            controller.play_quintic_arm_rad(
                "right",
                target,
                seconds=0.5,
                dt=0.02,
            )
            written = (
                Path(folder) / "single" / "joint_point.yaml"
            ).read_text(encoding="utf-8")
        self.assertEqual(len(calls), 1)
        self.assertEqual(controller.commanded_arm_rad("right"), target)
        self.assertIn("move_duration: 0.500", written)


if __name__ == "__main__":
    unittest.main()
