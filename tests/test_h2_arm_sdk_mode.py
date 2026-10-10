from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.h2_arm_sdk import arm_sdk_mode_ready  # noqa: E402


class ArmSdkModeReadyTests(unittest.TestCase):
    def test_named_ai_is_ready(self) -> None:
        ok, why = arm_sdk_mode_ready(0, {"form": "0", "name": "ai"}, 0, 703)
        self.assertTrue(ok)
        self.assertEqual(why, "ai")

    def test_blank_name_with_phase_walk_is_ready(self) -> None:
        ok, why = arm_sdk_mode_ready(0, {"form": "0", "name": ""}, 0, 703)
        self.assertTrue(ok)
        self.assertIn("703", why)
        self.assertIn("PhaseWalk", why)

    def test_failed_check_with_phase_walk_is_ready(self) -> None:
        ok, _why = arm_sdk_mode_ready(3104, None, 0, 703)
        self.assertTrue(ok)

    def test_blank_name_while_passive_is_not_ready(self) -> None:
        ok, why = arm_sdk_mode_ready(0, {"name": ""}, 0, 1)
        self.assertFalse(ok)
        self.assertEqual(why, "")

    def test_explicit_other_name_is_not_overridden_by_fsm(self) -> None:
        ok, why = arm_sdk_mode_ready(0, {"name": "normal"}, 0, 703)
        self.assertFalse(ok)
        self.assertEqual(why, "normal")

    def test_rpc_failure_without_fsm_is_not_ready(self) -> None:
        ok, why = arm_sdk_mode_ready(3104, None, None, None)
        self.assertFalse(ok)
        self.assertEqual(why, "")


if __name__ == "__main__":
    unittest.main()
