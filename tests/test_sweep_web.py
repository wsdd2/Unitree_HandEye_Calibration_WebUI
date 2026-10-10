from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.sweep_web import (  # noqa: E402
    apply_sweep_command,
    command_from_key,
    normalize_sweep_command,
)


class SweepCommandTests(unittest.TestCase):
    def test_normalize_and_keys(self) -> None:
        self.assertEqual(normalize_sweep_command("sweep_next"), "sweep_next")
        self.assertIsNone(normalize_sweep_command("save"))
        self.assertEqual(command_from_key("n"), "sweep_next")
        self.assertEqual(command_from_key("s"), "sweep_skip")
        self.assertEqual(command_from_key("q"), "sweep_quit")
        self.assertIsNone(command_from_key("w"))

    def test_menu_and_step(self) -> None:
        action, step, z_step, joint_step = apply_sweep_command(
            "sweep_next",
            controller=None,
            ik=None,
            arm="right",
            step_m=0.002,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            log=lambda _msg: None,
        )
        self.assertEqual(action, "next")
        self.assertEqual(step, 0.002)
        self.assertEqual(z_step, 0.02)
        self.assertEqual(joint_step, 2.0)
        _, step, z_step, joint_step = apply_sweep_command(
            "sweep_step_double",
            controller=None,
            ik=None,
            arm="right",
            step_m=0.002,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            log=lambda _msg: None,
        )
        self.assertAlmostEqual(step, 0.004)
        self.assertAlmostEqual(z_step, 0.04)
        _, step, z_step, joint_step = apply_sweep_command(
            "sweep_joint_step_double",
            controller=None,
            ik=None,
            arm="right",
            step_m=step,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            z_step_m=z_step,
            joint_step_deg=joint_step,
            log=lambda _msg: None,
        )
        self.assertEqual(joint_step, 4.0)

    def test_jog_calls_ik(self) -> None:
        class FakeIK:
            def fk_xyz_rpy(self, q):
                return [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]

            def apply_delta(self, q, dxyz, drpy):
                self.last = (list(q), list(dxyz), list(drpy))
                return [q[0] + dxyz[0], *q[1:]], True, 0.0, 0.0

        class FakeCtrl:
            def measured_arm_rad(self, arm):
                return [0.0] * 7

            def apply_arm_rad(self, arm, q, seconds, dt):
                self.last = list(q)

        ik = FakeIK()
        ctrl = FakeCtrl()
        action, step, z_step, _joint_step = apply_sweep_command(
            "sweep_x_plus",
            controller=ctrl,
            ik=ik,
            arm="right",
            step_m=0.002,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            log=lambda _msg: None,
        )
        self.assertIsNone(action)
        self.assertEqual(step, 0.002)
        self.assertEqual(ik.last[1][0], 0.002)
        self.assertAlmostEqual(ctrl.last[0], 0.002)

        class LiftIK(FakeIK):
            def __init__(self):
                self.z = 0.3

            def fk_xyz_rpy(self, q):
                return [0.1, 0.2, self.z], [0.0, 0.0, 0.0]

            def apply_delta(self, q, dxyz, drpy):
                self.last = (list(q), list(dxyz), list(drpy))
                self.z += dxyz[2]
                return list(q), True, 0.0, 0.0

        lift = LiftIK()
        apply_sweep_command(
            "sweep_z_plus",
            controller=ctrl,
            ik=lift,
            arm="right",
            step_m=0.002,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            z_step_m=0.02,
            log=lambda _msg: None,
        )
        self.assertAlmostEqual(lift.last[1][2], 0.02)

    def test_joint_jog_uses_quintic_and_limits(self) -> None:
        class FakeCtrl:
            def __init__(self):
                self.q = [0.0] * 7
                self.calls = []

            def commanded_arm_rad(self, arm):
                return list(self.q)

            def play_quintic_arm_rad(self, arm, q, seconds, dt):
                self.q = list(q)
                self.calls.append((arm, list(q), seconds, dt))

        ctrl = FakeCtrl()
        _action, _step, _z_step, joint_step = apply_sweep_command(
            "sweep_joint_1_minus",
            controller=ctrl,
            ik=None,
            arm="right",
            step_m=0.002,
            rot_deg=2.0,
            move_s=0.1,
            control_dt=0.02,
            joint_step_deg=2.0,
            log=lambda _msg: None,
        )
        self.assertEqual(joint_step, 2.0)
        self.assertEqual(len(ctrl.calls), 1)
        self.assertAlmostEqual(ctrl.q[0], -2.0 * 3.141592653589793 / 180.0)


class SweepHttpTests(unittest.TestCase):
    def test_buttons_wait_for_terminal_b(self) -> None:
        try:
            from handeye_calib.debug_stream import DebugStreamServer
        except ModuleNotFoundError as exc:
            if "flask" in str(exc).lower():
                self.skipTest("flask not installed")
            raise
        server = DebugStreamServer(host="127.0.0.1", port=0)
        client = server._app.test_client()
        html = client.get("/").get_data(as_text=True)
        self.assertIn("data-sweep", html)
        self.assertIn("sweep_next", html)
        self.assertIn("sweep_joint_1_plus", html)
        self.assertIn("data-sweep-jog", html)
        self.assertIn("等待终端 B 接入", html)
        state = client.get("/state").get_json()
        self.assertFalse(state["sweep"]["connected"])
        self.assertFalse(state["sweep"]["accepts_input"])
        rejected = client.post("/sweep/command", json={"command": "sweep_next"})
        self.assertEqual(rejected.status_code, 409)
        client.post(
            "/sweep/heartbeat",
            json={"accepts_input": False, "accepted": 0, "target": 0, "phase": "raising"},
        )
        busy = client.get("/state").get_json()
        self.assertTrue(busy["sweep"]["connected"])
        self.assertFalse(busy["sweep"]["accepts_input"])
        still_blocked = client.post("/sweep/command", json={"command": "sweep_next"})
        self.assertEqual(still_blocked.status_code, 409)
        client.post(
            "/sweep/heartbeat",
            json={"accepts_input": True, "accepted": 1, "target": 0, "phase": "dwell"},
        )
        ready = client.get("/state").get_json()
        self.assertTrue(ready["sweep"]["accepts_input"])
        accepted = client.post("/sweep/command", json={"command": "sweep_x_plus"})
        self.assertEqual(accepted.status_code, 200)
        # Repeated held-button jogs are coalesced instead of accumulating.
        accepted = client.post("/sweep/command", json={"command": "sweep_x_plus"})
        self.assertEqual(accepted.status_code, 200)
        popped = client.get("/sweep/command").get_json()
        self.assertEqual(popped["command"], "sweep_x_plus")
        empty = client.get("/sweep/command").get_json()
        self.assertIsNone(empty["command"])


if __name__ == "__main__":
    unittest.main()
