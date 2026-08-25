from __future__ import annotations

import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, WORKSPACE_ROOT / "robot_kinematics"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from handeye_calib.validation import mount_mode_mismatch  # noqa: E402
from ros_joint_state_bridge import ROSJointStateBridge  # noqa: E402
from unitree_sdk2_bridge import (  # noqa: E402
    JointSample,
    TimedJointState,
    UnitreeG1LowStateBridge,
)


class MountModeTests(unittest.TestCase):
    def test_fixed_mount_rejected_for_eye_in_hand(self) -> None:
        self.assertIsNotNone(mount_mode_mismatch("eye_in_hand", "abdomen"))

    def test_wrist_mount_rejected_for_eye_to_hand(self) -> None:
        self.assertIsNotNone(mount_mode_mismatch("eye_to_hand", "wrist"))

    def test_matching_mounts_are_allowed(self) -> None:
        self.assertIsNone(mount_mode_mismatch("eye_to_hand", "torso"))
        self.assertIsNone(mount_mode_mismatch("eye_in_hand", "hand"))


class LowStateHistoryTests(unittest.TestCase):
    def make_bridge(self) -> UnitreeG1LowStateBridge:
        bridge = UnitreeG1LowStateBridge.__new__(UnitreeG1LowStateBridge)
        bridge._lock = threading.Lock()
        bridge._history = deque(maxlen=8)
        bridge._latest = {}
        bridge._latest_host_monotonic_sec = None
        return bridge

    def test_nearest_state_uses_host_timestamp(self) -> None:
        bridge = self.make_bridge()
        now = time.monotonic()
        first = {"joint": JointSample(1.0, 0.0, 0.0)}
        second = {"joint": JointSample(2.0, 0.0, 0.0)}
        bridge._history.extend(
            [
                TimedJointState(now - 0.08, first),
                TimedJointState(now - 0.02, second),
            ]
        )
        matched = bridge.nearest_state(now - 0.03, max_delta_sec=0.02)
        self.assertEqual(matched.joints["joint"].q, 2.0)

    def test_nearest_state_rejects_excessive_delta(self) -> None:
        bridge = self.make_bridge()
        now = time.monotonic()
        bridge._history.append(
            TimedJointState(now - 0.5, {"joint": JointSample(1.0, 0.0, 0.0)})
        )
        with self.assertRaises(TimeoutError):
            bridge.nearest_state(now, max_delta_sec=0.1)


class ROSJointStateHistoryTests(unittest.TestCase):
    def test_callback_records_position_velocity_and_receipt_time(self) -> None:
        bridge = ROSJointStateBridge.__new__(ROSJointStateBridge)
        bridge._lock = threading.Lock()
        bridge._history = deque(maxlen=8)
        bridge._latest = {}
        bridge._latest_host_monotonic_sec = None
        bridge._ready = threading.Event()
        message = type(
            "JointStateMessage",
            (),
            {
                "name": ["right_shoulder_pitch_joint", "right_elbow_joint"],
                "position": [0.2, 0.4],
                "velocity": [0.01],
            },
        )()
        before = time.monotonic()
        bridge._on_joint_state(message)
        after = time.monotonic()
        matched = bridge.nearest_state(after, max_delta_sec=0.1)
        self.assertGreaterEqual(matched.host_monotonic_sec, before)
        self.assertEqual(
            matched.joints["right_shoulder_pitch_joint"].dq,
            0.01,
        )
        self.assertEqual(matched.joints["right_elbow_joint"].dq, 0.0)


if __name__ == "__main__":
    unittest.main()
