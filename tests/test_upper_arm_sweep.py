from __future__ import annotations

import random
import sys
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.h2_arm_sdk import (  # noqa: E402
    ARM_JOINTS,
    H2ArmSdkController,
    HOLD_JOINTS,
    RIGHT_ARM_JOINTS,
    WEIGHT_INDEX,
    _sdk_search_roots,
)
from handeye_calib.upper_arm_sweep import (  # noqa: E402
    DEFAULT_SEED_DEG,
    RAISE_POSE_DEG,
    URDF_LIMITS_RAD,
    ReachablePoseSampler,
    clamp_pose_deg,
    pose_l2_deg,
    reachable_box_deg,
    render_multi_joint_yaml,
    render_single_joint_yaml,
    sample_reachable_poses,
)


class UpperArmSweepTests(unittest.TestCase):
    def test_samples_requested_count_plus_seed_return(self) -> None:
        poses = sample_reachable_poses(24, rng=random.Random(7))
        self.assertEqual(len(poses), 26)
        self.assertEqual(poses[0].name, "raise")
        self.assertEqual(poses[-1].name, "raise")
        self.assertEqual(len(poses[0].deg), 7)
        self.assertEqual(poses[0].deg, RAISE_POSE_DEG["right"])

    def test_poses_stay_inside_reachable_and_urdf_box(self) -> None:
        box = reachable_box_deg("right")
        poses = sample_reachable_poses(20, rng=random.Random(3), return_to_seed=False)
        for pose in poses:
            for value, (lo, hi), (urdf_lo, urdf_hi) in zip(
                pose.deg, box, URDF_LIMITS_RAD["right"]
            ):
                self.assertGreaterEqual(value, lo - 1e-6)
                self.assertLessEqual(value, hi + 1e-6)
                self.assertGreaterEqual(value, __import__("math").degrees(urdf_lo) - 1e-3)
                self.assertLessEqual(value, __import__("math").degrees(urdf_hi) + 1e-3)

    def test_samples_stay_near_raised_pose(self) -> None:
        poses = sample_reachable_poses(20, rng=random.Random(3), return_to_seed=False)
        for pose in poses:
            self.assertLessEqual(pose.deg[0], -47.0)
            self.assertGreaterEqual(pose.deg[0], -63.0)

    def test_upper_arm_only_keeps_wrist_on_seed(self) -> None:
        seed = DEFAULT_SEED_DEG["right"]
        poses = sample_reachable_poses(
            12,
            seed_deg=seed,
            upper_arm_only=True,
            return_to_seed=False,
            rng=random.Random(11),
        )
        for pose in poses:
            self.assertAlmostEqual(pose.deg[4], seed[4], places=3)
            self.assertAlmostEqual(pose.deg[5], seed[5], places=3)
            self.assertAlmostEqual(pose.deg[6], seed[6], places=3)

    def test_min_separation_rejects_near_duplicates(self) -> None:
        poses = sample_reachable_poses(
            10,
            min_separation_deg=4.0,
            return_to_seed=False,
            rng=random.Random(21),
        )
        random_poses = [pose for pose in poses if pose.name != "raise"]
        for i, a in enumerate(random_poses):
            for b in random_poses[i + 1 :]:
                self.assertGreaterEqual(pose_l2_deg(a.deg, b.deg), 4.0 - 1e-6)

    def test_multi_yaml_matches_h2_arm_schema(self) -> None:
        poses = sample_reachable_poses(8, rng=random.Random(1), return_to_seed=False)
        text = render_multi_joint_yaml(poses, arm="right", move_duration=3.0)
        self.assertIn('control_mode: "right"', text)
        self.assertIn("waypoints:", text)
        waypoint_lines = [line for line in text.splitlines() if line.startswith("  - [")]
        self.assertEqual(len(waypoint_lines), 9)
        first = waypoint_lines[0]
        self.assertEqual(first.count(","), 6)

    def test_single_yaml_writes_right_target(self) -> None:
        poses = sample_reachable_poses(1, rng=random.Random(2), return_to_seed=False)
        text = render_single_joint_yaml(poses[0], arm="right")
        self.assertIn('control_mode: "right"', text)
        self.assertIn("right_target:", text)
        self.assertIn("left_target:", text)

    def test_sampler_can_keep_drawing(self) -> None:
        sampler = ReachablePoseSampler(rng=random.Random(5), recent_window=40)
        poses = [sampler.next_pose() for _ in range(80)]
        self.assertEqual(len(poses), 80)
        box = reachable_box_deg("right")
        for pose in poses:
            for value, (lo, hi) in zip(pose.deg, box):
                self.assertGreaterEqual(value, lo - 1e-6)
                self.assertLessEqual(value, hi + 1e-6)

    def test_clamp_keeps_values_inside_box(self) -> None:
        box = ((-10.0, 10.0),) * 7
        clamped = clamp_pose_deg((99.0, -99.0, 0.0, 0.0, 0.0, 0.0, 0.0), box)
        self.assertEqual(clamped[0], 10.0)
        self.assertEqual(clamped[1], -10.0)

    def test_sdk_joint_map_matches_h2_right_arm(self) -> None:
        self.assertEqual(RIGHT_ARM_JOINTS, (22, 23, 24, 25, 26, 27, 28))
        self.assertEqual(ARM_JOINTS["right"], RIGHT_ARM_JOINTS)
        self.assertEqual(len(ARM_JOINTS["left"]), 7)
        self.assertEqual(WEIGHT_INDEX, 31)
        self.assertIn(22, HOLD_JOINTS)
        self.assertIn(15, HOLD_JOINTS)

    def test_sdk_search_includes_mscapetech_copy(self) -> None:
        roots = [path.as_posix() for path in _sdk_search_roots()]
        self.assertTrue(any(path.endswith("unitree_sdk2_python") for path in roots))

    def test_engage_does_not_drop_weight_when_already_engaged(self) -> None:
        ctrl = object.__new__(H2ArmSdkController)
        ctrl._state_lock = threading.Lock()
        ctrl._servo_thread = object()
        ctrl._weight = 1.0
        ctrl.write = lambda *args, **kwargs: self.fail("already-engaged controller must not ramp weight")
        ctrl.engage(seconds=0.001, dt=0.001)

    def test_ramp_tracking_error_does_not_reengage_and_drop_arm(self) -> None:
        ctrl = object.__new__(H2ArmSdkController)
        ctrl.hold_positions = lambda: {joint: 0.0 for joint in HOLD_JOINTS}
        ctrl.measured_arm_deg = lambda arm: [0.0] * 7
        writes = []
        ctrl.write = lambda command, weight=1.0: writes.append((dict(command), weight))
        ctrl.engage = lambda *args, **kwargs: self.fail("tracking lag must not re-engage")
        ctrl.ramp_arm("right", (-30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.001, 0.001)
        self.assertEqual(len(writes), 2)
        self.assertTrue(all(weight == 1.0 for _, weight in writes))

    def test_camera_ik_small_x_step(self) -> None:
        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        seed = [__import__("math").radians(v) for v in RAISE_POSE_DEG["right"]]
        xyz0, _ = ik.fk_xyz_rpy(seed)
        q_new, ok, pos_err, _ = ik.apply_delta(seed, dxyz=(0.01, 0.0, 0.0))
        self.assertTrue(ok)
        self.assertLess(pos_err, 0.008)
        xyz1, _ = ik.fk_xyz_rpy(q_new)
        self.assertGreater(xyz1[0] - xyz0[0], 0.005)
        q_up, ok_up, pos_up, _ = ik.apply_delta(seed, dxyz=(0.0, 0.0, 0.01))
        self.assertTrue(ok_up)
        self.assertLess(pos_up, 0.008)
        xyz_up, _ = ik.fk_xyz_rpy(q_up)
        self.assertGreater(xyz_up[2] - xyz0[2], 0.005)

    def test_tip_steps_do_not_wind_the_wrist(self) -> None:
        import math

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        posture = [math.radians(value) for value in RAISE_POSE_DEG["right"]]
        q = posture[:]
        tip0 = ik.tip_xyz(q)
        for _ in range(12):
            q, ok, remain, jump = ik.apply_tip_delta(q, (0.004, 0.003, -0.002), posture)
            self.assertTrue(ok, f"remain={remain:.4f} jump={jump:.3f}")
        tip1 = ik.tip_xyz(q)
        self.assertGreater(tip1[0] - tip0[0], 0.015)
        self.assertGreater(tip1[1] - tip0[1], 0.02)
        self.assertLess(tip1[2] - tip0[2], -0.01)
        for index in (2, 4, 6):
            self.assertLess(abs(q[index] - posture[index]), math.radians(20.0))

    def test_tip_step_far_from_the_raise_uses_the_current_pose(self) -> None:
        import math

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        raise_pose = [math.radians(value) for value in (-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0)]
        here = [math.radians(value) for value in (-40.0, -20.0, 10.0, 20.0, -15.0, 10.0, 8.0)]
        step = (0.006, -0.001, -0.002)
        _, pulled_ok, pulled_remain, pulled_jump = ik.apply_tip_delta(here, step, raise_pose)
        _, ok, remain, jump = ik.apply_tip_delta(here, step, here)
        self.assertFalse(pulled_ok)
        self.assertGreater(pulled_jump, math.radians(14.0))
        self.assertTrue(ok, f"remain={remain:.4f} jump={jump:.3f}")
        self.assertLess(remain, 0.002)
        self.assertLess(jump, math.radians(5.0))

    def test_loaded_delta_keeps_support_offset_and_moves_actual_pose_forward(self) -> None:
        import math
        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        commanded = np.radians([-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0])
        measured = np.radians([-41.0, -19.0, 1.0, 50.0, 1.0, -5.0, -2.0])
        xyz0, _ = ik.fk_xyz_rpy(measured)
        q_new, ok, pos_err, _ = ik.apply_loaded_delta(
            commanded,
            measured,
            dxyz=(0.005, 0.0, 0.0),
        )
        self.assertTrue(ok, f"pos_err={pos_err:.4f}")
        inferred_actual = measured + (np.asarray(q_new) - commanded)
        xyz1, _ = ik.fk_xyz_rpy(inferred_actual)
        self.assertGreater(xyz1[0] - xyz0[0], 0.002)

    def test_loaded_tip_target_does_not_require_a_wrist_flip(self) -> None:
        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        commanded = np.radians([-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0])
        measured = np.radians([-50.3, -17.1, 1.9, 73.8, 0.1, 1.9, 0.1])
        goal = measured + np.radians([-2.0, 1.0, 1.0, -3.0, 1.0, -1.0, 1.0])
        target = np.asarray(ik.tip_xyz(goal))
        q_new, ok, remain, _ = ik.apply_loaded_tip_target(commanded, measured, target)
        self.assertTrue(ok, f"remain={remain:.4f}")
        inferred_actual = measured + (np.asarray(q_new) - commanded)
        tip = np.asarray(ik.tip_xyz(inferred_actual))
        self.assertLess(float(np.linalg.norm(target - tip)), 0.010)
        self.assertLess(float(np.max(np.abs(np.degrees(np.asarray(q_new) - commanded)))), 70.0)

    def test_loaded_tip_target_reaches_from_a_hang_when_seeded_at_the_raise(self) -> None:
        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        hang = np.radians([7.7, -4.3, 13.2, 74.9, 1.0, 1.1, 0.9])
        ready = np.radians([-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0])
        target = np.asarray(ik.tip_xyz(ready))
        q_new, ok, remain, _ = ik.apply_loaded_tip_target(
            hang,
            hang,
            target,
            freeze_wrist=True,
            posture_q_rad=ready,
        )
        self.assertTrue(ok, f"remain={remain:.4f}")
        self.assertLess(remain, 0.010)
        self.assertTrue(np.allclose(q_new[4:7], hang[4:7], atol=1e-9))
        jump = float(np.max(np.abs(np.degrees(np.asarray(q_new) - hang))))
        self.assertLess(jump, 120.0)
        self.assertLess(float(np.degrees(q_new[0])), -40.0)
        self.assertLess(abs(float(np.degrees(q_new[2]))), 40.0)

    def test_level_forearm_reaches_button_height_without_folding_the_elbow(self) -> None:
        import math

        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        hang = np.radians([7.7, -4.3, 13.2, 74.9, 1.0, 1.1, 0.9])
        level = np.radians([-90.0, -15.0, 0.0, 85.0, 0.0, 0.0, 0.0])
        target = np.array([0.61, -0.09, 0.40])
        q_new, ok, remain, _ = ik.apply_loaded_tip_target(
            hang,
            hang,
            target,
            freeze_wrist=True,
            posture_q_rad=level,
            level_forearm=True,
        )
        self.assertTrue(ok, f"remain={remain:.4f} q={np.degrees(q_new)}")
        self.assertLess(remain, 0.02)
        self.assertGreater(float(np.degrees(q_new[3])), 60.0)
        axis = ik._forearm_axis(q_new)
        self.assertGreater(float(axis[0]), 0.9)
        self.assertLess(abs(float(axis[2])), 0.12)
        folded, folded_ok, _, _ = ik.apply_loaded_tip_target(
            hang,
            hang,
            target,
            freeze_wrist=True,
            posture_q_rad=level,
            level_forearm=False,
        )
        self.assertTrue(folded_ok)
        self.assertLess(abs(float(np.degrees(folded[3]))), 40.0)

    def test_loaded_tip_target_can_freeze_all_wrist_joints(self) -> None:
        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        measured = np.radians([-55.0, -20.0, 5.0, 55.0, 8.0, -12.0, 6.0])
        goal = measured + np.radians([-1.0, 1.0, 1.0, -2.0, 0.0, 0.0, 0.0])
        target = np.asarray(ik.tip_xyz(goal))
        q_new, ok, remain, _ = ik.apply_loaded_tip_target(
            measured,
            measured,
            target,
            freeze_wrist=True,
        )
        self.assertTrue(ok, f"remain={remain:.4f}")
        self.assertTrue(np.allclose(q_new[4:7], measured[4:7], atol=1e-9))
        self.assertLess(
            float(np.linalg.norm(np.asarray(ik.tip_xyz(q_new)) - target)),
            0.010,
        )

    def test_loaded_tool_pose_targets_right_ee_link_offset(self) -> None:
        import numpy as np

        from handeye_calib.h2_ee_ik import H2CameraIK, _rpy_from_matrix, resolve_urdf

        try:
            urdf = resolve_urdf(str(PROJECT_ROOT.parent / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"))
        except RuntimeError:
            self.skipTest("H2.urdf not available")
        ik = H2CameraIK(urdf_path=str(urdf), arm="right")
        measured = np.radians([-76.0, -25.0, 0.0, 65.0, 0.0, 0.0, 0.0])
        goal = measured + np.radians([1.0, -1.0, 1.0, -1.0, 0.5, 0.5, -0.5])
        goal_pose = np.asarray(ik.fk_matrix(goal), dtype=float)
        offset = goal_pose[:3, :3] @ np.asarray([0.205, 0.0, 0.0])
        tool_xyz = goal_pose[:3, 3] + offset
        tool_rpy = _rpy_from_matrix(goal_pose)
        q_new, ok, pos_err, ori_err = ik.apply_loaded_tool_pose_target(
            measured,
            measured,
            tool_xyz,
            tool_rpy,
        )
        self.assertTrue(ok, f"pos={pos_err:.4f} ori={ori_err:.4f}")
        solved_pose = np.asarray(ik.fk_matrix(q_new), dtype=float)
        solved_tip = solved_pose[:3, 3] + solved_pose[:3, :3] @ np.asarray([0.205, 0.0, 0.0])
        self.assertLess(float(np.linalg.norm(solved_tip - tool_xyz)), 0.008)


if __name__ == "__main__":
    unittest.main()
