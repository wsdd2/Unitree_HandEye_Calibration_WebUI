# -*- coding: utf-8 -*-
"""Move the right wrist/tool toward a board-center coordinate from Plan B.

This is a guarded motion script. By default it only computes FK/IK and prints
the planned target. It publishes rt/arm_sdk commands only when both
``--execute`` and ``--confirm-touch-motion`` are provided.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_URDF = WORKSPACE_ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"

ROBOT_KINEMATICS_DIR = WORKSPACE_ROOT / "robot_kinematics"
JOINT_TO_POSE_DIR = ROBOT_KINEMATICS_DIR / "joint_to_pose"
POSE_TO_JOINT_DIR = ROBOT_KINEMATICS_DIR / "pose_to_joint"
for module_dir in (ROBOT_KINEMATICS_DIR, JOINT_TO_POSE_DIR, POSE_TO_JOINT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from fk_urdf import URDFFK, base_pose_matrix  # noqa: E402
from ik_urdf import URDFIK  # noqa: E402


ARM_SDK_WEIGHT = 29
WAIST_JOINTS = [12, 13, 14]
LEFT_ARM_JOINTS = list(range(15, 22))
RIGHT_ARM_JOINTS = list(range(22, 29))
UPPER_BODY_COMMAND_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

G1_JOINT_INDEX = {
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}
INDEX_TO_G1_JOINT = {index: name for name, index in G1_JOINT_INDEX.items()}
RIGHT_ARM_JOINT_NAMES = [INDEX_TO_G1_JOINT[index] for index in RIGHT_ARM_JOINTS]


class LowStateArmSDK:
    def __init__(self, network_interface: str, domain_id: int, kp: float, kd: float) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: WPS433
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: WPS433
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: WPS433
        from unitree_sdk2py.utils.crc import CRC  # noqa: WPS433

        self.kp = float(kp)
        self.kd = float(kd)
        self._low_cmd_default = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()
        self._latest_lowstate: Any = None

        if network_interface:
            ChannelFactoryInitialize(domain_id, network_interface)
        else:
            ChannelFactoryInitialize(domain_id)

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()

    def _on_lowstate(self, msg: Any) -> None:
        self._latest_lowstate = msg

    def wait_for_lowstate(self, timeout: float) -> Any:
        deadline = time.time() + timeout
        while self._latest_lowstate is None and time.time() < deadline:
            time.sleep(0.02)
        if self._latest_lowstate is None:
            raise RuntimeError("No rt/lowstate received. Check --network-interface and robot state.")
        return self._latest_lowstate

    def joint_positions_by_index(self) -> dict[int, float]:
        msg = self.wait_for_lowstate(2.0)
        return {index: float(msg.motor_state[index].q) for index in UPPER_BODY_COMMAND_JOINTS}

    def joint_positions_by_name(self) -> dict[str, float]:
        msg = self.wait_for_lowstate(2.0)
        return {
            name: float(msg.motor_state[index].q)
            for name, index in G1_JOINT_INDEX.items()
        }

    def write_arm_command(self, command_q: dict[int, float], weight: float, active_joints: Optional[list[int]] = None) -> None:
        cmd = self._low_cmd_default()
        lowstate = self._latest_lowstate
        if lowstate is not None:
            if hasattr(cmd, "mode_pr") and hasattr(lowstate, "mode_pr"):
                cmd.mode_pr = int(lowstate.mode_pr)
            if hasattr(cmd, "mode_machine") and hasattr(lowstate, "mode_machine"):
                cmd.mode_machine = int(lowstate.mode_machine)

        cmd.motor_cmd[ARM_SDK_WEIGHT].q = float(weight)
        active = set(UPPER_BODY_COMMAND_JOINTS if active_joints is None else active_joints)
        for joint in active:
            motor_cmd = cmd.motor_cmd[joint]
            if hasattr(motor_cmd, "mode"):
                motor_cmd.mode = 1
            motor_cmd.q = float(command_q[joint])
            motor_cmd.dq = 0.0
            motor_cmd.kp = self.kp
            motor_cmd.kd = self.kd
            motor_cmd.tau = 0.0
        cmd.crc = self._crc.Crc(cmd)
        self._publisher.Write(cmd)

    def ramp_to(self, target_q: dict[int, float], seconds: float, hz: float) -> None:
        start_q = self.joint_positions_by_index()
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for step in range(steps):
            ratio = float(step + 1) / float(steps)
            command_q = dict(start_q)
            for joint in UPPER_BODY_COMMAND_JOINTS:
                command_q[joint] = start_q[joint] * (1.0 - ratio) + target_q[joint] * ratio
            self.write_arm_command(command_q, weight=1.0)
            time.sleep(dt)

    def hold(self, target_q: dict[int, float], seconds: float, hz: float) -> None:
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            self.write_arm_command(target_q, weight=1.0)
            time.sleep(dt)

    def release(self, seconds: float, hz: float) -> None:
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            self.write_arm_command({}, weight=0.0, active_joints=[])
            time.sleep(dt)


def as_transform(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape={arr.shape}")
    return arr


def make_target_pose(current_wrist_pose: np.ndarray, target_base_xyz: np.ndarray, tool_offset_wrist_xyz: np.ndarray) -> np.ndarray:
    target_pose = np.array(current_wrist_pose, dtype=np.float64)
    target_pose[:3, 3] = target_base_xyz - current_wrist_pose[:3, :3] @ tool_offset_wrist_xyz
    return target_pose


def compute_fk(model: URDFFK, joint_values: dict[str, float], base_link: str, camera_link: str, wrist_link: str) -> tuple[np.ndarray, np.ndarray]:
    poses = model.compute_link_poses(
        joint_values=joint_values,
        targets=[camera_link, wrist_link],
        base_link=base_link,
        base_pose=base_pose_matrix(None, None, None),
        clamp_to_limits=False,
    )
    return as_transform(poses[camera_link].matrix), as_transform(poses[wrist_link].matrix)


def solve_right_arm_ik(
    urdf: str,
    base_link: str,
    wrist_link: str,
    current_joint_values: dict[str, float],
    target_wrist_pose: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    solver = URDFIK(urdf)
    return solver.solve(
        target_link=wrist_link,
        target_pose=target_wrist_pose.tolist(),
        initial_joint_values=current_joint_values,
        active_joints=RIGHT_ARM_JOINT_NAMES,
        base_link=base_link,
        base_pose=base_pose_matrix(None, None, None),
        max_iterations=args.ik_max_iterations,
        tolerance_position=args.ik_tolerance_position,
        tolerance_orientation=args.ik_tolerance_orientation,
        damping=args.ik_damping,
        step_scale=args.ik_step_scale,
        finite_difference_step=args.ik_finite_difference_step,
        position_weight=1.0,
        orientation_weight=args.ik_orientation_weight,
        clamp_to_limits=True,
    )


def target_xyz_in_base(args: argparse.Namespace, T_base_camera: np.ndarray) -> np.ndarray:
    if args.target_base_xyz is not None:
        return np.asarray(args.target_base_xyz, dtype=np.float64)
    if args.target_camera_xyz is None:
        raise ValueError("Provide --target-camera-xyz from Plan B web state, or --target-base-xyz.")
    target_camera = np.ones(4, dtype=np.float64)
    target_camera[:3] = np.asarray(args.target_camera_xyz, dtype=np.float64) + np.asarray(args.target_camera_offset_xyz, dtype=np.float64)
    return (T_base_camera @ target_camera)[:3]


def build_target_command(current_q_index: dict[int, float], solution_joint_values: dict[str, float]) -> dict[int, float]:
    target_q = dict(current_q_index)
    for name, value in solution_joint_values.items():
        index = G1_JOINT_INDEX[name]
        target_q[index] = float(value)
    return target_q


def max_right_arm_delta(current_q_name: dict[str, float], solution_joint_values: dict[str, float]) -> float:
    deltas = [
        abs(float(solution_joint_values[name]) - float(current_q_name[name]))
        for name in RIGHT_ARM_JOINT_NAMES
        if name in current_q_name and name in solution_joint_values
    ]
    return max(deltas) if deltas else 0.0


def run(args: argparse.Namespace) -> int:
    arm = LowStateArmSDK(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        kp=args.kp,
        kd=args.kd,
    )
    arm.wait_for_lowstate(args.state_timeout)
    current_q_name = arm.joint_positions_by_name()
    current_q_index = arm.joint_positions_by_index()

    model = URDFFK(args.urdf)
    T_base_camera, T_base_wrist = compute_fk(
        model=model,
        joint_values=current_q_name,
        base_link=args.base_link,
        camera_link=args.camera_link,
        wrist_link=args.wrist_link,
    )
    target_base_xyz = target_xyz_in_base(args, T_base_camera)
    tool_offset = np.asarray(args.tool_offset_wrist_xyz, dtype=np.float64)
    target_wrist_pose = make_target_pose(T_base_wrist, target_base_xyz, tool_offset)

    current_wrist_xyz = T_base_wrist[:3, 3]
    requested_move = float(np.linalg.norm(target_wrist_pose[:3, 3] - current_wrist_xyz))
    if requested_move > args.max_move_m:
        raise RuntimeError(
            f"Refusing move {requested_move:.4f}m > --max-move-m {args.max_move_m:.4f}m. "
            "Move the arm closer first or increase the limit deliberately."
        )

    solution = solve_right_arm_ik(
        urdf=args.urdf,
        base_link=args.base_link,
        wrist_link=args.wrist_link,
        current_joint_values=current_q_name,
        target_wrist_pose=target_wrist_pose,
        args=args,
    )
    joint_delta = max_right_arm_delta(current_q_name, solution.joint_values)
    if joint_delta > args.max_joint_delta_rad:
        raise RuntimeError(
            f"Refusing max joint delta {joint_delta:.4f}rad > --max-joint-delta-rad {args.max_joint_delta_rad:.4f}rad."
        )
    if not solution.success and solution.final_position_error_norm > args.allow_position_error_m:
        raise RuntimeError(
            "IK did not converge well enough: "
            f"success={solution.success}, pos_error={solution.final_position_error_norm:.4f}m, "
            f"message={solution.message}"
        )

    target_q = build_target_command(current_q_index, solution.joint_values)
    result = {
        "mode": "plan_b_touch_board",
        "execute": bool(args.execute),
        "base_link": args.base_link,
        "camera_link": args.camera_link,
        "wrist_link": args.wrist_link,
        "target_base_xyz_m": target_base_xyz.astype(float).tolist(),
        "current_wrist_base_xyz_m": current_wrist_xyz.astype(float).tolist(),
        "target_wrist_base_xyz_m": target_wrist_pose[:3, 3].astype(float).tolist(),
        "requested_wrist_move_m": requested_move,
        "tool_offset_wrist_xyz_m": tool_offset.astype(float).tolist(),
        "ik": {
            "success": bool(solution.success),
            "message": solution.message,
            "iterations": int(solution.iterations),
            "final_position_error_norm": float(solution.final_position_error_norm),
            "final_orientation_error_norm": float(solution.final_orientation_error_norm),
            "max_right_arm_joint_delta_rad": float(joint_delta),
            "joint_values": solution.joint_values,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not args.execute:
        print("[DRY-RUN] Not executing. Add --execute --confirm-touch-motion to publish arm_sdk commands.")
        return 0
    if not args.confirm_touch_motion:
        raise RuntimeError("Refusing to execute. Add --confirm-touch-motion after checking dry-run output and robot safety.")

    print("[EXECUTE] Publishing guarded right-arm touch motion...")
    arm.ramp_to(target_q, args.ramp_seconds, args.control_hz)
    arm.hold(target_q, args.hold_seconds, args.control_hz)
    if args.release_after_touch:
        arm.release(args.release_seconds, args.control_hz)
    print("[DONE]")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded Plan B right-wrist touch motion from board-center coordinates.")
    parser.add_argument("--target-camera-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Board center in camera_link frame, meters.")
    parser.add_argument("--target-camera-offset-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="Optional offset added in camera frame, meters.")
    parser.add_argument("--target-base-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Alternative target point in base_link frame, meters.")
    parser.add_argument("--tool-offset-wrist-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="Contact point offset from wrist_link in wrist frame, meters.")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--base-link", default="pelvis")
    parser.add_argument("--camera-link", default="d435_link")
    parser.add_argument("--wrist-link", default="right_wrist_yaw_link")
    parser.add_argument("--max-move-m", type=float, default=0.08)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.35)
    parser.add_argument("--allow-position-error-m", type=float, default=0.015)
    parser.add_argument("--ik-max-iterations", type=int, default=250)
    parser.add_argument("--ik-tolerance-position", type=float, default=0.005)
    parser.add_argument("--ik-tolerance-orientation", type=float, default=10.0)
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument("--ik-step-scale", type=float, default=0.4)
    parser.add_argument("--ik-finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.0, help="0 means position-only IK while keeping current orientation as a loose target.")
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=3.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--release-after-touch", action="store_true")
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-touch-motion", action="store_true")
    args = parser.parse_args()
    if args.target_camera_xyz is not None and args.target_base_xyz is not None:
        parser.error("Use only one of --target-camera-xyz or --target-base-xyz")
    if args.target_camera_xyz is None and args.target_base_xyz is None:
        parser.error("Provide --target-camera-xyz or --target-base-xyz")
    if args.max_move_m <= 0 or args.max_joint_delta_rad <= 0:
        parser.error("--max-move-m and --max-joint-delta-rad must be positive")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
