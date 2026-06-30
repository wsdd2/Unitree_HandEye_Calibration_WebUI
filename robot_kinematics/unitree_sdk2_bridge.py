"""Bridge Unitree SDK2 low-state data into the URDF FK/IK helpers.

This script is intended for WSL2 Ubuntu 22.04 with ``unitree_sdk2_python``.
It subscribes to Unitree ``LowState`` messages, converts motor indexes to URDF
joint names, then feeds those values into the local FK/IK modules.

By default this script is read-only. Low-level command publishing is only
enabled when explicitly requested with the publish safety flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
JOINT_TO_POSE_DIR = Path(__file__).resolve().parent / "joint_to_pose"
POSE_TO_JOINT_DIR = Path(__file__).resolve().parent / "pose_to_joint"
for module_dir in (JOINT_TO_POSE_DIR, POSE_TO_JOINT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from fk_urdf import (  # noqa: E402
    URDFFK,
    base_pose_matrix,
    parse_targets,
    pose_to_json,
)
from ik_urdf import (  # noqa: E402
    URDFIK,
    load_target_pose,
    parse_name_list,
    solution_to_json,
)

DEFAULT_G1_URDF = (
    ROOT_DIR / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
)

G1_29DOF_JOINT_INDEX: Dict[str, int] = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
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


@dataclass(frozen=True)
class JointSample:
    q: float
    dq: float
    tau_est: float


def import_unitree_sdk2() -> Dict[str, object]:
    """Import SDK2 symbols lazily so offline FK/IK code remains importable."""

    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (  # type: ignore
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # type: ignore
            LowCmd_,
            LowState_,
        )
        from unitree_sdk2py.utils.crc import CRC  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing unitree_sdk2_python. Install it inside WSL2 Ubuntu first, "
            "then run this script from that environment."
        ) from exc

    return {
        "ChannelFactoryInitialize": ChannelFactoryInitialize,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "LowCmd_": LowCmd_,
        "LowState_": LowState_,
        "LowCmdDefault": unitree_hg_msg_dds__LowCmd_,
        "CRC": CRC,
    }


class UnitreeG1LowStateBridge:
    """Subscribe Unitree low state and expose URDF-named joint values."""

    def __init__(
        self,
        network_interface: Optional[str],
        domain_id: int = 0,
        joint_index: Mapping[str, int] = G1_29DOF_JOINT_INDEX,
        enable_publisher: bool = False,
    ) -> None:
        self.sdk = import_unitree_sdk2()
        self.joint_index = dict(joint_index)
        self._latest: Dict[str, JointSample] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()

        init = self.sdk["ChannelFactoryInitialize"]
        if network_interface:
            init(domain_id, network_interface)
        else:
            init(domain_id)

        low_state_type = self.sdk["LowState_"]
        subscriber_cls = self.sdk["ChannelSubscriber"]
        self._subscriber = subscriber_cls("rt/lowstate", low_state_type)
        self._subscriber.Init(self._on_low_state, 10)

        self._publisher = None
        self._low_cmd_default = None
        self._crc = None
        if enable_publisher:
            low_cmd_type = self.sdk["LowCmd_"]
            publisher_cls = self.sdk["ChannelPublisher"]
            self._publisher = publisher_cls("rt/lowcmd", low_cmd_type)
            self._publisher.Init()
            self._low_cmd_default = self.sdk["LowCmdDefault"]
            self._crc = self.sdk["CRC"]()

    def _on_low_state(self, msg: object) -> None:
        motor_state = getattr(msg, "motor_state")
        latest: Dict[str, JointSample] = {}
        for joint_name, index in self.joint_index.items():
            if index >= len(motor_state):
                continue
            state = motor_state[index]
            latest[joint_name] = JointSample(
                q=float(getattr(state, "q", 0.0)),
                dq=float(getattr(state, "dq", 0.0)),
                tau_est=float(getattr(state, "tau_est", 0.0)),
            )

        with self._lock:
            self._latest = latest
        self._ready.set()

    def wait_for_state(self, timeout: float) -> Dict[str, JointSample]:
        if not self._ready.wait(timeout):
            raise TimeoutError(
                "Timed out waiting for rt/lowstate. Check robot power, WiFi, "
                "WSL2 networking, and --network-interface."
            )
        return self.latest_state()

    def latest_state(self) -> Dict[str, JointSample]:
        with self._lock:
            return dict(self._latest)

    def latest_joint_positions(self) -> Dict[str, float]:
        return {name: sample.q for name, sample in self.latest_state().items()}

    def publish_position_command(
        self,
        target_joint_values: Mapping[str, float],
        hold_joint_values: Optional[Mapping[str, float]] = None,
        kp: float = 20.0,
        kd: float = 2.0,
    ) -> None:
        if self._publisher is None or self._low_cmd_default is None or self._crc is None:
            raise RuntimeError("Publisher was not enabled for this bridge instance.")

        hold_values = dict(hold_joint_values or self.latest_joint_positions())
        command_values = dict(hold_values)
        command_values.update({name: float(value) for name, value in target_joint_values.items()})

        cmd = self._low_cmd_default()
        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = 0
        if hasattr(cmd, "mode_machine"):
            cmd.mode_machine = 0

        for joint_name, index in self.joint_index.items():
            if index >= len(cmd.motor_cmd):
                continue
            motor_cmd = cmd.motor_cmd[index]
            if hasattr(motor_cmd, "mode"):
                motor_cmd.mode = 1
            motor_cmd.q = float(command_values.get(joint_name, hold_values.get(joint_name, 0.0)))
            motor_cmd.dq = 0.0
            motor_cmd.kp = float(kp)
            motor_cmd.kd = float(kd)
            motor_cmd.tau = 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._publisher.Write(cmd)


def joint_samples_to_json(samples: Mapping[str, JointSample]) -> Dict[str, object]:
    return {
        name: {"q": sample.q, "dq": sample.dq, "tau_est": sample.tau_est}
        for name, sample in samples.items()
    }


def run_state(args: argparse.Namespace) -> Dict[str, object]:
    bridge = UnitreeG1LowStateBridge(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
    )
    samples = bridge.wait_for_state(args.state_timeout)
    return {"source": "rt/lowstate", "joints": joint_samples_to_json(samples)}


def run_fk(args: argparse.Namespace) -> Dict[str, object]:
    bridge = UnitreeG1LowStateBridge(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
    )
    bridge.wait_for_state(args.state_timeout)
    joint_values = bridge.latest_joint_positions()

    model = URDFFK(args.urdf)
    targets = parse_targets(args.target)
    if not targets:
        raise ValueError("Provide at least one --target for FK mode.")
    base_pose = base_pose_matrix(args.base_xyz, args.base_rpy, args.base_quat_xyzw)
    poses = model.compute_link_poses(
        joint_values=joint_values,
        targets=targets,
        base_link=args.base_link,
        base_pose=base_pose,
        clamp_to_limits=args.clamp_to_limits,
    )

    return {
        "source": "rt/lowstate",
        "urdf": str(Path(args.urdf)),
        "base_link": args.base_link or model.root_links[0],
        "joint_values": joint_values,
        "targets": {
            link_name: pose_to_json(pose, args.quat_order)
            for link_name, pose in poses.items()
        },
    }


def run_ik(args: argparse.Namespace) -> Dict[str, object]:
    enable_publish = args.publish_solution
    bridge = UnitreeG1LowStateBridge(
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        enable_publisher=enable_publish,
    )
    bridge.wait_for_state(args.state_timeout)
    current_joint_values = bridge.latest_joint_positions()

    solver = URDFIK(args.urdf)
    target_pose = load_target_pose(args.target_pose, args)
    base_pose = base_pose_matrix(args.base_xyz, args.base_rpy, args.base_quat_xyzw)
    active_joints = parse_name_list(args.active_joint)
    solution = solver.solve(
        target_link=args.target_link,
        target_pose=target_pose,
        initial_joint_values=current_joint_values,
        active_joints=active_joints,
        base_link=args.base_link,
        base_pose=base_pose,
        max_iterations=args.max_iterations,
        tolerance_position=args.tolerance_position,
        tolerance_orientation=args.tolerance_orientation,
        damping=args.damping,
        step_scale=args.step_scale,
        finite_difference_step=args.finite_difference_step,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        clamp_to_limits=not args.ignore_limits,
    )

    published = False
    if enable_publish:
        if not args.confirm_low_level_control:
            raise ValueError(
                "Refusing to publish. Add --confirm-low-level-control after "
                "verifying the robot is supported, suspended safely, and in the "
                "expected low-level control mode."
            )
        if not solution.success:
            raise ValueError("Refusing to publish an unsuccessful IK solution.")

        deadline = time.monotonic() + args.publish_duration
        while time.monotonic() < deadline:
            bridge.publish_position_command(
                target_joint_values=solution.joint_values,
                hold_joint_values=current_joint_values,
                kp=args.kp,
                kd=args.kd,
            )
            time.sleep(1.0 / args.publish_hz)
        published = True

    return {
        "source": "rt/lowstate",
        "urdf": str(Path(args.urdf)),
        "initial_joint_values_from_robot": current_joint_values,
        "ik_solution": solution_to_json(solution),
        "published_solution": published,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use unitree_sdk2_python low-state data with local URDF FK/IK."
    )
    parser.add_argument(
        "--mode",
        choices=("state", "fk", "ik"),
        default="fk",
        help="state: print joints, fk: compute target link poses, ik: solve from current joints.",
    )
    parser.add_argument(
        "--network-interface",
        help="WSL2/Ubuntu network interface connected to the robot WiFi, e.g. eth0.",
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-timeout", type=float, default=3.0)
    parser.add_argument("--urdf", default=str(DEFAULT_G1_URDF))
    parser.add_argument("--base-link", default="pelvis")
    parser.add_argument("--target", action="append", help="FK target link(s).")
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order for FK output.",
    )
    parser.add_argument("--clamp-to-limits", action="store_true")
    parser.add_argument("--target-link", help="IK target link.")
    parser.add_argument(
        "--target-pose",
        help="IK target JSON with position_xyz and orientation_quat_xyzw or orientation_rpy.",
    )
    parser.add_argument("--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-rpy", nargs=3, type=float, metavar=("R", "P", "Y"))
    parser.add_argument(
        "--target-quat-xyzw",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
    )
    parser.add_argument("--active-joint", action="append")
    parser.add_argument("--base-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--base-rpy", nargs=3, type=float, metavar=("R", "P", "Y"))
    parser.add_argument(
        "--base-quat-xyzw",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance-position", type=float, default=1e-4)
    parser.add_argument("--tolerance-orientation", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--ignore-limits", action="store_true")
    parser.add_argument("--publish-solution", action="store_true")
    parser.add_argument("--confirm-low-level-control", action="store_true")
    parser.add_argument("--publish-duration", type=float, default=0.2)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "fk" and not parse_targets(args.target):
        raise ValueError("FK mode requires --target.")
    if args.mode == "ik":
        if not args.target_link:
            raise ValueError("IK mode requires --target-link.")
        if args.publish_hz <= 0.0:
            raise ValueError("--publish-hz must be positive.")
        if args.publish_duration < 0.0:
            raise ValueError("--publish-duration must be non-negative.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)

    if args.mode == "state":
        output = run_state(args)
    elif args.mode == "fk":
        output = run_fk(args)
    else:
        output = run_ik(args)

    indent = 2 if args.pretty else None
    print(json.dumps(output, indent=indent))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
