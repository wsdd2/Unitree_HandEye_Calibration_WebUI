#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only FK comparison for H2 hand-eye debugging.

This script subscribes to ``rt/lowstate`` and compares:

1. Our transparent URDF FK (`robot_kinematics/joint_to_pose/fk_urdf.py`)
2. Hardware engineer's Pinocchio reduced-model FK (`H2CompatibleIK.R_ee`)

It never publishes to lowcmd/arm_sdk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]
ROBOT_KINEMATICS_DIR = PKG_ROOT / "robot_kinematics"
JOINT_TO_POSE_DIR = ROBOT_KINEMATICS_DIR / "joint_to_pose"
for module_dir in (ROBOT_KINEMATICS_DIR, JOINT_TO_POSE_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from fk_urdf import URDFFK, base_pose_matrix  # noqa: E402
from unitree_sdk2_bridge import G1_29DOF_JOINT_INDEX, UnitreeG1LowStateBridge  # noqa: E402


ARM_Q_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

WAIST_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


def import_hardware_fk(h2_root: Path):
    """Import hardware engineer's H2 Pinocchio FK without publishing commands."""

    scripts = h2_root / "scripts"
    unitree_sdk = h2_root / "third_party" / "unitree_sdk2_python"
    robot_control = h2_root / "third_party" / "xr_teleoperate" / "teleop" / "robot_control"
    for module_dir in (scripts, unitree_sdk, robot_control):
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

    from h2_xr_official_ik_demo import H2CompatibleIK, current_ee_poses  # noqa: WPS433

    return H2CompatibleIK, current_ee_poses


def our_r_ee_pose(
    fk_model: URDFFK,
    joint_values: dict[str, float],
    *,
    base_link: str,
    target_link: str,
    lock_waist: bool,
    offset_xyz_m: np.ndarray,
) -> np.ndarray:
    joints = dict(joint_values)
    if lock_waist:
        for name in WAIST_NAMES:
            joints[name] = 0.0
    poses = fk_model.compute_link_poses(
        joint_values=joints,
        targets=[target_link],
        base_link=base_link,
        base_pose=base_pose_matrix(None, None, None),
        clamp_to_limits=False,
    )
    T_base_wrist = np.asarray(poses[target_link].matrix, dtype=np.float64)
    T_wrist_ree = np.eye(4, dtype=np.float64)
    T_wrist_ree[:3, 3] = offset_xyz_m
    return T_base_wrist @ T_wrist_ree


def hardware_r_ee_pose(hw_ik, current_ee_poses, joint_values: dict[str, float]) -> np.ndarray:
    q = np.array([joint_values[name] for name in ARM_Q_NAMES], dtype=np.float64)
    _left, right = current_ee_poses(hw_ik, q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(right.rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(right.translation, dtype=np.float64)
    return T


def rotation_error_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    R = T_a[:3, :3].T @ T_b[:3, :3]
    cos_angle = float((np.trace(R) - 1.0) * 0.5)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def fmt_vec(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare H2 hand-eye FK variants.")
    parser.add_argument("--iface", default="eth0", help="DDS interface, e.g. eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=0, help="0 means run forever")
    parser.add_argument(
        "--h2-root",
        default=str(Path.home() / "H2_joint_cartesian"),
        help="Hardware engineer H2_joint_cartesian root",
    )
    parser.add_argument(
        "--our-urdf",
        default=str(Path.home() / "H2_joint_cartesian/third_party/xr_teleoperate/assets/h2/H2.urdf"),
        help="URDF used by our URDF FK comparator",
    )
    parser.add_argument("--base-link", default="pelvis")
    parser.add_argument("--target-link", default="right_wrist_yaw_link")
    parser.add_argument("--ee-offset-x", type=float, default=0.05)
    parser.add_argument("--lock-waist", action="store_true", help="Force waist joints to zero in our URDF FK")
    parser.add_argument("--no-hw-payload", action="store_true", help="Set hardware FK payload mass to zero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.period <= 0:
        raise ValueError("--period must be positive")

    h2_root = Path(args.h2_root).expanduser()
    our_urdf = Path(args.our_urdf).expanduser()
    H2CompatibleIK, current_ee_poses = import_hardware_fk(h2_root)

    bridge = UnitreeG1LowStateBridge(
        network_interface=args.iface,
        domain_id=args.domain_id,
        joint_index=G1_29DOF_JOINT_INDEX,
        enable_publisher=False,
    )
    fk_model = URDFFK(our_urdf)
    payload_mass = 0.0 if args.no_hw_payload else 0.55
    hw_ik = H2CompatibleIK(gripper_payload_mass=payload_mass)
    offset = np.array([args.ee_offset_x, 0.0, 0.0], dtype=np.float64)

    print("h2_fk_compare_start")
    print(f"iface={args.iface} base_link={args.base_link} target_link={args.target_link}")
    print(f"our_urdf={our_urdf}")
    print(f"h2_root={h2_root}")
    print(f"offset_xyz_m={offset.tolist()} lock_waist={args.lock_waist} hw_payload_mass={payload_mass}")

    index = 0
    while args.count <= 0 or index < args.count:
        bridge.wait_for_state(2.0)
        joints = bridge.latest_joint_positions()
        missing = [name for name in ARM_Q_NAMES if name not in joints]
        if missing:
            raise RuntimeError(f"missing arm joints from lowstate: {missing}")

        T_ours = our_r_ee_pose(
            fk_model,
            joints,
            base_link=args.base_link,
            target_link=args.target_link,
            lock_waist=args.lock_waist,
            offset_xyz_m=offset,
        )
        T_hw = hardware_r_ee_pose(hw_ik, current_ee_poses, joints)

        p_ours = T_ours[:3, 3]
        p_hw = T_hw[:3, 3]
        delta = p_ours - p_hw
        waist = np.array([joints.get(name, 0.0) for name in WAIST_NAMES], dtype=np.float64)
        rot_deg = rotation_error_deg(T_ours, T_hw)

        print(
            "fk_compare "
            f"idx={index} "
            f"waist_rad={fmt_vec(waist, 5)} "
            f"ours_xyz={fmt_vec(p_ours)} "
            f"hw_xyz={fmt_vec(p_hw)} "
            f"delta_mm={fmt_vec(delta * 1000.0, 3)} "
            f"norm_mm={np.linalg.norm(delta) * 1000.0:.3f} "
            f"rot_deg={rot_deg:.3f}"
        )

        index += 1
        time.sleep(args.period)

    return 0


if __name__ == "__main__":
    # Keep Pinocchio's optional relaunch logic in h2_xr_official_ik_demo happy by
    # preserving the user's H2 environment from scripts/setup_env.sh.
    os.environ.setdefault("H2_XR_PINOCCHIO_ENV", "clean")
    raise SystemExit(main())
