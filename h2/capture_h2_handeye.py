#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2-specific hand-eye capture wrapper.

This wrapper keeps ``Hand_Eye_Calib/capture_handeye.py`` untouched, but adds the
same virtual right end-effector frame used by the H2 Cartesian keyboard scripts:

    R_ee = right_wrist_yaw_joint frame + [0.05, 0, 0] meters

The frame is intended to represent the mechanical arm to gripper mounting point,
without including Dex1-1 kinematics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PKG_ROOT = Path(__file__).resolve().parents[1]
HAND_EYE_DIR = PKG_ROOT
if str(HAND_EYE_DIR) not in sys.path:
    sys.path.insert(0, str(HAND_EYE_DIR))

import capture_handeye as base  # noqa: E402


H2_VIRTUAL_EE_FRAME = "R_ee"
H2_WRIST_YAW_LINK = "right_wrist_yaw_link"
H2_EE_OFFSET_M = np.array([0.05, 0.0, 0.0], dtype=np.float64)
H2_LOCKED_WAIST_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
H2_RIGHT_ARM_JOINT_NAMES = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


_original_expand_fk_targets = base.expand_fk_targets
_original_snapshot = base.FKStateProvider.snapshot


def _is_h2_virtual_frame(frame_name: str) -> bool:
    return frame_name.strip() in {H2_VIRTUAL_EE_FRAME, "right_R_ee", "right_ee"}


def expand_fk_targets_h2(targets: list[str], hand_frame: str) -> tuple[list[str], list[str]]:
    """Map virtual H2 end-effector frames to the real wrist link for FK."""

    if not _is_h2_virtual_frame(hand_frame):
        return _original_expand_fk_targets(targets, hand_frame)

    public: list[str] = []
    for name in [*targets, H2_WRIST_YAW_LINK, H2_VIRTUAL_EE_FRAME]:
        if name and name not in public:
            public.append(name)

    compute = [name for name in public if name != H2_VIRTUAL_EE_FRAME]
    if H2_WRIST_YAW_LINK not in compute:
        compute.append(H2_WRIST_YAW_LINK)
    return public, compute


def snapshot_h2(self) -> dict:
    """Compute H2 FK with waist locked, then add the virtual ``R_ee`` pose."""

    if not _is_h2_virtual_frame(getattr(self, "hand_frame", "")):
        return _original_snapshot(self)

    self._bridge.wait_for_state(self._state_timeout)
    joint_values = self._bridge.latest_joint_positions()
    measured_waist = {
        name: float(joint_values.get(name, 0.0))
        for name in H2_LOCKED_WAIST_JOINTS
    }
    for name in H2_LOCKED_WAIST_JOINTS:
        joint_values[name] = 0.0

    poses = self._model.compute_link_poses(
        joint_values=joint_values,
        targets=self._compute_targets,
        base_link=self.base_link,
        base_pose=self._base_pose,
        clamp_to_limits=False,
    )

    targets = {
        link_name: self._pose_to_json(poses[link_name], "xyzw")
        for link_name in self._public_targets
        if link_name in poses and link_name != H2_VIRTUAL_EE_FRAME
    }
    if H2_WRIST_YAW_LINK not in targets:
        return {
            "source": "rt/lowstate",
            "base_link": self.base_link,
            "hand_frame": self.hand_frame,
            "targets": targets,
            "right_arm_joints": {
                name: joint_values[name]
                for name in sorted(joint_values)
                if name in H2_RIGHT_ARM_JOINT_NAMES
            },
            "h2_virtual_ee": {
                "frame": H2_VIRTUAL_EE_FRAME,
                "parent": H2_WRIST_YAW_LINK,
                "offset_xyz_m": H2_EE_OFFSET_M.astype(float).tolist(),
                "waist_locked": True,
                "measured_waist_rad": measured_waist,
                "error": f"missing FK target: {H2_WRIST_YAW_LINK}",
            },
        }

    wrist_matrix = np.asarray(
        targets[H2_WRIST_YAW_LINK]["transform_matrix"],
        dtype=np.float64,
    )
    T_wrist_ree = np.eye(4, dtype=np.float64)
    T_wrist_ree[:3, 3] = H2_EE_OFFSET_M
    T_base_ree = wrist_matrix @ T_wrist_ree

    pose = self._Pose(link=H2_VIRTUAL_EE_FRAME, matrix=T_base_ree.tolist())
    targets[H2_VIRTUAL_EE_FRAME] = self._pose_to_json(pose, "xyzw")

    return {
        "source": "rt/lowstate",
        "base_link": self.base_link,
        "hand_frame": self.hand_frame,
        "targets": targets,
        "right_arm_joints": {
            name: joint_values[name]
            for name in sorted(joint_values)
            if name in H2_RIGHT_ARM_JOINT_NAMES
        },
        "h2_virtual_ee": {
            "frame": H2_VIRTUAL_EE_FRAME,
            "parent": H2_WRIST_YAW_LINK,
            "offset_xyz_m": H2_EE_OFFSET_M.astype(float).tolist(),
            "waist_locked": True,
            "measured_waist_rad": measured_waist,
            "note": (
                "Matches H2 keyboard Cartesian control R_ee definition; "
                "waist joints are locked to zero and Dex1-1 is not included."
            ),
        },
    }


def h2_default_argv() -> list[str]:
    """Sensible H2 defaults; override camera serial / URDF / NIC via CLI."""

    default_urdf = Path.home() / "H2_joint_cartesian/third_party/xr_teleoperate/assets/h2/H2.urdf"
    return [
        "--mode",
        "eye-in-hand",
        "--stream-debug",
        "--stream-fk",
        "--headless",
        "--color-only",
        "--no-cam-fallback",
        "--robot-model",
        "Unitree H2",
        "--fk-urdf",
        str(default_urdf),
        "--fk-base-link",
        "pelvis",
        "--hand-frame",
        H2_VIRTUAL_EE_FRAME,
        "--camera-mount",
        "wrist",
        "--camera-name",
        "right_wrist_d435",
        "--fk-target",
        f"{H2_WRIST_YAW_LINK},{H2_VIRTUAL_EE_FRAME}",
        "--fk-network-interface",
        "",
        "--width",
        "1280",
        "--height",
        "720",
        "--fps",
        "30",
        "--cols",
        "11",
        "--rows",
        "8",
        "--square-mm",
        "20",
        "--min-samples",
        "12",
        "--stream-host",
        "0.0.0.0",
        "--stream-port",
        "8080",
        "--notes",
        "H2 wrist D435 eye-in-hand; FK hand frame is virtual R_ee without Dex1-1.",
    ]


def main() -> int:
    base.expand_fk_targets = expand_fk_targets_h2
    base.FKStateProvider.snapshot = snapshot_h2
    argv = h2_default_argv() + sys.argv[1:]
    return base.run_capture(base.parse_args_from(argv) if hasattr(base, "parse_args_from") else _parse_args(argv))


def _parse_args(argv: list[str]):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return base.parse_args()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
