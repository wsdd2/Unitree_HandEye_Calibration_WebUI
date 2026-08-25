from __future__ import annotations

from typing import Optional


def mount_mode_mismatch(mode: str, mount: str) -> Optional[str]:
    """Describe a known camera-mount/hand-eye-mode mismatch."""

    if not mount:
        return None
    mode_key = mode.strip().lower().replace("-", "_")
    mount_key = mount.strip().lower().replace("-", "_")
    eye_in_hand_mounts = {
        "hand", "wrist", "flange", "palm", "tool", "end_effector", "gripper",
    }
    eye_to_hand_mounts = {
        "head", "fixed", "external", "tripod", "base", "world",
        "abdomen", "torso", "body",
    }
    if mode_key == "eye_in_hand" and mount_key in eye_to_hand_mounts:
        return f"mode={mode_key} 但 camera_mount={mount} 是固定相机安装位"
    if mode_key == "eye_to_hand" and mount_key in eye_in_hand_mounts:
        return f"mode={mode_key} 但 camera_mount={mount} 是末端相机安装位"
    return None
