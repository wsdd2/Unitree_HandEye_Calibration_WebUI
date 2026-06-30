"""Example: compute zero-joint poses for common Unitree G1 links."""

from __future__ import annotations

import json
from pathlib import Path

from fk_urdf import URDFFK, pose_to_json


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    urdf_path = (
        workspace
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof_rev_1_0.urdf"
    )

    model = URDFFK(urdf_path)
    targets = ["d435_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    poses = model.compute_link_poses(joint_values={}, targets=targets)

    print(
        json.dumps(
            {
                "urdf": str(urdf_path),
                "base_link": model.root_links[0],
                "joint_values": "all active joints default to zero",
                "targets": {
                    link_name: pose_to_json(pose, quat_order="xyzw")
                    for link_name, pose in poses.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
