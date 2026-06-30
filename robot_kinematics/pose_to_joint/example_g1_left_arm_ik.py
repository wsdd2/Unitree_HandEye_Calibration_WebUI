"""Example: solve Unitree G1 left wrist IK for a generated reachable pose."""

from __future__ import annotations

import json
from pathlib import Path

from ik_urdf import URDFIK, solution_to_json


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    urdf_path = (
        workspace
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof_rev_1_0.urdf"
    )

    target_link = "left_wrist_yaw_link"
    active_joints = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    known_joint_values = {
        "left_shoulder_pitch_joint": 0.25,
        "left_shoulder_roll_joint": 0.12,
        "left_shoulder_yaw_joint": -0.08,
        "left_elbow_joint": 0.55,
        "left_wrist_roll_joint": 0.05,
        "left_wrist_pitch_joint": -0.04,
        "left_wrist_yaw_joint": 0.1,
    }

    solver = URDFIK(urdf_path)
    target_pose = solver.fk.compute_link_poses(
        joint_values=known_joint_values,
        targets=[target_link],
    )[target_link].matrix

    solution = solver.solve(
        target_link=target_link,
        target_pose=target_pose,
        initial_joint_values={},
        active_joints=active_joints,
        max_iterations=300,
        tolerance_position=1e-5,
        tolerance_orientation=1e-4,
        damping=1e-3,
        step_scale=0.6,
    )

    print(
        json.dumps(
            {
                "urdf": str(urdf_path),
                "target_link": target_link,
                "known_joint_values_used_to_generate_target": known_joint_values,
                "ik_solution": solution_to_json(solution),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
