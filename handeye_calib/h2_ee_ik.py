# -*- coding: utf-8 -*-
"""Single-arm SE3 IK for the H2 wrist/camera using robot_kinematics."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence


ARM_JOINT_NAMES = {
    "left": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}
EE_LINK = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}
DEFAULT_URDF_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "robots" / "h2" / "H2.urdf",
    Path(__file__).resolve().parents[1] / "robots" / "H2.urdf",
)


def ensure_robot_kinematics() -> None:
    roots = (
        Path(__file__).resolve().parents[1] / "robot_kinematics",
    )
    for root in roots:
        if (root / "pose_to_joint" / "ik_urdf.py").is_file():
            pose_dir = str(root / "pose_to_joint")
            fk_dir = str(root / "joint_to_pose")
            if pose_dir not in sys.path:
                sys.path.insert(0, pose_dir)
            if fk_dir not in sys.path:
                sys.path.insert(0, fk_dir)
            return
    raise RuntimeError("找不到 robot_kinematics。请保留仓库里的 robot_kinematics 目录")


def resolve_urdf(path: str = "") -> Path:
    if path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise RuntimeError(f"URDF 不存在: {candidate}")
    for candidate in DEFAULT_URDF_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError("找不到 H2.urdf。用 --urdf 指定，或放到 robots/h2/H2.urdf")


def _rpy_from_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(matrix[2][0]) < 0.999999:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return roll, pitch, yaw


class H2CameraIK:
    def __init__(self, urdf_path: str = "", arm: str = "right", ee_link: str = "") -> None:
        ensure_robot_kinematics()
        from ik_urdf import URDFIK

        self.arm = "left" if str(arm).lower() == "left" else "right"
        self.urdf = resolve_urdf(urdf_path)
        self.ee_link = ee_link or EE_LINK[self.arm]
        self.joint_names = ARM_JOINT_NAMES[self.arm]
        self.solver = URDFIK(self.urdf)
        print(f"[IK] urdf={self.urdf} ee={self.ee_link} arm={self.arm}")

    def joint_dict(self, q_rad: Sequence[float]) -> dict[str, float]:
        if len(q_rad) != 7:
            raise ValueError("arm q must have 7 values")
        return {name: float(value) for name, value in zip(self.joint_names, q_rad)}

    def fk_xyz_rpy(self, q_rad: Sequence[float]) -> tuple[list[float], list[float]]:
        from fk_urdf import URDFFK

        fk = self.solver.fk if hasattr(self.solver, "fk") else URDFFK(self.urdf)
        pose = fk.compute_link_poses(
            joint_values=self.joint_dict(q_rad),
            targets=[self.ee_link],
            base_link="torso_link",
        )[self.ee_link].matrix
        xyz = [pose[0][3], pose[1][3], pose[2][3]]
        rpy = list(_rpy_from_matrix(pose))
        return xyz, rpy

    def fk_matrix(self, q_rad: Sequence[float]):
        return self.solver.fk.compute_link_poses(
            joint_values=self.joint_dict(q_rad),
            targets=[self.ee_link],
            base_link="torso_link",
        )[self.ee_link].matrix

    def apply_delta(
        self,
        q_rad: Sequence[float],
        dxyz: Sequence[float] = (0.0, 0.0, 0.0),
        drpy: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        rotation_frame: str = "tool",
    ) -> tuple[list[float], bool, float, float]:
        from fk_urdf import matmul4, rpy_matrix, translation_matrix

        current = self.fk_matrix(q_rad)
        # XYZ is torso/base. RPY stays in the camera/tool frame.
        moved = matmul4(
            translation_matrix((float(dxyz[0]), float(dxyz[1]), float(dxyz[2]))),
            current,
        )
        rotation = rpy_matrix((float(drpy[0]), float(drpy[1]), float(drpy[2])))
        if rotation_frame == "base":
            target = matmul4(rotation, moved)
        else:
            target = matmul4(moved, rotation)
        solution = self.solver.solve(
            target_link=self.ee_link,
            target_pose=target,
            initial_joint_values=self.joint_dict(q_rad),
            active_joints=self.joint_names,
            base_link="torso_link",
            max_iterations=80,
            tolerance_position=0.002,
            tolerance_orientation=0.02,
            damping=1e-2,
            step_scale=0.45,
        )
        q_out = [float(solution.joint_values[name]) for name in self.joint_names]
        ok = bool(solution.success) or (
            solution.final_position_error_norm < 0.008
            and solution.final_orientation_error_norm < 0.08
        )
        return q_out, ok, solution.final_position_error_norm, solution.final_orientation_error_norm

    def apply_loaded_delta(
        self,
        commanded_q_rad: Sequence[float],
        measured_q_rad: Sequence[float],
        dxyz: Sequence[float] = (0.0, 0.0, 0.0),
        drpy: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        rotation_frame: str = "base",
    ) -> tuple[list[float], bool, float, float]:
        """Solve at the measured pose, then add that joint increment to the hold command.

        A loaded H2 arm can lag the arm_sdk target by tens of degrees. Solving
        around the commanded pose makes a Cartesian +X click move the physical
        hand backward/down. Replacing the command with the measured joints is
        also unsafe because it ratchets gravity sag. This keeps the command-
        measurement support offset while using the measured Jacobian.
        """
        from fk_urdf import matmul4, rpy_matrix, translation_matrix

        commanded = [float(value) for value in commanded_q_rad]
        measured = [float(value) for value in measured_q_rad]
        current = self.fk_matrix(measured)
        moved = matmul4(
            translation_matrix((float(dxyz[0]), float(dxyz[1]), float(dxyz[2]))),
            current,
        )
        rotation = rpy_matrix((float(drpy[0]), float(drpy[1]), float(drpy[2])))
        target = matmul4(rotation, moved) if rotation_frame == "base" else matmul4(moved, rotation)
        solution = self.solver.solve(
            target_link=self.ee_link,
            target_pose=target,
            initial_joint_values=self.joint_dict(measured),
            active_joints=self.joint_names,
            base_link="torso_link",
            max_iterations=80,
            tolerance_position=0.002,
            tolerance_orientation=0.02,
            damping=2e-2,
            step_scale=0.40,
        )
        measured_solution = [float(solution.joint_values[name]) for name in self.joint_names]
        target_command = {
            name: command + solved - actual
            for name, command, solved, actual in zip(
                self.joint_names,
                commanded,
                measured_solution,
                measured,
            )
        }
        target_command = self.solver._clamp_joint_values(target_command, self.joint_names)
        q_out = [float(target_command[name]) for name in self.joint_names]
        ok = bool(solution.success) or (
            solution.final_position_error_norm < 0.008
            and solution.final_orientation_error_norm < 0.08
        )
        return q_out, ok, solution.final_position_error_norm, solution.final_orientation_error_norm

    def solve_loaded_tool_pose(
        self,
        commanded_q_rad: Sequence[float],
        measured_q_rad: Sequence[float],
        target_tool_xyz_m: Sequence[float],
        target_tool_rpy_rad: Sequence[float],
        tool_offset_wrist_m: Sequence[float] = (0.205, 0.0, 0.0),
        seed_q_rad: Sequence[float] | None = None,
    ) -> dict[str, object]:
        """Solve right_ee and keep the measured-space joints for the next seed.

        q_command is the arm_sdk target (command + solved - measured). q_solved
        stays in the measured joint basin so a Cartesian sample chain does not
        re-seed from the gravity offset.
        """
        from fk_urdf import rpy_matrix

        target = rpy_matrix(
            (
                float(target_tool_rpy_rad[0]),
                float(target_tool_rpy_rad[1]),
                float(target_tool_rpy_rad[2]),
            )
        )
        offset = [
            sum(float(target[row][col]) * float(tool_offset_wrist_m[col]) for col in range(3))
            for row in range(3)
        ]
        for row in range(3):
            target[row][3] = float(target_tool_xyz_m[row]) - offset[row]
        commanded = [float(value) for value in commanded_q_rad]
        measured = [float(value) for value in measured_q_rad]
        seed = measured if seed_q_rad is None else [float(value) for value in seed_q_rad]
        if len(seed) != len(measured):
            seed = measured
        solution = self.solver.solve(
            target_link=self.ee_link,
            target_pose=target,
            initial_joint_values=self.joint_dict(seed),
            active_joints=self.joint_names,
            base_link="torso_link",
            max_iterations=120,
            tolerance_position=0.002,
            tolerance_orientation=0.02,
            damping=2e-2,
            step_scale=0.35,
        )
        solved = [float(solution.joint_values[name]) for name in self.joint_names]
        target_command = {
            name: command + result - actual
            for name, command, result, actual in zip(
                self.joint_names,
                commanded,
                solved,
                measured,
            )
        }
        target_command = self.solver._clamp_joint_values(target_command, self.joint_names)
        q_out = [float(target_command[name]) for name in self.joint_names]
        ok = bool(solution.success) or (
            solution.final_position_error_norm < 0.008
            and solution.final_orientation_error_norm < 0.08
        )
        return {
            "q_command": q_out,
            "q_solved": solved,
            "ok": ok,
            "pos_err": solution.final_position_error_norm,
            "ori_err": solution.final_orientation_error_norm,
        }

    def apply_loaded_tool_pose_target(
        self,
        commanded_q_rad: Sequence[float],
        measured_q_rad: Sequence[float],
        target_tool_xyz_m: Sequence[float],
        target_tool_rpy_rad: Sequence[float],
        tool_offset_wrist_m: Sequence[float] = (0.205, 0.0, 0.0),
        seed_q_rad: Sequence[float] | None = None,
    ) -> tuple[list[float], bool, float, float]:
        """Target right_ee_link while the solver chain ends at right_wrist_yaw_link.

        seed_q_rad only chooses the numerical basin. The command is still the
        solved increment added to the existing arm_sdk target.
        """
        result = self.solve_loaded_tool_pose(
            commanded_q_rad,
            measured_q_rad,
            target_tool_xyz_m,
            target_tool_rpy_rad,
            tool_offset_wrist_m,
            seed_q_rad,
        )
        return (
            result["q_command"],
            bool(result["ok"]),
            float(result["pos_err"]),
            float(result["ori_err"]),
        )

    def apply_loaded_tip_target(
        self,
        commanded_q_rad: Sequence[float],
        measured_q_rad: Sequence[float],
        target_tip_xyz_m: Sequence[float],
        offset_m: Sequence[float] = (0.205, 0.0, 0.0),
        freeze_wrist: bool = False,
        posture_q_rad: Sequence[float] | None = None,
        level_forearm: bool = False,
    ) -> tuple[list[float], bool, float, float]:
        """Position-only loaded-arm IK for a direct fingertip target.

        Wrist orientation is deliberately free. Locking all six wrist pose
        dimensions made a reachable button target require 100+ degree wrist
        flips. The solved measured-joint increment is added to the existing
        arm_sdk command so gravity sag is not fed back as the next target.

        A drooped measured pose is a bad seed: shoulder yaw runs into the
        joint stop and the fingertip stays put. Pass the raise pose as
        posture_q_rad so the one-shot starts in that basin.
        """
        import numpy as np

        commanded = np.asarray(commanded_q_rad, dtype=np.float64).reshape(7)
        measured = np.asarray(measured_q_rad, dtype=np.float64).reshape(7)
        target = np.asarray(target_tip_xyz_m, dtype=np.float64).reshape(3)
        if posture_q_rad is None:
            posture = measured.copy()
            q = measured.copy()
            iterations = 80
            step = 0.06
            bias_gain = 0.10
        else:
            posture = np.asarray(posture_q_rad, dtype=np.float64).reshape(7).copy()
            q = posture.copy()
            iterations = 100
            step = 0.08
            bias_gain = 0.20
        if freeze_wrist:
            q[4:7] = measured[4:7]
            posture[4:7] = measured[4:7]
        if level_forearm:
            return self._solve_level_forearm(
                commanded, measured, q, target, offset_m, freeze_wrist
            )
        damping = 0.04
        for _ in range(iterations):
            tip = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
            error = target - tip
            if float(np.linalg.norm(error)) < 0.002:
                break
            jacobian = np.zeros((3, 7), dtype=np.float64)
            eps = 1e-4
            for index in range(7):
                bumped = q.copy()
                bumped[index] += eps
                jacobian[:, index] = (
                    np.asarray(self.tip_xyz(bumped, offset_m), dtype=np.float64) - tip
                ) / eps
            if freeze_wrist:
                jacobian[:, 4:7] = 0.0
            normal = jacobian @ jacobian.T + (damping ** 2) * np.eye(3)
            task = jacobian.T @ np.linalg.solve(normal, error)
            bias = bias_gain * (posture - q)
            task = task + bias - jacobian.T @ np.linalg.solve(normal, jacobian @ bias)
            if freeze_wrist:
                task[4:7] = 0.0
            q = q + np.clip(task, -step, step)
            q = np.asarray(
                [
                    self.solver._clamp_joint_values({name: float(value)}, self.joint_names)[name]
                    for name, value in zip(self.joint_names, q)
                ],
                dtype=np.float64,
            )
            if freeze_wrist:
                q[4:7] = measured[4:7]
        remain = float(
            np.linalg.norm(target - np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64))
        )
        target_command = {
            name: float(command + solved - actual)
            for name, command, solved, actual in zip(
                self.joint_names,
                commanded,
                q, # q is the solved joints
                measured,
            )
        }
        target_command = self.solver._clamp_joint_values(target_command, self.joint_names)
        q_out = [float(target_command[name]) for name in self.joint_names]
        if freeze_wrist:
            q_out[4:7] = [float(value) for value in commanded[4:7]]
        return q_out, remain < 0.010, remain, 0.0

    def _forearm_axis(self, q_rad: Sequence[float]):
        import numpy as np

        poses = self.solver.fk.compute_link_poses(
            joint_values=self.joint_dict(q_rad),
            targets=[f"{self.arm}_elbow_link", f"{self.arm}_wrist_yaw_link"],
            base_link="torso_link",
        )
        elbow = np.array([poses[f"{self.arm}_elbow_link"].matrix[row][3] for row in range(3)])
        wrist = np.array([poses[f"{self.arm}_wrist_yaw_link"].matrix[row][3] for row in range(3)])
        delta = wrist - elbow
        norm = float(np.linalg.norm(delta))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return delta / norm

    def _solve_level_forearm(
        self,
        commanded,
        measured,
        q,
        target,
        offset_m: Sequence[float],
        freeze_wrist: bool,
    ) -> tuple[list[float], bool, float, float]:
        """Match button height and lateral position with a horizontal forearm.

        Chasing a closer X at this height folds the elbow into a V, because a
        level forearm already reaches about as far as the panel. Forward poke
        is left to IBVS.
        """
        import numpy as np

        q = np.asarray(q, dtype=np.float64).copy()
        for _ in range(80):
            tip = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
            axis = self._forearm_axis(q)
            yz = target[1:3] - tip[1:3]
            if float(np.linalg.norm(yz)) < 0.01 and abs(float(axis[2])) < 0.08 and float(axis[0]) > 0.9:
                break
            jacobian = np.zeros((3, 7), dtype=np.float64)
            eps = 1e-4
            for index in range(4):
                bumped = q.copy()
                bumped[index] += eps
                bumped_tip = np.asarray(self.tip_xyz(bumped, offset_m), dtype=np.float64)
                bumped_axis = self._forearm_axis(bumped)
                jacobian[0, index] = (bumped_tip[1] - tip[1]) / eps
                jacobian[1, index] = (bumped_tip[2] - tip[2]) / eps
                jacobian[2, index] = (bumped_axis[2] - axis[2]) / eps
            error = np.array([yz[0], yz[1], -float(axis[2])], dtype=np.float64)
            normal = jacobian @ jacobian.T + (0.05 ** 2) * np.eye(3)
            task = jacobian.T @ np.linalg.solve(normal, error)
            task[4:7] = 0.0
            q = q + np.clip(task, -0.08, 0.08)
            q = np.asarray(
                [
                    self.solver._clamp_joint_values({name: float(value)}, self.joint_names)[name]
                    for name, value in zip(self.joint_names, q)
                ],
                dtype=np.float64,
            )
            if freeze_wrist:
                q[4:7] = measured[4:7]
        tip = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
        axis = self._forearm_axis(q)
        remain = float(np.linalg.norm(target[1:3] - tip[1:3]))
        ok = remain < 0.02 and abs(float(axis[2])) < 0.12 and float(axis[0]) > 0.9
        target_command = {
            name: float(command + solved - actual)
            for name, command, solved, actual in zip(
                self.joint_names,
                commanded,
                q,
                measured,
            )
        }
        target_command = self.solver._clamp_joint_values(target_command, self.joint_names)
        q_out = [float(target_command[name]) for name in self.joint_names]
        if freeze_wrist:
            q_out[4:7] = [float(value) for value in commanded[4:7]]
        return q_out, ok, remain, 0.0

    def tip_xyz(self, q_rad: Sequence[float], offset_m: Sequence[float] = (0.205, 0.0, 0.0)) -> list[float]:
        """Closed-jaw point. The wrist origin is not the fingertip."""
        pose = self.fk_matrix(q_rad)
        tip = [float(pose[row][3]) for row in range(3)]
        for row in range(3):
            tip[row] += sum(float(pose[row][col]) * float(offset_m[col]) for col in range(3))
        return tip

    def apply_tip_delta(
        self,
        q_rad: Sequence[float],
        dxyz: Sequence[float],
        posture_rad: Sequence[float] | None = None,
        offset_m: Sequence[float] = (0.205, 0.0, 0.0),
    ) -> tuple[list[float], bool, float, float]:
        """Move the fingertip. Leave the spare joints near the raise pose.

        The old step locked all six wrist pose numbers. A 6 mm poke then had
        to twist shoulder yaw and the wrist to keep that orientation, so the
        arm wound up over successive steps.
        """
        import numpy as np

        q = np.asarray(q_rad, dtype=np.float64).reshape(7)
        start = q.copy()
        tip0 = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
        target = tip0 + np.asarray(dxyz, dtype=np.float64).reshape(3)
        posture = None if posture_rad is None else np.asarray(posture_rad, dtype=np.float64).reshape(7)
        damping = 0.02
        for _ in range(6):
            tip = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
            error = target - tip
            if float(np.linalg.norm(error)) < 0.0015:
                break
            jacobian = np.zeros((3, 7), dtype=np.float64)
            eps = 1e-4
            for index in range(7):
                bumped = q.copy()
                bumped[index] += eps
                jacobian[:, index] = (np.asarray(self.tip_xyz(bumped, offset_m)) - tip) / eps
            normal = jacobian @ jacobian.T + (damping ** 2) * np.eye(3)
            task = jacobian.T @ np.linalg.solve(normal, error)
            if posture is not None:
                bias = 0.35 * (posture - q)
                task = task + bias - jacobian.T @ np.linalg.solve(normal, jacobian @ bias)
            q = q + np.clip(task, -0.06, 0.06)
            q = np.asarray(
                [
                    self.solver._clamp_joint_values({name: float(value)}, self.joint_names)[name]
                    for name, value in zip(self.joint_names, q)
                ],
                dtype=np.float64,
            )
        tip1 = np.asarray(self.tip_xyz(q, offset_m), dtype=np.float64)
        remain = float(np.linalg.norm(target - tip1))
        command = np.asarray(dxyz, dtype=np.float64).reshape(3)
        command_norm = float(np.linalg.norm(command))
        moved = tip1 - tip0
        same_way = True
        if command_norm > 1e-6 and float(np.linalg.norm(moved)) > 1e-6:
            same_way = float(np.dot(moved, command)) >= 0.5 * command_norm * float(np.linalg.norm(moved))
        jump = float(np.max(np.abs(q - start)))
        moved_norm = float(np.linalg.norm(moved))
        # A damped step may stop a few millimetres short. Apply it.
        # Reject a step that goes the wrong way or swings a joint hard.
        ok = same_way and jump < 0.25 and (command_norm < 1e-4 or moved_norm > 0.001)
        return [float(value) for value in q], ok, remain, jump
