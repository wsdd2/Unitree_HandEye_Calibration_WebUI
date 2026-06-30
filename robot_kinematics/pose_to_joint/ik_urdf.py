"""Generic numerical inverse kinematics for URDF robots.

This module solves one target-link pose with damped least squares. It reuses
the pure-Python forward kinematics implementation from ../joint_to_pose.

Pose convention:
    - Position is xyz in meters.
    - Quaternion inputs use x, y, z, w order.
    - Joint values are radians for revolute/continuous joints and meters for
      prismatic joints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

JOINT_TO_POSE_DIR = Path(__file__).resolve().parents[1] / "joint_to_pose"
if str(JOINT_TO_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_TO_POSE_DIR))

from fk_urdf import (  # noqa: E402
    Matrix4,
    URDFFK,
    base_pose_matrix,
    load_joint_values,
    quaternion_xyzw_to_matrix,
    rpy_matrix,
)

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class IKSolution:
    success: bool
    iterations: int
    joint_values: Dict[str, float]
    final_position_error_norm: float
    final_orientation_error_norm: float
    final_error_norm: float
    target_link: str
    active_joints: List[str]
    message: str


class URDFIK:
    """Numerical IK solver for one URDF target link."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.fk = URDFFK(urdf_path)
        self.parent_joint_by_child = {joint.child: joint for joint in self.fk.joints}

    def default_chain_joints(
        self, target_link: str, base_link: Optional[str] = None
    ) -> List[str]:
        selected_base = base_link or self.fk.root_links[0]
        if target_link not in self.fk.links:
            raise ValueError(f"Target link '{target_link}' is not in URDF")
        if selected_base not in self.fk.links:
            raise ValueError(f"Base link '{selected_base}' is not in URDF")

        joints: List[str] = []
        current = target_link
        while current != selected_base:
            joint = self.parent_joint_by_child.get(current)
            if joint is None:
                raise ValueError(
                    f"Cannot trace a chain from '{selected_base}' to '{target_link}'"
                )
            if joint.joint_type in {"revolute", "continuous", "prismatic"}:
                joints.append(joint.name)
            current = joint.parent
        joints.reverse()
        return joints

    def solve(
        self,
        target_link: str,
        target_pose: Matrix4,
        initial_joint_values: Optional[Mapping[str, float]] = None,
        active_joints: Optional[Sequence[str]] = None,
        base_link: Optional[str] = None,
        base_pose: Optional[Matrix4] = None,
        max_iterations: int = 200,
        tolerance_position: float = 1e-4,
        tolerance_orientation: float = 1e-3,
        damping: float = 1e-3,
        step_scale: float = 0.5,
        finite_difference_step: float = 1e-6,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        clamp_to_limits: bool = True,
    ) -> IKSolution:
        selected_active_joints = list(
            active_joints
            if active_joints is not None
            else self.default_chain_joints(target_link, base_link)
        )
        self._validate_active_joints(selected_active_joints)

        q = {
            name: float(value)
            for name, value in (initial_joint_values or {}).items()
            if name in self.fk.joint_by_name
        }
        for name in selected_active_joints:
            q.setdefault(name, 0.0)
        if clamp_to_limits:
            q = self._clamp_joint_values(q, selected_active_joints)

        message = "Maximum iterations reached"
        pos_norm = math.inf
        orient_norm = math.inf
        error_norm = math.inf
        iterations_done = 0

        for iteration in range(max_iterations + 1):
            current_pose = self._compute_pose(q, target_link, base_link, base_pose)
            error = pose_error(current_pose, target_pose)
            pos_norm = vector_norm(error[:3])
            orient_norm = vector_norm(error[3:])
            weighted_error = weight_error(error, position_weight, orientation_weight)
            error_norm = vector_norm(weighted_error)
            iterations_done = iteration

            if pos_norm <= tolerance_position and orient_norm <= tolerance_orientation:
                message = "Converged"
                return IKSolution(
                    success=True,
                    iterations=iterations_done,
                    joint_values={name: q[name] for name in selected_active_joints},
                    final_position_error_norm=pos_norm,
                    final_orientation_error_norm=orient_norm,
                    final_error_norm=error_norm,
                    target_link=target_link,
                    active_joints=selected_active_joints,
                    message=message,
                )

            if iteration == max_iterations:
                break

            jacobian = numerical_jacobian(
                solver=self,
                joint_values=q,
                active_joints=selected_active_joints,
                target_link=target_link,
                base_link=base_link,
                base_pose=base_pose,
                finite_difference_step=finite_difference_step,
                current_pose=current_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            try:
                delta_q = damped_least_squares_step(
                    jacobian=jacobian,
                    weighted_error=weighted_error,
                    damping=damping,
                )
            except ValueError as exc:
                message = str(exc)
                break

            for name, delta in zip(selected_active_joints, delta_q):
                q[name] = q.get(name, 0.0) + step_scale * delta
            if clamp_to_limits:
                q = self._clamp_joint_values(q, selected_active_joints)

        return IKSolution(
            success=False,
            iterations=iterations_done,
            joint_values={name: q[name] for name in selected_active_joints},
            final_position_error_norm=pos_norm,
            final_orientation_error_norm=orient_norm,
            final_error_norm=error_norm,
            target_link=target_link,
            active_joints=selected_active_joints,
            message=message,
        )

    def _compute_pose(
        self,
        joint_values: Mapping[str, float],
        target_link: str,
        base_link: Optional[str],
        base_pose: Optional[Matrix4],
    ) -> Matrix4:
        return self.fk.compute_link_poses(
            joint_values=joint_values,
            targets=[target_link],
            base_link=base_link,
            base_pose=base_pose,
            clamp_to_limits=False,
        )[target_link].matrix

    def _validate_active_joints(self, active_joints: Sequence[str]) -> None:
        if not active_joints:
            raise ValueError("No active joints selected for IK")
        known_active = set(self.fk.active_joint_names())
        unknown = [name for name in active_joints if name not in known_active]
        if unknown:
            raise ValueError(f"Unknown or non-active IK joint(s): {unknown}")

    def _clamp_joint_values(
        self, joint_values: Mapping[str, float], joint_names: Iterable[str]
    ) -> Dict[str, float]:
        clamped = dict(joint_values)
        for name in joint_names:
            joint = self.fk.joint_by_name[name]
            value = clamped.get(name, 0.0)
            if joint.joint_type != "continuous":
                if joint.limit.lower is not None:
                    value = max(value, joint.limit.lower)
                if joint.limit.upper is not None:
                    value = min(value, joint.limit.upper)
            clamped[name] = value
        return clamped


def numerical_jacobian(
    solver: URDFIK,
    joint_values: Mapping[str, float],
    active_joints: Sequence[str],
    target_link: str,
    base_link: Optional[str],
    base_pose: Optional[Matrix4],
    finite_difference_step: float,
    current_pose: Matrix4,
    position_weight: float,
    orientation_weight: float,
) -> List[List[float]]:
    columns: List[List[float]] = []
    for name in active_joints:
        q_perturbed = dict(joint_values)
        q_perturbed[name] = q_perturbed.get(name, 0.0) + finite_difference_step
        perturbed_pose = solver._compute_pose(
            q_perturbed, target_link, base_link, base_pose
        )
        delta = pose_delta(current_pose, perturbed_pose)
        weighted_delta = weight_error(delta, position_weight, orientation_weight)
        columns.append([value / finite_difference_step for value in weighted_delta])

    return [
        [columns[col][row] for col in range(len(active_joints))]
        for row in range(6)
    ]


def damped_least_squares_step(
    jacobian: List[List[float]], weighted_error: Sequence[float], damping: float
) -> List[float]:
    if not jacobian or not jacobian[0]:
        raise ValueError("Jacobian is empty")

    row_count = len(jacobian)
    col_count = len(jacobian[0])
    jjt = [
        [
            sum(jacobian[row][col] * jacobian[other][col] for col in range(col_count))
            for other in range(row_count)
        ]
        for row in range(row_count)
    ]
    damping_square = damping * damping
    for i in range(row_count):
        jjt[i][i] += damping_square

    y = solve_linear_system(jjt, list(weighted_error))
    return [
        sum(jacobian[row][col] * y[row] for row in range(row_count))
        for col in range(col_count)
    ]


def solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("IK linear solve failed: singular damped system")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        for idx in range(col, size + 1):
            augmented[col][idx] /= pivot

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            for idx in range(col, size + 1):
                augmented[row][idx] -= factor * augmented[col][idx]

    return [augmented[row][size] for row in range(size)]


def pose_error(current: Matrix4, target: Matrix4) -> List[float]:
    pos_error = [
        target[0][3] - current[0][3],
        target[1][3] - current[1][3],
        target[2][3] - current[2][3],
    ]
    rot_error = rotation_vector(relative_rotation(current, target))
    return pos_error + rot_error


def pose_delta(from_pose: Matrix4, to_pose: Matrix4) -> List[float]:
    pos_delta = [
        to_pose[0][3] - from_pose[0][3],
        to_pose[1][3] - from_pose[1][3],
        to_pose[2][3] - from_pose[2][3],
    ]
    rot_delta = rotation_vector(relative_rotation(from_pose, to_pose))
    return pos_delta + rot_delta


def relative_rotation(current: Matrix4, target: Matrix4) -> List[List[float]]:
    current_rot_t = [[current[col][row] for col in range(3)] for row in range(3)]
    target_rot = [[target[row][col] for col in range(3)] for row in range(3)]
    return [
        [
            sum(target_rot[row][k] * current_rot_t[k][col] for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def rotation_vector(rotation: List[List[float]]) -> List[float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    theta = math.acos(cos_theta)

    skew = [
        rotation[2][1] - rotation[1][2],
        rotation[0][2] - rotation[2][0],
        rotation[1][0] - rotation[0][1],
    ]

    if theta < 1e-9:
        return [0.5 * value for value in skew]

    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        scale = theta / max(vector_norm(skew), 1e-12)
        return [scale * value for value in skew]

    scale = theta / (2.0 * sin_theta)
    return [scale * value for value in skew]


def target_pose_matrix(
    xyz: Sequence[float],
    rpy: Optional[Sequence[float]],
    quat_xyzw: Optional[Sequence[float]],
) -> Matrix4:
    position = sequence_to_vector3(xyz, "target xyz")
    if quat_xyzw is not None:
        transform = quaternion_xyzw_to_matrix(quat_xyzw)
    else:
        transform = rpy_matrix(sequence_to_vector3(rpy or (0.0, 0.0, 0.0), "target rpy"))
    transform[0][3] = position[0]
    transform[1][3] = position[1]
    transform[2][3] = position[2]
    return transform


def load_target_pose(path: Optional[str], args: argparse.Namespace) -> Matrix4:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "position_xyz" in data:
            xyz = data["position_xyz"]
        elif "xyz" in data:
            xyz = data["xyz"]
        else:
            raise ValueError("Target JSON must contain 'position_xyz' or 'xyz'")

        quat = data.get("orientation_quat_xyzw") or data.get("quat_xyzw")
        rpy = data.get("orientation_rpy") or data.get("rpy")
        if quat is None and rpy is None:
            raise ValueError(
                "Target JSON must contain 'orientation_quat_xyzw'/'quat_xyzw' "
                "or 'orientation_rpy'/'rpy'"
            )
        return target_pose_matrix(xyz=xyz, rpy=rpy, quat_xyzw=quat)

    if args.target_xyz is None:
        raise ValueError("Provide --target-xyz or --target-pose")
    return target_pose_matrix(
        xyz=args.target_xyz,
        rpy=args.target_rpy,
        quat_xyzw=args.target_quat_xyzw,
    )


def parse_name_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    names: List[str] = []
    for item in values:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return names


def solution_to_json(solution: IKSolution) -> Dict[str, object]:
    return {
        "success": solution.success,
        "message": solution.message,
        "iterations": solution.iterations,
        "target_link": solution.target_link,
        "active_joints": solution.active_joints,
        "joint_values": solution.joint_values,
        "final_position_error_norm": solution.final_position_error_norm,
        "final_orientation_error_norm": solution.final_orientation_error_norm,
        "final_error_norm": solution.final_error_norm,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve one target-link pose with numerical URDF IK."
    )
    parser.add_argument("--urdf", required=True, help="Path to a URDF file.")
    parser.add_argument("--target-link", required=True, help="Target link to solve.")
    parser.add_argument(
        "--target-pose",
        help="JSON file with position_xyz and orientation_quat_xyzw or orientation_rpy.",
    )
    parser.add_argument(
        "--target-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Target position in output/base frame.",
    )
    parser.add_argument(
        "--target-rpy",
        nargs=3,
        type=float,
        metavar=("R", "P", "Y"),
        help="Target roll pitch yaw in radians.",
    )
    parser.add_argument(
        "--target-quat-xyzw",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
        help="Target quaternion in x y z w order.",
    )
    parser.add_argument("--initial-joints", help="JSON file containing initial joints.")
    parser.add_argument(
        "--joint",
        action="append",
        help="Override or provide one initial joint as name=value. Can repeat.",
    )
    parser.add_argument(
        "--active-joint",
        action="append",
        help="Joint allowed to move. Can repeat or use comma-separated names. "
        "Defaults to the active joints on the base-to-target chain.",
    )
    parser.add_argument("--base-link", help="Base/root link for the output frame.")
    parser.add_argument(
        "--base-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="World-to-base translation in meters.",
    )
    parser.add_argument(
        "--base-rpy",
        nargs=3,
        type=float,
        metavar=("R", "P", "Y"),
        help="World-to-base roll pitch yaw in radians.",
    )
    parser.add_argument(
        "--base-quat-xyzw",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
        help="World-to-base quaternion in x y z w order.",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance-position", type=float, default=1e-4)
    parser.add_argument("--tolerance-orientation", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument(
        "--ignore-limits",
        action="store_true",
        help="Do not clamp solved joints to URDF limits.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    solver = URDFIK(args.urdf)
    initial_joints = load_joint_values(args.initial_joints, args.joint)
    active_joints = parse_name_list(args.active_joint)
    target_pose = load_target_pose(args.target_pose, args)
    base_pose = base_pose_matrix(args.base_xyz, args.base_rpy, args.base_quat_xyzw)

    solution = solver.solve(
        target_link=args.target_link,
        target_pose=target_pose,
        initial_joint_values=initial_joints,
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

    output = {
        "urdf": str(Path(args.urdf)),
        "root_links": solver.fk.root_links,
        "base_link": args.base_link or solver.fk.root_links[0],
        **solution_to_json(solution),
    }
    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0 if solution.success else 2


def weight_error(
    error: Sequence[float], position_weight: float, orientation_weight: float
) -> List[float]:
    return [
        position_weight * error[0],
        position_weight * error[1],
        position_weight * error[2],
        orientation_weight * error[3],
        orientation_weight * error[4],
        orientation_weight * error[5],
    ]


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def sequence_to_vector3(values: Sequence[object], name: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
