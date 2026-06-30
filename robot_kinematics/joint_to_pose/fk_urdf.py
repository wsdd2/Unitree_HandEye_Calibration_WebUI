"""Generic URDF forward kinematics.

This module computes link poses from joint values without depending on ROS,
Pinocchio, or other robotics libraries. It is intended as a transparent
baseline for Unitree G1/H-series URDF files and other standard URDF robots.

Pose convention:
    - Position is xyz in meters.
    - Quaternion output defaults to ROS order: x, y, z, w.
    - Joint values are radians for revolute/continuous joints and meters for
      prismatic joints.
"""
from __future__ import annotations

"""
说明一下在做什么
输入：URDF文件，关节值
输出：目标link的位姿
具体做法：
1. 从URDF文件中解析出机器人运动学树
2. 对每个关节，按下面的形式连乘变换：
T_parent_child(q) = T_origin_xyz_rpy * T_joint_motion(q)
3. 最终得到：
T_base_target = T_base_link1 * T_link1_link2 * ... * T_linkN_target
4. 如果额外传入了机器人根节点在世界系下的位姿，则会输出：
T_world_target = T_world_base * T_base_target
5. 输出JSON格式，包含：
- urdf：本次加载的URDF文件路径。
- root_links：URDF中没有父关节的根link。G1通常是pelvis。
- base_link：本次输出位姿所使用的基准link。
- missing_joint_values_defaulted_to_zero：没有输入、被默认设为0.0的active关节。
- targets：每个目标link的计算结果。
- position_xyz：目标link原点在输出坐标系中的位置，单位米。
- orientation_quat_xyzw：目标link相对输出坐标系的姿态四元数，顺序为x, y, z, w。
- orientation_quat_wxyz：如果使用--quat-order wxyz，输出字段会变成这个，顺序为w, x, y, z。
- transform_matrix：4x4齐次变换矩阵，表示T_output_target。
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
Matrix4 = List[List[float]]


@dataclass(frozen=True)
class JointLimit:
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: Vector3
    origin_rpy: Vector3
    axis: Vector3
    limit: JointLimit


@dataclass(frozen=True)
class Pose:
    link: str
    matrix: Matrix4

    @property
    def position_xyz(self) -> Vector3:
        return (self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])

    def quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        return rotation_matrix_to_quaternion_xyzw(self.matrix)


class URDFFK:
    """Forward kinematics model loaded from one URDF file."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.urdf_path = Path(urdf_path)
        self.links: List[str] = []
        self.joints: List[Joint] = []
        self.children_by_parent: Dict[str, List[Joint]] = {}
        self.joint_by_name: Dict[str, Joint] = {}
        self.root_links: List[str] = []
        self._load()

    def _load(self) -> None:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        self.links = [
            str(link.attrib["name"])
            for link in root.findall(_tag("link"))
            if "name" in link.attrib
        ]

        joints: List[Joint] = []
        child_links = set()
        for node in root.findall(_tag("joint")):
            name = str(node.attrib.get("name", ""))
            joint_type = str(node.attrib.get("type", "fixed"))
            parent_node = node.find(_tag("parent"))
            child_node = node.find(_tag("child"))
            if not name or parent_node is None or child_node is None:
                continue

            parent = str(parent_node.attrib["link"])
            child = str(child_node.attrib["link"])
            child_links.add(child)

            origin_node = node.find(_tag("origin"))
            axis_node = node.find(_tag("axis"))
            limit_node = node.find(_tag("limit"))

            joint = Joint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=_read_vector(origin_node, "xyz", (0.0, 0.0, 0.0)),
                origin_rpy=_read_vector(origin_node, "rpy", (0.0, 0.0, 0.0)),
                axis=_normalize(
                    _read_vector(axis_node, "xyz", (1.0, 0.0, 0.0))
                ),
                limit=_read_limit(limit_node),
            )
            joints.append(joint)

        self.joints = joints
        self.joint_by_name = {joint.name: joint for joint in joints}
        self.children_by_parent = {}
        for joint in joints:
            self.children_by_parent.setdefault(joint.parent, []).append(joint)

        self.root_links = [link for link in self.links if link not in child_links]
        if not self.root_links and self.links:
            self.root_links = [self.links[0]]

    def active_joint_names(self) -> List[str]:
        return [
            joint.name
            for joint in self.joints
            if joint.joint_type in {"revolute", "continuous", "prismatic"}
        ]

    def compute_all_link_poses(
        self,
        joint_values: Mapping[str, float],
        base_link: Optional[str] = None,
        base_pose: Optional[Matrix4] = None,
        clamp_to_limits: bool = False,
    ) -> Dict[str, Pose]:
        """Compute poses for all reachable links.

        Args:
            joint_values: Map from URDF joint name to joint value. Missing
                active joints are treated as zero.
            base_link: Link to attach the provided base pose to. Defaults to
                the first root link in the URDF.
            base_pose: Transform from output/world frame to base_link. If not
                provided, identity is used, so results are relative to base_link.
            clamp_to_limits: Clamp revolute/prismatic joint values to URDF
                limits before computing transforms.
        """

        if not self.root_links:
            raise ValueError(f"No links found in URDF: {self.urdf_path}")

        selected_base = base_link or self.root_links[0]
        if selected_base not in self.links:
            raise ValueError(f"Base link '{selected_base}' is not in URDF")

        poses: Dict[str, Pose] = {}
        root_pose = base_pose if base_pose is not None else identity_matrix()

        def visit(link_name: str, link_pose: Matrix4) -> None:
            poses[link_name] = Pose(link=link_name, matrix=link_pose)
            for joint in self.children_by_parent.get(link_name, []):
                q = float(joint_values.get(joint.name, 0.0))
                if clamp_to_limits:
                    q = _clamp_joint_value(joint, q)
                joint_tf = joint_transform(joint, q)
                child_pose = matmul4(link_pose, joint_tf)
                visit(joint.child, child_pose)

        visit(selected_base, root_pose)
        return poses

    def compute_link_poses(
        self,
        joint_values: Mapping[str, float],
        targets: Iterable[str],
        base_link: Optional[str] = None,
        base_pose: Optional[Matrix4] = None,
        clamp_to_limits: bool = False,
    ) -> Dict[str, Pose]:
        all_poses = self.compute_all_link_poses(
            joint_values=joint_values,
            base_link=base_link,
            base_pose=base_pose,
            clamp_to_limits=clamp_to_limits,
        )
        missing = [target for target in targets if target not in all_poses]
        if missing:
            raise ValueError(f"Target link(s) not reachable or not found: {missing}")
        return {target: all_poses[target] for target in targets}


def joint_transform(joint: Joint, value: float) -> Matrix4:
    if joint.joint_type == "floating":
        raise NotImplementedError(
            "Floating joints are not handled as scalar joint values. "
            "Pass the robot base pose with --base-xyz/--base-rpy or "
            "--base-quat-xyzw instead."
        )
    if joint.joint_type == "planar":
        raise NotImplementedError("Planar joints are not supported by this FK helper.")

    transform = transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
    if joint.joint_type in {"revolute", "continuous"}:
        return matmul4(transform, rotation_about_axis(joint.axis, value))
    if joint.joint_type == "prismatic":
        axis = joint.axis
        return matmul4(
            transform,
            translation_matrix((axis[0] * value, axis[1] * value, axis[2] * value)),
        )
    return transform


def transform_from_xyz_rpy(xyz: Vector3, rpy: Vector3) -> Matrix4:
    return matmul4(translation_matrix(xyz), rpy_matrix(rpy))


def identity_matrix() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_matrix(xyz: Vector3) -> Matrix4:
    matrix = identity_matrix()
    matrix[0][3] = xyz[0]
    matrix[1][3] = xyz[1]
    matrix[2][3] = xyz[2]
    return matrix


def rpy_matrix(rpy: Vector3) -> Matrix4:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # URDF uses fixed-axis roll-pitch-yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_about_axis(axis: Vector3, angle: float) -> Matrix4:
    x, y, z = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def rotation_matrix_to_quaternion_xyzw(matrix: Matrix4) -> Tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0][0], matrix[0][1], matrix[0][2]
    m10, m11, m12 = matrix[1][0], matrix[1][1], matrix[1][2]
    m20, m21, m22 = matrix[2][0], matrix[2][1], matrix[2][2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return _normalize_quaternion((qx, qy, qz, qw))


def quaternion_xyzw_to_matrix(quat: Sequence[float]) -> Matrix4:
    if len(quat) != 4:
        raise ValueError("Quaternion must have exactly 4 values: x y z w")
    x, y, z, w = _normalize_quaternion(
        (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    )
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def base_pose_matrix(
    xyz: Optional[Sequence[float]],
    rpy: Optional[Sequence[float]],
    quat_xyzw: Optional[Sequence[float]],
) -> Matrix4:
    position = _sequence_to_vector3(xyz or (0.0, 0.0, 0.0), "base xyz")
    if quat_xyzw is not None:
        rotation = quaternion_xyzw_to_matrix(quat_xyzw)
    else:
        rotation = rpy_matrix(_sequence_to_vector3(rpy or (0.0, 0.0, 0.0), "base rpy"))
    transform = rotation
    transform[0][3] = position[0]
    transform[1][3] = position[1]
    transform[2][3] = position[2]
    return transform


def load_joint_values(path: Optional[str], pairs: Optional[Sequence[str]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "joints" in data:
            data = data["joints"]
        if not isinstance(data, dict):
            raise ValueError("Joint JSON must be an object or contain a 'joints' object")
        values.update({str(name): float(value) for name, value in data.items()})

    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Joint override must look like name=value, got: {pair}")
        name, raw_value = pair.split("=", 1)
        values[name.strip()] = float(raw_value)
    return values


def parse_targets(target_args: Optional[Sequence[str]]) -> List[str]:
    targets: List[str] = []
    for item in target_args or []:
        targets.extend(part.strip() for part in item.split(",") if part.strip())
    return targets


def pose_to_json(pose: Pose, quat_order: str) -> Dict[str, object]:
    quat_xyzw = pose.quaternion_xyzw()
    if quat_order == "wxyz":
        quat: Sequence[float] = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
        quat_key = "orientation_quat_wxyz"
    else:
        quat = quat_xyzw
        quat_key = "orientation_quat_xyzw"
    return {
        "position_xyz": list(pose.position_xyz),
        quat_key: list(quat),
        "transform_matrix": pose.matrix,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute link poses from URDF joint values."
    )
    parser.add_argument("--urdf", required=True, help="Path to a URDF file.")
    parser.add_argument("--joints", help="JSON file containing joint values.")
    parser.add_argument(
        "--joint",
        action="append",
        help="Override or provide one joint value as name=value. Can repeat.",
    )
    parser.add_argument(
        "--target",
        action="append",
        help="Target link name. Can repeat or use comma-separated names.",
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
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order for output. Default: xyzw.",
    )
    parser.add_argument(
        "--clamp-to-limits",
        action="store_true",
        help="Clamp provided joint values to URDF limits.",
    )
    parser.add_argument(
        "--list-active-joints",
        action="store_true",
        help="Print active joint names and exit.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model = URDFFK(args.urdf)

    if args.list_active_joints:
        print(json.dumps(model.active_joint_names(), indent=2))
        return 0

    targets = parse_targets(args.target)
    if not targets:
        raise ValueError("Provide at least one --target link name.")

    joint_values = load_joint_values(args.joints, args.joint)
    base_pose = base_pose_matrix(args.base_xyz, args.base_rpy, args.base_quat_xyzw)
    poses = model.compute_link_poses(
        joint_values=joint_values,
        targets=targets,
        base_link=args.base_link,
        base_pose=base_pose,
        clamp_to_limits=args.clamp_to_limits,
    )

    output = {
        "urdf": str(Path(args.urdf)),
        "root_links": model.root_links,
        "base_link": args.base_link or model.root_links[0],
        "missing_joint_values_defaulted_to_zero": [
            name for name in model.active_joint_names() if name not in joint_values
        ],
        "targets": {
            link_name: pose_to_json(pose, args.quat_order)
            for link_name, pose in poses.items()
        },
    }
    indent = 2 if args.pretty else None
    print(json.dumps(output, indent=indent))
    return 0


def _tag(name: str) -> str:
    return f".//{{*}}{name}"


def _read_vector(
    node: Optional[ET.Element], attribute: str, default: Vector3
) -> Vector3:
    if node is None or attribute not in node.attrib:
        return default
    return _sequence_to_vector3(node.attrib[attribute].split(), attribute)


def _sequence_to_vector3(values: Sequence[object], name: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _read_limit(node: Optional[ET.Element]) -> JointLimit:
    if node is None:
        return JointLimit(lower=None, upper=None)
    lower = float(node.attrib["lower"]) if "lower" in node.attrib else None
    upper = float(node.attrib["upper"]) if "upper" in node.attrib else None
    return JointLimit(lower=lower, upper=upper)


def _normalize(vector: Vector3) -> Vector3:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm == 0.0:
        return (1.0, 0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _normalize_quaternion(
    quat_xyzw: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat_xyzw))
    if norm == 0.0:
        raise ValueError("Quaternion norm cannot be zero")
    return tuple(value / norm for value in quat_xyzw)  # type: ignore[return-value]


def _clamp_joint_value(joint: Joint, value: float) -> float:
    if joint.joint_type == "continuous":
        return value
    if joint.limit.lower is not None:
        value = max(value, joint.limit.lower)
    if joint.limit.upper is not None:
        value = min(value, joint.limit.upper)
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
