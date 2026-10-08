# -*- coding: utf-8 -*-
"""Static gravity feedforward for H2 arm_sdk from URDF masses.

Does not need Pinocchio. Uses the already-uploaded robot_kinematics FK plus
link inertials, then virtual work: tau_i = axis · Σ (r_com × mg).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Mapping

from .h2_ee_ik import ARM_JOINT_NAMES, ensure_robot_kinematics, resolve_urdf


GRAVITY_MPS2 = (0.0, 0.0, -9.80665)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
MOTOR_TO_JOINT = {
    12: "waist_yaw_joint",
    13: "waist_roll_joint",
    14: "waist_pitch_joint",
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    20: "left_wrist_pitch_joint",
    21: "left_wrist_yaw_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}
JOINT_TO_MOTOR = {name: index for index, name in MOTOR_TO_JOINT.items()}
# Dex1-1 ≈ 0.55 kg at 7 cm, plus wrist D435/mount ≈ 0.20 kg.
DEFAULT_WRIST_PAYLOAD_KG = 0.75
DEFAULT_WRIST_PAYLOAD_COM = (0.07, 0.0, 0.0)
TAU_ABS_LIMIT_NM = 50.0


class H2GravityCompensator:
    def __init__(
        self,
        urdf_path: str = "",
        *,
        wrist_payload_kg: float = DEFAULT_WRIST_PAYLOAD_KG,
        wrist_payload_com: tuple[float, float, float] = DEFAULT_WRIST_PAYLOAD_COM,
        payload_arm: str = "right",
        tau_scale: float = 1.0,
    ) -> None:
        ensure_robot_kinematics()
        from fk_urdf import URDFFK, joint_transform, matmul4

        self._joint_transform = joint_transform
        self._matmul4 = matmul4
        self.urdf = resolve_urdf(urdf_path)
        self.fk = URDFFK(self.urdf)
        self.tau_scale = float(tau_scale)
        self.mass_by_link = _load_link_masses(self.urdf)
        payload = max(0.0, float(wrist_payload_kg))
        if payload > 0.0:
            wrist_link = ARM_JOINT_NAMES["left" if payload_arm == "left" else "right"][6]
            wrist_link = wrist_link.replace("_joint", "_link")
            self._add_payload(wrist_link, payload, wrist_payload_com)
        self._descendants = {
            joint.name: _descendant_links(self.fk, joint.child)
            for joint in self.fk.joints
            if joint.joint_type in {"revolute", "continuous"}
        }
        self._command_joints = tuple(MOTOR_TO_JOINT[index] for index in range(15, 29))

    def _add_payload(
        self,
        link: str,
        mass: float,
        com: tuple[float, float, float],
    ) -> None:
        old_mass, old_com = self.mass_by_link.get(link, (0.0, (0.0, 0.0, 0.0)))
        total = old_mass + mass
        if total <= 1e-9:
            return
        merged = tuple(
            (old_mass * old_com[i] + mass * com[i]) / total for i in range(3)
        )
        self.mass_by_link[link] = (total, merged)

    def joint_values(
        self,
        command_q: Mapping[int, float],
        lowstate: Any | None = None,
    ) -> dict[str, float]:
        values = {name: 0.0 for name in self.fk.active_joint_names()}
        if lowstate is not None and getattr(lowstate, "motor_state", None) is not None:
            for index, name in MOTOR_TO_JOINT.items():
                if index < len(lowstate.motor_state):
                    values[name] = float(lowstate.motor_state[index].q)
        for index, q in command_q.items():
            name = MOTOR_TO_JOINT.get(int(index))
            if name is not None:
                values[name] = float(q)
        return values

    def motor_tau(
        self,
        command_q: Mapping[int, float],
        lowstate: Any | None = None,
    ) -> dict[int, float]:
        values = self.joint_values(command_q, lowstate)
        poses = self.fk.compute_all_link_poses(values)
        tau_by_joint = self._joint_tau(poses)
        out: dict[int, float] = {}
        for name in self._command_joints:
            motor = JOINT_TO_MOTOR[name]
            tau = self.tau_scale * float(tau_by_joint.get(name, 0.0))
            out[motor] = max(-TAU_ABS_LIMIT_NM, min(TAU_ABS_LIMIT_NM, tau))
        return out

    def _joint_tau(self, poses: Mapping[str, Any]) -> dict[str, float]:
        gx, gy, gz = GRAVITY_MPS2
        taus: dict[str, float] = {}
        for joint in self.fk.joints:
            if joint.joint_type not in {"revolute", "continuous"}:
                continue
            parent_pose = poses.get(joint.parent)
            if parent_pose is None:
                continue
            origin = self._matmul4(
                parent_pose.matrix,
                self._joint_transform(joint, 0.0),
            )
            axis = _rotate(origin, joint.axis)
            origin_xyz = (origin[0][3], origin[1][3], origin[2][3])
            tau = 0.0
            for link in self._descendants.get(joint.name, ()):
                inertial = self.mass_by_link.get(link)
                link_pose = poses.get(link)
                if inertial is None or link_pose is None:
                    continue
                mass, com_local = inertial
                if mass <= 1e-9:
                    continue
                com = _transform_point(link_pose.matrix, com_local)
                rx = com[0] - origin_xyz[0]
                ry = com[1] - origin_xyz[1]
                rz = com[2] - origin_xyz[2]
                fx, fy, fz = mass * gx, mass * gy, mass * gz
                # (r × mg) · axis
                tau += (
                    axis[0] * (ry * fz - rz * fy)
                    + axis[1] * (rz * fx - rx * fz)
                    + axis[2] * (rx * fy - ry * fx)
                )
            taus[joint.name] = tau
        return taus


def _descendant_links(fk: Any, root_link: str) -> tuple[str, ...]:
    found = [root_link]
    stack = [root_link]
    seen = {root_link}
    while stack:
        link = stack.pop()
        for child in fk.children_by_parent.get(link, ()):
            if child.child in seen:
                continue
            seen.add(child.child)
            found.append(child.child)
            stack.append(child.child)
    return tuple(found)


def _load_link_masses(urdf_path) -> dict[str, tuple[float, tuple[float, float, float]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    masses: dict[str, tuple[float, tuple[float, float, float]]] = {}
    for link in root.findall(".//{*}link"):
        name = str(link.attrib.get("name") or "")
        inertial = link.find("{*}inertial")
        if not name or inertial is None:
            continue
        mass_node = inertial.find("{*}mass")
        origin_node = inertial.find("{*}origin")
        if mass_node is None:
            continue
        mass = float(mass_node.attrib.get("value") or 0.0)
        com = (0.0, 0.0, 0.0)
        if origin_node is not None and "xyz" in origin_node.attrib:
            parts = origin_node.attrib["xyz"].split()
            if len(parts) == 3:
                com = (float(parts[0]), float(parts[1]), float(parts[2]))
        if mass > 0.0:
            masses[name] = (mass, com)
    return masses


def _rotate(matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _transform_point(matrix, point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3],
    )


def format_tau_summary(tau: Mapping[int, float], arm: str = "right") -> str:
    names = ARM_JOINT_NAMES["left" if arm == "left" else "right"]
    motors = range(15, 22) if arm == "left" else range(22, 29)
    parts = [
        f"{name.split('_', 1)[1]}={tau.get(motor, 0.0):+.2f}"
        for name, motor in zip(names, motors)
    ]
    return " ".join(parts)
