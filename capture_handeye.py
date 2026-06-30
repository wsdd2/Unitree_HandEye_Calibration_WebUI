# -*- coding: utf-8 -*-
"""RealSense D435i 手眼标定采集与求解。

支持两种安装方式：
- eye-in-hand / hand-in-eye：相机安装在灵巧手/末端上，棋盘格固定在外部。
- eye-to-hand / hand-to-eye：相机固定在外部，棋盘格固定在灵巧手/末端上。

每次采集输入机器人当前灵巧手/末端 6D 位姿：x y z rx ry rz。
默认解释为 T_base_hand，平移 mm，旋转 deg，欧拉角顺序 xyz。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from handeye_calib.camera import RealSenseD435i
from handeye_calib.calibration_target import build_object_points, solve_target_pose
from handeye_calib.chessboard import (
    find_chessboard_corners,
    gamma_correct_bgr,
    put_text_bgr_adaptive,
)
from handeye_calib.debug_stream import DebugStreamServer
from handeye_calib.io_utils import load_camera_params, save_capture_record
from handeye_calib.solver import load_capture_records, normalize_mode, opencv_method_from_name, solve_handeye
from handeye_calib.transforms import parse_pose_text, pose_to_transform


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_ARM_PRESETS_JSON = PROJECT_ROOT / "data" / "arm_presets.json"
DEFAULT_WRIST_CAM_SERIAL = ""
DEFAULT_HEAD_CAM_SERIAL = ""
DEFAULT_FK_URDF = PROJECT_ROOT / "robots" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf"
DEFAULT_HAND_FRAME = "right_dex1_gripper_tcp"
RIGHT_DEX1_BASE_LINK = "right_dex1_base_link"
RIGHT_DEX1_FINGER_LINK_1 = "right_dex1_finger_link_1"
RIGHT_DEX1_FINGER_LINK_2 = "right_dex1_finger_link_2"
RIGHT_DEX1_GRIPPER_TCP = "right_dex1_gripper_tcp"
RIGHT_DEX1_FINGER_JOINTS = (
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
DEX1_GRIPPER_FRAMES = {
    RIGHT_DEX1_GRIPPER_TCP,
    RIGHT_DEX1_BASE_LINK,
    RIGHT_DEX1_FINGER_LINK_1,
    RIGHT_DEX1_FINGER_LINK_2,
}
DEFAULT_RIGHT_ARM_FK_TARGETS = (
    "right_shoulder_pitch_link,right_elbow_link,right_wrist_yaw_link,"
    f"{RIGHT_DEX1_GRIPPER_TCP}"
)
RIGHT_ARM_JOINT_NAMES = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
ARM_SDK_WEIGHT = 29
WAIST_JOINTS = [12, 13, 14]
LEFT_ARM_JOINTS = list(range(15, 22))
RIGHT_ARM_JOINTS = list(range(22, 29))
RIGHT_WRIST_JOINTS = [26, 27, 28]
RIGHT_SHOULDER_ELBOW_JOINTS = [22, 23, 24, 25]
RIGHT_ARM_JOINT_SHORT_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


@dataclass
class ArmSessionState:
    """Track commanded posture so joint nudges accumulate and arm_sdk keeps holding."""

    commanded_q: Optional[dict[int, float]] = None
    preset_reference_q: dict[int, float] = field(default_factory=dict)
    active_preset_name: str = ""
    hold_active: bool = False
    last_republish_ts: float = 0.0
    last_safety_check_ts: float = 0.0

    def base_q(self, controller: "G1ArmWaypointController") -> dict[int, float]:
        if self.commanded_q is not None:
            return dict(self.commanded_q)
        return controller.command_joint_positions()

    def set_commanded(self, target_q: dict[int, float]) -> None:
        self.commanded_q = dict(target_q)
        self.hold_active = True

    def clear_hold(self) -> None:
        self.hold_active = False

    def set_preset_reference(self, preset_name: str, target_q: dict[int, float]) -> None:
        self.active_preset_name = preset_name
        self.preset_reference_q = {joint: float(target_q[joint]) for joint in RIGHT_ARM_JOINTS}

    def right_arm_preset_deltas(self) -> dict[str, float]:
        if self.commanded_q is None or not self.preset_reference_q:
            return {}
        deltas: dict[str, float] = {}
        for joint in RIGHT_ARM_JOINTS:
            short_name = INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")
            deltas[short_name] = self.commanded_q[joint] - self.preset_reference_q[joint]
        return deltas

    def to_state_dict(self, presets: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "hold_active": self.hold_active,
            "active_preset": self.active_preset_name,
            "preset_deltas_rad": self.right_arm_preset_deltas(),
            "urdf_default_joints_rad": urdf_default_right_arm_joint_values(),
            "presets": {
                name: {
                    "label": str(info.get("label") or name),
                    "description": str(info.get("description") or ""),
                }
                for name, info in presets.items()
            },
        }


UPPER_BODY_COMMAND_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
ARM_WAYPOINT_JOINTS = RIGHT_ARM_JOINTS
G1_ARM_JOINT_INDEX = {
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
INDEX_TO_G1_ARM_JOINT = {index: name for name, index in G1_ARM_JOINT_INDEX.items()}
JOINT_ALIASES = {
    name.replace("_joint", ""): index
    for name, index in G1_ARM_JOINT_INDEX.items()
}


def resolve_arm_joint_name(name: str) -> int:
    key = name.strip()
    if key in G1_ARM_JOINT_INDEX:
        index = G1_ARM_JOINT_INDEX[key]
        if index not in RIGHT_ARM_JOINTS:
            raise ValueError(f"arm waypoint 只允许右臂 7 关节，不允许: {name}")
        return index
    if key in JOINT_ALIASES:
        index = JOINT_ALIASES[key]
        if index not in RIGHT_ARM_JOINTS:
            raise ValueError(f"arm waypoint 只允许右臂 7 关节，不允许: {name}")
        return index
    raise ValueError(f"未知 arm waypoint 关节名: {name}")


class ArmMotionAborted(Exception):
    """Arm ramp/hold interrupted by quit."""


class LoopControl:
    """Poll web commands; prioritize quit and allow aborting blocking arm motions."""

    def __init__(self, stream_server: Optional[DebugStreamServer]) -> None:
        self.stream_server = stream_server
        self.quit_requested = False
        self._pending_payload: Optional[dict[str, Any]] = None
        self._pending_command: Optional[str] = None

    def poll_web(self) -> bool:
        if self.stream_server is None:
            return self.quit_requested
        last_payload: Optional[dict[str, Any]] = None
        while True:
            payload = self.stream_server.pop_command()
            if payload is None:
                break
            cmd = DebugStreamServer.command_name(payload) or ""
            if cmd == "quit":
                self.quit_requested = True
                self._pending_payload = None
                self._pending_command = None
                print("[QUIT] web quit requested")
                return True
            last_payload = payload
        if last_payload is not None:
            self._pending_payload = last_payload
            self._pending_command = DebugStreamServer.command_name(last_payload)
        return self.quit_requested

    def take_web_command(self) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        payload = self._pending_payload
        command = self._pending_command
        self._pending_payload = None
        self._pending_command = None
        return payload, command

    def should_abort_motion(self) -> bool:
        return self.poll_web()


class G1ArmWaypointController:
    """Publish conservative arm waypoints through Unitree rt/arm_sdk."""

    def __init__(self, network_interface: str, domain_id: int, kp: float, kd: float) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: WPS433
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: WPS433
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: WPS433
        from unitree_sdk2py.utils.crc import CRC  # noqa: WPS433

        self.kp = float(kp)
        self.kd = float(kd)
        self._low_cmd_default = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()
        self._latest_lowstate = None

        if network_interface:
            ChannelFactoryInitialize(domain_id, network_interface)
        else:
            ChannelFactoryInitialize(domain_id)

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()

    def ensure_sdk_motion_mode(self, retries: int = 3) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: WPS433

        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        status, result = client.CheckMode()
        print("MotionSwitcher before arm_sdk command:", status, result)
        if not isinstance(result, dict) or not result.get("name"):
            print("Motion mode already released for SDK control.")
            return
        for attempt in range(1, retries + 1):
            code, data = client.ReleaseMode()
            print(f"ReleaseMode attempt {attempt}: code={code}, data={data}")
            time.sleep(1.0)
            status, result = client.CheckMode()
            print("MotionSwitcher after ReleaseMode:", status, result)
            if isinstance(result, dict) and not result.get("name"):
                return
        raise RuntimeError("MotionSwitcher mode is still active; arm_sdk command may be ignored.")

    def check_motion_mode(self) -> dict[str, Any]:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: WPS433

        client = MotionSwitcherClient()
        client.SetTimeout(2.0)
        client.Init()
        status, result = client.CheckMode()
        return {"status": status, "result": result}

    def _on_lowstate(self, msg: Any) -> None:
        self._latest_lowstate = msg

    def wait_for_lowstate(self, timeout: float) -> Any:
        deadline = time.time() + timeout
        while self._latest_lowstate is None and time.time() < deadline:
            time.sleep(0.02)
        if self._latest_lowstate is None:
            raise RuntimeError("No rt/lowstate received for arm waypoint controller.")
        return self._latest_lowstate

    def command_joint_positions(self) -> dict[int, float]:
        msg = self.wait_for_lowstate(2.0)
        return {joint: float(msg.motor_state[joint].q) for joint in UPPER_BODY_COMMAND_JOINTS}

    def named_joint_positions(self) -> dict[str, float]:
        q = self.command_joint_positions()
        return {
            INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", ""): q[joint]
            for joint in ARM_WAYPOINT_JOINTS
        }

    def write_arm_command(
        self,
        command_q: dict[int, float],
        weight: float,
        active_joints: Optional[list[int]] = None,
    ) -> None:
        cmd = self._low_cmd_default()
        lowstate = self._latest_lowstate
        if lowstate is not None:
            if hasattr(cmd, "mode_pr") and hasattr(lowstate, "mode_pr"):
                cmd.mode_pr = int(lowstate.mode_pr)
            if hasattr(cmd, "mode_machine") and hasattr(lowstate, "mode_machine"):
                cmd.mode_machine = int(lowstate.mode_machine)
        cmd.motor_cmd[ARM_SDK_WEIGHT].q = float(weight)
        active = set(UPPER_BODY_COMMAND_JOINTS if active_joints is None else active_joints)
        for joint in active:
            motor_cmd = cmd.motor_cmd[joint]
            if hasattr(motor_cmd, "mode"):
                motor_cmd.mode = 1
            motor_cmd.q = float(command_q[joint])
            motor_cmd.dq = 0.0
            motor_cmd.kp = self.kp
            motor_cmd.kd = self.kd
            motor_cmd.tau = 0.0
        cmd.crc = self._crc.Crc(cmd)
        self._publisher.Write(cmd)

    def ramp_to(
        self,
        target_q: dict[int, float],
        seconds: float,
        hz: float,
        active_joints: Optional[list[int]] = None,
        start_q: Optional[dict[int, float]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        if start_q is None:
            start_q = self.command_joint_positions()
        else:
            start_q = dict(start_q)
        joints = UPPER_BODY_COMMAND_JOINTS if active_joints is None else active_joints
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for step in range(steps):
            if should_abort and should_abort():
                raise ArmMotionAborted("ramp aborted")
            ratio = float(step + 1) / float(steps)
            command_q = dict(start_q)
            for joint in joints:
                command_q[joint] = start_q[joint] * (1.0 - ratio) + target_q[joint] * ratio
            self.write_arm_command(command_q, weight=1.0, active_joints=list(joints))
            time.sleep(dt)

    def hold(
        self,
        target_q: dict[int, float],
        seconds: float,
        hz: float,
        active_joints: Optional[list[int]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        joints = UPPER_BODY_COMMAND_JOINTS if active_joints is None else active_joints
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            if should_abort and should_abort():
                raise ArmMotionAborted("hold aborted")
            self.write_arm_command(target_q, weight=1.0, active_joints=list(joints))
            time.sleep(dt)

    def release(self, seconds: float, hz: float) -> None:
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            self.write_arm_command({}, weight=0.0, active_joints=[])
            time.sleep(dt)


def load_arm_waypoints(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("arm waypoint JSON 必须是 list")
    return payload


def save_arm_waypoints(path: Path, waypoints: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(waypoints, ensure_ascii=False, indent=2), encoding="utf-8")


def load_arm_presets(path: Path) -> dict[str, dict[str, Any]]:
    presets = default_arm_presets()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("arm preset JSON 必须是对象 {name: {...}}")
        presets.update(payload)
    if "urdf_default" not in presets:
        presets["urdf_default"] = default_arm_presets()["urdf_default"]
    return presets


def default_arm_presets() -> dict[str, dict[str, Any]]:
    return {
        "urdf_default": {
            "label": "URDF 默认",
            "description": "右臂 7 关节 URDF 零位 (0 rad)",
            "use_urdf_default": True,
        }
    }


def urdf_default_right_arm_joint_values() -> dict[str, float]:
    return {
        INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", ""): 0.0
        for joint in RIGHT_ARM_JOINTS
    }


def preset_uses_urdf_default(preset: dict[str, Any]) -> bool:
    if preset.get("use_urdf_default") is True:
        return True
    if str(preset.get("source", "")).strip().lower() in {"urdf_default", "urdf_zero", "urdf"}:
        return True
    joints = preset.get("joints")
    if joints is None:
        return True
    if joints == "urdf_default":
        return True
    if isinstance(joints, dict) and not joints:
        return True
    return False


def preset_target_q(preset: dict[str, Any], current_q: dict[int, float]) -> dict[int, float]:
    target_q = dict(current_q)
    if preset_uses_urdf_default(preset):
        for joint in RIGHT_ARM_JOINTS:
            target_q[joint] = 0.0
        return target_q
    joints = preset.get("joints") or {}
    if not isinstance(joints, dict):
        raise ValueError("preset.joints 必须是对象或省略以使用 URDF 默认零位")
    for name, value in joints.items():
        target_q[resolve_arm_joint_name(str(name))] = float(value)
    return target_q


def clamp_to_preset_envelope(
    target_q: dict[int, float],
    preset_reference_q: dict[int, float],
    max_delta_rad: float,
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    if not preset_reference_q or max_delta_rad <= 0:
        return target_q, {}
    clamped: dict[str, dict[str, float]] = {}
    for joint in RIGHT_ARM_JOINTS:
        ref = preset_reference_q.get(joint)
        if ref is None:
            continue
        lo = ref - max_delta_rad
        hi = ref + max_delta_rad
        requested = target_q[joint]
        bounded = max(lo, min(hi, requested))
        if abs(bounded - requested) > 1e-9:
            short_name = INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")
            clamped[short_name] = {
                "requested": requested,
                "clamped": bounded,
                "preset_ref": ref,
                "limit_rad": max_delta_rad,
            }
            target_q[joint] = bounded
    return target_q, clamped


def prepare_arm_target_q(
    target_q: dict[int, float],
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    limit_margin_rad: float,
    arm_session: ArmSessionState,
    preset_max_delta_rad: float,
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    target_q, clamped = clamp_right_arm_target_q(target_q, arm_joint_limits, limit_margin_rad)
    if arm_session.preset_reference_q:
        preset_clamped: dict[str, dict[str, float]] = {}
        target_q, preset_clamped = clamp_to_preset_envelope(
            target_q,
            arm_session.preset_reference_q,
            preset_max_delta_rad,
        )
        clamped.update(preset_clamped)
    return target_q, clamped


def apply_arm_preset(
    preset_name: str,
    presets: dict[str, dict[str, Any]],
    controller: G1ArmWaypointController,
    arm_session: ArmSessionState,
    args: argparse.Namespace,
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    loop_control: Optional[LoopControl] = None,
) -> str:
    if preset_name not in presets:
        available = ", ".join(sorted(presets)) or "<none>"
        raise ValueError(f"未知 preset: {preset_name}，可用: {available}")
    preset = presets[preset_name]
    current_q = controller.command_joint_positions()
    target_q = preset_target_q(preset, current_q)
    target_q, clamped = clamp_right_arm_target_q(target_q, arm_joint_limits, args.arm_limit_margin_rad)
    label = str(preset.get("label") or preset_name)
    print(f"[ARM] applying preset {preset_name} ({label})")
    if clamped:
        print(f"[ARM] preset clamped: {clamped}")
    start_q = arm_session.base_q(controller)
    should_abort = loop_control.should_abort_motion if loop_control is not None else None
    controller.ramp_to(
        target_q,
        args.arm_ramp_seconds,
        args.arm_control_hz,
        active_joints=UPPER_BODY_COMMAND_JOINTS,
        start_q=start_q,
        should_abort=should_abort,
    )
    controller.hold(
        target_q,
        args.arm_hold_seconds,
        args.arm_control_hz,
        active_joints=UPPER_BODY_COMMAND_JOINTS,
        should_abort=should_abort,
    )
    arm_session.set_commanded(target_q)
    arm_session.set_preset_reference(preset_name, target_q)
    return f"applied preset {preset_name} ({label})"


def load_urdf_joint_limits(urdf_path: str | Path) -> dict[str, tuple[Optional[float], Optional[float]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        if limit is None:
            limits[name] = (None, None)
            continue
        lower = float(limit.attrib["lower"]) if "lower" in limit.attrib else None
        upper = float(limit.attrib["upper"]) if "upper" in limit.attrib else None
        limits[name] = (lower, upper)
    return limits


def clamp_joint_value(
    joint_index: int,
    value: float,
    joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    margin_rad: float,
) -> tuple[float, bool]:
    joint_name = INDEX_TO_G1_ARM_JOINT[joint_index]
    lower, upper = joint_limits.get(joint_name, (None, None))
    clamped = float(value)
    if lower is not None:
        clamped = max(clamped, lower + margin_rad)
    if upper is not None:
        clamped = min(clamped, upper - margin_rad)
    return clamped, abs(clamped - float(value)) > 1e-9


def waypoint_target_q(waypoint: dict[str, Any], current_q: dict[int, float]) -> dict[int, float]:
    joints = waypoint.get("joints") or waypoint.get("joint_positions") or {}
    if not isinstance(joints, dict):
        raise ValueError("waypoint.joints 必须是对象")
    target_q = dict(current_q)
    for name, value in joints.items():
        target_q[resolve_arm_joint_name(str(name))] = float(value)
    return target_q


def release_motion_switcher_mode(retries: int = 3) -> dict[str, Any]:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: WPS433

    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    status, result = client.CheckMode()
    print("MotionSwitcher before release:", status, result)
    if not isinstance(result, dict) or not result.get("name"):
        return {"status": status, "result": result, "released": True}
    for attempt in range(1, retries + 1):
        code, data = client.ReleaseMode()
        print(f"ReleaseMode attempt {attempt}: code={code}, data={data}")
        time.sleep(1.0)
        status, result = client.CheckMode()
        print("MotionSwitcher after ReleaseMode:", status, result)
        if isinstance(result, dict) and not result.get("name"):
            return {"status": status, "result": result, "released": True}
    raise RuntimeError("MotionSwitcher mode is still active")


def run_test_move(
    controller: G1ArmWaypointController,
    args: argparse.Namespace,
    arm_session: ArmSessionState,
    loop_control: LoopControl,
) -> str:
    joint = resolve_arm_joint_name(args.test_joint)
    joint_name = INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")
    controller.ensure_sdk_motion_mode()
    start_q = arm_session.base_q(controller)
    target_q = dict(start_q)
    target_q[joint] = start_q[joint] + args.test_delta_rad
    print(f"[TEST] moving {joint_name}: {start_q[joint]:+.4f} -> {target_q[joint]:+.4f}")
    controller.ramp_to(
        target_q,
        args.arm_ramp_seconds,
        args.arm_control_hz,
        active_joints=ARM_WAYPOINT_JOINTS,
        start_q=start_q,
        should_abort=loop_control.should_abort_motion,
    )
    controller.hold(
        target_q,
        args.test_hold_seconds,
        args.arm_control_hz,
        active_joints=ARM_WAYPOINT_JOINTS,
        should_abort=loop_control.should_abort_motion,
    )
    current_q = controller.command_joint_positions()
    actual_delta = current_q[joint] - start_q[joint]
    print(f"[TEST] observed {joint_name} delta={actual_delta:+.4f} rad")
    message = f"test {joint_name}: observed delta={actual_delta:+.4f} rad"
    if abs(actual_delta) < args.motion_eps_rad:
        message += " (NO MOTION: arm_sdk command may be ignored)"
        print(f"[TEST][WARN] {message}")
    if args.test_return_to_start and not loop_control.quit_requested:
        print("[TEST] returning test joint to start posture")
        controller.ramp_to(
            start_q,
            args.arm_ramp_seconds,
            args.arm_control_hz,
            active_joints=ARM_WAYPOINT_JOINTS,
            start_q=current_q,
            should_abort=loop_control.should_abort_motion,
        )
        controller.hold(
            start_q,
            args.arm_hold_seconds,
            args.arm_control_hz,
            active_joints=ARM_WAYPOINT_JOINTS,
            should_abort=loop_control.should_abort_motion,
        )
        arm_session.set_commanded(start_q)
    else:
        arm_session.set_commanded(target_q)
    return message


def request_robot_default_pose() -> list[str]:
    """Ask high-level arm service to release arms only; never command lower body."""
    messages: list[str] = []
    try:
        from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient  # noqa: WPS433

        arm = G1ArmActionClient()
        arm.SetTimeout(10.0)
        arm.Init()
        code = arm.ExecuteAction(99)
        messages.append(f"release_arm(99) code={code}")
    except Exception as exc:
        messages.append(f"release_arm failed: {exc}")
    return messages


def random_right_arm_target_q(
    current_q: dict[int, float],
    shoulder_elbow_max_delta_rad: float,
    wrist_max_delta_rad: float,
    joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    limit_margin_rad: float,
    rng: random.Random,
) -> tuple[dict[int, float], dict[str, float], dict[str, dict[str, float]]]:
    target_q = dict(current_q)
    deltas: dict[str, float] = {}
    clamped_targets: dict[str, dict[str, float]] = {}
    for joint in RIGHT_SHOULDER_ELBOW_JOINTS:
        delta = rng.uniform(-shoulder_elbow_max_delta_rad, shoulder_elbow_max_delta_rad)
        requested = current_q[joint] + delta
        target, was_clamped = clamp_joint_value(joint, requested, joint_limits, limit_margin_rad)
        target_q[joint] = target
        short_name = INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")
        deltas[short_name] = target - current_q[joint]
        if was_clamped:
            clamped_targets[short_name] = {"requested": requested, "clamped": target}
    for joint in RIGHT_WRIST_JOINTS:
        delta = rng.uniform(-wrist_max_delta_rad, wrist_max_delta_rad)
        requested = current_q[joint] + delta
        target, was_clamped = clamp_joint_value(joint, requested, joint_limits, limit_margin_rad)
        target_q[joint] = target
        short_name = INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")
        deltas[short_name] = target - current_q[joint]
        if was_clamped:
            clamped_targets[short_name] = {"requested": requested, "clamped": target}
    return target_q, deltas, clamped_targets


def joint_random_max_delta_rad(joint_index: int, args: argparse.Namespace) -> float:
    if joint_index in RIGHT_WRIST_JOINTS:
        return float(args.arm_random_wrist_max_delta_rad)
    return float(args.arm_random_shoulder_elbow_max_delta_rad)


def clamp_right_arm_target_q(
    target_q: dict[int, float],
    joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    limit_margin_rad: float,
) -> tuple[dict[int, float], dict[str, dict[str, float]]]:
    clamped: dict[str, dict[str, float]] = {}
    for joint in RIGHT_ARM_JOINTS:
        target, was_clamped = clamp_joint_value(joint, target_q[joint], joint_limits, limit_margin_rad)
        if was_clamped:
            clamped[INDEX_TO_G1_ARM_JOINT[joint].replace("_joint", "")] = {
                "requested": target_q[joint],
                "clamped": target,
            }
            target_q[joint] = target
    return target_q, clamped


def single_joint_target_q(
    current_q: dict[int, float],
    joint_index: int,
    mode: str,
    value: Optional[float],
    max_delta_rad: float,
    joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    limit_margin_rad: float,
    rng: random.Random,
) -> tuple[dict[int, float], float, bool, dict[str, dict[str, float]]]:
    target_q = dict(current_q)
    short_name = INDEX_TO_G1_ARM_JOINT[joint_index].replace("_joint", "")
    if mode == "delta":
        if value is None:
            raise ValueError("delta 模式需要 delta_rad")
        requested = current_q[joint_index] + float(value)
    elif mode == "abs":
        if value is None:
            raise ValueError("abs 模式需要 value_rad")
        requested = float(value)
    elif mode == "random":
        span = float(max_delta_rad)
        if value is not None and float(value) > 0:
            span = float(value)
        requested = current_q[joint_index] + rng.uniform(-span, span)
    else:
        raise ValueError(f"未知单关节模式: {mode}")
    target, was_clamped = clamp_joint_value(joint_index, requested, joint_limits, limit_margin_rad)
    target_q[joint_index] = target
    delta = target - current_q[joint_index]
    clamped: dict[str, dict[str, float]] = {}
    if was_clamped:
        clamped[short_name] = {"requested": requested, "clamped": target}
    return target_q, delta, was_clamped, clamped


def execute_arm_motion(
    controller: G1ArmWaypointController,
    target_q: dict[int, float],
    args: argparse.Namespace,
    message_prefix: str,
    arm_session: ArmSessionState,
    start_q: Optional[dict[int, float]] = None,
    loop_control: Optional[LoopControl] = None,
) -> str:
    print(f"[ARM] {message_prefix}")
    if start_q is None:
        start_q = arm_session.base_q(controller)
    should_abort = loop_control.should_abort_motion if loop_control is not None else None
    controller.ramp_to(
        target_q,
        args.arm_ramp_seconds,
        args.arm_control_hz,
        active_joints=UPPER_BODY_COMMAND_JOINTS,
        start_q=start_q,
        should_abort=should_abort,
    )
    controller.hold(
        target_q,
        args.arm_hold_seconds,
        args.arm_control_hz,
        active_joints=UPPER_BODY_COMMAND_JOINTS,
        should_abort=should_abort,
    )
    arm_session.set_commanded(target_q)
    if args.arm_release_after_move:
        controller.release(args.arm_release_seconds, args.arm_control_hz)
        arm_session.clear_hold()
        return f"{message_prefix} and released arm_sdk"
    return f"holding after {message_prefix}"


def maintain_arm_session(
    controller: G1ArmWaypointController,
    arm_session: ArmSessionState,
    args: argparse.Namespace,
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    loop_control: Optional[LoopControl] = None,
) -> Optional[str]:
    if not arm_session.hold_active or arm_session.commanded_q is None:
        return None
    now = time.monotonic()
    republish_dt = 1.0 / max(args.arm_control_hz, 1.0)
    if now - arm_session.last_republish_ts >= republish_dt:
        controller.write_arm_command(
            arm_session.commanded_q,
            weight=1.0,
            active_joints=UPPER_BODY_COMMAND_JOINTS,
        )
        arm_session.last_republish_ts = now

    if not arm_session.preset_reference_q or args.arm_preset_max_delta_rad <= 0:
        return None
    if now - arm_session.last_safety_check_ts < args.arm_safety_check_period:
        return None
    arm_session.last_safety_check_ts = now

    if controller._latest_lowstate is None:
        return None
    actual_q = {
        joint: float(controller._latest_lowstate.motor_state[joint].q)
        for joint in RIGHT_ARM_JOINTS
    }
    recover_q = dict(arm_session.commanded_q)
    needs_recover = False
    for joint in RIGHT_ARM_JOINTS:
        ref = arm_session.preset_reference_q[joint]
        actual_delta = actual_q[joint] - ref
        if abs(actual_delta) <= args.arm_preset_max_delta_rad:
            continue
        bounded = ref + max(-args.arm_preset_max_delta_rad, min(args.arm_preset_max_delta_rad, actual_delta))
        if abs(bounded - recover_q[joint]) > 1e-6:
            recover_q[joint] = bounded
            needs_recover = True
    if not needs_recover:
        return None
    recover_q, _ = prepare_arm_target_q(
        recover_q,
        arm_joint_limits,
        args.arm_limit_margin_rad,
        arm_session,
        args.arm_preset_max_delta_rad,
    )
    print("[ARM][SAFETY] preset envelope recovery triggered")
    should_abort = loop_control.should_abort_motion if loop_control is not None else None
    controller.ramp_to(
        recover_q,
        max(0.5, args.arm_ramp_seconds * 0.5),
        args.arm_control_hz,
        active_joints=UPPER_BODY_COMMAND_JOINTS,
        start_q=arm_session.commanded_q,
        should_abort=should_abort,
    )
    arm_session.set_commanded(recover_q)
    return "safety: recovered to preset envelope"


def make_no_camera_placeholder(width: int, height: int, lines: list[str]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y = 40
    for line in lines:
        put_text_bgr_adaptive(frame, line, (24, y), 0.75)
        y += 34
    return frame


def build_camera_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(serial: str, camera_name: str, camera_mount: str, role: str) -> None:
        key = serial.strip()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "serial": key,
                "camera_name": camera_name,
                "camera_mount": camera_mount,
                "role": role,
            }
        )

    primary_serial = (args.cam_serial or DEFAULT_WRIST_CAM_SERIAL).strip()
    primary_name = args.camera_name.strip() or "right_hand_d435"
    primary_mount = args.camera_mount.strip() or "wrist"
    add(primary_serial, primary_name, primary_mount, "wrist")

    if args.cam_fallback:
        fallback_serial = (args.cam_serial_fallback or DEFAULT_HEAD_CAM_SERIAL).strip()
        fallback_name = args.camera_fallback_name.strip() or "head_d435"
        fallback_mount = args.camera_fallback_mount.strip() or "head"
        add(fallback_serial, fallback_name, fallback_mount, "head")

    return candidates


def open_camera(
    args: argparse.Namespace,
    *,
    serial: Optional[str] = None,
    camera_name: Optional[str] = None,
    camera_mount: Optional[str] = None,
) -> RealSenseD435i:
    use_serial = (serial if serial is not None else args.cam_serial or DEFAULT_WRIST_CAM_SERIAL).strip()
    use_name = (camera_name if camera_name is not None else args.camera_name).strip()
    use_mount = (camera_mount if camera_mount is not None else args.camera_mount).strip()
    RealSenseD435i.set_emitter(args.cam_index if args.enable_emitter else None, use_serial)
    cam = RealSenseD435i(
        index=args.cam_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=use_serial,
        camera_name=use_name,
        mount=use_mount,
        color_only=args.color_only,
    )
    cam.open()
    cam.start()
    return cam


def try_open_camera(args: argparse.Namespace) -> tuple[Optional[RealSenseD435i], dict, Optional[str]]:
    if args.arm_only:
        return None, {"source": "arm_only", "available": False}, None

    candidates = build_camera_candidates(args) if args.cam_fallback else [
        {
            "serial": (args.cam_serial or DEFAULT_WRIST_CAM_SERIAL).strip(),
            "camera_name": args.camera_name.strip() or "right_hand_d435",
            "camera_mount": args.camera_mount.strip() or "wrist",
            "role": "primary",
        }
    ]
    errors: list[str] = []
    for index, profile in enumerate(candidates):
        serial = profile["serial"]
        try:
            cam = open_camera(
                args,
                serial=serial,
                camera_name=profile["camera_name"],
                camera_mount=profile["camera_mount"],
            )
            info = cam.capture_metadata()
            info["available"] = True
            info["fallback_index"] = index
            info["camera_role"] = profile["role"]
            info["selected_serial"] = serial
            if index == 0:
                print(f"[CAMERA] opened primary ({profile['role']}) serial={serial} name={profile['camera_name']}")
            else:
                print(
                    f"[CAMERA] primary unavailable, opened fallback ({profile['role']}) "
                    f"serial={serial} name={profile['camera_name']}"
                )
            return cam, info, None
        except Exception as exc:
            msg = f"{profile['role']}:{serial} -> {exc}"
            errors.append(msg)
            print(f"[CAMERA][WARN] {msg}", file=sys.stderr)

    combined = "; ".join(errors) if errors else "no camera candidates"
    if args.cam_fallback or args.allow_no_camera:
        print(f"[CAMERA] all options failed, entering no-camera mode: {combined}")
        return (
            None,
            {
                "source": "unavailable",
                "available": False,
                "fallback_attempts": errors,
                "error": combined,
            },
            combined,
        )
    raise RuntimeError(f"相机打开失败: {combined}")


def resolve_intrinsics_without_camera(args: argparse.Namespace) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    loaded = load_camera_params(
        camera_matrix_npy=args.camera_matrix_npy,
        dist_coeffs_npy=args.dist_coeffs_npy,
        camera_json=args.camera_json,
    )
    if loaded is not None:
        camera_matrix, dist_coeffs, info = loaded
        info["source"] = info.get("source", "file") + "_no_camera"
        return camera_matrix, dist_coeffs, info
    return None, None, {"source": "none", "note": "no camera and no intrinsics file"}


def process_arm_web_command(
    web_payload: dict[str, Any],
    controller: G1ArmWaypointController,
    arm_waypoints: list[dict[str, Any]],
    arm_waypoint_index: int,
    arm_waypoint_file: Path,
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    arm_session: ArmSessionState,
    arm_presets: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    loop_control: Optional[LoopControl] = None,
) -> tuple[str, int, dict[str, float], dict[str, dict[str, float]]]:
    web_command = DebugStreamServer.command_name(web_payload) or ""
    arm_random_deltas: dict[str, float] = {}
    arm_random_clamped: dict[str, dict[str, float]] = {}

    try:
        return _process_arm_web_command_impl(
            web_payload,
            web_command,
            controller,
            arm_waypoints,
            arm_waypoint_index,
            arm_waypoint_file,
            arm_joint_limits,
            arm_session,
            arm_presets,
            args,
            loop_control,
            arm_random_deltas,
            arm_random_clamped,
        )
    except ArmMotionAborted:
        if loop_control is not None and loop_control.quit_requested:
            raise
        return "arm motion aborted", arm_waypoint_index, arm_random_deltas, arm_random_clamped


def _process_arm_web_command_impl(
    web_payload: dict[str, Any],
    web_command: str,
    controller: G1ArmWaypointController,
    arm_waypoints: list[dict[str, Any]],
    arm_waypoint_index: int,
    arm_waypoint_file: Path,
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]],
    arm_session: ArmSessionState,
    arm_presets: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    loop_control: Optional[LoopControl],
    arm_random_deltas: dict[str, float],
    arm_random_clamped: dict[str, dict[str, float]],
) -> tuple[str, int, dict[str, float], dict[str, dict[str, float]]]:
    if web_command == "arm_preset":
        preset_name = str(web_payload.get("preset", "")).strip()
        if not preset_name:
            raise ValueError("arm_preset 需要 preset 名称")
        msg = apply_arm_preset(
            preset_name, arm_presets, controller, arm_session, args, arm_joint_limits, loop_control
        )
        return msg, arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_prev":
        if arm_waypoints:
            arm_waypoint_index = (arm_waypoint_index - 1) % len(arm_waypoints)
        return f"selected waypoint {arm_waypoint_index + 1}/{len(arm_waypoints)}", arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_next":
        if arm_waypoints:
            arm_waypoint_index = (arm_waypoint_index + 1) % len(arm_waypoints)
        return f"selected waypoint {arm_waypoint_index + 1}/{len(arm_waypoints)}", arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_save_current":
        name = f"waypoint_{len(arm_waypoints) + 1:03d}"
        arm_waypoints.append({"name": name, "joints": controller.named_joint_positions()})
        arm_waypoint_index = len(arm_waypoints) - 1
        save_arm_waypoints(arm_waypoint_file, arm_waypoints)
        return f"saved current arm waypoint: {name}", arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_move":
        if not arm_waypoints:
            raise RuntimeError("no arm waypoints; click Save Current first")
        selected = arm_waypoints[arm_waypoint_index]
        start_q = arm_session.base_q(controller)
        target_q = waypoint_target_q(selected, start_q)
        target_q, arm_random_clamped = prepare_arm_target_q(
            target_q,
            arm_joint_limits,
            args.arm_limit_margin_rad,
            arm_session,
            args.arm_preset_max_delta_rad,
        )
        name = str(selected.get("name") or f"waypoint_{arm_waypoint_index + 1:03d}")
        msg = execute_arm_motion(
            controller, target_q, args, f"moving to {name}", arm_session, start_q=start_q, loop_control=loop_control
        )
        return msg, arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_random_right":
        current_q = arm_session.base_q(controller)
        target_q, arm_random_deltas, arm_random_clamped = random_right_arm_target_q(
            current_q,
            args.arm_random_shoulder_elbow_max_delta_rad,
            args.arm_random_wrist_max_delta_rad,
            arm_joint_limits,
            args.arm_limit_margin_rad,
            random.Random(),
        )
        target_q, preset_clamped = clamp_to_preset_envelope(
            target_q,
            arm_session.preset_reference_q,
            args.arm_preset_max_delta_rad,
        )
        arm_random_clamped.update(preset_clamped)
        prefix = (
            "moving random right arm "
            f"(shoulder/elbow={args.arm_random_shoulder_elbow_max_delta_rad:.3f} rad, "
            f"wrist={args.arm_random_wrist_max_delta_rad:.3f} rad)"
        )
        print(f"[ARM] {prefix}: {arm_random_deltas}")
        msg = execute_arm_motion(
            controller, target_q, args, prefix, arm_session, start_q=current_q, loop_control=loop_control
        )
        return msg, arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command in {"arm_joint_delta", "arm_joint_abs", "arm_joint_random"}:
        joint_name = str(web_payload.get("joint", "")).strip()
        joint_index = resolve_arm_joint_name(joint_name)
        short_name = INDEX_TO_G1_ARM_JOINT[joint_index].replace("_joint", "")
        current_q = arm_session.base_q(controller)
        mode = {"arm_joint_delta": "delta", "arm_joint_abs": "abs", "arm_joint_random": "random"}[web_command]
        value = None
        if mode == "delta":
            value = float(web_payload.get("delta_rad", args.arm_joint_default_delta_rad))
        elif mode == "abs":
            if "value_rad" not in web_payload:
                raise ValueError("arm_joint_abs 需要 value_rad")
            value = float(web_payload["value_rad"])
        elif mode == "random":
            if "delta_rad" in web_payload:
                value = float(web_payload["delta_rad"])
            elif "max_delta_rad" in web_payload:
                value = float(web_payload["max_delta_rad"])
        max_delta = joint_random_max_delta_rad(joint_index, args)
        target_q, delta, _, joint_clamped = single_joint_target_q(
            current_q,
            joint_index,
            mode,
            value,
            max_delta,
            arm_joint_limits,
            args.arm_limit_margin_rad,
            random.Random(),
        )
        preset_clamped: dict[str, dict[str, float]] = {}
        target_q, preset_clamped = clamp_to_preset_envelope(
            target_q,
            arm_session.preset_reference_q,
            args.arm_preset_max_delta_rad,
        )
        arm_random_clamped = {**joint_clamped, **preset_clamped}
        arm_random_deltas = {short_name: delta}
        prefix = f"moving {short_name} {mode} delta={delta:.4f} rad"
        print(f"[ARM] {prefix}")
        msg = execute_arm_motion(
            controller, target_q, args, prefix, arm_session, start_q=current_q, loop_control=loop_control
        )
        return msg, arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_hold_current":
        target_q = arm_session.base_q(controller)
        controller.hold(target_q, args.arm_hold_seconds, args.arm_control_hz)
        arm_session.set_commanded(target_q)
        return "holding current arm posture", arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "arm_release":
        controller.release(args.arm_release_seconds, args.arm_control_hz)
        arm_session.clear_hold()
        return "released arm_sdk", arm_waypoint_index, arm_random_deltas, arm_random_clamped
    if web_command == "robot_default_pose":
        controller.release(args.arm_release_seconds, args.arm_control_hz)
        arm_session.clear_hold()
        messages = request_robot_default_pose()
        return "default pose requested: " + "; ".join(messages), arm_waypoint_index, arm_random_deltas, arm_random_clamped
    raise RuntimeError(f"unsupported arm command: {web_command}")


class Dex1RightStateBridge:
    """Subscribe rt/dex1/right/state for the single Dex1-1 motor position."""

    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber  # noqa: WPS433
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_  # noqa: WPS433

        self._lock = threading.Lock()
        self._finger_q = 0.0
        self._available = False
        self._subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
        self._subscriber.Init(self._on_state, 10)

    def _on_state(self, msg: object) -> None:
        states = getattr(msg, "states", None) or []
        if not states:
            return
        with self._lock:
            self._finger_q = float(getattr(states[0], "q", 0.0))
            self._available = True

    def finger_joint_positions(self) -> tuple[dict[str, float], bool]:
        with self._lock:
            finger_q = self._finger_q
            available = self._available
        return (
            {
                RIGHT_DEX1_FINGER_JOINTS[0]: finger_q,
                RIGHT_DEX1_FINGER_JOINTS[1]: finger_q,
            },
            available,
        )


class FKStateProvider:
    """Compute live FK from Unitree lowstate for selected target links."""

    def __init__(
        self,
        network_interface: str,
        urdf: str,
        base_link: str,
        targets: list[str],
        hand_frame: str,
        domain_id: int,
        state_timeout: float,
        *,
        use_dex1_gripper: bool,
    ) -> None:
        robot_kinematics_dir = PROJECT_ROOT / "robot_kinematics"
        if str(robot_kinematics_dir) not in sys.path:
            sys.path.insert(0, str(robot_kinematics_dir))

        from unitree_sdk2_bridge import UnitreeG1LowStateBridge  # noqa: WPS433
        from fk_urdf import Pose, URDFFK, base_pose_matrix, pose_to_json  # noqa: WPS433

        self.hand_frame = hand_frame
        self.use_dex1_gripper = use_dex1_gripper
        self._public_targets, self._compute_targets = expand_fk_targets(targets, hand_frame)
        self.base_link = base_link
        self._state_timeout = state_timeout
        self._bridge = UnitreeG1LowStateBridge(
            network_interface=network_interface,
            domain_id=domain_id,
        )
        self._model = URDFFK(urdf)
        self._base_pose = base_pose_matrix(None, None, None)
        self._pose_to_json = pose_to_json
        self._Pose = Pose
        self._dex1_bridge: Optional[Dex1RightStateBridge] = None
        if use_dex1_gripper:
            try:
                self._dex1_bridge = Dex1RightStateBridge()
            except Exception as exc:
                print(f"[STREAM][FK][WARN] Dex1 state unavailable, finger joints default to 0: {exc}", file=sys.stderr)

    def snapshot(self) -> dict:
        self._bridge.wait_for_state(self._state_timeout)
        joint_values = self._bridge.latest_joint_positions()
        dex1_state_available = False
        dex1_finger_q = 0.0
        if self.use_dex1_gripper:
            if self._dex1_bridge is not None:
                dex1_joints, dex1_state_available = self._dex1_bridge.finger_joint_positions()
                joint_values.update(dex1_joints)
                dex1_finger_q = dex1_joints[RIGHT_DEX1_FINGER_JOINTS[0]]
            else:
                for joint_name in RIGHT_DEX1_FINGER_JOINTS:
                    joint_values.setdefault(joint_name, 0.0)
        poses = self._model.compute_link_poses(
            joint_values=joint_values,
            targets=self._compute_targets,
            base_link=self.base_link,
            base_pose=self._base_pose,
            clamp_to_limits=False,
        )
        if self.use_dex1_gripper and (
            RIGHT_DEX1_GRIPPER_TCP in self._public_targets
            or self.hand_frame == RIGHT_DEX1_GRIPPER_TCP
        ):
            tcp_matrix = compute_dex1_gripper_tcp_matrix(
                np.asarray(poses[RIGHT_DEX1_FINGER_LINK_1].matrix, dtype=np.float64),
                np.asarray(poses[RIGHT_DEX1_FINGER_LINK_2].matrix, dtype=np.float64),
                np.asarray(poses[RIGHT_DEX1_BASE_LINK].matrix, dtype=np.float64),
            )
            poses[RIGHT_DEX1_GRIPPER_TCP] = self._Pose(
                link=RIGHT_DEX1_GRIPPER_TCP,
                matrix=tcp_matrix.tolist(),
            )
        return {
            "source": "rt/lowstate",
            "base_link": self.base_link,
            "hand_frame": self.hand_frame,
            "targets": {
                link_name: self._pose_to_json(poses[link_name], "xyzw")
                for link_name in self._public_targets
                if link_name in poses
            },
            "right_arm_joints": {
                name: joint_values[name]
                for name in sorted(joint_values)
                if name in RIGHT_ARM_JOINT_NAMES
            },
            "dex1": {
                "enabled": self.use_dex1_gripper,
                "finger_q": dex1_finger_q,
                "state_available": dex1_state_available,
                "finger_joints": {
                    name: joint_values.get(name, 0.0)
                    for name in RIGHT_DEX1_FINGER_JOINTS
                },
            },
        }


def safe_path_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return token.strip("_") or "camera"


def camera_dir_name(args: argparse.Namespace) -> str:
    if args.camera_name:
        return safe_path_token(args.camera_name)
    if args.cam_serial:
        return f"serial_{safe_path_token(args.cam_serial)}"
    return f"camera_{args.cam_index}"


def robot_context(args: argparse.Namespace) -> dict:
    return {
        "robot_model": args.robot_model,
        "robot_host": args.robot_host,
        "robot_user": args.robot_user,
        "ros_distro": args.ros_distro,
        "ros_domain_id": args.ros_domain_id,
        "notes": args.notes,
    }


def warn_if_mount_mismatch(mode: str, mount: str) -> None:
    if not mount:
        return
    mount_key = mount.strip().lower().replace("-", "_")
    eye_in_hand_mounts = {"hand", "wrist", "flange", "palm", "tool", "end_effector", "gripper"}
    eye_to_hand_mounts = {"head", "fixed", "external", "tripod", "base", "world"}
    if mode == "eye_in_hand" and mount_key in eye_to_hand_mounts:
        print(f"[WARN] mode={mode} 但 camera_mount={mount} 看起来像固定/头部相机，请确认模式是否选反。")
    if mode == "eye_to_hand" and mount_key in eye_in_hand_mounts:
        print(f"[WARN] mode={mode} 但 camera_mount={mount} 看起来像末端相机，请确认模式是否选反。")


def parse_name_list(values: Optional[list[str]]) -> list[str]:
    names: list[str] = []
    for value in values or []:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def uses_dex1_gripper(hand_frame: str, urdf_path: str) -> bool:
    frame = hand_frame.strip()
    if frame in DEX1_GRIPPER_FRAMES or frame.startswith("right_dex1_"):
        return True
    return "dex1" in Path(urdf_path).name.lower()


def expand_fk_targets(targets: list[str], hand_frame: str) -> tuple[list[str], list[str]]:
    public: list[str] = []
    for name in [*targets, hand_frame]:
        if name and name not in public:
            public.append(name)
    compute = list(public)
    if RIGHT_DEX1_GRIPPER_TCP in public or hand_frame == RIGHT_DEX1_GRIPPER_TCP:
        for link_name in (
            RIGHT_DEX1_BASE_LINK,
            RIGHT_DEX1_FINGER_LINK_1,
            RIGHT_DEX1_FINGER_LINK_2,
        ):
            if link_name not in compute:
                compute.append(link_name)
    return public, compute


def compute_dex1_gripper_tcp_matrix(
    finger1_tf: np.ndarray,
    finger2_tf: np.ndarray,
    base_tf: np.ndarray,
) -> np.ndarray:
    """Dex1-1 TCP: midpoint of the two finger links, orientation from gripper base."""
    tcp = np.eye(4, dtype=np.float64)
    tcp[:3, :3] = base_tf[:3, :3]
    tcp[:3, 3] = 0.5 * (finger1_tf[:3, 3] + finger2_tf[:3, 3])
    return tcp


def build_fk_provider(args: argparse.Namespace) -> Optional[FKStateProvider]:
    if not args.stream_fk:
        return None
    hand_frame = normalize_hand_frame(args.hand_frame)
    targets = parse_name_list(args.fk_target)
    if not targets:
        targets = parse_name_list([DEFAULT_RIGHT_ARM_FK_TARGETS])
    dex1_mode = uses_dex1_gripper(hand_frame, args.fk_urdf)
    try:
        return FKStateProvider(
            network_interface=args.fk_network_interface,
            urdf=args.fk_urdf,
            base_link=args.fk_base_link,
            targets=targets,
            hand_frame=hand_frame,
            domain_id=args.fk_domain_id,
            state_timeout=args.fk_state_timeout,
            use_dex1_gripper=dex1_mode,
        )
    except Exception as exc:
        print(f"[STREAM][FK][WARN] disabled: {exc}", file=sys.stderr)
        return None


def normalize_hand_frame(hand_frame: str) -> str:
    frame = hand_frame.strip()
    if frame in {"", "hand", "gripper", "end_effector", "tool", "flange", "palm"}:
        return DEFAULT_HAND_FRAME
    return frame


def euler_xyz_from_rotation(rotation: np.ndarray, unit: str) -> list[float]:
    sy = float(rotation[0, 2])
    sy = max(-1.0, min(1.0, sy))
    y = float(np.arcsin(sy))
    cy = float(np.cos(y))
    if abs(cy) > 1e-8:
        x = float(np.arctan2(-rotation[1, 2], rotation[2, 2]))
        z = float(np.arctan2(-rotation[0, 1], rotation[0, 0]))
    else:
        x = float(np.arctan2(rotation[2, 1], rotation[1, 1]))
        z = 0.0
    values = [x, y, z]
    if unit == "deg":
        values = np.rad2deg(values).astype(float).tolist()
    return values


def pose_values_from_transform(transform: np.ndarray, translation_unit: str, rotation_unit: str) -> list[float]:
    translation = transform[:3, 3].astype(float)
    if translation_unit == "mm":
        translation = translation * 1000.0
    return translation.tolist() + euler_xyz_from_rotation(transform[:3, :3], rotation_unit)


def fk_hand_transform(
    last_fk_state: dict,
    hand_frame_name: str,
    translation_unit: str,
    rotation_unit: str,
) -> tuple[np.ndarray, str, list[float]]:
    if not last_fk_state:
        raise RuntimeError("FK 状态为空，请确认 --stream-fk 正常刷新")
    if "error" in last_fk_state:
        raise RuntimeError(f"FK 状态错误: {last_fk_state['error']}")
    targets = last_fk_state.get("targets") or {}
    target = targets.get(hand_frame_name)
    if target is None:
        available = ", ".join(sorted(targets)) or "<none>"
        raise RuntimeError(f"FK 中没有 {hand_frame_name}，当前 targets: {available}")
    transform = np.asarray(target.get("transform_matrix"), dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"FK transform_matrix 形状错误: {transform.shape}")
    pose_text = f"fk:{last_fk_state.get('base_link', 'base')}->{hand_frame_name}"
    pose_values = pose_values_from_transform(transform, translation_unit, rotation_unit)
    return transform, pose_text, pose_values


def prompt_pose_6d(hand_frame_name: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()
        try:
            while True:
                text = simpledialog.askstring(
                    f"输入 {hand_frame_name} 6D 位姿",
                    "请输入 x,y,z,rx,ry,rz\n可用英文逗号或空格分隔：",
                    parent=root,
                )
                if text is None:
                    return None
                try:
                    parse_pose_text(text)
                    return text
                except ValueError as exc:
                    messagebox.showerror("位姿格式错误", str(exc), parent=root)
        finally:
            root.destroy()
    except Exception:
        text = input(f"请输入 {hand_frame_name} 位姿 x,y,z,rx,ry,rz（空输入取消）: ").strip()
        return text or None


def connect_arm_sdk(args: argparse.Namespace) -> G1ArmWaypointController:
    controller = G1ArmWaypointController(
        network_interface=args.arm_network_interface,
        domain_id=args.arm_domain_id,
        kp=args.arm_kp,
        kd=args.arm_kd,
    )
    controller.wait_for_lowstate(args.fk_state_timeout)
    if args.arm_release_on_startup:
        controller.release(args.arm_release_seconds, args.arm_control_hz)
    print("[ARM] arm_sdk connected after web confirmation")
    return controller


def disconnect_arm_sdk(
    controller: Optional[G1ArmWaypointController],
    arm_session: ArmSessionState,
    args: argparse.Namespace,
) -> tuple[None, str]:
    if controller is None:
        return None, "arm_sdk not connected"
    controller.release(args.arm_release_seconds, args.arm_control_hz)
    arm_session.clear_hold()
    print("[ARM] arm_sdk disconnected; external controller can move arm again")
    return None, "arm_sdk disconnected"


def arm_waypoint_state(
    controller: Optional[G1ArmWaypointController],
    waypoints: list[dict[str, Any]],
    waypoint_index: int,
    waypoint_file: Optional[Path],
    last_message: str,
    *,
    arm_ui_enabled: bool,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "enabled": arm_ui_enabled,
        "arm_sdk_connected": controller is not None,
        "arm_sdk_detached": arm_ui_enabled and controller is None,
        "count": len(waypoints),
        "index": waypoint_index,
        "file": str(waypoint_file) if waypoint_file is not None else "",
        "last_message": last_message,
    }
    if waypoints:
        selected = waypoints[max(0, min(waypoint_index, len(waypoints) - 1))]
        state["selected_name"] = str(selected.get("name") or f"waypoint_{waypoint_index + 1:03d}")
    if controller is not None:
        try:
            state["current_joints_rad"] = controller.named_joint_positions()
        except Exception as exc:
            state["error"] = str(exc)
    return state


def resolve_camera_intrinsics(args: argparse.Namespace, cam: RealSenseD435i) -> tuple[np.ndarray, np.ndarray, dict]:
    loaded = load_camera_params(
        camera_matrix_npy=args.camera_matrix_npy,
        dist_coeffs_npy=args.dist_coeffs_npy,
        camera_json=args.camera_json,
    )
    if loaded is not None:
        return loaded
    camera_matrix, dist_coeffs, info = cam.color_intrinsics()
    info["source"] = "realsense_profile"
    return camera_matrix, dist_coeffs, info


def run_capture(args: argparse.Namespace) -> int:
    mode = normalize_mode(args.mode)
    args.hand_frame = normalize_hand_frame(args.hand_frame)
    if args.stream_fk and uses_dex1_gripper(args.hand_frame, args.fk_urdf):
        urdf_text = Path(args.fk_urdf).read_text(encoding="utf-8")
        if RIGHT_DEX1_BASE_LINK not in urdf_text:
            raise ValueError(
                f"hand_frame={args.hand_frame} 需要 Dex1 URDF，"
                f"请使用 {DEFAULT_FK_URDF.name} 或显式传入 --fk-urdf"
            )
    if args.stream_fk and args.stream_debug:
        print(f"[HAND] frame={args.hand_frame} urdf={Path(args.fk_urdf).name}")
    if args.stream_fk and args.euler_order != "xyz":
        raise ValueError("网页 FK 自动采集当前只记录 xyz 欧拉角，请使用默认 --euler-order xyz")
    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    session_name = args.session_name or f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = data_root / session_name / camera_dir_name(args)
    saved_count = len(list(session_dir.glob("*.json"))) if session_dir.exists() else 0
    last_msg = ""
    last_msg_ts = 0.0
    last_warning: dict[str, Any] = {}
    warning_seq = 0
    warn_if_mount_mismatch(mode, args.camera_mount)
    stream_server = None
    fk_provider = None
    arm_controller = None
    arm_waypoint_file = Path(args.arm_waypoints_json) if args.arm_waypoints_json else session_dir / "arm_waypoints.json"
    arm_waypoints: list[dict[str, Any]] = []
    arm_waypoint_index = 0
    arm_last_msg = ""
    arm_random_deltas: dict[str, float] = {}
    arm_random_clamped: dict[str, dict[str, float]] = {}
    arm_joint_limits: dict[str, tuple[Optional[float], Optional[float]]] = {}
    arm_session = ArmSessionState()
    arm_presets: dict[str, dict[str, Any]] = {}
    last_fk_state: dict = {}
    last_fk_ts = 0.0
    loop_control = LoopControl(None)
    if args.stream_debug:
        stream_server = DebugStreamServer(
            host=args.stream_host,
            port=args.stream_port,
            jpeg_quality=args.stream_jpeg_quality,
        )
        stream_server.start()
        print(f"[STREAM] http://{args.stream_host}:{args.stream_port}")
        fk_provider = build_fk_provider(args)
        loop_control = LoopControl(stream_server)
        if args.enable_arm_waypoints:
            arm_joint_limits = load_urdf_joint_limits(args.fk_urdf)
            arm_waypoints = load_arm_waypoints(arm_waypoint_file)
            arm_presets = load_arm_presets(Path(args.arm_presets_json))
            arm_last_msg = "arm_sdk detached; use external controller, then Save. Web takeover needs confirm."
            if args.arm_sdk_on_startup:
                arm_controller = connect_arm_sdk(args)
                arm_last_msg = "arm_sdk connected on startup"
                if args.arm_preset_on_startup and args.arm_preset:
                    try:
                        arm_last_msg = apply_arm_preset(
                            args.arm_preset,
                            arm_presets,
                            arm_controller,
                            arm_session,
                            args,
                            arm_joint_limits,
                            loop_control,
                        )
                        print(f"[ARM] startup: {arm_last_msg}")
                    except ArmMotionAborted:
                        if loop_control.quit_requested:
                            print("[QUIT] during startup preset")
                            return 0
                    except Exception as exc:
                        arm_last_msg = f"startup preset failed: {exc}"
                        print(f"[ARM][WARN] {arm_last_msg}", file=sys.stderr)

    cam, camera_info, camera_error = try_open_camera(args)
    camera_available = cam is not None
    arm_debug_mode = args.arm_only or not camera_available
    if arm_debug_mode:
        if camera_error:
            print(f"[ARM-DEBUG] camera unavailable, arm-only debug mode: {camera_error}")
        else:
            print("[ARM-DEBUG] arm-only mode: skipping camera, web arm control only")
        if not args.enable_arm_waypoints:
            raise RuntimeError("无相机调试模式需要 --enable-arm-waypoints")
        if stream_server is None:
            raise RuntimeError("无相机调试模式需要 --stream-debug")

    try:
        if camera_available:
            camera_matrix, dist_coeffs, camera_intrinsics = resolve_camera_intrinsics(args, cam)
        else:
            camera_matrix, dist_coeffs, camera_intrinsics = resolve_intrinsics_without_camera(args)
            camera_info = dict(camera_info)
            camera_info["intrinsics"] = camera_intrinsics
        win = f"D435i HandEye {mode} cam{args.cam_index} (SPACE=capture ENTER/S=solve ESC/q=quit)"
        if camera_available and not args.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, args.width, args.height)
        print(f"[SESSION] {session_dir.resolve()}")
        print(f"[MODE] {mode}")
        print(f"[CAMERA] {camera_info}")
        print(f"[ROBOT] {robot_context(args)}")
        print(f"[INTRINSICS] {camera_intrinsics.get('source')}")
        if arm_debug_mode:
            print("[ARM-DEBUG] capture/solve disabled until camera is available")
        else:
            print("[KEYS] SPACE=capture, ENTER/S=solve, ESC/q=quit")

        last_vis = None
        last_corners = None
        last_detected = False
        last_detect_method = "none"
        fetch_timeout_ms = min(200, args.timeout_ms) if stream_server is not None else args.timeout_ms

        while True:
            if loop_control.poll_web():
                break
            web_payload, web_command = loop_control.take_web_command()

            if arm_controller is not None:
                try:
                    safety_msg = maintain_arm_session(
                        arm_controller, arm_session, args, arm_joint_limits, loop_control
                    )
                except ArmMotionAborted:
                    if loop_control.quit_requested:
                        break
                    safety_msg = None
                if safety_msg:
                    arm_last_msg = safety_msg
                    last_msg = safety_msg
                    last_msg_ts = time.monotonic()

            detected = False
            detect_method = "none"
            corners = None
            frame_bgr = None
            if camera_available:
                frame = cam.fetch(timeout_ms=fetch_timeout_ms)
                if frame is None or frame.get("rgb") is None:
                    if loop_control.poll_web():
                        break
                    if last_vis is not None:
                        vis = last_vis.copy()
                        detected = last_detected
                        detect_method = last_detect_method
                        corners = last_corners
                    else:
                        time.sleep(0.02)
                        continue
                else:
                    frame_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
                    preview = gamma_correct_bgr(frame_bgr, args.gamma)
                    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                    corners, detect_method = find_chessboard_corners(gray, pattern_size, args.gamma)
                    detected = corners is not None
                    vis = preview.copy()
                    if detected:
                        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
                    last_vis = vis.copy()
                    last_detected = detected
                    last_detect_method = detect_method
                    last_corners = corners
            else:
                vis = make_no_camera_placeholder(
                    args.width,
                    args.height,
                    [
                        "ARM DEBUG MODE",
                        "wrist + head camera unavailable",
                        "Use Right Arm Joints panel",
                        "Save/Solve disabled",
                    ],
                )

            put_text_bgr_adaptive(
                vis,
                f"mode={mode} cam={'on' if camera_available else 'off'} chessboard={int(detected)} saved={saved_count}",
                (10, 30),
                0.72,
            )
            put_text_bgr_adaptive(
                vis,
                f"board={args.cols}x{args.rows} square={args.square_mm:g}mm hand_frame={args.hand_frame}",
                (10, 60),
                0.6,
            )
            if camera_available:
                put_text_bgr_adaptive(vis, "SPACE=capture  ENTER/S=solve  ESC/q=quit", (10, 90), 0.6)
            else:
                put_text_bgr_adaptive(vis, "ARM DEBUG: joint buttons only  ESC/q=quit", (10, 90), 0.6)
            if last_msg and time.monotonic() - last_msg_ts < 3.0:
                put_text_bgr_adaptive(vis, last_msg, (10, 120), 0.6)
            if camera_available and not args.headless:
                cv2.imshow(win, vis)
            if stream_server is not None:
                now = time.monotonic()
                if fk_provider is not None and now - last_fk_ts >= args.fk_update_period:
                    try:
                        last_fk_state = fk_provider.snapshot()
                    except Exception as exc:
                        last_fk_state = {"error": str(exc)}
                    last_fk_ts = now
                stream_server.update_frame(vis)
                stream_server.update_state(
                    {
                        "mode": mode,
                        "arm_debug_mode": arm_debug_mode,
                        "camera_available": camera_available,
                        "camera": camera_info,
                        "chessboard_detected": detected,
                        "saved_count": saved_count,
                        "session_dir": str(session_dir.resolve()),
                        "last_message": last_msg,
                        "warning": last_warning,
                        "fk": last_fk_state,
                        "arm_waypoints": {
                            **arm_waypoint_state(
                                arm_controller,
                                arm_waypoints,
                                arm_waypoint_index,
                                arm_waypoint_file,
                                arm_last_msg,
                                arm_ui_enabled=args.enable_arm_waypoints,
                            ),
                            "last_random_right_arm_delta_rad": arm_random_deltas,
                            "last_random_right_arm_clamped": arm_random_clamped,
                            "joint_limit_margin_rad": args.arm_limit_margin_rad,
                            "joint_default_delta_rad": args.arm_joint_default_delta_rad,
                            "preset_max_delta_rad": args.arm_preset_max_delta_rad,
                            **arm_session.to_state_dict(arm_presets),
                        },
                    }
                )

            key = (cv2.waitKey(1) & 0xFF) if (camera_available and not args.headless) else 255
            if key in (27, ord("q"), ord("Q")):
                print("[QUIT] keyboard quit")
                break
            if web_command == "arm_sdk_enable":
                if not args.enable_arm_waypoints:
                    arm_last_msg = "arm_sdk disabled; start with --enable-arm-waypoints"
                    last_msg = arm_last_msg
                    last_msg_ts = time.monotonic()
                    continue
                if not bool((web_payload or {}).get("confirm")):
                    arm_last_msg = "arm_sdk_enable rejected: confirm required"
                    last_msg = arm_last_msg
                    last_msg_ts = time.monotonic()
                    continue
                if arm_controller is None:
                    try:
                        arm_controller = connect_arm_sdk(args)
                        arm_last_msg = "arm_sdk connected; web arm controls enabled"
                        print(f"[ARM] {arm_last_msg}")
                    except Exception as exc:
                        arm_last_msg = f"arm_sdk connect failed: {exc}"
                        print(f"[ARM][ERROR] {exc}", file=sys.stderr)
                else:
                    arm_last_msg = "arm_sdk already connected"
                last_msg = arm_last_msg
                last_msg_ts = time.monotonic()
                continue
            if web_command == "arm_sdk_disable":
                arm_controller, arm_last_msg = disconnect_arm_sdk(arm_controller, arm_session, args)
                last_msg = arm_last_msg
                last_msg_ts = time.monotonic()
                continue
            if web_command == "switch_sdk":
                try:
                    mode = release_motion_switcher_mode()
                    last_msg = f"switch sdk done: {mode}"
                    print(f"[SDK] {last_msg}")
                except Exception as exc:
                    last_msg = f"switch sdk failed: {exc}"
                    print(f"[SDK][ERROR] {exc}", file=sys.stderr)
                last_msg_ts = time.monotonic()
                continue
            if web_command == "test_move":
                if not args.enable_arm_waypoints:
                    last_msg = "test_move disabled; start with --enable-arm-waypoints"
                    last_msg_ts = time.monotonic()
                    continue
                if arm_controller is None:
                    last_msg = "test_move needs arm_sdk; use 接管 Arm SDK with double confirm first"
                    last_msg_ts = time.monotonic()
                    continue
                try:
                    last_msg = run_test_move(arm_controller, args, arm_session, loop_control)
                except ArmMotionAborted:
                    if loop_control.quit_requested:
                        break
                    last_msg = "test_move aborted"
                except Exception as exc:
                    last_msg = f"test_move failed: {exc}"
                    print(f"[TEST][ERROR] {exc}", file=sys.stderr)
                last_msg_ts = time.monotonic()
                continue
            if web_command == "start_calib":
                last_msg = "manual calib mode: external controller + Save/Solve (no auto motion)"
                last_msg_ts = time.monotonic()
                continue
            if web_command == "stop":
                last_msg = "stop acknowledged; motion abort if running"
                last_msg_ts = time.monotonic()
                continue
            if web_command and (web_command.startswith("arm_") or web_command == "robot_default_pose"):
                if not args.enable_arm_waypoints:
                    arm_last_msg = "arm waypoints disabled; start with --enable-arm-waypoints"
                    last_msg = arm_last_msg
                    last_msg_ts = time.monotonic()
                    continue
                if arm_controller is None:
                    arm_last_msg = "arm_sdk not connected; use 接管 Arm SDK with double confirm first"
                    last_msg = arm_last_msg
                    last_msg_ts = time.monotonic()
                    continue
                try:
                    arm_last_msg, arm_waypoint_index, arm_random_deltas, arm_random_clamped = process_arm_web_command(
                        web_payload or {"command": web_command},
                        arm_controller,
                        arm_waypoints,
                        arm_waypoint_index,
                        arm_waypoint_file,
                        arm_joint_limits,
                        arm_session,
                        arm_presets,
                        args,
                        loop_control,
                    )
                    last_msg = arm_last_msg
                except ArmMotionAborted:
                    if loop_control.quit_requested:
                        break
                    last_msg = "arm motion aborted"
                except Exception as exc:
                    arm_last_msg = f"arm command failed: {exc}"
                    last_msg = arm_last_msg
                    print(f"[ARM][ERROR] {exc}", file=sys.stderr)
                last_msg_ts = time.monotonic()
                continue
            if key in (13, 10, ord("s"), ord("S")) or web_command == "solve":
                if arm_debug_mode:
                    last_msg = "solve rejected: arm debug mode (no camera capture)"
                    last_msg_ts = time.monotonic()
                    continue
                try:
                    records = load_capture_records(session_dir)
                    path = solve_handeye(
                        records,
                        mode=mode,
                        output_dir=output_dir,
                        min_samples=args.min_samples,
                        method=opencv_method_from_name(args.handeye_method),
                    )
                    last_msg = f"solve saved: {path.name}"
                    print(f"[SOLVE] saved: {path.resolve()}")
                except Exception as exc:
                    last_msg = f"solve failed: {exc}"
                    print(f"[SOLVE][ERROR] {exc}", file=sys.stderr)
                last_msg_ts = time.monotonic()
                continue
            capture_requested = key == ord(" ") or web_command == "save"
            if not capture_requested:
                if not camera_available:
                    time.sleep(0.02)
                continue

            if arm_debug_mode:
                last_msg = "capture rejected: arm debug mode (no camera)"
                last_msg_ts = time.monotonic()
                print("[CAPTURE] 拒绝：当前为无相机调试模式")
                continue

            if corners is None:
                last_msg = "capture rejected: no chessboard"
                last_msg_ts = time.monotonic()
                print("[CAPTURE] 拒绝：当前画面未检测到棋盘格")
                continue

            try:
                if web_command == "save" and fk_provider is not None:
                    T_hand2base, pose_text, pose_values = fk_hand_transform(
                        last_fk_state,
                        args.hand_frame,
                        args.pose_translation_unit,
                        args.pose_rotation_unit,
                    )
                else:
                    pose_text = prompt_pose_6d(args.hand_frame)
                    if pose_text is None:
                        last_msg = "capture cancelled"
                        last_msg_ts = time.monotonic()
                        print("[CAPTURE] cancelled")
                        continue
                    pose_values = parse_pose_text(pose_text)
                    T_hand2base = pose_to_transform(
                        pose_values,
                        translation_unit=args.pose_translation_unit,
                        rotation_unit=args.pose_rotation_unit,
                        euler_order=args.euler_order,
                    )
                target_rvec, target_tvec, target_rms = solve_target_pose(objp, corners, camera_matrix, dist_coeffs)
                if target_rms > args.capture_max_reproj_rms_px:
                    warning_seq += 1
                    last_msg = (
                        f"capture rejected: pnp_rms={target_rms:.3f}px "
                        f"> {args.capture_max_reproj_rms_px:.3f}px"
                    )
                    last_warning = {
                        "id": warning_seq,
                        "kind": "capture_rejected_high_reprojection",
                        "message": last_msg,
                        "target_reprojection_rms_px": float(target_rms),
                        "max_reprojection_rms_px": float(args.capture_max_reproj_rms_px),
                    }
                    last_msg_ts = time.monotonic()
                    print(f"[CAPTURE][WARN] {last_msg}")
                    continue
                image_path = save_capture_record(
                    out_dir=session_dir,
                    image_bgr=frame_bgr,
                    mode=mode,
                    camera_index=args.cam_index,
                    camera_info=camera_info,
                    camera_intrinsics=camera_intrinsics,
                    robot_context=robot_context(args),
                    pattern_size=pattern_size,
                    square_mm=args.square_mm,
                    detection_method=detect_method,
                    corners=corners,
                    target_rvec=target_rvec,
                    target_tvec=target_tvec,
                    target_reproj_rms=target_rms,
                    pose_text=pose_text,
                    pose_values=pose_values,
                    pose_units={
                        "translation": args.pose_translation_unit,
                        "rotation": args.pose_rotation_unit,
                        "euler_order": args.euler_order,
                    },
                    T_hand2base=T_hand2base,
                    hand_frame_name=args.hand_frame,
                )
            except Exception as exc:
                last_msg = f"capture failed: {exc}"
                last_msg_ts = time.monotonic()
                print(f"[CAPTURE][ERROR] {exc}", file=sys.stderr)
                continue

            saved_count += 1
            last_msg = f"saved: {image_path.name}, pnp_rms={target_rms:.3f}px"
            last_msg_ts = time.monotonic()
            print(f"[CAPTURE] saved: {image_path.resolve()} pnp_rms={target_rms:.4f}px")
    finally:
        if cam is not None:
            cam.close()
        RealSenseD435i.set_emitter(None)
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense D435i 灵巧手手眼标定采集")
    parser.add_argument("--mode", required=True, help="eye-in-hand/hand-in-eye 或 eye-to-hand/hand-to-eye")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--cam-serial", type=str, default="", help=f"主相机序列号，默认腕部 D435 ({DEFAULT_WRIST_CAM_SERIAL})")
    parser.add_argument(
        "--cam-serial-fallback",
        type=str,
        default="",
        help=f"主相机失败后尝试的序列号，默认头部 D435I ({DEFAULT_HEAD_CAM_SERIAL})",
    )
    parser.add_argument("--cam-fallback", dest="cam_fallback", action="store_true", default=True, help="腕部->头部->无相机 顺序尝试（默认开启）")
    parser.add_argument("--no-cam-fallback", dest="cam_fallback", action="store_false", help="只尝试主相机，不自动切换头部/无相机")
    parser.add_argument("--camera-fallback-name", type=str, default="", help="fallback 头部相机名称，默认 head_d435")
    parser.add_argument("--camera-fallback-mount", type=str, default="", help="fallback 头部相机 mount，默认 head")
    parser.add_argument("--camera-name", type=str, default="", help="主相机名称，默认 right_hand_d435")
    parser.add_argument("--camera-mount", type=str, default="", help="主相机安装位，默认 wrist")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--cols", type=int, default=11, help="棋盘横向内角点数")
    parser.add_argument("--rows", type=int, default=8, help="棋盘纵向内角点数")
    parser.add_argument("--square-mm", type=float, default=30.0, help="棋盘方格边长 mm")
    parser.add_argument("--gamma", type=float, default=0.85, help="预览/检测 gamma")
    parser.add_argument("--camera-matrix-npy", type=str, default="", help="可选：相机内参 camera_matrix.npy")
    parser.add_argument("--dist-coeffs-npy", type=str, default="", help="可选：相机畸变 dist_coeffs.npy")
    parser.add_argument("--camera-json", type=str, default="", help="可选：包含 color.fx/fy/ppx/ppy/coeffs 的内参 JSON")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--session-name", type=str, default="")
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--capture-max-reproj-rms-px", type=float, default=0.5, help="Save 时允许的单帧棋盘重投影 RMS 上限，超过则拒绝保存")
    parser.add_argument("--pose-translation-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--pose-rotation-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--euler-order", choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"), default="xyz")
    parser.add_argument("--hand-frame", type=str, default=DEFAULT_HAND_FRAME, help="Save/FK 位姿所属 link；默认 right_dex1_gripper_tcp（Dex1-1 两指中点）")
    parser.add_argument("--handeye-method", choices=("tsai", "park", "horaud", "andreff", "daniilidis"), default="tsai")
    parser.add_argument("--enable-emitter", action="store_true", help="开启当前 D435i 深度发射器；默认关闭以减少棋盘反光")
    parser.add_argument("--color-only", action="store_true", help="只打开彩色流，不打开深度流；适合棋盘格/内参采集")
    parser.add_argument("--robot-model", type=str, default="Unitree H2")
    parser.add_argument("--robot-host", type=str, default="", help="可选：Ubuntu ROS2 机器人主机 IP/hostname，用于记录 SSH 调试环境")
    parser.add_argument("--robot-user", type=str, default="", help="可选：SSH 用户名")
    parser.add_argument("--ros-distro", type=str, default="", help="可选：humble/iron/jazzy 等")
    parser.add_argument("--ros-domain-id", type=str, default="", help="可选：ROS_DOMAIN_ID")
    parser.add_argument("--notes", type=str, default="", help="可选：采集备注")
    parser.add_argument("--stream-debug", action="store_true", help="开启网页调试流，显示 OpenCV 标定画面和状态面板")
    parser.add_argument("--headless", action="store_true", help="不创建 OpenCV 本地窗口，只通过网页调试流操作")
    parser.add_argument("--stream-host", type=str, default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--stream-jpeg-quality", type=int, default=80)
    parser.add_argument("--stream-fk", action="store_true", help="在网页状态面板中显示实时 FK 位姿")
    parser.add_argument("--fk-network-interface", type=str, default="eth0", help="Unitree DDS 网卡，如 eth0")
    parser.add_argument("--fk-domain-id", type=int, default=0)
    parser.add_argument("--fk-state-timeout", type=float, default=1.0)
    parser.add_argument("--fk-update-period", type=float, default=0.5, help="FK 面板刷新间隔，单位秒")
    parser.add_argument(
        "--fk-urdf",
        type=str,
        default=str(DEFAULT_FK_URDF),
        help="FK URDF；默认 g1_29dof_mode_15_with_dex1_1.urdf（含 Dex1-1）",
    )
    parser.add_argument("--fk-base-link", type=str, default="pelvis")
    parser.add_argument(
        "--fk-target",
        action="append",
        help="要显示 FK 的 link，可重复传入或逗号分隔；默认右臂相关 link",
    )
    parser.add_argument("--enable-arm-waypoints", action="store_true", help="网页显示控臂面板；默认不接管 arm_sdk，需二次确认后才连接")
    parser.add_argument("--arm-sdk-on-startup", dest="arm_sdk_on_startup", action="store_true", default=False, help="启动时立即连接 arm_sdk（旧行为）")
    parser.add_argument("--arm-waypoints-json", type=str, default="", help="arm waypoint JSON；默认保存到当前 session 下")
    parser.add_argument("--arm-network-interface", type=str, default="", help="Unitree DDS 网卡；默认使用 DDS 默认网卡")
    parser.add_argument("--arm-domain-id", type=int, default=0)
    parser.add_argument("--arm-control-hz", type=float, default=50.0)
    parser.add_argument("--arm-ramp-seconds", type=float, default=1.5)
    parser.add_argument("--arm-hold-seconds", type=float, default=1.0)
    parser.add_argument("--arm-release-seconds", type=float, default=0.5)
    parser.add_argument("--arm-release-on-startup", dest="arm_release_on_startup", action="store_true", default=False, help="连接 arm_sdk 后先 release 一次")
    parser.add_argument("--no-arm-release-on-startup", dest="arm_release_on_startup", action="store_false", help="连接 arm_sdk 后不自动 release")
    parser.add_argument("--arm-release-after-move", dest="arm_release_after_move", action="store_true", default=False, help="Move/Random 到位短暂稳定后自动释放 arm_sdk")
    parser.add_argument("--no-arm-release-after-move", dest="arm_release_after_move", action="store_false", help="Move/Random 到位后保持 arm_sdk（默认）")
    parser.add_argument("--arm-presets-json", type=str, default=str(DEFAULT_ARM_PRESETS_JSON), help="右臂 preset 姿态 JSON")
    parser.add_argument("--arm-preset", type=str, default="urdf_default", help="启动时应用的 preset 名称（默认 urdf_default = URDF 零位）")
    parser.add_argument("--arm-preset-on-startup", dest="arm_preset_on_startup", action="store_true", default=False, help="连接 arm_sdk 后自动运动到 --arm-preset")
    parser.add_argument("--no-arm-preset-on-startup", dest="arm_preset_on_startup", action="store_false", help="连接 arm_sdk 后不自动应用 preset")
    parser.add_argument("--arm-preset-max-delta-rad", type=float, default=0.35, help="相对 preset 的每关节最大允许偏移；超出会被 clamp 或自动拉回")
    parser.add_argument("--arm-safety-check-period", type=float, default=0.5, help="preset 安全边界检查周期（秒）")
    parser.add_argument("--arm-random-max-delta-rad", type=float, default=0.08, help="兼容旧参数：未设置分组幅度时作为肩肘随机幅度")
    parser.add_argument("--arm-random-shoulder-elbow-max-delta-rad", type=float, default=None, help="Random Right Arm 肩/肘关节最大随机扰动")
    parser.add_argument("--arm-random-wrist-max-delta-rad", type=float, default=0.18, help="Random Right Arm 手腕关节最大随机扰动")
    parser.add_argument("--arm-limit-margin-rad", type=float, default=0.03, help="按 URDF joint limit clamp 时预留的安全边界")
    parser.add_argument("--arm-kp", type=float, default=60.0)
    parser.add_argument("--arm-kd", type=float, default=1.5)
    parser.add_argument("--arm-joint-default-delta-rad", type=float, default=0.02, help="网页单关节 Δ 默认值")
    parser.add_argument("--test-joint", type=str, default="right_shoulder_pitch", help="网页 Test Move 使用的关节名")
    parser.add_argument("--test-delta-rad", type=float, default=0.20, help="网页 Test Move 的关节偏移(rad)")
    parser.add_argument("--test-hold-seconds", type=float, default=1.0, help="Test Move 到位后保持时间")
    parser.add_argument("--test-return-to-start", dest="test_return_to_start", action="store_true", default=True, help="Test Move 后回到起始关节角")
    parser.add_argument("--no-test-return-to-start", dest="test_return_to_start", action="store_false", help="Test Move 后保持测试姿态")
    parser.add_argument("--motion-eps-rad", type=float, default=0.02, help="Test Move 低于该关节变化量则认为未实际运动")
    parser.add_argument("--allow-no-camera", action="store_true", help="与 --no-cam-fallback 联用：主相机失败时进入无相机模式（--cam-fallback 已含此行为）")
    parser.add_argument("--arm-only", action="store_true", help="跳过相机，仅开启网页控臂/FK 调试")
    args = parser.parse_args()
    if args.cols < 2 or args.rows < 2:
        parser.error("--cols/--rows 必须 >= 2")
    if args.square_mm <= 0:
        parser.error("--square-mm 必须 > 0")
    if args.gamma <= 0:
        parser.error("--gamma 必须 > 0")
    if args.min_samples < 3:
        parser.error("--min-samples 必须 >= 3")
    if args.capture_max_reproj_rms_px <= 0:
        parser.error("--capture-max-reproj-rms-px 必须 > 0")
    if args.stream_port <= 0:
        parser.error("--stream-port 必须 > 0")
    if args.fk_update_period <= 0:
        parser.error("--fk-update-period 必须 > 0")
    if args.stream_fk and not args.stream_debug:
        parser.error("--stream-fk 需要同时指定 --stream-debug")
    if args.enable_arm_waypoints and not args.stream_debug:
        parser.error("--enable-arm-waypoints 需要同时指定 --stream-debug")
    if args.arm_control_hz <= 0:
        parser.error("--arm-control-hz 必须 > 0")
    if args.arm_ramp_seconds <= 0:
        parser.error("--arm-ramp-seconds 必须 > 0")
    if args.arm_hold_seconds <= 0:
        parser.error("--arm-hold-seconds 必须 > 0")
    if args.arm_release_seconds <= 0:
        parser.error("--arm-release-seconds 必须 > 0")
    if args.arm_random_max_delta_rad <= 0:
        parser.error("--arm-random-max-delta-rad 必须 > 0")
    if args.arm_random_shoulder_elbow_max_delta_rad is None:
        args.arm_random_shoulder_elbow_max_delta_rad = args.arm_random_max_delta_rad
    if args.arm_random_shoulder_elbow_max_delta_rad <= 0:
        parser.error("--arm-random-shoulder-elbow-max-delta-rad 必须 > 0")
    if args.arm_random_wrist_max_delta_rad <= 0:
        parser.error("--arm-random-wrist-max-delta-rad 必须 > 0")
    if args.arm_limit_margin_rad < 0:
        parser.error("--arm-limit-margin-rad 必须 >= 0")
    if args.arm_kp < 0 or args.arm_kd < 0:
        parser.error("--arm-kp/--arm-kd 必须 >= 0")
    if args.arm_joint_default_delta_rad <= 0:
        parser.error("--arm-joint-default-delta-rad 必须 > 0")
    if args.test_delta_rad <= 0:
        parser.error("--test-delta-rad 必须 > 0")
    if args.test_hold_seconds <= 0:
        parser.error("--test-hold-seconds 必须 > 0")
    if args.motion_eps_rad <= 0:
        parser.error("--motion-eps-rad 必须 > 0")
    if args.arm_preset_max_delta_rad <= 0:
        parser.error("--arm-preset-max-delta-rad 必须 > 0")
    if args.arm_safety_check_period <= 0:
        parser.error("--arm-safety-check-period 必须 > 0")
    if args.arm_preset_on_startup and args.enable_arm_waypoints and not args.arm_sdk_on_startup:
        parser.error("--arm-preset-on-startup 需要同时指定 --arm-sdk-on-startup")
    if args.arm_preset_on_startup and args.enable_arm_waypoints:
        presets_path = Path(args.arm_presets_json)
        if presets_path.exists():
            presets = load_arm_presets(presets_path)
            if args.arm_preset and args.arm_preset not in presets:
                parser.error(f"--arm-preset {args.arm_preset} 不在 {presets_path}")
    if args.arm_only and not args.allow_no_camera:
        args.allow_no_camera = True
    if args.arm_only and (not args.stream_debug or not args.enable_arm_waypoints):
        parser.error("--arm-only 需要同时指定 --stream-debug 和 --enable-arm-waypoints")
    args.hand_frame = normalize_hand_frame(args.hand_frame)
    if uses_dex1_gripper(args.hand_frame, args.fk_urdf):
        urdf_path = Path(args.fk_urdf)
        if urdf_path.exists():
            urdf_text = urdf_path.read_text(encoding="utf-8")
            if RIGHT_DEX1_BASE_LINK not in urdf_text:
                parser.error(
                    f"--hand-frame {args.hand_frame} 需要 Dex1 URDF，"
                    f"请改用 {DEFAULT_FK_URDF}"
                )
    normalize_mode(args.mode)
    return args


if __name__ == "__main__":
    sys.exit(run_capture(parse_args()))
