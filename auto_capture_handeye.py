# -*- coding: utf-8 -*-
"""Move the G1 arm through small offsets and capture hand-eye samples automatically.

The script is intentionally conservative:
- motion starts only with --confirm-robot-motion;
- targets are relative offsets from the current posture;
- each sample is saved only when the chessboard is detected after the arm settles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from handeye_calib.calibration_target import build_object_points, solve_target_pose
from handeye_calib.camera import RealSenseD435i
from handeye_calib.chessboard import find_chessboard_corners, gamma_correct_bgr
from handeye_calib.debug_stream import DebugStreamServer
from handeye_calib.io_utils import load_camera_params, save_capture_record
from handeye_calib.solver import normalize_mode


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_URDF = PROJECT_ROOT / "robots" / "g1" / "g1_29dof_rev_1_0.urdf"

ARM_SDK_WEIGHT = 29
WAIST_JOINTS = [12, 13, 14]
LEFT_ARM_JOINTS = list(range(15, 22))
RIGHT_ARM_JOINTS = list(range(22, 29))
COMMAND_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

G1_JOINT_INDEX = {
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
INDEX_TO_URDF_JOINT = {index: name for name, index in G1_JOINT_INDEX.items()}
JOINT_ALIASES = {
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

DEFAULT_RIGHT_ARM_OFFSETS = [
    {"name": "home", "offsets": {}},
    {"name": "shoulder_pitch_pos", "offsets": {"right_shoulder_pitch": 0.10}},
    {"name": "shoulder_pitch_neg", "offsets": {"right_shoulder_pitch": -0.10}},
    {"name": "shoulder_roll_pos", "offsets": {"right_shoulder_roll": 0.08}},
    {"name": "shoulder_roll_neg", "offsets": {"right_shoulder_roll": -0.08}},
    {"name": "elbow_pos", "offsets": {"right_elbow": 0.12}},
    {"name": "elbow_neg", "offsets": {"right_elbow": -0.12}},
    {"name": "wrist_pitch_pos", "offsets": {"right_wrist_pitch": 0.12}},
    {"name": "wrist_pitch_neg", "offsets": {"right_wrist_pitch": -0.12}},
    {"name": "wrist_yaw_pos", "offsets": {"right_wrist_yaw": 0.12}},
    {"name": "wrist_yaw_neg", "offsets": {"right_wrist_yaw": -0.12}},
    {"name": "combo_pitch_yaw", "offsets": {"right_shoulder_pitch": 0.06, "right_wrist_yaw": -0.10}},
]


class G1ArmSdkController:
    def __init__(self, network_interface: str, domain_id: int, kp: float, kd: float) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: WPS433
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: WPS433
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: WPS433
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: WPS433
        from unitree_sdk2py.utils.crc import CRC  # noqa: WPS433

        self.kp = float(kp)
        self.kd = float(kd)
        self._low_cmd_default = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()
        self._latest_lowstate = None
        self._motion_switcher_cls = MotionSwitcherClient

        if network_interface:
            ChannelFactoryInitialize(domain_id, network_interface)
        else:
            ChannelFactoryInitialize(domain_id)

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_lowstate, 10)
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()

    def _on_lowstate(self, msg: Any) -> None:
        self._latest_lowstate = msg

    def wait_for_lowstate(self, timeout: float) -> Any:
        deadline = time.time() + timeout
        while self._latest_lowstate is None and time.time() < deadline:
            time.sleep(0.02)
        if self._latest_lowstate is None:
            raise RuntimeError("No rt/lowstate received. Check DDS interface and robot state.")
        return self._latest_lowstate

    def ensure_sdk_motion_mode(self, retries: int = 3) -> None:
        client = self._motion_switcher_cls()
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
        client = self._motion_switcher_cls()
        client.SetTimeout(2.0)
        client.Init()
        status, result = client.CheckMode()
        return {"status": status, "result": result}

    def command_joint_positions(self) -> dict[int, float]:
        msg = self.wait_for_lowstate(2.0)
        return {joint: float(msg.motor_state[joint].q) for joint in COMMAND_JOINTS}

    def urdf_joint_positions(self) -> dict[str, float]:
        msg = self.wait_for_lowstate(2.0)
        return {
            urdf_name: float(msg.motor_state[index].q)
            for index, urdf_name in INDEX_TO_URDF_JOINT.items()
        }

    def write_arm_command(self, command_q: dict[int, float], weight: float) -> None:
        cmd = self._low_cmd_default()
        lowstate = self._latest_lowstate
        if lowstate is not None:
            if hasattr(cmd, "mode_pr") and hasattr(lowstate, "mode_pr"):
                cmd.mode_pr = int(lowstate.mode_pr)
            elif hasattr(cmd, "mode_pr"):
                cmd.mode_pr = 0
            if hasattr(cmd, "mode_machine") and hasattr(lowstate, "mode_machine"):
                cmd.mode_machine = int(lowstate.mode_machine)
        cmd.motor_cmd[ARM_SDK_WEIGHT].q = float(weight)
        for joint in COMMAND_JOINTS:
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

    def hold(
        self,
        target_q: dict[int, float],
        seconds: float,
        hz: float,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> bool:
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            if stop_requested is not None and stop_requested():
                return False
            self.write_arm_command(target_q, weight=1.0)
            time.sleep(dt)
        return True

    def ramp_to(
        self,
        target_q: dict[int, float],
        seconds: float,
        hz: float,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> bool:
        start_q = self.command_joint_positions()
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for step in range(steps):
            if stop_requested is not None and stop_requested():
                return False
            ratio = float(step + 1) / float(steps)
            command_q = {
                joint: start_q[joint] * (1.0 - ratio) + target_q[joint] * ratio
                for joint in COMMAND_JOINTS
            }
            self.write_arm_command(command_q, weight=1.0)
            time.sleep(dt)
        return True

    def release(self, hold_q: dict[int, float], seconds: float, hz: float) -> None:
        steps = max(1, int(seconds * hz))
        dt = 1.0 / hz
        for _ in range(steps):
            self.write_arm_command(hold_q, weight=0.0)
            time.sleep(dt)


def load_fk_model(urdf: str):
    robot_kinematics_dir = PROJECT_ROOT / "robot_kinematics"
    joint_to_pose_dir = robot_kinematics_dir / "joint_to_pose"
    for path in (robot_kinematics_dir, joint_to_pose_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from fk_urdf import URDFFK, base_pose_matrix  # noqa: WPS433

    return URDFFK(urdf), base_pose_matrix(None, None, None)


def resolve_joint_name(name: str) -> int:
    key = name.strip()
    if key in JOINT_ALIASES:
        return JOINT_ALIASES[key]
    if key in G1_JOINT_INDEX:
        return G1_JOINT_INDEX[key]
    if key.endswith("_joint") and key in G1_JOINT_INDEX:
        return G1_JOINT_INDEX[key]
    raise ValueError(f"Unknown joint name in auto pose offsets: {name}")


def load_auto_poses(path: str) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_RIGHT_ARM_OFFSETS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Auto pose JSON must be a list of {name, offsets} objects.")
    return payload


def build_target_q(
    base_q: dict[int, float],
    pose: dict[str, Any],
    max_abs_offset_rad: float,
) -> dict[int, float]:
    target_q = dict(base_q)
    offsets = pose.get("offsets") or {}
    if not isinstance(offsets, dict):
        raise ValueError(f"Pose {pose.get('name', '<unnamed>')} offsets must be an object.")
    for joint_name, offset in offsets.items():
        joint = resolve_joint_name(str(joint_name))
        value = float(offset)
        if abs(value) > max_abs_offset_rad:
            raise ValueError(
                f"Pose {pose.get('name', '<unnamed>')} offset for {joint_name} is {value:.3f} rad, "
                f"larger than --max-offset-rad {max_abs_offset_rad:.3f}"
            )
        target_q[joint] = base_q[joint] + value
    return target_q


def max_actual_delta_for_pose(
    start_q: dict[int, float],
    target_q: dict[int, float],
    current_q: dict[int, float],
) -> tuple[float, float]:
    commanded_joints = [
        joint
        for joint in COMMAND_JOINTS
        if abs(target_q[joint] - start_q[joint]) > 1e-6
    ]
    if not commanded_joints:
        return 0.0, 0.0
    expected = max(abs(target_q[joint] - start_q[joint]) for joint in commanded_joints)
    actual = max(abs(current_q[joint] - start_q[joint]) for joint in commanded_joints)
    return expected, actual


def transform_for_hand_frame(
    controller: G1ArmSdkController,
    model: Any,
    base_pose: Any,
    base_link: str,
    hand_frame: str,
) -> np.ndarray:
    poses = model.compute_link_poses(
        joint_values=controller.urdf_joint_positions(),
        targets=[hand_frame],
        base_link=base_link,
        base_pose=base_pose,
        clamp_to_limits=False,
    )
    if hand_frame not in poses:
        raise RuntimeError(f"FK did not return hand frame: {hand_frame}")
    return np.asarray(poses[hand_frame].matrix, dtype=np.float64)


def telemetry_state(
    controller: Optional[G1ArmSdkController],
    model: Any,
    base_pose: Any,
    args: argparse.Namespace,
    include_motion_mode: bool = False,
) -> dict[str, Any]:
    if controller is None:
        return {"robot": {"ready": False}}
    state: dict[str, Any] = {"robot": {"ready": True}}
    try:
        lowstate = controller.wait_for_lowstate(0.2)
        state["lowstate_mode"] = {
            "mode_pr": getattr(lowstate, "mode_pr", None),
            "mode_machine": getattr(lowstate, "mode_machine", None),
        }
        command_q = controller.command_joint_positions()
        state["right_arm_joints_rad"] = {
            INDEX_TO_URDF_JOINT[joint].replace("_joint", ""): command_q[joint]
            for joint in RIGHT_ARM_JOINTS
        }
        state["waist_joints_rad"] = {
            INDEX_TO_URDF_JOINT[joint].replace("_joint", ""): command_q[joint]
            for joint in WAIST_JOINTS
        }
    except Exception as exc:
        state["joint_error"] = str(exc)

    if model is not None and base_pose is not None:
        try:
            T_hand2base = transform_for_hand_frame(
                controller,
                model,
                base_pose,
                base_link=args.fk_base_link,
                hand_frame=args.hand_frame,
            )
            state["fk"] = {
                "base_link": args.fk_base_link,
                "target": args.hand_frame,
                "position_xyz_m": T_hand2base[:3, 3].astype(float).tolist(),
                "transform_matrix": T_hand2base.astype(float).tolist(),
            }
        except Exception as exc:
            state["fk_error"] = str(exc)

    if include_motion_mode:
        try:
            state["motion_mode"] = controller.check_motion_mode()
        except Exception as exc:
            state["motion_mode_error"] = str(exc)
    return state


def run_test_move(
    controller: G1ArmSdkController,
    args: argparse.Namespace,
    stream_server: DebugStreamServer,
    cam: RealSenseD435i,
    session_dir: Path,
    saved_count: int,
    model: Any,
    base_pose: Any,
) -> str:
    joint = resolve_joint_name(args.test_joint)
    start_q = controller.command_joint_positions()
    target_q = dict(start_q)
    target_q[joint] = start_q[joint] + args.test_delta_rad
    joint_name = INDEX_TO_URDF_JOINT.get(joint, str(joint))
    flags = {"stop": False, "quit": False}

    def should_stop() -> bool:
        poll_stream_command(stream_server, flags)
        return flags["stop"] or flags["quit"]

    print(f"[TEST] moving {joint_name}: {start_q[joint]:+.4f} -> {target_q[joint]:+.4f}")
    stream_server.update_state(
        {
            "status": "test_moving",
            "test_joint": joint_name,
            "test_delta_rad": args.test_delta_rad,
            "session_dir": str(session_dir.resolve()),
            "saved_count": saved_count,
            "last_message": f"testing {joint_name}; Stop is available",
        }
    )
    moved_out = controller.ramp_to(target_q, args.ramp_seconds, args.control_hz, should_stop)
    if moved_out:
        controller.hold(target_q, args.test_hold_seconds, args.control_hz, should_stop)
    current_q = controller.command_joint_positions()
    actual_delta = current_q[joint] - start_q[joint]
    print(f"[TEST] observed {joint_name} delta={actual_delta:+.4f} rad")
    message = f"test {joint_name}: observed delta={actual_delta:+.4f} rad"
    if abs(actual_delta) < args.motion_eps_rad:
        message += " (NO MOTION: arm_sdk command may be ignored)"
        print(f"[TEST][WARN] {message}")

    if args.test_return_to_start and not flags["stop"]:
        print("[TEST] returning test joint to start posture")
        controller.ramp_to(start_q, args.ramp_seconds, args.control_hz, should_stop)
        controller.hold(start_q, args.hold_seconds, args.control_hz, should_stop)
    controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
    update_idle_stream(
        cam,
        stream_server,
        args,
        session_dir,
        saved_count,
        status="idle_waiting_for_start",
        last_message=message,
        controller=controller,
        model=model,
        base_pose=base_pose,
    )
    return message


def open_camera(args: argparse.Namespace) -> RealSenseD435i:
    RealSenseD435i.set_emitter(args.cam_index if args.enable_emitter else None, args.cam_serial)
    cam = RealSenseD435i(
        index=args.cam_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.cam_serial,
        camera_name=args.camera_name,
        mount=args.camera_mount,
        color_only=args.color_only,
    )
    cam.open()
    cam.start()
    return cam


def wait_for_detected_frame(
    cam: RealSenseD435i,
    pattern_size: tuple[int, int],
    gamma: float,
    timeout_seconds: float,
    stop_requested: Optional[Callable[[], bool]] = None,
    stream_server: Optional[DebugStreamServer] = None,
    state: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if stop_requested is not None and stop_requested():
            return None
        frame = cam.fetch(timeout_ms=200)
        if frame is None or frame.get("rgb") is None:
            continue
        frame_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        preview = gamma_correct_bgr(frame_bgr, gamma)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        corners, detect_method = find_chessboard_corners(gray, pattern_size, gamma)
        if corners is not None:
            cv2.drawChessboardCorners(preview, pattern_size, corners, True)
        if stream_server is not None:
            stream_server.update_frame(preview)
            stream_server.update_state(
                {
                    **(state or {}),
                    "chessboard_detected": corners is not None,
                    "detect_method": detect_method,
                    "last_message": "detecting before auto save",
                }
            )
        if corners is not None:
            return frame_bgr, corners, detect_method
    return None


def poll_stream_command(stream_server: Optional[DebugStreamServer], flags: dict[str, bool]) -> Optional[str]:
    if stream_server is None:
        return None
    payload = stream_server.pop_command()
    cmd = DebugStreamServer.command_name(payload)
    if cmd == "stop":
        flags["stop"] = True
    elif cmd == "quit":
        flags["quit"] = True
        flags["stop"] = True
    return cmd


def update_idle_stream(
    cam: RealSenseD435i,
    stream_server: DebugStreamServer,
    args: argparse.Namespace,
    session_dir: Path,
    saved_count: int,
    status: str,
    last_message: str,
    controller: Optional[G1ArmSdkController] = None,
    model: Any = None,
    base_pose: Any = None,
) -> None:
    telemetry = telemetry_state(controller, model, base_pose, args, include_motion_mode=True)
    frame = cam.fetch(timeout_ms=200)
    if frame is None or frame.get("rgb") is None:
        stream_server.update_state(
            {
                "status": status,
                "session_dir": str(session_dir.resolve()),
                "saved_count": saved_count,
                "last_message": "waiting for camera frame",
                **telemetry,
            }
        )
        return
    frame_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
    preview = gamma_correct_bgr(frame_bgr, args.gamma)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    pattern_size = (args.cols, args.rows)
    corners, detect_method = find_chessboard_corners(gray, pattern_size, args.gamma)
    if corners is not None:
        cv2.drawChessboardCorners(preview, pattern_size, corners, True)
    cv2.putText(
        preview,
        f"{status} saved={saved_count} board={int(corners is not None)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if corners is not None else (0, 160, 255),
        2,
        cv2.LINE_AA,
    )
    stream_server.update_frame(preview)
    stream_server.update_state(
        {
            "status": status,
            "session_dir": str(session_dir.resolve()),
            "saved_count": saved_count,
            "chessboard_detected": corners is not None,
            "detect_method": detect_method,
            "last_message": last_message,
            "controls": {
                "start_calib": "start automatic robot motion",
                "switch_sdk": "release MotionSwitcher for arm_sdk control",
                "test_move": "move one right-arm joint and report observed delta",
                "stop": "stop current automatic run and keep program alive",
                "quit": "exit program",
            },
            **telemetry,
        }
    )


def wait_for_start_or_quit(
    cam: RealSenseD435i,
    stream_server: DebugStreamServer,
    args: argparse.Namespace,
    session_dir: Path,
    saved_count: int,
    last_message: str,
    controller: Optional[G1ArmSdkController],
    model: Any,
    base_pose: Any,
) -> str:
    print("[AUTO] waiting for browser Start Calib/Switch SDK/Test Move command")
    while True:
        update_idle_stream(
            cam,
            stream_server,
            args,
            session_dir,
            saved_count,
            status="idle_waiting_for_start",
            last_message=last_message,
            controller=controller,
            model=model,
            base_pose=base_pose,
        )
        cmd = DebugStreamServer.command_name(stream_server.pop_command())
        if cmd == "start_calib":
            print("[AUTO] Start Calib received")
            return "start_calib"
        if cmd == "switch_sdk":
            print("[AUTO] Switch SDK Mode received")
            return "switch_sdk"
        if cmd == "test_move":
            print("[AUTO] Test Move received")
            return "test_move"
        if cmd == "quit":
            print("[AUTO] Quit received while idle")
            return "quit"
        if cmd == "stop":
            last_message = "already idle; stop ignored"
        time.sleep(0.05)


def robot_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "robot_model": args.robot_model,
        "robot_host": args.robot_host,
        "robot_user": args.robot_user,
        "ros_distro": args.ros_distro,
        "ros_domain_id": args.ros_domain_id,
        "notes": args.notes,
        "auto_capture": True,
    }


def camera_dir_name(args: argparse.Namespace) -> str:
    if args.camera_name:
        return args.camera_name
    if args.cam_serial:
        return f"serial_{args.cam_serial}"
    return f"camera_{args.cam_index}"


def sleep_with_countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Auto hand-eye motion starts in {remaining}s... Ctrl+C to cancel.")
        time.sleep(1.0)


def run_auto_capture(args: argparse.Namespace) -> int:
    if not args.confirm_robot_motion:
        raise RuntimeError("Refusing to move robot without --confirm-robot-motion.")

    mode = normalize_mode(args.mode)
    if mode != "eye_in_hand":
        raise ValueError("Auto robot-motion capture currently supports only eye-in-hand.")

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    session_name = args.session_name or f"{mode}_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = data_root / session_name / camera_dir_name(args)

    cam = open_camera(args)
    stream_server = DebugStreamServer(
        host=args.stream_host,
        port=args.stream_port,
        jpeg_quality=args.stream_jpeg_quality,
    )
    stream_server.start()
    print(f"[STREAM] http://{args.stream_host}:{args.stream_port}")

    controller: Optional[G1ArmSdkController] = None
    model = None
    base_pose = None
    sdk_mode_released = False
    saved_count = 0
    last_message = "open browser and press Start Calib"
    try:
        camera_matrix, dist_coeffs, camera_intrinsics = load_camera_params(
            camera_matrix_npy=args.camera_matrix_npy,
            dist_coeffs_npy=args.dist_coeffs_npy,
            camera_json=args.camera_json,
        ) or cam.color_intrinsics()
        camera_info = cam.capture_metadata()
        auto_poses = load_auto_poses(args.auto_poses_json)
        controller = G1ArmSdkController(
            network_interface=args.network_interface,
            domain_id=args.domain_id,
            kp=args.kp,
            kd=args.kd,
        )
        controller.wait_for_lowstate(args.state_timeout)
        model, base_pose = load_fk_model(args.fk_urdf)

        print(f"[SESSION] {session_dir.resolve()}")
        print(f"[AUTO] poses={len(auto_poses)} hand_frame={args.hand_frame}")
        print(f"[CAMERA] {camera_info}")

        while True:
            idle_command = wait_for_start_or_quit(
                cam,
                stream_server,
                args,
                session_dir,
                saved_count,
                last_message,
                controller,
                model,
                base_pose,
            )
            if idle_command == "quit":
                return 0

            flags = {"stop": False, "quit": False}

            def should_stop() -> bool:
                poll_stream_command(stream_server, flags)
                return flags["stop"] or flags["quit"]

            if idle_command == "switch_sdk":
                if not args.skip_release_mode:
                    controller.ensure_sdk_motion_mode()
                    sdk_mode_released = True
                last_message = f"switch sdk done: {controller.check_motion_mode()}"
                continue

            if idle_command == "test_move":
                if not args.skip_release_mode and not sdk_mode_released:
                    controller.ensure_sdk_motion_mode()
                    sdk_mode_released = True
                last_message = run_test_move(
                    controller,
                    args,
                    stream_server,
                    cam,
                    session_dir,
                    saved_count,
                    model,
                    base_pose,
                )
                continue

            if not args.skip_release_mode and not sdk_mode_released:
                controller.ensure_sdk_motion_mode()
                sdk_mode_released = True
            base_q = controller.command_joint_positions()
            last_message = "running automatic calibration"
            print("[AUTO] Start Calib accepted. Current posture is the base for relative offsets.")

            if args.wait > 0:
                for remaining in range(args.wait, 0, -1):
                    update_idle_stream(
                        cam,
                        stream_server,
                        args,
                        session_dir,
                        saved_count,
                        status="starting_countdown",
                        last_message=f"motion starts in {remaining}s; Stop cancels",
                        controller=controller,
                        model=model,
                        base_pose=base_pose,
                    )
                    if should_stop():
                        break
                    time.sleep(1.0)

            for index, pose in enumerate(auto_poses, start=1):
                if should_stop():
                    break
                name = str(pose.get("name") or f"pose_{index:02d}")
                target_q = build_target_q(base_q, pose, args.max_offset_rad)
                print(f"[POSE {index}/{len(auto_poses)}] moving to {name}")
                stream_server.update_state(
                    {
                        "status": "moving",
                        "pose_index": index,
                        "pose_count": len(auto_poses),
                        "pose_name": name,
                        "session_dir": str(session_dir.resolve()),
                        "saved_count": saved_count,
                        "last_message": f"moving to {name}; Stop is available",
                        **telemetry_state(controller, model, base_pose, args),
                    }
                )
                if not controller.ramp_to(target_q, args.ramp_seconds, args.control_hz, should_stop):
                    break
                if not controller.hold(target_q, args.settle_seconds, args.control_hz, should_stop):
                    break
                expected_delta, actual_delta = max_actual_delta_for_pose(
                    base_q,
                    target_q,
                    controller.command_joint_positions(),
                )
                if expected_delta > 0.0 and actual_delta < args.motion_eps_rad:
                    last_message = (
                        f"NO MOTION at {name}: expected {expected_delta:.3f} rad, "
                        f"observed {actual_delta:.3f} rad"
                    )
                    print(f"[AUTO][WARN] {last_message}")
                    stream_server.update_state(
                        {
                            "status": "motion_not_observed",
                            "pose_index": index,
                            "pose_count": len(auto_poses),
                            "pose_name": name,
                            "expected_delta_rad": expected_delta,
                            "observed_delta_rad": actual_delta,
                            "session_dir": str(session_dir.resolve()),
                            "saved_count": saved_count,
                            "last_message": last_message,
                        }
                    )
                    flags["stop"] = True
                    break

                detected = wait_for_detected_frame(
                    cam,
                    pattern_size=pattern_size,
                    gamma=args.gamma,
                    timeout_seconds=args.detect_timeout,
                    stop_requested=should_stop,
                    stream_server=stream_server,
                    state={
                        "status": "detecting",
                        "pose_index": index,
                        "pose_count": len(auto_poses),
                        "pose_name": name,
                        "session_dir": str(session_dir.resolve()),
                        "saved_count": saved_count,
                        **telemetry_state(controller, model, base_pose, args),
                    },
                )
                if should_stop():
                    break
                if detected is None:
                    print(f"[SKIP] {name}: chessboard not detected within {args.detect_timeout:.1f}s")
                    last_message = f"skipped {name}: no chessboard"
                    continue

                frame_bgr, corners, detect_method = detected
                if model is None or base_pose is None:
                    raise RuntimeError("FK model is not initialized.")
                T_hand2base = transform_for_hand_frame(
                    controller,
                    model,
                    base_pose,
                    base_link=args.fk_base_link,
                    hand_frame=args.hand_frame,
                )
                target_rvec, target_tvec, target_rms = solve_target_pose(
                    objp,
                    corners,
                    camera_matrix,
                    dist_coeffs,
                )
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
                    pose_text=f"auto_fk:{args.fk_base_link}->{args.hand_frame}:{name}",
                    pose_values=T_hand2base.reshape(-1).astype(float).tolist(),
                    pose_units={
                        "translation": "m",
                        "rotation": "matrix",
                        "euler_order": "fk_transform",
                    },
                    T_hand2base=T_hand2base,
                    hand_frame_name=args.hand_frame,
                )
                saved_count += 1
                last_message = f"saved {image_path.name}, pnp_rms={target_rms:.3f}px"
                print(f"[SAVE] {image_path.resolve()} pnp_rms={target_rms:.4f}px")

            if flags["quit"]:
                print("[AUTO] Quit received during run")
                controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
                return 0

            if flags["stop"]:
                print("[AUTO] Stop received; releasing arm_sdk and returning to idle")
                last_message = "stopped; press Start Calib to run again"
                controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
                continue

            if args.return_to_start:
                print("[AUTO] returning to start posture")
                if controller.ramp_to(base_q, args.ramp_seconds, args.control_hz, should_stop):
                    controller.hold(base_q, args.hold_seconds, args.control_hz, should_stop)
                if flags["quit"]:
                    print("[AUTO] Quit received while returning to start")
                    controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
                    return 0
                if flags["stop"]:
                    print("[AUTO] Stop received while returning to start; releasing arm_sdk and returning to idle")
                    last_message = "stopped while returning to start; press Start Calib to run again"
                    controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
                    continue

            print("[AUTO] releasing arm_sdk")
            controller.release(controller.command_joint_positions(), args.release_seconds, args.control_hz)
            last_message = f"completed; saved={saved_count}; press Start Calib to run again"
            print(f"[DONE] saved={saved_count} session={session_dir.resolve()}")
    finally:
        cam.close()
        RealSenseD435i.set_emitter(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动移动 G1 右臂并采集 eye-in-hand 手眼标定样本")
    parser.add_argument("--confirm-robot-motion", action="store_true", help="确认允许脚本发布 arm_sdk 运动命令")
    parser.add_argument("--network-interface", default="eth0", help="DDS 网卡，如 eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--skip-release-mode", action="store_true")
    parser.add_argument("--wait", type=int, default=0, help="Start Calib 后再等待多少秒开始运动")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--hold-seconds", type=float, default=0.3)
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--max-offset-rad", type=float, default=0.25)
    parser.add_argument("--motion-eps-rad", type=float, default=0.02, help="低于该关节变化量则认为机器人没有实际运动")
    parser.add_argument("--return-to-start", action="store_true")
    parser.add_argument("--auto-poses-json", default="", help="可选：自定义相对关节偏移序列 JSON")
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--stream-host", default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--stream-jpeg-quality", type=int, default=80)
    parser.add_argument("--test-joint", default="right_shoulder_pitch", help="网页 Test Move 使用的关节名")
    parser.add_argument("--test-delta-rad", type=float, default=0.20, help="网页 Test Move 的关节偏移")
    parser.add_argument("--test-hold-seconds", type=float, default=1.0)
    parser.add_argument("--test-return-to-start", action="store_true", default=True)

    parser.add_argument("--mode", default="eye-in-hand")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--cam-serial", default="")
    parser.add_argument("--camera-name", default="")
    parser.add_argument("--camera-mount", default="wrist")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--enable-emitter", action="store_true")
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--detect-timeout", type=float, default=3.0)
    parser.add_argument("--camera-matrix-npy", default="")
    parser.add_argument("--dist-coeffs-npy", default="")
    parser.add_argument("--camera-json", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--session-name", default="")

    parser.add_argument("--hand-frame", default="right_wrist_yaw_link")
    parser.add_argument("--fk-base-link", default="pelvis")
    parser.add_argument("--fk-urdf", default=str(DEFAULT_URDF))

    parser.add_argument("--robot-model", default="Unitree G1")
    parser.add_argument("--robot-host", default="")
    parser.add_argument("--robot-user", default="")
    parser.add_argument("--ros-distro", default="")
    parser.add_argument("--ros-domain-id", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_auto_capture(parse_args()))
