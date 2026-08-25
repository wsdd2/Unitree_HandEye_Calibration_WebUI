# -*- coding: utf-8 -*-
"""Plan B: use the URDF head D435 link to observe the right wrist/board target.

This script is intentionally read-only: it opens the head camera, detects the
chessboard, computes the board center in the camera frame, computes the current
right wrist pose from lowstate + URDF, and streams both the annotated image and
pose data to a browser.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from handeye_calib import camera as camera_module
from handeye_calib.calibration_target import build_object_points, solve_target_pose
from handeye_calib.camera import OpenCVVideoCamera, RealSenseD435i
from handeye_calib.chessboard import find_chessboard_corners, gamma_correct_bgr, put_text_bgr_adaptive
from handeye_calib.debug_stream import DebugStreamServer
from handeye_calib.io_utils import load_camera_params
from handeye_calib.transforms import invert_transform, transform_from_rvec_tvec, transform_to_record
from capture_handeye import (
    G1ArmWaypointController,
    INDEX_TO_G1_ARM_JOINT,
    UPPER_BODY_COMMAND_JOINTS,
)
from plan_b_touch_board import (
    build_target_command,
    compute_fk,
    make_target_pose,
    max_right_arm_delta,
    solve_right_arm_ik,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_URDF = WORKSPACE_ROOT / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"


class PlanBFKProvider:
    def __init__(
        self,
        network_interface: str,
        domain_id: int,
        urdf: str,
        base_link: str,
        camera_link: str,
        wrist_link: str,
        state_timeout: float,
    ) -> None:
        robot_kinematics_dir = WORKSPACE_ROOT / "robot_kinematics"
        joint_to_pose_dir = robot_kinematics_dir / "joint_to_pose"
        for path in (robot_kinematics_dir, joint_to_pose_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from fk_urdf import URDFFK, base_pose_matrix, pose_to_json  # noqa: WPS433
        from unitree_sdk2_bridge import UnitreeG1LowStateBridge  # noqa: WPS433

        self.base_link = base_link
        self.camera_link = camera_link
        self.wrist_link = wrist_link
        self.state_timeout = state_timeout
        self.bridge = UnitreeG1LowStateBridge(
            network_interface=network_interface,
            domain_id=domain_id,
        )
        self.model = URDFFK(urdf)
        self.base_pose = base_pose_matrix(None, None, None)
        self.pose_to_json = pose_to_json

    def snapshot(self) -> dict[str, Any]:
        self.bridge.wait_for_state(self.state_timeout)
        joint_values = self.bridge.latest_joint_positions()
        poses = self.model.compute_link_poses(
            joint_values=joint_values,
            targets=[self.camera_link, self.wrist_link],
            base_link=self.base_link,
            base_pose=self.base_pose,
            clamp_to_limits=False,
        )
        if self.camera_link not in poses:
            raise RuntimeError(f"URDF/FK missing camera link: {self.camera_link}")
        if self.wrist_link not in poses:
            raise RuntimeError(f"URDF/FK missing wrist link: {self.wrist_link}")

        T_base_camera = np.asarray(poses[self.camera_link].matrix, dtype=np.float64)
        T_base_wrist = np.asarray(poses[self.wrist_link].matrix, dtype=np.float64)
        T_camera_wrist = invert_transform(T_base_camera) @ T_base_wrist

        return {
            "base_link": self.base_link,
            "camera_link": self.camera_link,
            "wrist_link": self.wrist_link,
            "base_to_camera": self.pose_to_json(poses[self.camera_link], "xyzw"),
            "base_to_wrist": self.pose_to_json(poses[self.wrist_link], "xyzw"),
            "camera_to_wrist": transform_to_record(T_camera_wrist),
            "right_arm_joints_rad": {
                name: joint_values[name]
                for name in (
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                )
                if name in joint_values
            },
        }


class PlanBTouchExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.arm = G1ArmWaypointController(
            network_interface=args.fk_network_interface,
            domain_id=args.fk_domain_id,
            kp=args.touch_kp,
            kd=args.touch_kd,
        )
        self.arm.wait_for_lowstate(args.fk_state_timeout)
        robot_kinematics_dir = WORKSPACE_ROOT / "robot_kinematics"
        joint_to_pose_dir = robot_kinematics_dir / "joint_to_pose"
        for path in (robot_kinematics_dir, joint_to_pose_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from fk_urdf import URDFFK  # noqa: WPS433

        self.model = URDFFK(args.fk_urdf)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.status: dict[str, Any] = {
            "enabled": True,
            "running": False,
            "last_message": "ready; press Touch to execute",
        }

    def start_touch(self, target_camera_xyz: list[float]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.status["last_message"] = "touch already running"
                print("[TOUCH] ignored: already running")
                return
            self._stop_event.clear()
            print(f"[TOUCH] requested target_camera_xyz_m={target_camera_xyz}")
            self._thread = threading.Thread(
                target=self._run_touch,
                args=(list(target_camera_xyz),),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.status["last_message"] = "stop requested"
        print("[TOUCH] stop requested")

    def release(self) -> None:
        self._stop_event.set()
        try:
            self.arm.release(self.args.touch_release_seconds, self.args.touch_control_hz)
            self.status["last_message"] = "arm_sdk released"
            print("[TOUCH] arm_sdk released")
        except Exception as exc:
            self.status["last_message"] = f"release failed: {exc}"
            print(f"[TOUCH][WARN] release failed: {exc}")

    def _run_touch(self, target_camera_xyz: list[float]) -> None:
        with self._lock:
            self.status = {
                "enabled": True,
                "running": True,
                "last_message": "planning touch",
                "target_camera_xyz_m": target_camera_xyz,
            }
        try:
            target_q, plan = self._plan(target_camera_xyz)
            with self._lock:
                self.status.update(plan)
                self.status["last_message"] = "executing touch"
            print(f"[TOUCH] plan: {plan}")

            self._ramp_to(target_q)
            if not self._stop_event.is_set():
                self._hold(target_q)
            if self.args.touch_release_after_touch or self._stop_event.is_set():
                self.arm.release(self.args.touch_release_seconds, self.args.touch_control_hz)

            with self._lock:
                self.status["running"] = False
                self.status["last_message"] = "stopped" if self._stop_event.is_set() else "touch complete"
            print("[TOUCH] stopped" if self._stop_event.is_set() else "[TOUCH] complete")
        except Exception as exc:
            self._stop_event.set()
            try:
                self.arm.release(self.args.touch_release_seconds, self.args.touch_control_hz)
            except Exception:
                pass
            with self._lock:
                self.status["running"] = False
                self.status["last_message"] = f"touch failed: {exc}"
                self.status["error"] = str(exc)
            print(f"[TOUCH][ERROR] {exc}")

    def _plan(self, target_camera_xyz: list[float]) -> tuple[dict[int, float], dict[str, Any]]:
        current_q_index = self.arm.command_joint_positions()
        current_q_name = {
            INDEX_TO_G1_ARM_JOINT[index]: value
            for index, value in current_q_index.items()
        }
        T_base_camera, T_base_wrist = compute_fk(
            model=self.model,
            joint_values=current_q_name,
            base_link=self.args.fk_base_link,
            camera_link=self.args.camera_link,
            wrist_link=self.args.wrist_link,
        )
        target_camera = np.ones(4, dtype=np.float64)
        target_camera[:3] = np.asarray(target_camera_xyz, dtype=np.float64) + np.asarray(
            self.args.touch_target_camera_offset_xyz,
            dtype=np.float64,
        )
        target_base_xyz = (T_base_camera @ target_camera)[:3]
        tool_offset = np.asarray(self.args.touch_tool_offset_wrist_xyz, dtype=np.float64)
        target_wrist_pose = make_target_pose(T_base_wrist, target_base_xyz, tool_offset)
        requested_move = float(np.linalg.norm(target_wrist_pose[:3, 3] - T_base_wrist[:3, 3]))
        if requested_move > self.args.touch_max_move_m:
            raise RuntimeError(
                f"Refusing move {requested_move:.4f}m > touch_max_move_m {self.args.touch_max_move_m:.4f}m"
            )

        ik_args = argparse.Namespace(
            ik_max_iterations=self.args.touch_ik_max_iterations,
            ik_tolerance_position=self.args.touch_ik_tolerance_position,
            ik_tolerance_orientation=self.args.touch_ik_tolerance_orientation,
            ik_damping=self.args.touch_ik_damping,
            ik_step_scale=self.args.touch_ik_step_scale,
            ik_finite_difference_step=self.args.touch_ik_finite_difference_step,
            ik_orientation_weight=self.args.touch_ik_orientation_weight,
        )
        solution = solve_right_arm_ik(
            urdf=self.args.fk_urdf,
            base_link=self.args.fk_base_link,
            wrist_link=self.args.wrist_link,
            current_joint_values=current_q_name,
            target_wrist_pose=target_wrist_pose,
            args=ik_args,
        )
        joint_delta = max_right_arm_delta(current_q_name, solution.joint_values)
        if joint_delta > self.args.touch_max_joint_delta_rad:
            raise RuntimeError(
                f"Refusing max joint delta {joint_delta:.4f}rad > {self.args.touch_max_joint_delta_rad:.4f}rad"
            )
        if not solution.success and solution.final_position_error_norm > self.args.touch_allow_position_error_m:
            raise RuntimeError(
                f"IK failed: success={solution.success}, pos_error={solution.final_position_error_norm:.4f}m, "
                f"message={solution.message}"
            )

        target_q = build_target_command(current_q_index, solution.joint_values)
        plan = {
            "target_base_xyz_m": target_base_xyz.astype(float).tolist(),
            "current_wrist_base_xyz_m": T_base_wrist[:3, 3].astype(float).tolist(),
            "target_wrist_base_xyz_m": target_wrist_pose[:3, 3].astype(float).tolist(),
            "requested_wrist_move_m": requested_move,
            "max_right_arm_joint_delta_rad": float(joint_delta),
            "ik_success": bool(solution.success),
            "ik_message": solution.message,
            "ik_position_error_m": float(solution.final_position_error_norm),
        }
        return target_q, plan

    def _ramp_to(self, target_q: dict[int, float]) -> None:
        start_q = self.arm.command_joint_positions()
        steps = max(1, int(self.args.touch_ramp_seconds * self.args.touch_control_hz))
        dt = 1.0 / self.args.touch_control_hz
        for step in range(steps):
            if self._stop_event.is_set():
                break
            ratio = float(step + 1) / float(steps)
            command_q = dict(start_q)
            for joint in UPPER_BODY_COMMAND_JOINTS:
                command_q[joint] = start_q[joint] * (1.0 - ratio) + target_q[joint] * ratio
            self.arm.write_arm_command(command_q, weight=1.0, active_joints=UPPER_BODY_COMMAND_JOINTS)
            time.sleep(dt)

    def _hold(self, target_q: dict[int, float]) -> None:
        steps = max(1, int(self.args.touch_hold_seconds * self.args.touch_control_hz))
        dt = 1.0 / self.args.touch_control_hz
        for _ in range(steps):
            if self._stop_event.is_set():
                break
            self.arm.write_arm_command(target_q, weight=1.0, active_joints=UPPER_BODY_COMMAND_JOINTS)
            time.sleep(dt)


def maybe_reset_realsense(args: argparse.Namespace) -> None:
    if not args.rs_hardware_reset:
        return
    if camera_module.rs is None:
        raise RuntimeError("pyrealsense2 is not available; cannot hardware reset RealSense")
    target_serial = args.cam_serial.strip()
    matched = False
    for dev in camera_module.rs.context().query_devices():
        serial = dev.get_info(camera_module.rs.camera_info.serial_number) if dev.supports(camera_module.rs.camera_info.serial_number) else ""
        if target_serial and serial != target_serial:
            continue
        print(f"[CAMERA] hardware_reset RealSense serial={serial or '?'}")
        dev.hardware_reset()
        matched = True
    if target_serial and not matched:
        raise RuntimeError(f"RealSense serial not found for hardware reset: {target_serial}")
    if matched:
        time.sleep(args.rs_reset_wait_seconds)


def open_realsense_camera(args: argparse.Namespace) -> RealSenseD435i:
    maybe_reset_realsense(args)
    last_exc: Exception | None = None
    reset_after_busy = False
    for attempt in range(1, args.camera_start_retries + 1):
        cam = RealSenseD435i(
            index=args.cam_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.cam_serial,
            camera_name=args.camera_name,
            mount="head",
            color_only=True,
        )
        try:
            cam.open()
            cam.start()
            if attempt > 1:
                print(f"[CAMERA] started on retry {attempt}")
            return cam
        except Exception as exc:
            last_exc = exc
            cam.close()
            message = str(exc)
            if (
                args.rs_reset_on_busy
                and not args.rs_hardware_reset
                and not reset_after_busy
                and ("resource busy" in message.lower() or "errno=16" in message.lower() or "VIDIOC_S_FMT" in message)
            ):
                print("[CAMERA] busy detected; hardware_reset target RealSense before retry")
                reset_after_busy = True
                args.rs_hardware_reset = True
                maybe_reset_realsense(args)
                args.rs_hardware_reset = False
            if attempt >= args.camera_start_retries:
                break
            print(
                f"[CAMERA] start failed on attempt {attempt}/{args.camera_start_retries}: {exc}; "
                f"retrying in {args.camera_start_retry_delay:.1f}s"
            )
            time.sleep(args.camera_start_retry_delay)
    raise RuntimeError(f"failed to start RealSense after {args.camera_start_retries} attempts: {last_exc}") from last_exc


def open_opencv_camera(args: argparse.Namespace) -> OpenCVVideoCamera:
    cam = OpenCVVideoCamera(
        device=args.opencv_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    cam.open()
    cam.start()
    return cam


def open_camera(args: argparse.Namespace) -> RealSenseD435i | OpenCVVideoCamera:
    if args.camera_backend == "realsense":
        return open_realsense_camera(args)
    if args.camera_backend == "opencv":
        return open_opencv_camera(args)

    try:
        return open_realsense_camera(args)
    except Exception as exc:
        print(f"[CAMERA][WARN] RealSense backend failed: {exc}")
        print(f"[CAMERA][WARN] Falling back to OpenCV/V4L2 device {args.opencv_device}")
        return open_opencv_camera(args)


def realsense_profile_color_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if camera_module.rs is None:
        return None
    target_serial = args.cam_serial.strip()
    ctx = camera_module.rs.context()
    for dev in ctx.query_devices():
        serial = dev.get_info(camera_module.rs.camera_info.serial_number) if dev.supports(camera_module.rs.camera_info.serial_number) else ""
        if target_serial and serial != target_serial:
            continue
        for sensor in dev.query_sensors():
            profiles = sensor.get_stream_profiles()
            color_profiles = []
            for profile in profiles:
                try:
                    if profile.stream_type() != camera_module.rs.stream.color:
                        continue
                    video_profile = profile.as_video_stream_profile()
                    intr = video_profile.get_intrinsics()
                    color_profiles.append((profile, intr))
                except Exception:
                    continue
            if not color_profiles:
                continue

            selected_profile, selected_intr = color_profiles[0]
            for profile, intr in color_profiles:
                try:
                    if int(intr.width) == int(args.width) and int(intr.height) == int(args.height) and int(profile.fps()) == int(args.fps):
                        selected_profile, selected_intr = profile, intr
                        break
                except Exception:
                    pass

            coeffs = np.asarray(selected_intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
            camera_matrix = np.array(
                [
                    [selected_intr.fx, 0.0, selected_intr.ppx],
                    [0.0, selected_intr.fy, selected_intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            info = {
                "source": "realsense_device_color_profile",
                "serial": serial,
                "width": int(selected_intr.width),
                "height": int(selected_intr.height),
                "fps": int(selected_profile.fps()) if hasattr(selected_profile, "fps") else None,
                "fx": float(selected_intr.fx),
                "fy": float(selected_intr.fy),
                "ppx": float(selected_intr.ppx),
                "ppy": float(selected_intr.ppy),
                "distortion_model": str(selected_intr.model),
                "coeffs": coeffs.reshape(-1).astype(float).tolist(),
            }
            return camera_matrix, coeffs, info
    return None


def resolve_camera_intrinsics(args: argparse.Namespace, cam: RealSenseD435i | OpenCVVideoCamera) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    loaded = load_camera_params(
        camera_matrix_npy=args.camera_matrix_npy,
        dist_coeffs_npy=args.dist_coeffs_npy,
        camera_json=args.camera_json,
    )
    if loaded is not None:
        return loaded
    if not hasattr(cam, "color_intrinsics"):
        profile_intrinsics = realsense_profile_color_intrinsics(args)
        if profile_intrinsics is not None:
            return profile_intrinsics
        raise RuntimeError("OpenCV/V4L2 camera backend could not read RealSense color profile intrinsics.")
    camera_matrix, dist_coeffs, info = cam.color_intrinsics()
    info["source"] = "realsense_profile"
    return camera_matrix, dist_coeffs, info


def board_center_object_point(cols: int, rows: int, square_mm: float) -> np.ndarray:
    square_m = float(square_mm) / 1000.0
    return np.array(
        [0.5 * (cols - 1) * square_m, 0.5 * (rows - 1) * square_m, 0.0, 1.0],
        dtype=np.float64,
    )


def draw_detection(
    vis: np.ndarray,
    corners: np.ndarray | None,
    pattern_size: tuple[int, int],
    rvec: np.ndarray | None,
    tvec: np.ndarray | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    center_obj: np.ndarray,
) -> None:
    if corners is not None:
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    if rvec is None or tvec is None:
        return
    try:
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec.reshape(3, 1), tvec.reshape(3, 1), 0.08)
    except Exception:
        pass
    center_px, _ = cv2.projectPoints(
        center_obj[:3].reshape(1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    u, v = center_px.reshape(2)
    cv2.circle(vis, (int(round(u)), int(round(v))), 8, (0, 0, 255), -1)
    cv2.putText(vis, "board center", (int(round(u)) + 10, int(round(v)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def run(args: argparse.Namespace) -> int:
    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    center_obj = board_center_object_point(args.cols, args.rows, args.square_mm)

    cam = open_camera(args)
    camera_info = cam.capture_metadata()
    camera_matrix, dist_coeffs, intrinsics = resolve_camera_intrinsics(args, cam)

    stream_server = DebugStreamServer(
        host=args.stream_host,
        port=args.stream_port,
        jpeg_quality=args.stream_jpeg_quality,
    )
    stream_server.start()
    print(f"[STREAM] http://{args.stream_host}:{args.stream_port}")

    fk_provider = PlanBFKProvider(
        network_interface=args.fk_network_interface,
        domain_id=args.fk_domain_id,
        urdf=args.fk_urdf,
        base_link=args.fk_base_link,
        camera_link=args.camera_link,
        wrist_link=args.wrist_link,
        state_timeout=args.fk_state_timeout,
    )
    touch_executor: PlanBTouchExecutor | None = None
    if args.execute:
        if not args.confirm_touch_motion:
            raise RuntimeError("Refusing to enable web Touch. Add --confirm-touch-motion after safety checks.")
        touch_executor = PlanBTouchExecutor(args)

    try:
        last_fk: dict[str, Any] = {}
        last_fk_ts = 0.0
        last_msg = ""

        print(f"[CAMERA] {camera_info}")
        print(f"[INTRINSICS] {intrinsics.get('source')}")
        print(f"[URDF] {args.fk_urdf}")
        print("[NOTE] Assuming URDF d435_link matches the OpenCV color camera frame. Verify optical-frame convention before precision use.")

        while True:
            frame = cam.fetch(timeout_ms=args.timeout_ms)
            if frame is None or frame.get("rgb") is None:
                continue

            frame_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            preview = gamma_correct_bgr(frame_bgr, args.gamma)
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            corners, detect_method = find_chessboard_corners(gray, pattern_size, args.gamma)

            rvec = None
            tvec = None
            board_state: dict[str, Any] = {"detected": corners is not None, "method": detect_method}
            if corners is not None:
                try:
                    rvec, tvec, rms = solve_target_pose(objp, corners, camera_matrix, dist_coeffs)
                    T_camera_board = transform_from_rvec_tvec(rvec, tvec)
                    center_camera = T_camera_board @ center_obj
                    board_state.update(
                        {
                            "target_reprojection_rms_px": rms,
                            "camera_to_board": transform_to_record(T_camera_board),
                            "board_center_in_camera_xyz_m": center_camera[:3].astype(float).tolist(),
                        }
                    )
                except Exception as exc:
                    last_msg = f"PnP failed: {exc}"
                    board_state["error"] = str(exc)

            now = time.monotonic()
            if now - last_fk_ts >= args.fk_update_period:
                try:
                    last_fk = fk_provider.snapshot()
                except Exception as exc:
                    last_fk = {"error": str(exc)}
                last_fk_ts = now

            if "camera_to_wrist" in last_fk and "board_center_in_camera_xyz_m" in board_state:
                wrist_xyz = np.asarray(last_fk["camera_to_wrist"]["tvec_m"], dtype=np.float64)
                center_xyz = np.asarray(board_state["board_center_in_camera_xyz_m"], dtype=np.float64)
                delta = center_xyz - wrist_xyz
                board_state["wrist_to_board_center_delta_in_camera_m"] = delta.astype(float).tolist()
                board_state["wrist_to_board_center_distance_m"] = float(np.linalg.norm(delta))

            vis = preview.copy()
            draw_detection(vis, corners, pattern_size, rvec, tvec, camera_matrix, dist_coeffs, center_obj)
            put_text_bgr_adaptive(vis, f"Plan B head cam board={int(corners is not None)} saved=0", (10, 30), 0.72)
            put_text_bgr_adaptive(vis, f"method={detect_method or '-'} camera_link={args.camera_link} wrist={args.wrist_link}", (10, 60), 0.58)
            if "wrist_to_board_center_distance_m" in board_state:
                put_text_bgr_adaptive(vis, f"wrist->board center dist={board_state['wrist_to_board_center_distance_m']:.3f} m", (10, 90), 0.58)
            if last_msg:
                put_text_bgr_adaptive(vis, last_msg, (10, 120), 0.58)

            stream_server.update_frame(vis)
            stream_server.update_state(
                {
                    "mode": "plan_b_head_camera_execute" if args.execute else "plan_b_head_camera_readonly",
                    "camera": camera_info,
                    "camera_intrinsics": intrinsics,
                    "board": board_state,
                    "fk": last_fk,
                    "touch": touch_executor.status if touch_executor is not None else {"enabled": False},
                    "warning": (
                        "Execute enabled: robot moves only after web Touch. Stop/Release can interrupt."
                        if args.execute
                        else "Read-only. No robot motion is sent. URDF d435_link vs OpenCV optical frame must be verified."
                    ),
                }
            )

            command = DebugStreamServer.command_name(stream_server.pop_command())
            if command == "quit":
                if touch_executor is not None:
                    touch_executor.stop()
                    touch_executor.release()
                break
            if command == "touch":
                if touch_executor is None:
                    last_msg = "Touch ignored: start with --execute --confirm-touch-motion"
                elif "board_center_in_camera_xyz_m" not in board_state:
                    last_msg = "Touch ignored: no board center detected"
                else:
                    touch_executor.start_touch(board_state["board_center_in_camera_xyz_m"])
                    last_msg = "Touch requested"
            elif command == "stop":
                if touch_executor is not None:
                    touch_executor.stop()
                    last_msg = "Stop requested"
            elif command in {"release", "arm_release"}:
                if touch_executor is not None:
                    touch_executor.release()
                    last_msg = "Release requested"
            time.sleep(0.005)
    finally:
        if touch_executor is not None:
            touch_executor.stop()
            touch_executor.release()
        cam.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan B head-camera chessboard and URDF FK test.")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--cam-serial", type=str, default="", help="头部 D435 序列号；为空则按 index 选择")
    parser.add_argument("--camera-name", type=str, default="head_d435")
    parser.add_argument("--camera-backend", choices=("auto", "realsense", "opencv"), default="auto")
    parser.add_argument("--opencv-device", type=str, default="/dev/video0", help="OpenCV/V4L2 fallback color device")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-start-retries", type=int, default=3)
    parser.add_argument("--camera-start-retry-delay", type=float, default=1.5)
    parser.add_argument("--rs-hardware-reset", action="store_true", help="启动前对目标 RealSense 执行 hardware_reset")
    parser.add_argument("--no-rs-reset-on-busy", dest="rs_reset_on_busy", action="store_false", help="相机 busy 时不自动 hardware_reset")
    parser.set_defaults(rs_reset_on_busy=True)
    parser.add_argument("--rs-reset-wait-seconds", type=float, default=6.0)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--camera-matrix-npy", type=str, default="")
    parser.add_argument("--dist-coeffs-npy", type=str, default="")
    parser.add_argument("--camera-json", type=str, default="")
    parser.add_argument("--stream-host", type=str, default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8081)
    parser.add_argument("--stream-jpeg-quality", type=int, default=80)
    parser.add_argument("--fk-network-interface", type=str, default="eth0")
    parser.add_argument("--fk-domain-id", type=int, default=0)
    parser.add_argument("--fk-state-timeout", type=float, default=1.0)
    parser.add_argument("--fk-update-period", type=float, default=0.2)
    parser.add_argument("--fk-urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--fk-base-link", type=str, default="pelvis")
    parser.add_argument("--camera-link", type=str, default="d435_link")
    parser.add_argument("--wrist-link", type=str, default="right_wrist_yaw_link")
    parser.add_argument("--execute", action="store_true", help="启用网页 Touch 运动能力；不会自动触碰")
    parser.add_argument("--confirm-touch-motion", action="store_true", help="二次确认：允许网页 Touch 发送 arm_sdk")
    parser.add_argument("--touch-target-camera-offset-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--touch-tool-offset-wrist-xyz", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--touch-max-move-m", type=float, default=0.08)
    parser.add_argument("--touch-max-joint-delta-rad", type=float, default=0.35)
    parser.add_argument("--touch-allow-position-error-m", type=float, default=0.015)
    parser.add_argument("--touch-ik-max-iterations", type=int, default=250)
    parser.add_argument("--touch-ik-tolerance-position", type=float, default=0.005)
    parser.add_argument("--touch-ik-tolerance-orientation", type=float, default=10.0)
    parser.add_argument("--touch-ik-damping", type=float, default=1e-3)
    parser.add_argument("--touch-ik-step-scale", type=float, default=0.4)
    parser.add_argument("--touch-ik-finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--touch-ik-orientation-weight", type=float, default=0.0)
    parser.add_argument("--touch-kp", type=float, default=60.0)
    parser.add_argument("--touch-kd", type=float, default=3.0)
    parser.add_argument("--touch-ramp-seconds", type=float, default=2.0)
    parser.add_argument("--touch-hold-seconds", type=float, default=1.0)
    parser.add_argument("--touch-control-hz", type=float, default=50.0)
    parser.add_argument("--touch-release-after-touch", action="store_true")
    parser.add_argument("--touch-release-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.cols < 2 or args.rows < 2:
        parser.error("--cols/--rows must be >= 2")
    if args.square_mm <= 0:
        parser.error("--square-mm must be > 0")
    if args.stream_port <= 0:
        parser.error("--stream-port must be > 0")
    if args.camera_start_retries < 1:
        parser.error("--camera-start-retries must be >= 1")
    if args.camera_start_retry_delay < 0:
        parser.error("--camera-start-retry-delay must be >= 0")
    if args.rs_reset_wait_seconds < 0:
        parser.error("--rs-reset-wait-seconds must be >= 0")
    if args.execute and not args.confirm_touch_motion:
        parser.error("--execute requires --confirm-touch-motion")
    if args.touch_max_move_m <= 0 or args.touch_max_joint_delta_rad <= 0:
        parser.error("--touch-max-move-m and --touch-max-joint-delta-rad must be > 0")
    if args.touch_control_hz <= 0:
        parser.error("--touch-control-hz must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
