# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from handeye_calib.camera import OpenCVVideoCamera, RealSenseD435i
    from handeye_calib.debug_stream import DebugStreamServer
except ModuleNotFoundError:  # Allows: python handeye_calib/chessboard.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from handeye_calib.camera import OpenCVVideoCamera, RealSenseD435i
    from handeye_calib.debug_stream import DebugStreamServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRS = {
    "camera_calib": PROJECT_ROOT / "camera_calib_image",
    "handeye": PROJECT_ROOT / "handeye_img",
}


def gamma_correct_bgr(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) <= 1e-6:
        return image_bgr
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image_bgr, table)


def gray_variants(gray: np.ndarray, gamma: float) -> list[tuple[str, np.ndarray]]:
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return [
        ("raw", gray),
        ("gamma", cv2.LUT(gray, table)),
        ("equalize", cv2.equalizeHist(gray)),
        ("clahe", clahe.apply(gray)),
    ]


def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    gamma: float = 0.85,
) -> tuple[Optional[np.ndarray], str]:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    sb_flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
    sb_flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))

    for variant_name, candidate in gray_variants(gray, gamma):
        ok, corners = cv2.findChessboardCorners(candidate, pattern_size, flags=classic_flags)
        if ok and corners is not None:
            corners = corners.astype(np.float32)
            refined = cv2.cornerSubPix(candidate, corners, (11, 11), (-1, -1), criteria)
            return refined, f"{variant_name}/classic"

        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(candidate, pattern_size, flags=sb_flags)
            if ok and corners is not None:
                corners = corners.astype(np.float32)
                refined = cv2.cornerSubPix(candidate, corners, (11, 11), (-1, -1), criteria)
                return refined, f"{variant_name}/sb"

    return None, ""


def put_text_bgr_adaptive(
    vis: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float = 0.65,
    thickness: int = 2,
    sample_w: int = 920,
) -> None:
    h, w = vis.shape[:2]
    ox, oy = int(org[0]), int(org[1])
    x0 = max(0, min(w - 1, ox))
    x1 = max(x0 + 1, min(w, ox + max(48, sample_w)))
    y0 = max(0, min(h - 1, oy - 26))
    y1 = max(y0 + 1, min(h, oy + 6))
    roi = vis[y0:y1, x0:x1]
    if roi.size == 0:
        fg, edge = (250, 250, 250), (0, 0, 0)
    else:
        b, g, r = roi.reshape(-1, 3).astype(np.float32).mean(axis=0)
        lum = float(0.114 * b + 0.587 * g + 0.299 * r)
        fg, edge = ((28, 28, 28), (255, 255, 255)) if lum >= 130.0 else ((250, 250, 250), (0, 0, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = max(2, thickness + 2)
    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(vis, text, (ox + du, oy + dv), font, font_scale, edge, outline, cv2.LINE_AA)
    cv2.putText(vis, text, (ox, oy), font, font_scale, fg, thickness, cv2.LINE_AA)


def output_dir_for_task(task: str, output_dir: str = "") -> Path:
    if output_dir:
        return Path(output_dir)
    return DEFAULT_OUTPUT_DIRS[task]


def save_chessboard_image(image_bgr: np.ndarray, output_dir: Path, task: str, cam_index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"{stamp}_cam{cam_index}_{task}.jpg"
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")
    return path


def draw_preview_overlay(
    vis: np.ndarray,
    *,
    task: str,
    pattern_size: tuple[int, int],
    detected: bool,
    detect_method: str,
    saved_count: int,
    output_dir: Path,
    last_msg: str,
    last_msg_ts: float,
) -> None:
    status = "YES" if detected else "NO"
    put_text_bgr_adaptive(vis, f"task={task} chessboard={status} pattern={pattern_size[0]}x{pattern_size[1]}", (10, 30), 0.72)
    put_text_bgr_adaptive(vis, f"method={detect_method or '-'} saved={saved_count}", (10, 60), 0.62)
    put_text_bgr_adaptive(vis, f"output={output_dir}", (10, 90), 0.55)
    put_text_bgr_adaptive(vis, "SPACE=save detected image  ESC/q=quit", (10, 120), 0.62)
    if last_msg and time.monotonic() - last_msg_ts < 3.0:
        put_text_bgr_adaptive(vis, last_msg, (10, 150), 0.62)


def open_camera(args: argparse.Namespace):
    if args.camera_backend == "opencv":
        device = resolve_opencv_video_device(args)
        cam = OpenCVVideoCamera(
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        cam.open()
        cam.start()
        return cam

    RealSenseD435i.set_emitter(args.cam_index if args.enable_emitter else None, args.cam_serial)
    cam = RealSenseD435i(
        index=args.cam_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.cam_serial,
        color_only=args.color_only,
    )
    cam.open()
    cam.start()
    return cam


def resolve_opencv_video_device(args: argparse.Namespace) -> str:
    if args.cam_serial:
        by_id_dir = Path("/dev/v4l/by-id")
        if by_id_dir.exists():
            matches = sorted(
                path
                for path in by_id_dir.iterdir()
                if args.cam_serial in path.name and f"video-index{args.video_index}" in path.name
            )
            if matches:
                resolved = str(matches[0].resolve())
                print(f"[CAMERA] serial={args.cam_serial} video-index={args.video_index} -> {resolved}")
                return resolved
        print(
            f"[CAMERA][WARN] 未在 /dev/v4l/by-id 中找到 serial={args.cam_serial} "
            f"video-index={args.video_index}，回退到 --video-device={args.video_device}",
            file=sys.stderr,
        )
    return args.video_device


def run_preview(args: argparse.Namespace) -> int:
    pattern_size = (args.cols, args.rows)
    output_dir = output_dir_for_task(args.task, args.output_dir)
    saved_count = len(list(output_dir.glob("*.jpg"))) if output_dir.exists() else 0
    last_msg = ""
    last_msg_ts = 0.0
    stream_server = None
    if args.stream_debug:
        stream_server = DebugStreamServer(
            host=args.stream_host,
            port=args.stream_port,
            jpeg_quality=args.stream_jpeg_quality,
        )
        stream_server.start()
        print(f"[STREAM] http://{args.stream_host}:{args.stream_port}")

    cam = open_camera(args)
    try:
        win = f"Chessboard Capture {args.task} cam{args.cam_index}"
        if not args.headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, args.width, args.height)
        print(f"[TASK] {args.task}")
        print(f"[PATTERN] {args.cols}x{args.rows}")
        print(f"[OUTPUT] {output_dir.resolve()}")
        print("[KEYS] SPACE=save detected image, ESC/q=quit")

        while True:
            frame = cam.fetch(timeout_ms=args.timeout_ms)
            if frame is None or frame.get("rgb") is None:
                continue

            frame_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            preview = gamma_correct_bgr(frame_bgr, args.gamma)
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            corners, detect_method = find_chessboard_corners(gray, pattern_size, args.gamma)
            detected = corners is not None

            vis = preview.copy()
            if detected:
                cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            draw_preview_overlay(
                vis,
                task=args.task,
                pattern_size=pattern_size,
                detected=detected,
                detect_method=detect_method,
                saved_count=saved_count,
                output_dir=output_dir,
                last_msg=last_msg,
                last_msg_ts=last_msg_ts,
            )
            if not args.headless:
                cv2.imshow(win, vis)
            if stream_server is not None:
                stream_server.update_frame(vis)
                stream_server.update_state(
                    {
                        "task": args.task,
                        "pattern": {"cols": args.cols, "rows": args.rows},
                        "chessboard_detected": detected,
                        "detect_method": detect_method,
                        "saved_count": saved_count,
                        "output_dir": str(output_dir.resolve()),
                        "last_message": last_msg,
                    }
                )

            key = (cv2.waitKey(1) & 0xFF) if not args.headless else 255
            web_command = DebugStreamServer.command_name(
                stream_server.pop_command() if stream_server is not None else None
            )
            if key in (27, ord("q"), ord("Q")) or web_command == "quit":
                break
            save_requested = key == ord(" ") or web_command == "save"
            if not save_requested:
                continue
            if not detected and not args.save_without_board:
                last_msg = "save rejected: no chessboard"
                last_msg_ts = time.monotonic()
                print("[SAVE] 拒绝：当前画面未检测到棋盘格")
                continue

            try:
                image_path = save_chessboard_image(frame_bgr, output_dir, args.task, args.cam_index)
            except Exception as exc:
                last_msg = f"save failed: {exc}"
                last_msg_ts = time.monotonic()
                print(f"[SAVE][ERROR] {exc}", file=sys.stderr)
                continue
            saved_count += 1
            last_msg = f"saved: {image_path.name}"
            last_msg_ts = time.monotonic()
            print(f"[SAVE] {image_path.resolve()}")
    finally:
        cam.close()
        RealSenseD435i.set_emitter(None)
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense 棋盘格视频预览与采图工具")
    parser.add_argument("--task", choices=("camera_calib", "handeye"), default="camera_calib", help="任务类型决定默认保存路径")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--cam-serial", type=str, default="", help="可选：按 RealSense 序列号选择相机")
    parser.add_argument("--camera-backend", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--video-device", type=str, default="/dev/video0", help="OpenCV/V4L2 后端使用的视频设备")
    parser.add_argument("--video-index", type=int, default=0, help="OpenCV/V4L2 后端按 --cam-serial 查找 /dev/v4l/by-id 时使用的 video-index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--cols", type=int, default=11, help="棋盘横向内角点数")
    parser.add_argument("--rows", type=int, default=8, help="棋盘纵向内角点数")
    parser.add_argument("--gamma", type=float, default=0.85, help="预览/检测 gamma")
    parser.add_argument("--output-dir", type=str, default="", help="可选：覆盖任务类型对应的默认保存路径")
    parser.add_argument("--enable-emitter", action="store_true", help="开启当前 D435i 深度发射器；默认关闭以减少棋盘反光")
    parser.add_argument("--color-only", action="store_true", help="只打开彩色流，不打开深度流；适合相机内参标定")
    parser.add_argument("--save-without-board", action="store_true", help="允许未检测到棋盘格时也保存图片")
    parser.add_argument("--stream-debug", action="store_true", help="开启网页调试流，显示棋盘格 OpenCV 画面和采图状态")
    parser.add_argument("--stream-host", type=str, default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--stream-jpeg-quality", type=int, default=80)
    parser.add_argument("--headless", action="store_true", help="不创建本地 OpenCV 窗口；配合 --stream-debug 在网页端操作")
    args = parser.parse_args()
    if args.cols < 2 or args.rows < 2:
        parser.error("--cols/--rows 必须 >= 2")
    if args.gamma <= 0:
        parser.error("--gamma 必须 > 0")
    if args.stream_port <= 0:
        parser.error("--stream-port 必须 > 0")
    if args.headless and not args.stream_debug:
        parser.error("--headless 需要同时指定 --stream-debug，否则无法操作保存/退出")
    return args


if __name__ == "__main__":
    sys.exit(run_preview(parse_args()))
