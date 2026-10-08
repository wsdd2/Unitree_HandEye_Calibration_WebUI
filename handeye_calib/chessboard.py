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


PREVIEW_DETECT_MAX_SIDE = 640
PREVIEW_MISS_PERIOD_S = 0.20
PREVIEW_HIT_PERIOD_S = 0.05
PREVIEW_TILE_COVERAGE = 0.62
PREVIEW_TILE_SCALE = 2.0
PREVIEW_TILE_MIN_EDGE = 1.0


def gray_variants(gray: np.ndarray, gamma: float) -> list[tuple[str, np.ndarray]]:
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.uint8) # Its principle is to adjust the brightness of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # This is used for improving the contrast of the image
    return [
        ("raw", gray),
        ("gamma", cv2.LUT(gray, table)),
        ("equalize", cv2.equalizeHist(gray)),
        ("clahe", clahe.apply(gray)),
    ]


def preview_gray_variants(gray: np.ndarray, gamma: float) -> list[tuple[str, np.ndarray]]:
    variants = [("raw", gray)]
    if abs(gamma - 1.0) > 1e-6:
        table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.uint8)
        variants.append(("gamma", cv2.LUT(gray, table)))
    return variants


def preview_detect_due(last_ts: float, last_hit: bool, now: Optional[float] = None) -> bool:
    """True when the live preview should run another cheap chessboard search."""
    stamp = time.monotonic() if now is None else now
    period = PREVIEW_HIT_PERIOD_S if last_hit else PREVIEW_MISS_PERIOD_S
    return stamp - last_ts >= period


def _downscale_gray(gray: np.ndarray, max_side: int = PREVIEW_DETECT_MAX_SIDE) -> tuple[np.ndarray, float]:
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return gray, 1.0
    scale = max_side / float(longest)
    small = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def _scale_corners(corners: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(corners, dtype=np.float32)
    if scale == 1.0:
        return out
    out = out.copy()
    out[..., :2] /= scale
    return out


def _offset_corners(corners: np.ndarray, origin_xy: tuple[int, int]) -> np.ndarray:
    ox, oy = origin_xy
    if ox == 0 and oy == 0:
        return corners
    out = np.asarray(corners, dtype=np.float32).copy()
    out[..., 0] += float(ox)
    out[..., 1] += float(oy)
    return out


def _preview_tiles(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Overlapping crops so a board sitting in a corner is not shrunk away."""
    tile_h = max(32, int(round(height * PREVIEW_TILE_COVERAGE)))
    tile_w = max(32, int(round(width * PREVIEW_TILE_COVERAGE)))
    if tile_h >= height and tile_w >= width:
        return []
    ys = (0, max(0, height - tile_h))
    xs = (0, max(0, width - tile_w))
    tiles: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            tile = (y0, y0 + tile_h, x0, x0 + tile_w)
            if tile not in tiles:
                tiles.append(tile)
    return tiles


def _mean_abs_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return float(np.mean(np.abs(lap)))


def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    gamma: float = 0.85,
    *,
    mode: str = "full",
) -> tuple[Optional[np.ndarray], str]:
    """Detect inner corners.

    ``mode="preview"`` first tries a downscaled FAST_CHECK so an empty scene
    stays cheap. If that misses, it searches overlapping full-resolution tiles
    enlarged 2x, which is what finds a small board parked in a corner.
    ``mode="full"`` keeps the thorough search for Save / offline.
    """
    kind = (mode or "full").strip().lower()
    if kind not in {"full", "preview"}:
        raise ValueError(f"unsupported chessboard detect mode: {mode}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    classic_flags += int(getattr(cv2, "CALIB_CB_FAST_CHECK", 0))
    if kind == "preview":
        work, scale = _downscale_gray(gray)
        variants = preview_gray_variants(work, gamma)
        use_sb = False
        refine_gray = gray
    else:
        work, scale = gray, 1.0
        variants = gray_variants(work, gamma)
        use_sb = True
        refine_gray = gray
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        sb_flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        sb_flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))

    def accept(corners: np.ndarray, found_scale: float, origin_xy: tuple[int, int], label: str):
        mapped = _offset_corners(_scale_corners(corners, found_scale), origin_xy)
        refined = cv2.cornerSubPix(refine_gray, mapped, (11, 11), (-1, -1), criteria)
        return refined, label

    for variant_name, candidate in variants:
        ok, corners = cv2.findChessboardCorners(candidate, pattern_size, flags=classic_flags)
        if ok and corners is not None:
            return accept(corners, scale, (0, 0), f"{variant_name}/classic")

        if use_sb and hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(candidate, pattern_size, flags=sb_flags)
            if ok and corners is not None:
                return accept(corners, scale, (0, 0), f"{variant_name}/sb")

    if kind != "preview":
        return None, ""

    height, width = gray.shape[:2]
    for y0, y1, x0, x1 in _preview_tiles(height, width):
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0 or _mean_abs_laplacian(crop) < PREVIEW_TILE_MIN_EDGE:
            continue
        enlarged = cv2.resize(
            crop,
            None,
            fx=PREVIEW_TILE_SCALE,
            fy=PREVIEW_TILE_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        ok, corners = cv2.findChessboardCorners(enlarged, pattern_size, flags=classic_flags)
        if ok and corners is not None:
            return accept(corners, PREVIEW_TILE_SCALE, (x0, y0), "tile/classic")

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
    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)): # (+-1, +-1) means the text will be put in the surrounding of the original text
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


def camera_overlay_text(cam) -> tuple[str, dict]:
    info = dict(cam.capture_metadata()) if hasattr(cam, "capture_metadata") else {}
    try:
        devices = RealSenseD435i.list_devices()
    except Exception:
        devices = []
    serial = str(info.get("serial") or getattr(cam, "serial", "") or "").strip()
    info["enumerated"] = devices
    info["device_count"] = len(devices)
    info["device_index"] = next(
        (index for index, item in enumerate(devices) if str(item.get("serial") or "") == serial),
        0,
    )
    for item in devices:
        if str(item.get("serial") or "") == serial:
            info["model"] = item.get("model") or info.get("model") or "RealSense"
            info["serial"] = serial
            break
    info.setdefault("model", "RealSense")
    info.setdefault("serial", serial or "?")
    info["label"] = RealSenseD435i.format_device_label(info)
    slot = f"  ({info['device_index'] + 1}/{info['device_count']})" if info["device_count"] else ""
    return f"{info['label']}{slot}", info


def switch_preview_camera(cam, args: argparse.Namespace):
    if args.camera_backend != "realsense":
        raise RuntimeError("Next Camera 仅支持 RealSense 后端")
    devices = RealSenseD435i.list_devices()
    current = str(getattr(cam, "serial", "") or "").strip()
    nxt = RealSenseD435i.cycle_listed_device(devices, current)
    cam.close()
    args.cam_serial = str(nxt.get("serial") or "")
    args.cam_index = int(nxt.get("index") or 0)
    return open_camera(args)


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
    camera_line: str = "",
) -> None:
    status = "YES" if detected else "NO"
    put_text_bgr_adaptive(vis, camera_line or "camera=?", (10, 30), 0.72)
    put_text_bgr_adaptive(vis, f"task={task} chessboard={status} pattern={pattern_size[0]}x{pattern_size[1]}", (10, 60), 0.62)
    put_text_bgr_adaptive(vis, f"method={detect_method or '-'} saved={saved_count}", (10, 90), 0.62)
    put_text_bgr_adaptive(vis, f"output={output_dir}", (10, 120), 0.55)
    put_text_bgr_adaptive(vis, "SPACE=save  Next Camera on web  ESC/q=quit", (10, 150), 0.58)
    if last_msg and time.monotonic() - last_msg_ts < 3.0:
        put_text_bgr_adaptive(vis, last_msg, (10, 180), 0.62)


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
    last_detect_ts = 0.0
    last_detected = False
    last_detect_method = "none"
    last_corners = None
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
            now = time.monotonic()
            if preview_detect_due(last_detect_ts, last_detected, now):
                corners, detect_method = find_chessboard_corners(
                    gray,
                    pattern_size,
                    args.gamma,
                    mode="preview",
                )
                detected = corners is not None
                last_detect_ts = now
                last_detected = detected
                last_detect_method = detect_method
                last_corners = corners
            else:
                corners = last_corners
                detect_method = last_detect_method
                detected = last_detected

            vis = preview.copy()
            if detected:
                cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            camera_line, camera_info = camera_overlay_text(cam)
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
                camera_line=camera_line,
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
                        "camera": camera_info,
                    }
                )

            key = (cv2.waitKey(1) & 0xFF) if not args.headless else 255
            web_command = DebugStreamServer.command_name(
                stream_server.pop_command() if stream_server is not None else None
            )
            if web_command == "next_camera" or key in (ord("n"), ord("N")):
                try:
                    cam = switch_preview_camera(cam, args)
                    last_msg = f"switched to {camera_overlay_text(cam)[0]}"
                    print(f"[CAMERA] {last_msg}")
                except Exception as exc:
                    last_msg = f"next camera failed: {exc}"
                    print(f"[CAMERA][ERROR] {exc}", file=sys.stderr)
                last_msg_ts = time.monotonic()
                continue
            if key in (27, ord("q"), ord("Q")) or web_command == "quit":
                break
            save_requested = key == ord(" ") or web_command == "save"
            if not save_requested:
                continue
            if not detected:
                corners, detect_method = find_chessboard_corners(
                    gray,
                    pattern_size,
                    args.gamma,
                    mode="full",
                )
                detected = corners is not None
                last_detected = detected
                last_detect_method = detect_method
                last_corners = corners
                last_detect_ts = time.monotonic()
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
