# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import re
import sys
import time
from pathlib import Path

import numpy as np
import cv2


def _is_usable_rs(module) -> bool:
    return bool(module) and (
        (hasattr(module, "context") or hasattr(module, "Context"))
        and hasattr(module, "pipeline")
    )


def _purge_rs_modules() -> None:
    for name in list(sys.modules):
        if name == "pyrealsense2" or name.startswith("pyrealsense2."):
            del sys.modules[name]


def _user_site_path() -> Optional[str]:
    try:
        import site

        user = site.getusersitepackages()
    except Exception:
        return None
    return user if user else None


def _strip_user_site() -> Optional[str]:
    """Temporarily drop ~/.local so pip's empty pyrealsense2 stub cannot win."""
    user = _user_site_path()
    if not user:
        return None
    sys.path[:] = [item for item in sys.path if item != user]
    return user


def _restore_user_site(user: Optional[str]) -> None:
    """Put user site back at the end so Flask etc. still import."""
    if user and user not in sys.path:
        sys.path.append(user)


def _bind_search_dirs() -> list[Path]:
    home = Path.home()
    dirs: list[Path] = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/lib/python3/dist-packages/pyrealsense2"),
        Path("/usr/local/lib/python3/dist-packages"),
        home / "librealsense" / "build" / "wrappers" / "python",
        home / "librealsense" / "build" / "Release",
        Path("/opt/realsense/lib"),
    ]
    for root in (
        Path("/usr/lib"),
        Path("/usr/local/lib"),
        home / "anaconda3",
        home / "miniconda3",
        Path("/opt"),
    ):
        if not root.exists():
            continue
        dirs.extend(sorted(root.glob("python3.*/dist-packages")))
        dirs.extend(sorted(root.glob("python3.*/site-packages")))
        dirs.extend(sorted(root.glob("envs/*/lib/python*/site-packages")))
    try:
        import shutil

        binary = shutil.which("rs-enumerate-devices")
        if binary:
            prefix = Path(binary).resolve().parent.parent
            dirs.append(prefix / "lib" / "python3" / "dist-packages")
            dirs.extend(sorted((prefix / "lib").glob("python3.*/dist-packages")))
            dirs.extend(sorted((prefix / "lib").glob("python3.*/site-packages")))
    except Exception:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        folder = item.parent if item.suffix == ".so" else item
        key = str(folder)
        if key in seen or not folder.is_dir():
            continue
        if not (folder / "pyrealsense2").exists() and not list(folder.glob("pyrealsense2*.so")):
            continue
        seen.add(key)
        out.append(folder)
    return out


def _load_pyrealsense2():
    """Skip pip stubs; prefer librealsense bindings that actually have pipeline."""
    user = _strip_user_site()
    try:
        try:
            import pyrealsense2 as module
        except ImportError as exc:
            module, first_error = None, exc
        else:
            first_error = None
            if _is_usable_rs(module):
                return module, None
            try:
                import pyrealsense2.pyrealsense2 as inner
            except ImportError:
                inner = None
            if _is_usable_rs(inner):
                return inner, None
        for folder in _bind_search_dirs():
            try:
                path = str(folder)
                if path in sys.path:
                    sys.path.remove(path)
                sys.path.insert(0, path)
                _purge_rs_modules()
                import pyrealsense2 as candidate
            except Exception:
                continue
            if _is_usable_rs(candidate):
                print(f"[CAMERA] using pyrealsense2 from {folder}")
                return candidate, None
            try:
                import pyrealsense2.pyrealsense2 as inner
            except Exception:
                continue
            if _is_usable_rs(inner):
                print(f"[CAMERA] using pyrealsense2.pyrealsense2 from {folder}")
                return inner, None
        return None, first_error or RuntimeError(
            "当前 python 里的 pyrealsense2 是空壳（没有 context/pipeline）。"
            "不要 pip install pyrealsense2。请在机上找 rs-enumerate-devices 或 pyrealsense2*.so，"
            "把其所在目录加入 PYTHONPATH。"
        )
    finally:
        _restore_user_site(user)


rs, _IMPORT_ERROR = _load_pyrealsense2()


def _rs_context():
    if not _is_usable_rs(rs):
        raise RuntimeError(
            "未找到可用的 librealsense Python 绑定。"
            "在 H2 上执行: which rs-enumerate-devices; find /usr /opt /home/unitree -name 'pyrealsense2*.so'"
            f" 错误={_IMPORT_ERROR}"
        ) from _IMPORT_ERROR
    factory = getattr(rs, "context", None) or getattr(rs, "Context", None)
    return factory()


class RealSenseD435i:
    """RealSense D435i color/depth capture wrapper for calibration."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial: str = "",
        camera_name: str = "",
        mount: str = "",
        color_only: bool = False,
    ) -> None:
        if not _is_usable_rs(rs):
            raise RuntimeError(
                "未找到可用的 librealsense Python 绑定（当前 pyrealsense2 是空壳）。"
                "不要 pip install pyrealsense2。"
                f" 错误={_IMPORT_ERROR}"
            ) from _IMPORT_ERROR
        self.index = index
        self.serial = serial.strip()
        self.camera_name = camera_name.strip()
        self.mount = mount.strip()
        self.color_only = bool(color_only)
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._profile = None
        self._started = False
        self.depth_scale = 1.0

    @staticmethod
    def list_devices() -> list[dict]:
        if rs is None:
            return []
        devices = []
        for i, dev in enumerate(_rs_context().query_devices()):
            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"
            firmware = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else ""
            product_line = dev.get_info(rs.camera_info.product_line) if dev.supports(rs.camera_info.product_line) else ""
            devices.append({"index": i, "serial": serial, "model": name, "firmware": firmware, "product_line": product_line})
        return devices

    @staticmethod
    def cycle_listed_device(
        devices: list[dict],
        current_serial: str = "",
    ) -> dict:
        """Pick the next RealSense in the enumerated list, wrapping around."""
        if not devices:
            raise RuntimeError("没有枚举到 RealSense 相机")
        serials = [str(item.get("serial") or "").strip() for item in devices]
        current = str(current_serial or "").strip()
        if current in serials:
            nxt = (serials.index(current) + 1) % len(serials)
        else:
            nxt = 0
        return dict(devices[nxt])

    @staticmethod
    def format_device_label(device: dict | None) -> str:
        info = device or {}
        model = str(info.get("model") or "RealSense")
        serial = str(info.get("serial") or info.get("selected_serial") or "?")
        return f"{model}  SN={serial}"

    @staticmethod
    def set_emitter(active_index: Optional[int], active_serial: str = "") -> None:
        """Enable the emitter for one camera and disable it for the others."""
        if rs is None:
            return
        active_serial = active_serial.strip()
        for idx, dev in enumerate(_rs_context().query_devices()):
            try:
                serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
                is_active = bool(active_serial and serial == active_serial) or (not active_serial and active_index == idx)
                sensor = dev.first_depth_sensor()
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, 1.0 if is_active else 0.0)
            except Exception:
                pass

    def selected_device_info(self) -> dict:
        devices = self.list_devices()
        if self.serial:
            for dev in devices:
                if dev.get("serial") == self.serial:
                    return dict(dev)
            return {"index": self.index, "serial": self.serial}
        if devices and self.index < len(devices):
            return dict(devices[self.index])
        return {"index": self.index}

    def capture_metadata(self) -> dict:
        info = self.selected_device_info()
        info.update(
            {
                "camera_name": self.camera_name,
                "mount": self.mount,
                "requested_index": int(self.index),
                "requested_serial": self.serial,
                "width": int(self.width),
                "height": int(self.height),
                "fps": int(self.fps),
                "color_only": self.color_only,
            }
        )
        return info

    def open(self) -> None:
        if self._pipeline is not None:
            return
        self._config = rs.config()
        device = self.selected_device_info()
        serial = str(device.get("serial") or "")
        if serial:
            self._config.enable_device(serial)
        if not self.color_only:
            self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self._pipeline = rs.pipeline()
        self._align = None if self.color_only else rs.align(rs.stream.color)

    def start(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("RealSenseD435i not opened; call open() first")
        if self._started:
            return
        self._profile = self._pipeline.start(self._config)
        if not self.color_only:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
        self._started = True

    def stop(self) -> None:
        if not self._started or self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        finally:
            self._started = False
            self._profile = None

    def close(self) -> None:
        if self._started:
            self.stop()
        self._pipeline = None
        self._config = None
        self._align = None

    def fetch(self, timeout_ms: int = 3000) -> Optional[dict[str, object]]: # 获取深度图和彩色图
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseD435i not started; call start() first")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            host_monotonic_sec = time.monotonic()
        except RuntimeError:
            return None
        if frames is None:
            return None
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        if self.color_only:
            depth_mm = np.zeros(rgb.shape[:2], dtype=np.uint16)
        else:
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return None
            raw_depth = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            depth_mm = raw_depth.astype(np.float32) * self.depth_scale * 1000.0
            depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth": np.ascontiguousarray(depth_mm),
            "host_monotonic_sec": host_monotonic_sec,
            "frame_timestamp_ms": float(color_frame.get_timestamp()),
            "frame_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
            "frame_number": int(color_frame.get_frame_number()),
        }

    def color_intrinsics(self) -> tuple[np.ndarray, np.ndarray, dict]: # 获取相机内参
        if self._profile is None:
            raise RuntimeError("RealSense pipeline 尚未启动，无法读取内参")
        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        # D435/D435i color is factory-tagged inverse_brown_conrady.
        # Same 5 Brown-Conrady coeffs; OpenCV undistort/solvePnP is the inverse map.
        allowed_models = {
            model
            for model in (
                getattr(rs.distortion, "none", None),
                getattr(rs.distortion, "brown_conrady", None),
                getattr(rs.distortion, "modified_brown_conrady", None),
                getattr(rs.distortion, "inverse_brown_conrady", None),
            )
            if model is not None
        }
        if intr.model not in allowed_models:
            allowed_names = "none, brown_conrady, modified_brown_conrady, inverse_brown_conrady"
            raise RuntimeError(
                f"RealSense color distortion model {intr.model!s} is not OpenCV-compatible; "
                f"supported models: {allowed_names}"
            )
        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coeffs = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
        info = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "distortion_model": str(intr.model),
            "coeffs": coeffs.reshape(-1).astype(float).tolist(),
            "depth_scale": float(self.depth_scale),
        }
        return camera_matrix, coeffs, info

    def __enter__(self) -> "RealSenseD435i":
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class OpenCVVideoCamera:
    """Minimal V4L2/OpenCV color camera wrapper for chessboard capture."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if self._cap is not None:
            return
        device_id: int | str
        device_text = str(self.device)
        match = re.fullmatch(r"/dev/video(\d+)", device_text)
        if match:
            device_id = int(match.group(1))
        else:
            device_id = int(device_text) if device_text.isdigit() else device_text
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Failed to open OpenCV camera device: {self.device}")

    def start(self) -> None:
        if self._cap is None:
            self.open()

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def close(self) -> None:
        self.stop()

    def capture_metadata(self) -> dict:
        return {
            "backend": "opencv_v4l2",
            "device": self.device,
            "width": int(self.width),
            "height": int(self.height),
            "fps": int(self.fps),
        }

    def fetch(self, timeout_ms: int = 3000) -> Optional[dict[str, object]]:
        del timeout_ms
        if self._cap is None:
            raise RuntimeError("OpenCVVideoCamera not opened; call open() first")
        ok, frame_bgr = self._cap.read()
        host_monotonic_sec = time.monotonic()
        if not ok or frame_bgr is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth": depth,
            "host_monotonic_sec": host_monotonic_sec,
        }
