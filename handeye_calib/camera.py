# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import re

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local hardware package
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


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
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2，请运行: pip install pyrealsense2") from _IMPORT_ERROR
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
        for i, dev in enumerate(rs.context().query_devices()):
            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"
            firmware = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else ""
            product_line = dev.get_info(rs.camera_info.product_line) if dev.supports(rs.camera_info.product_line) else ""
            devices.append({"index": i, "serial": serial, "model": name, "firmware": firmware, "product_line": product_line})
        return devices

    @staticmethod
    def set_emitter(active_index: Optional[int], active_serial: str = "") -> None:
        """Enable the emitter for one camera and disable it for the others."""
        if rs is None:
            return
        active_serial = active_serial.strip()
        for idx, dev in enumerate(rs.context().query_devices()):
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

    def fetch(self, timeout_ms: int = 3000) -> Optional[dict[str, np.ndarray]]: # 获取深度图和彩色图
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseD435i not started; call start() first")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
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
        return {"rgb": np.ascontiguousarray(rgb), "depth": np.ascontiguousarray(depth_mm)}

    def color_intrinsics(self) -> tuple[np.ndarray, np.ndarray, dict]: # 获取相机内参
        if self._profile is None:
            raise RuntimeError("RealSense pipeline 尚未启动，无法读取内参")
        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
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

    def fetch(self, timeout_ms: int = 3000) -> Optional[dict[str, np.ndarray]]:
        del timeout_ms
        if self._cap is None:
            raise RuntimeError("OpenCVVideoCamera not opened; call open() first")
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        return {"rgb": np.ascontiguousarray(rgb), "depth": depth}
