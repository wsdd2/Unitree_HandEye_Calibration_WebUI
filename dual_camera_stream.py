#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve two RealSense camera streams with web start/release controls.

This tool is intentionally separate from ``capture_handeye.py`` so it can be
used as a lightweight camera monitor. Stopping a camera from the web page calls
``pipeline.stop()`` and releases the device for other programs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, make_response, request

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.camera import RealSenseD435i  # noqa: E402


DEFAULT_WRIST_CAM_SERIAL = ""
DEFAULT_HEAD_CAM_SERIAL = ""


def now_s() -> float:
    return time.time()


def bgr_placeholder(width: int, height: int, text: str) -> np.ndarray:
    frame = np.zeros((max(120, height), max(240, width), 3), dtype=np.uint8)
    frame[:] = (24, 24, 24)
    cv2.putText(
        frame,
        text,
        (24, max(48, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return frame


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._") or "camera"


def timestamp_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass
class CameraSpec:
    key: str
    label: str
    serial: str
    index: int


class CameraWorker:
    """Own one RealSense pipeline and the latest JPEG frame."""

    def __init__(
        self,
        spec: CameraSpec,
        *,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        color_only: bool,
        output_dir: Path,
        autostart: bool,
    ) -> None:
        self.spec = spec
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.color_only = bool(color_only)
        self.output_dir = output_dir
        self._lock = threading.RLock()
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._camera: Optional[RealSenseD435i] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_bgr: Optional[np.ndarray] = None
        self._frame_count = 0
        self._last_frame_at = 0.0
        self._last_error = ""
        self._state = "released"
        self._metadata: dict[str, Any] = {}
        self._started_at = 0.0
        self._last_command_at = 0.0
        self._recording_state = "idle"
        self._recording_path = ""
        self._recording_started_at = 0.0
        self._recording_frames = 0
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._set_placeholder("released")
        if autostart:
            self.start()

    def _set_placeholder(self, text: str) -> None:
        self._update_frame(bgr_placeholder(self.width, self.height, f"{self.spec.label}: {text}"))

    def _update_frame(self, frame_bgr: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._latest_bgr = frame_bgr
            self._latest_jpeg = encoded.tobytes()
            self._last_frame_at = now_s()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._state in {"starting", "running"}:
                return self.status()
            self._state = "starting"
            self._last_error = ""
            self._last_command_at = now_s()
            self._stop_event.clear()
            self._set_placeholder("starting")
            self._capture_thread = threading.Thread(
                target=self._run_capture,
                name=f"camera-{self.spec.key}",
                daemon=True,
            )
            self._capture_thread.start()
            return self.status()

    def release(self) -> dict[str, Any]:
        thread: Optional[threading.Thread]
        with self._lock:
            self._last_command_at = now_s()
            self._stop_recording_locked()
            self._stop_event.set()
            thread = self._capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            self._close_camera_locked()
            self._state = "released"
            self._capture_thread = None
            self._set_placeholder("released")
            return self.status()

    def shutdown(self) -> None:
        self.release()

    def _close_camera_locked(self) -> None:
        self._stop_recording_locked()
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as exc:
                self._last_error = str(exc)
            self._camera = None

    def _run_capture(self) -> None:
        camera: Optional[RealSenseD435i] = None
        try:
            serial = self._resolve_serial()
            camera = RealSenseD435i(
                index=self.spec.index,
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=serial,
                camera_name=self.spec.label,
                color_only=self.color_only,
            )
            camera.open()
            camera.start()
            metadata = camera.capture_metadata()
            with self._lock:
                self._camera = camera
                self._metadata = metadata
                self._state = "running"
                self._started_at = now_s()
            while not self._stop_event.is_set():
                frame = camera.fetch(timeout_ms=1000)
                if frame is None:
                    continue
                rgb = frame["rgb"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._draw_overlay(bgr)
                with self._lock:
                    self._write_recording_frame_locked(bgr)
                self._update_frame(bgr)
                with self._lock:
                    self._frame_count += 1
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = "error"
                self._set_placeholder(f"error: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            with self._lock:
                if self._camera is camera:
                    self._camera = None
                if self._state == "running":
                    self._state = "released"

    def _resolve_serial(self) -> str:
        requested = self.spec.serial.strip()
        devices = RealSenseD435i.list_devices()
        if requested:
            if any(str(dev.get("serial", "")) == requested for dev in devices):
                return requested
            with self._lock:
                self._last_error = (
                    f"requested serial {requested} is not connected; "
                    f"falling back to index {self.spec.index}"
                )
        if 0 <= self.spec.index < len(devices):
            return str(devices[self.spec.index].get("serial", "") or "")
        return requested

    def _draw_overlay(self, bgr: np.ndarray) -> None:
        text = f"{self.spec.label} serial={self.spec.serial or 'index ' + str(self.spec.index)}"
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def photo(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "running":
                raise RuntimeError(f"{self.spec.label} is not running")
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
        if frame is None:
            raise RuntimeError(f"{self.spec.label} has no frame to save")
        path = self.output_dir / "photos" / f"{timestamp_token()}_{safe_token(self.spec.label)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"failed to save photo: {path}")
        return {"saved_path": str(path), "camera": self.spec.key, "label": self.spec.label}

    def record_start(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "running":
                raise RuntimeError(f"{self.spec.label} is not running")
            if self._recording_state == "recording":
                return self.recording_status()
            if self._recording_state == "paused" and self._video_writer is not None:
                self._recording_state = "recording"
                return self.recording_status()
            frame = self._latest_bgr
            if frame is None:
                raise RuntimeError(f"{self.spec.label} has no frame yet; start camera first")
            h, w = frame.shape[:2]
            path = self.output_dir / "videos" / f"{timestamp_token()}_{safe_token(self.spec.label)}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.fps),
                (int(w), int(h)),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {path}")
            self._video_writer = writer
            self._recording_path = str(path)
            self._recording_frames = 0
            self._recording_started_at = now_s()
            self._recording_state = "recording"
            return self.recording_status()

    def record_pause(self) -> dict[str, Any]:
        with self._lock:
            if self._recording_state == "recording":
                self._recording_state = "paused"
            return self.recording_status()

    def record_stop(self) -> dict[str, Any]:
        with self._lock:
            return self._stop_recording_locked()

    def _write_recording_frame_locked(self, frame_bgr: np.ndarray) -> None:
        if self._recording_state != "recording" or self._video_writer is None:
            return
        self._video_writer.write(frame_bgr)
        self._recording_frames += 1

    def _stop_recording_locked(self) -> dict[str, Any]:
        saved_path = self._recording_path
        frames = self._recording_frames
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            finally:
                self._video_writer = None
        self._recording_state = "idle"
        self._recording_path = ""
        self._recording_frames = 0
        self._recording_started_at = 0.0
        return {
            "state": "idle",
            "saved_path": saved_path,
            "frames": frames,
            "camera": self.spec.key,
            "label": self.spec.label,
        }

    def recording_status(self) -> dict[str, Any]:
        return {
            "state": self._recording_state,
            "path": self._recording_path,
            "frames": self._recording_frames,
            "duration_s": 0.0 if self._recording_started_at <= 0 else now_s() - self._recording_started_at,
            "camera": self.spec.key,
            "label": self.spec.label,
        }

    def jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        with self._lock:
            fps_est = 0.0
            if self._started_at > 0 and self._frame_count > 0:
                fps_est = self._frame_count / max(0.001, now_s() - self._started_at)
            return {
                "key": self.spec.key,
                "label": self.spec.label,
                "serial": self.spec.serial,
                "index": self.spec.index,
                "state": self._state,
                "frame_count": self._frame_count,
                "fps_est": fps_est,
                "last_frame_age_s": None if self._last_frame_at <= 0 else now_s() - self._last_frame_at,
                "last_error": self._last_error,
                "metadata": self._metadata,
                "last_command_at": self._last_command_at,
                "recording": self.recording_status(),
            }


class DualCameraStreamServer:
    def __init__(self, workers: dict[str, CameraWorker], *, host: str, port: int, output_dir: Path) -> None:
        self.workers = workers
        self.host = host
        self.port = int(port)
        self.output_dir = output_dir
        self.app = Flask(__name__)
        self._configure_routes()

    def _configure_routes(self) -> None:
        @self.app.get("/")
        def index() -> Response:
            response = make_response(self._html())
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response

        @self.app.get("/stream/<key>")
        def stream(key: str) -> Response:
            worker = self.workers.get(key)
            if worker is None:
                return Response(f"unknown camera: {key}", status=404)
            return Response(
                self._mjpeg_frames(worker),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.get("/state")
        def state() -> Response:
            return jsonify(self.state())

        @self.app.post("/camera/all/<action>")
        def all_action(action: str) -> Response:
            if action == "start":
                return jsonify({"ok": True, "cameras": {k: w.start() for k, w in self.workers.items()}})
            if action in {"release", "stop"}:
                return jsonify({"ok": True, "cameras": {k: w.release() for k, w in self.workers.items()}})
            return jsonify({"ok": False, "error": f"unsupported action: {action}"}), 400

        @self.app.post("/camera/<key>/<action>")
        def camera_action(key: str, action: str) -> Response:
            worker = self.workers.get(key)
            if worker is None:
                return jsonify({"ok": False, "error": f"unknown camera: {key}"}), 404
            try:
                if action == "start":
                    return jsonify({"ok": True, "camera": worker.start()})
                if action in {"release", "stop"}:
                    return jsonify({"ok": True, "camera": worker.release()})
                if action == "photo":
                    return jsonify({"ok": True, "photo": worker.photo()})
                if action == "record_start":
                    return jsonify({"ok": True, "recording": worker.record_start()})
                if action == "record_pause":
                    return jsonify({"ok": True, "recording": worker.record_pause()})
                if action == "record_stop":
                    return jsonify({"ok": True, "recording": worker.record_stop()})
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500
            return jsonify({"ok": False, "error": f"unsupported action: {action}"}), 400

        @self.app.get("/devices")
        def devices() -> Response:
            return jsonify({"devices": RealSenseD435i.list_devices()})

        @self.app.post("/quit")
        def quit_server() -> Response:
            def delayed_exit() -> None:
                time.sleep(0.2)
                for worker in self.workers.values():
                    worker.shutdown()
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=delayed_exit, daemon=True).start()
            return jsonify({"ok": True, "message": "server exiting"})

        @self.app.get("/quit-info")
        def quit_info() -> Response:
            host = request.host.split(":")[0]
            remote_dir = str(self.output_dir)
            return jsonify(
                {
                    "output_dir": remote_dir,
                    "scp": f"scp -r unitree@{host}:{remote_dir} ./dual_camera_stream_output/",
                    "tar": f"ssh unitree@{host} 'tar -C {remote_dir} -czf - .' > dual_camera_stream_output.tgz",
                    "message": "Run one of these commands on the computer that should receive the photos/videos.",
                }
            )

    def state(self) -> dict[str, Any]:
        return {
            "updated_at": now_s(),
            "cameras": {key: worker.status() for key, worker in self.workers.items()},
            "devices": RealSenseD435i.list_devices(),
            "output_dir": str(self.output_dir),
        }

    def run(self) -> None:
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

    @staticmethod
    def _mjpeg_frames(worker: CameraWorker):
        while True:
            jpeg = worker.jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.01)

    @staticmethod
    def _html() -> str:
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dual Camera Stream</title>
  <meta http-equiv="Cache-Control" content="no-store" />
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #101010; color: #eee; }
    header { height: 52px; display: flex; align-items: center; gap: 10px; padding: 0 14px; background: #181818; border-bottom: 1px solid #333; }
    h1 { font-size: 18px; margin: 0; font-weight: 600; }
    main { display: grid; grid-template-columns: 1fr 1fr 360px; gap: 0; height: calc(100vh - 52px); }
    .cam { position: relative; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid #333; }
    .cam-title { height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 10px; background: #202020; }
    .video { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; background: #000; }
    .video img { max-width: 100%; max-height: 100%; object-fit: contain; }
    aside { overflow: auto; padding: 12px; background: #181818; }
    button {
      padding: 7px 10px;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 5px;
      background: #2f5f80;
      color: #fff;
      cursor: pointer;
      box-shadow: 0 2px 0 rgba(0,0,0,0.42), 0 1px 5px rgba(0,0,0,0.25);
      transition: transform 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease, border-color 0.12s ease;
    }
    button:hover { background: #397095; transform: translateY(-1px); border-color: rgba(255,255,255,0.28); box-shadow: 0 4px 10px rgba(0,0,0,0.35); }
    button:active, button.is-pressing { transform: translateY(1px) scale(0.98); box-shadow: inset 0 2px 5px rgba(0,0,0,0.45); }
    button:focus-visible { outline: none; box-shadow: 0 0 0 2px #101010, 0 0 0 4px rgba(160,210,255,0.85); }
    button.btn-clicked { animation: btn-click-flash 0.28s ease; }
    @keyframes btn-click-flash {
      0% { transform: translateY(1px) scale(0.98); }
      50% { transform: translateY(-1px) scale(1.02); box-shadow: 0 0 0 3px rgba(255,255,255,0.22); }
      100% { transform: translateY(0) scale(1); }
    }
    button.release { background: #8b3f3f; }
    button.release:hover { background: #a04a4a; }
    button.all { background: #5f7f3f; }
    button.all:hover { background: #6d9048; }
    button.quit { background: #9b2c2c; }
    button.quit:hover { background: #b03535; }
    button.capture { background: #315a9b; }
    button.capture:hover { background: #3a68ad; }
    button.record { background: #8f3f66; }
    button.record:hover { background: #a54a77; }
    button.pause { background: #7b5db0; }
    button.pause:hover { background: #8a6bbf; }
    button.record-stop { background: #b36b00; }
    button.record-stop:hover { background: #c77a0a; }
    .status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: #777; }
    .status-dot.running { background: #57c76d; }
    .status-dot.starting { background: #d8b64c; }
    .status-dot.error { background: #d95b5b; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.35; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; height: auto; }
      .cam { height: 46vh; border-right: 0; border-bottom: 1px solid #333; }
      aside { min-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Dual Camera Stream</h1>
    <div class="controls">
      <button class="all" onclick="cameraAction('all','start', this)">Start All</button>
      <button class="release" onclick="cameraAction('all','release', this)">Release All</button>
      <button class="quit" onclick="quitServer(this)">Quit</button>
    </div>
  </header>
  <main>
    <section class="cam">
      <div class="cam-title">
        <strong><span id="dot-a" class="status-dot"></span><span id="label-a">Camera A</span></strong>
        <div class="controls">
          <button onclick="cameraAction('a','start', this)">Start</button>
          <button class="release" onclick="cameraAction('a','release', this)">Release</button>
          <button class="capture" onclick="cameraAction('a','photo', this)">Photo</button>
          <button class="record" onclick="cameraAction('a','record_start', this)">Rec</button>
          <button class="pause" onclick="cameraAction('a','record_pause', this)">Pause</button>
          <button class="record-stop" onclick="cameraAction('a','record_stop', this)">End</button>
        </div>
      </div>
      <div class="video"><img src="/stream/a"></div>
    </section>
    <section class="cam">
      <div class="cam-title">
        <strong><span id="dot-b" class="status-dot"></span><span id="label-b">Camera B</span></strong>
        <div class="controls">
          <button onclick="cameraAction('b','start', this)">Start</button>
          <button class="release" onclick="cameraAction('b','release', this)">Release</button>
          <button class="capture" onclick="cameraAction('b','photo', this)">Photo</button>
          <button class="record" onclick="cameraAction('b','record_start', this)">Rec</button>
          <button class="pause" onclick="cameraAction('b','record_pause', this)">Pause</button>
          <button class="record-stop" onclick="cameraAction('b','record_stop', this)">End</button>
        </div>
      </div>
      <div class="video"><img src="/stream/b"></div>
    </section>
    <aside>
      <h2>State</h2>
      <pre id="state">loading...</pre>
    </aside>
  </main>
  <script>
    function pulseButton(btn) {
      if (!btn) return;
      btn.classList.remove('btn-clicked');
      void btn.offsetWidth;
      btn.classList.add('btn-clicked');
      window.setTimeout(() => btn.classList.remove('btn-clicked'), 300);
    }
    function clearPressingButtons() {
      document.querySelectorAll('button.is-pressing').forEach((btn) => btn.classList.remove('is-pressing'));
    }
    function bindButtonPressFeedback() {
      document.addEventListener('pointerdown', (event) => {
        const btn = event.target.closest('button');
        if (btn) btn.classList.add('is-pressing');
      });
      document.addEventListener('pointerup', clearPressingButtons);
      document.addEventListener('pointercancel', clearPressingButtons);
    }
    async function cameraAction(key, action, btn) {
      pulseButton(btn);
      const response = await fetch(`/camera/${key}/${action}`, {method: 'POST'});
      if (!response.ok) {
        const text = await response.text();
        alert(text);
      } else {
        const data = await response.json();
        maybeAlertSavedPath(action, data);
      }
      await refreshState();
    }
    function maybeAlertSavedPath(action, data) {
      if (action === 'photo' && data.photo && data.photo.saved_path) {
        alert(`Photo saved:\n${data.photo.saved_path}`);
      }
      if (action === 'record_start' && data.recording && data.recording.path) {
        alert(`Recording started:\n${data.recording.path}`);
      }
      if (action === 'record_pause' && data.recording) {
        alert(`Recording paused:\n${data.recording.path || '(no active recording)'}`);
      }
      if (action === 'record_stop' && data.recording && data.recording.saved_path) {
        alert(`Recording saved:\n${data.recording.saved_path}\nframes=${data.recording.frames}`);
      }
    }
    async function quitServer(btn) {
      if (!window.confirm('Quit dual camera stream and release both cameras?')) return;
      pulseButton(btn);
      try {
        const infoResponse = await fetch('/quit-info', {cache: 'no-store'});
        const info = await infoResponse.json();
        alert(
          `Photos/videos are saved on the robot at:\n${info.output_dir}\n\n` +
          `Copy to another computer with one of these commands:\n\n` +
          `${info.scp}\n\n${info.tar}`
        );
      } catch (error) {
        alert(`Unable to fetch copy commands: ${error}`);
      }
      await fetch('/quit', {method: 'POST'});
      document.getElementById('state').textContent = 'server exiting...';
    }
    function setDot(key, state) {
      const dot = document.getElementById(`dot-${key}`);
      if (!dot) return;
      dot.className = `status-dot ${state || ''}`;
    }
    async function refreshState() {
      try {
        const response = await fetch('/state', {cache: 'no-store'});
        const data = await response.json();
        document.getElementById('state').textContent = JSON.stringify(data, null, 2);
        for (const key of ['a', 'b']) {
          const cam = data.cameras && data.cameras[key];
          if (!cam) continue;
          setDot(key, cam.state);
          const label = document.getElementById(`label-${key}`);
          if (label) label.textContent = `${cam.label} (${cam.state})`;
        }
      } catch (error) {
        document.getElementById('state').textContent = String(error);
      }
    }
    bindButtonPressFeedback();
    setInterval(refreshState, 500);
    refreshState();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual RealSense camera web stream with release toggles.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--color-only", action="store_true", help="Do not enable depth streams.")
    parser.add_argument("--no-autostart", action="store_true", help="Start with both cameras released.")
    parser.add_argument("--serial-a", default=DEFAULT_WRIST_CAM_SERIAL, help="Empty means auto-select by --index-a.")
    parser.add_argument("--serial-b", default=DEFAULT_HEAD_CAM_SERIAL, help="Empty means auto-select by --index-b.")
    parser.add_argument("--label-a", default="wrist")
    parser.add_argument("--label-b", default="head")
    parser.add_argument("--index-a", type=int, default=0)
    parser.add_argument("--index-b", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "dual_camera_stream"),
        help="Directory on the robot for saved photos and videos.",
    )
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        print(json.dumps({"devices": RealSenseD435i.list_devices()}, ensure_ascii=False, indent=2))
        return 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = {
        "a": CameraWorker(
            CameraSpec("a", args.label_a, args.serial_a.strip(), args.index_a),
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            color_only=args.color_only,
            output_dir=output_dir,
            autostart=not args.no_autostart,
        ),
        "b": CameraWorker(
            CameraSpec("b", args.label_b, args.serial_b.strip(), args.index_b),
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            color_only=args.color_only,
            output_dir=output_dir,
            autostart=not args.no_autostart,
        ),
    }
    server = DualCameraStreamServer(workers, host=args.host, port=args.port, output_dir=output_dir)

    def shutdown(_signum: int, _frame: Any) -> None:
        for worker in workers.values():
            worker.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Dual camera stream: http://127.0.0.1:{args.port}/")
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
