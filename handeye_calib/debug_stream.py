from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

import cv2
from flask import Flask, Response, jsonify, make_response, request


RIGHT_ARM_JOINT_UI = [
    ("right_shoulder_pitch", "肩 pitch"),
    ("right_shoulder_roll", "肩 roll"),
    ("right_shoulder_yaw", "肩 yaw"),
    ("right_elbow", "肘"),
    ("right_wrist_roll", "腕 roll"),
    ("right_wrist_pitch", "腕 pitch"),
    ("right_wrist_yaw", "腕 yaw"),
]

SIMPLE_COMMANDS = {
    "save",
    "solve",
    "quit",
    "start_calib",
    "switch_sdk",
    "test_move",
    "stop",
    "touch",
    "release",
    "arm_prev",
    "arm_next",
    "arm_move",
    "arm_random_right",
    "arm_hold_current",
    "arm_save_current",
    "arm_release",
    "arm_sdk_enable",
    "arm_sdk_disable",
    "robot_default_pose",
}

JOINT_COMMANDS = {
    "arm_joint_delta",
    "arm_joint_abs",
    "arm_joint_random",
}


def _joint_control_rows_html() -> str:
    rows = []
    for joint_key, label in RIGHT_ARM_JOINT_UI:
        rows.append(
            f"""
        <div class="joint-row">
          <span class="joint-label" title="{joint_key}">{label}</span>
          <input id="delta_{joint_key}" type="number" step="0.01" value="0.02" class="joint-input" />
          <input id="abs_{joint_key}" type="number" step="0.01" value="0.00" class="joint-input" />
          <button class="arm" onclick="sendJointCommand('arm_joint_delta', '{joint_key}', 'delta_{joint_key}', this)">Δ</button>
          <button class="arm" onclick="sendJointCommand('arm_joint_abs', '{joint_key}', 'abs_{joint_key}', this)">Go</button>
          <button class="arm" onclick="sendJointCommand('arm_joint_random', '{joint_key}', 'delta_{joint_key}', this)">Rand</button>
        </div>"""
        )
    return "".join(rows)


class DebugStreamServer:
    """Serve the latest OpenCV debug frame and structured state over HTTP."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 80) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._app = Flask(__name__)
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_state: dict[str, Any] = {}
        self._commands: list[dict[str, Any]] = []
        self._started = False
        self._configure_routes()

    def _configure_routes(self) -> None:
        joint_rows = _joint_control_rows_html()

        @self._app.route("/")
        def index() -> Response:
            html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hand-Eye Debug Stream</title>
  <meta http-equiv="Cache-Control" content="no-store" />
  <meta name="ui-version" content="20260626-capture-reject-alert" />
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
    .layout {{ display: flex; height: 100vh; }}
    .video {{ flex: 1 1 auto; display: flex; align-items: center; justify-content: center; background: #000; }}
    .video img {{ max-width: 100%; max-height: 100%; object-fit: contain; pointer-events: none; }}
    .panel {{ width: 520px; padding: 14px; overflow: auto; background: #1b1b1b; border-left: 1px solid #333; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    button {{
      padding: 8px 12px;
      border: 1px solid transparent;
      border-radius: 6px;
      color: #fff;
      cursor: pointer;
      user-select: none;
      touch-action: manipulation;
      -webkit-tap-highlight-color: transparent;
      position: relative;
      transition: transform 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease, border-color 0.12s ease;
      box-shadow: 0 2px 0 rgba(0, 0, 0, 0.38), 0 1px 3px rgba(0, 0, 0, 0.22);
      outline: none;
    }}
    @media (hover: hover) and (pointer: fine) {{
      button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.36), 0 0 0 1px rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.18);
      }}
    }}
    button:active,
    button.is-pressing {{
      transform: translateY(1px) scale(0.98);
      box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.45);
      border-color: rgba(0, 0, 0, 0.35);
    }}
    button:focus-visible {{
      box-shadow: 0 0 0 2px #111, 0 0 0 4px rgba(159, 208, 255, 0.85);
    }}
    button.btn-clicked {{
      animation: btn-click-flash 0.32s ease;
    }}
    @keyframes btn-click-flash {{
      0% {{ transform: translateY(1px) scale(0.98); }}
      45% {{ transform: translateY(-1px) scale(1.02); box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.28); }}
      100% {{ transform: translateY(0) scale(1); }}
    }}
    .start {{ background: #6b8e23; }}
    .start:hover {{ background: #7da52a; }}
    .mode {{ background: #4f6f8f; }}
    .mode:hover {{ background: #5d809f; }}
    .test {{ background: #7b5db0; }}
    .test:hover {{ background: #8a6bbf; }}
    .stop {{ background: #c26a1b; }}
    .stop:hover {{ background: #d47a28; }}
    .save {{ background: #1f7a3a; }}
    .save:hover {{ background: #259044; }}
    .solve {{ background: #315a9b; }}
    .solve:hover {{ background: #3a68ad; }}
    .arm {{ background: #5f7f3f; }}
    .arm:hover {{ background: #6d9048; }}
    .preset-active {{
      background: #3f6f8f;
      border-color: #9fd0ff;
      box-shadow: inset 0 0 0 1px #9fd0ff, 0 2px 0 rgba(0, 0, 0, 0.38);
    }}
    .preset-active:hover {{ background: #4a7fa0; }}
    .touch {{ background: #b36b00; }}
    .touch:hover {{ background: #c77a0a; }}
    .release {{ background: #8b3f3f; }}
    .release:hover {{ background: #a04a4a; }}
    .quit {{ background: #9b2c2c; }}
    .quit:hover {{ background: #b03535; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.35; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 14px 0 8px; font-size: 14px; color: #bbb; }}
    .joint-grid {{ display: grid; gap: 6px; margin-bottom: 12px; }}
    .joint-head, .joint-row {{ display: grid; grid-template-columns: 92px 72px 72px 42px 42px 52px; gap: 6px; align-items: center; }}
    .joint-head {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
    .joint-label {{ font-size: 12px; }}
    .joint-input {{
      width: 100%;
      box-sizing: border-box;
      padding: 4px 6px;
      border-radius: 4px;
      border: 1px solid #444;
      background: #111;
      color: #eee;
      transition: border-color 0.14s ease, box-shadow 0.14s ease, background 0.14s ease;
    }}
    .joint-input:hover {{ border-color: #666; background: #161616; }}
    .joint-input:focus {{
      border-color: #9fd0ff;
      background: #141820;
      outline: none;
      box-shadow: 0 0 0 2px rgba(159, 208, 255, 0.22);
    }}
    .hint {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
    .hint-warn {{ font-size: 12px; color: #d9b26a; margin-bottom: 8px; line-height: 1.45; }}
    .panel-disabled {{ opacity: 0.42; pointer-events: none; filter: grayscale(0.15); }}
    .arm-sdk-row {{ pointer-events: auto; opacity: 1; filter: none; }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="video"><img src="/stream" /></div>
    <div class="panel">
      <h2>FK / Capture State</h2>
      <div class="controls">
        <button class="start" onclick="sendCommand('start_calib', this)">Start Calib</button>
        <button class="mode" onclick="sendCommand('switch_sdk', this)">Switch SDK Mode</button>
        <button class="test" onclick="sendCommand('test_move', this)">Test Move</button>
        <button class="stop" onclick="sendCommand('stop', this)">Stop</button>
        <button class="save" onclick="sendCommand('save', this)">Save / SPACE</button>
        <button class="solve" onclick="sendCommand('solve', this)">Solve / S</button>
        <button class="quit" onclick="sendCommand('quit', this)">Quit / Q</button>
      </div>
      <div id="arm-ui-root" style="display:none">
      <h2>Arm SDK</h2>
      <div class="hint-warn">默认不接管 arm_sdk。请先用外部控制器把手臂摆到标定位姿，再点 Save。只有需要网页控臂时才二次确认接管。</div>
      <div class="controls arm-sdk-row" id="arm-sdk-controls">
        <button id="btn-arm-sdk-enable" class="mode" onclick="requestArmSdkEnable(this)">接管 Arm SDK</button>
        <button id="btn-arm-sdk-disable" class="release" onclick="sendCommand('arm_sdk_disable', this)" style="display:none">断开 Arm SDK</button>
      </div>
      <div id="arm-motion-panel" class="panel-disabled">
      <h2>Arm Waypoints</h2>
      <div class="controls">
        <button class="arm" onclick="sendCommand('arm_prev', this)">Prev</button>
        <button class="arm" onclick="sendCommand('arm_next', this)">Next</button>
        <button class="arm" onclick="sendCommand('arm_move', this)">Move</button>
        <button class="arm" onclick="sendCommand('arm_random_right', this)">Random Right Arm</button>
        <button class="arm" onclick="sendCommand('arm_hold_current', this)">Hold Current</button>
        <button class="arm" onclick="sendCommand('arm_save_current', this)">Save Current</button>
        <button class="release" onclick="sendCommand('robot_default_pose', this)">Arm Default</button>
        <button class="release" onclick="sendCommand('arm_release', this)">Release Arm SDK</button>
      </div>
      <h2>Arm Presets</h2>
      <div class="hint">Preset 回到 URDF 零位 (0 rad)；关节微调会累积，默认保持不 release</div>
      <div class="controls" id="preset-buttons"></div>
      <h2>Right Arm Joints</h2>
      <div class="hint">Δ=相对当前角位移(rad)，Go=绝对目标(rad)，Rand=按 Δ 列幅度随机</div>
      <div class="joint-grid">
        <div class="joint-head">
          <span>关节</span><span>Δ rad</span><span>Go rad</span><span></span><span></span><span></span>
        </div>
        {joint_rows}
      </div>
      <h2>Plan B Touch</h2>
      <div class="controls">
        <button class="touch" onclick="sendCommand('touch', this)">Touch</button>
        <button class="stop" onclick="sendCommand('stop', this)">Stop</button>
        <button class="release" onclick="sendCommand('release', this)">Release</button>
        <button class="quit" onclick="sendCommand('quit', this)">Quit</button>
      </div>
      </div>
      </div>
      <pre id="state">waiting...</pre>
    </div>
  </div>
  <script>
    function pulseButton(btn) {{
      if (!btn) return;
      btn.classList.remove('btn-clicked');
      void btn.offsetWidth;
      btn.classList.add('btn-clicked');
      window.setTimeout(() => btn.classList.remove('btn-clicked'), 320);
    }}

    function clearPressingButtons() {{
      document.querySelectorAll('button.is-pressing').forEach((btn) => btn.classList.remove('is-pressing'));
    }}

    function bindButtonPressFeedback() {{
      const panel = document.querySelector('.panel');
      if (!panel) return;
      panel.addEventListener('pointerdown', (event) => {{
        const btn = event.target.closest('button');
        if (btn) btn.classList.add('is-pressing');
      }});
      document.addEventListener('pointerup', clearPressingButtons);
      document.addEventListener('pointercancel', clearPressingButtons);
    }}

    async function sendCommand(command, btn) {{
      pulseButton(btn);
      const body = {{ command }};
      if (command === 'arm_sdk_enable') {{
        body.confirm = true;
      }}
      await fetch('/command', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body)
      }});
      refreshState();
    }}

    async function requestArmSdkEnable(btn) {{
      const msg1 = '确认接管 Arm SDK？\\n\\n接管后网页将发布 rt/arm_sdk 控制右臂。\\n请确保外部控制器已停止发送手臂指令。';
      if (!window.confirm(msg1)) return;
      const msg2 = '二次确认：现在接管 Arm SDK？\\n\\n若对面控制器仍在控臂，可能发生冲突。';
      if (!window.confirm(msg2)) return;
      pulseButton(btn);
      await fetch('/command', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ command: 'arm_sdk_enable', confirm: true }})
      }});
      refreshState();
    }}

    function refreshArmSdkUi(armState) {{
      const root = document.getElementById('arm-ui-root');
      const panel = document.getElementById('arm-motion-panel');
      const showArmUi = !!(armState && armState.enabled);
      if (root) root.style.display = showArmUi ? '' : 'none';
      if (!showArmUi) return;
      const connected = !!armState.arm_sdk_connected;
      if (panel) panel.classList.toggle('panel-disabled', !connected);
      const enableBtn = document.getElementById('btn-arm-sdk-enable');
      const disableBtn = document.getElementById('btn-arm-sdk-disable');
      if (enableBtn) enableBtn.style.display = connected ? 'none' : '';
      if (disableBtn) disableBtn.style.display = connected ? '' : 'none';
    }}

    async function sendPreset(preset, btn) {{
      pulseButton(btn);
      await fetch('/command', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ command: 'arm_preset', preset }})
      }});
      refreshState();
    }}

    async function sendJointCommand(command, joint, inputId, btn) {{
      pulseButton(btn);
      const body = {{ command, joint }};
      const input = document.getElementById(inputId);
      if (input) {{
        const value = parseFloat(input.value);
        if (!Number.isFinite(value)) {{
          alert('请输入有效数字');
          return;
        }}
        if (command === 'arm_joint_delta' || command === 'arm_joint_random') {{
          body.delta_rad = value;
        }} else if (command === 'arm_joint_abs') {{
          body.value_rad = value;
        }}
      }}
      await fetch('/command', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body)
      }});
      refreshState();
    }}

    function refreshPresetButtons(armState) {{
      const container = document.getElementById('preset-buttons');
      if (!container || !armState.presets) return;
      const active = armState.active_preset || '';
      const existing = new Map(
        [...container.querySelectorAll('button[data-preset]')].map((btn) => [btn.dataset.preset, btn])
      );
      for (const [name, info] of Object.entries(armState.presets)) {{
        let btn = existing.get(name);
        if (!btn) {{
          btn = document.createElement('button');
          btn.dataset.preset = name;
          btn.onclick = () => sendPreset(name, btn);
          container.appendChild(btn);
        }}
        const nextClass = active === name ? 'arm preset-active' : 'arm';
        if (btn.className !== nextClass) btn.className = nextClass;
        const label = info.label || name;
        if (btn.textContent !== label) btn.textContent = label;
        const title = info.description || name;
        if (btn.title !== title) btn.title = title;
        existing.delete(name);
      }}
      for (const btn of existing.values()) btn.remove();
    }}

    let lastWarningId = 0;

    function maybeShowWarning(data) {{
      const warning = data && data.warning;
      if (!warning || !warning.id || warning.id === lastWarningId) return;
      lastWarningId = warning.id;
      if (warning.kind === 'capture_rejected_high_reprojection') {{
        alert(
          '本次 Save 已被拒绝：棋盘重投影误差过高\\n\\n' +
          `当前 RMS: ${{Number(warning.target_reprojection_rms_px).toFixed(3)}} px\\n` +
          `上限: ${{Number(warning.max_reprojection_rms_px).toFixed(3)}} px\\n\\n` +
          '请等手臂/本体稳定、调整棋盘位置后重新 Save。'
        );
      }} else if (warning.message) {{
        alert(warning.message);
      }}
    }}

    async function refreshState() {{
      try {{
        const response = await fetch('/state', {{ cache: 'no-store' }});
        const data = await response.json();
        document.getElementById('state').textContent = JSON.stringify(data, null, 2);
        maybeShowWarning(data);
        if (data.arm_waypoints && data.arm_waypoints.current_joints_rad) {{
          const joints = data.arm_waypoints.current_joints_rad;
          for (const [key, value] of Object.entries(joints)) {{
            const absInput = document.getElementById('abs_' + key);
            if (absInput && document.activeElement !== absInput) {{
              absInput.value = Number(value).toFixed(3);
            }}
          }}
        }}
        refreshPresetButtons(data.arm_waypoints || {{}});
        refreshArmSdkUi(data.arm_waypoints || {{}});
      }} catch (error) {{
        document.getElementById('state').textContent = String(error);
      }}
    }}

    bindButtonPressFeedback();
    setInterval(refreshState, 500);
    refreshState();
  </script>
</body>
</html>
"""
            response = make_response(html)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response

        @self._app.route("/stream")
        def stream() -> Response:
            return Response(
                self._mjpeg_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self._app.route("/state")
        def state() -> Response:
            with self._lock:
                payload = dict(self._latest_state)
            return jsonify(payload)

        @self._app.route("/command", methods=["POST"])
        def command() -> Response:
            payload = request.get_json(silent=True) or {}
            cmd = str(payload.get("command", "")).strip().lower()
            if cmd in SIMPLE_COMMANDS:
                normalized = {"command": cmd}
                if cmd == "arm_sdk_enable":
                    if not payload.get("confirm"):
                        return jsonify({"ok": False, "error": "arm_sdk_enable requires confirm=true"}), 400
                    normalized["confirm"] = True
            elif cmd == "arm_preset":
                preset = str(payload.get("preset", "")).strip()
                if not preset:
                    return jsonify({"ok": False, "error": "arm_preset requires preset"}), 400
                normalized = {"command": cmd, "preset": preset}
            elif cmd in JOINT_COMMANDS:
                joint = str(payload.get("joint", "")).strip()
                if not joint:
                    return jsonify({"ok": False, "error": "joint command requires joint"}), 400
                normalized = {"command": cmd, "joint": joint}
                if "delta_rad" in payload:
                    normalized["delta_rad"] = float(payload["delta_rad"])
                if "value_rad" in payload:
                    normalized["value_rad"] = float(payload["value_rad"])
                if "max_delta_rad" in payload:
                    normalized["max_delta_rad"] = float(payload["max_delta_rad"])
            else:
                return jsonify({"ok": False, "error": f"unsupported command: {cmd}"}), 400
            with self._lock:
                self._commands.append(normalized)
                self._latest_state = dict(self._latest_state)
                self._latest_state["last_web_command"] = normalized
                self._latest_state["last_web_command_at"] = time.time()
            return jsonify({"ok": True, "command": normalized})

    def start(self) -> None:
        if self._started:
            return
        thread = threading.Thread(
            target=lambda: self._app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False,
            ),
            daemon=True,
        )
        thread.start()
        self._started = True

    def update_frame(self, frame_bgr: Any) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()

    def update_state(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state["updated_at"] = time.time()
        with self._lock:
            self._latest_state = state

    def pop_command(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self._commands:
                return None
            return self._commands.pop(0)

    @staticmethod
    def command_name(payload: Optional[dict[str, Any]]) -> Optional[str]:
        if not payload:
            return None
        return str(payload.get("command", "")).strip().lower() or None

    def _mjpeg_frames(self):
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            time.sleep(0.01)


def json_ready(value: Any) -> Any:
    """Best-effort conversion for values before passing them to the stream."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
