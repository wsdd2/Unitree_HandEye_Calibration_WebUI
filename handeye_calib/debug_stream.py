from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

import cv2
from flask import Flask, Response, jsonify, make_response, request

from handeye_calib.sweep_web import (
    SWEEP_JOG_COMMANDS,
    SWEEP_JOINT_COMMANDS,
    SWEEP_QUEUE_MAX,
    SWEEP_STALE_SEC,
    normalize_sweep_command,
)


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
    "next_camera",
    "test_move",
    "stop",
    "touch",
    "release",
    "arm_prev",
    "arm_next",
    "arm_move",
    "arm_random_right",
    "arm_random_sweep",
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


SWEEP_PANEL_HTML = """
      <h2>扫掠 / 末端微调</h2>
      <div id="sweep-status" class="hint sweep-wait">等待终端 B 接入…</div>
      <div class="hint">终端 A 只负责预览和 Save。下面按钮要等终端 B 到位后才可点。</div>
      <div id="sweep-panel" class="panel-disabled">
      <div class="controls">
        <button class="save" data-sweep disabled onclick="sendSweep('sweep_next', this)">下一个 n</button>
        <button class="stop" data-sweep disabled onclick="sendSweep('sweep_skip', this)">再抽 s</button>
        <button class="quit" data-sweep disabled onclick="sendSweep('sweep_quit', this)">结束 q</button>
      </div>
      <div class="hint">平移是躯干系：Z+ 向上抬（默认 20 mm），X/Y 默认 2 mm。旋转仍是相机系。当前 <span id="sweep-step">XY 2.0 mm / Z 20.0 mm / 2.0°</span></div>
      <div class="jog-grid">
        <span class="jog-label">X</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_x_plus" disabled>X+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_x_minus" disabled>X-</button>
        <span class="jog-label">Y</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_y_plus" disabled>Y+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_y_minus" disabled>Y-</button>
        <span class="jog-label">Z</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_z_plus" disabled>Z+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_z_minus" disabled>Z-</button>
        <span class="jog-label">roll</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_roll_plus" disabled>R+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_roll_minus" disabled>R-</button>
        <span class="jog-label">pitch</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_pitch_plus" disabled>P+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_pitch_minus" disabled>P-</button>
        <span class="jog-label">yaw</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_yaw_plus" disabled>Y+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_yaw_minus" disabled>Y-</button>
      </div>
      <div class="controls">
        <button class="mode" data-sweep disabled onclick="sendSweep('sweep_step_halve', this)">步长 /2</button>
        <button class="mode" data-sweep disabled onclick="sendSweep('sweep_step_double', this)">步长 ×2</button>
        <button class="test" data-sweep disabled onclick="sendSweep('sweep_follow', this)">跟随实测</button>
      </div>
      <h3>右臂关节平滑微调</h3>
      <div class="hint">沿用运控关节限位；单击走一步，按住连续走。当前 <span id="sweep-joint-step">2.0°</span></div>
      <div class="jog-grid">
        <span class="jog-label">肩 pitch</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_1_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_1_minus" disabled>−</button>
        <span class="jog-label">肩 roll</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_2_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_2_minus" disabled>−</button>
        <span class="jog-label">肩 yaw</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_3_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_3_minus" disabled>−</button>
        <span class="jog-label">肘</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_4_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_4_minus" disabled>−</button>
        <span class="jog-label">腕 roll</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_5_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_5_minus" disabled>−</button>
        <span class="jog-label">腕 pitch</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_6_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_6_minus" disabled>−</button>
        <span class="jog-label">腕 yaw</span>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_7_plus" disabled>+</button>
        <button class="arm" data-sweep data-sweep-jog="sweep_joint_7_minus" disabled>−</button>
      </div>
      <div class="controls">
        <button class="mode" data-sweep disabled onclick="sendSweep('sweep_joint_step_halve', this)">关节步长 /2</button>
        <button class="mode" data-sweep disabled onclick="sendSweep('sweep_joint_step_double', this)">关节步长 ×2</button>
      </div>
      <div class="hint" id="sweep-joint-state"></div>
      </div>
"""

SWEEP_PANEL_JS = """
    let sweepHoldDelay = null;
    let sweepHoldRepeat = null;

    async function sendSweep(command, btn) {
      if (btn && btn.disabled) return;
      pulseButton(btn);
      const response = await fetch('/sweep/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command })
      });
      if (!response.ok) {
        let detail = '';
        try {
          const data = await response.json();
          detail = data.error || '';
        } catch (error) {}
        console.warn('sweep command rejected', command, detail);
      }
      refreshState();
    }

    function stopSweepHold() {
      if (sweepHoldDelay !== null) window.clearTimeout(sweepHoldDelay);
      if (sweepHoldRepeat !== null) window.clearInterval(sweepHoldRepeat);
      sweepHoldDelay = null;
      sweepHoldRepeat = null;
    }

    function startSweepHold(event, btn) {
      if (!btn || btn.disabled) return;
      event.preventDefault();
      stopSweepHold();
      const command = btn.dataset.sweepJog;
      sendSweep(command, btn);
      sweepHoldDelay = window.setTimeout(() => {
        sweepHoldRepeat = window.setInterval(() => {
          if (!btn.disabled) sendSweep(command, btn);
        }, 450);
      }, 420);
    }

    function bindSweepJogControls() {
      document.querySelectorAll('[data-sweep-jog]').forEach((btn) => {
        btn.addEventListener('pointerdown', (event) => startSweepHold(event, btn));
        btn.addEventListener('pointerup', stopSweepHold);
        btn.addEventListener('pointercancel', stopSweepHold);
        btn.addEventListener('pointerleave', stopSweepHold);
        btn.addEventListener('contextmenu', (event) => event.preventDefault());
      });
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) stopSweepHold();
      });
    }

    function refreshSweepUi(sweep) {
      const statusEl = document.getElementById('sweep-status');
      const panel = document.getElementById('sweep-panel');
      const stepEl = document.getElementById('sweep-step');
      const jointStepEl = document.getElementById('sweep-joint-step');
      const jointStateEl = document.getElementById('sweep-joint-state');
      sweep = sweep || {};
      const connected = !!sweep.connected;
      const ready = !!sweep.accepts_input;
      let text = '等待终端 B 接入…';
      if (connected && ready) {
        const goal = Number(sweep.target) > 0 ? sweep.target : '∞';
        text = '终端 B 已就绪  已确认 ' + (sweep.accepted || 0) + '/' + goal;
        if (sweep.message) text += '  ' + sweep.message;
      } else if (connected) {
        text = '终端 B 已连接，手臂移动中，到位后按钮可用';
        if (sweep.message) text += '  ' + sweep.message;
      }
      if (statusEl) {
        statusEl.textContent = text;
        statusEl.className = ready ? 'hint sweep-ready' : (connected ? 'hint sweep-busy' : 'hint sweep-wait');
      }
      if (panel) panel.classList.toggle('panel-disabled', !ready);
      document.querySelectorAll('[data-sweep]').forEach((btn) => {
        btn.disabled = !ready;
      });
      if (stepEl && sweep.step_mm != null) {
        stepEl.textContent =
          'XY ' + Number(sweep.step_mm).toFixed(1) +
          ' mm / Z ' + Number(sweep.z_step_mm || 20).toFixed(1) +
          ' mm / ' + Number(sweep.rot_deg || 2).toFixed(1) + '°';
      }
      if (jointStepEl) {
        jointStepEl.textContent = Number(sweep.joint_step_deg || 2).toFixed(1) + '°';
      }
      if (jointStateEl) {
        const names = ['肩P', '肩R', '肩Y', '肘', '腕R', '腕P', '腕Y'];
        const values = Array.isArray(sweep.joint_deg) ? sweep.joint_deg : [];
        jointStateEl.textContent = values.length === 7
          ? names.map((name, index) => name + ' ' + Number(values[index]).toFixed(1) + '°').join('  ')
          : '';
      }
    }
"""


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
        self._sweep_commands: list[str] = []
        self._sweep_status: dict[str, Any] = {}
        self._sweep_heartbeat_at = 0.0
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
  <meta name="ui-version" content="20260918-sweep-buttons" />
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
    .camera-box {{
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #3a3a3a;
      border-radius: 8px;
      background: #151515;
    }}
    .camera-now {{ font-size: 14px; color: #9fd0ff; line-height: 1.4; margin-bottom: 6px; }}
    .camera-list {{ font-size: 12px; color: #aaa; line-height: 1.45; white-space: pre-wrap; }}
    .panel-disabled {{ opacity: 0.42; pointer-events: none; filter: grayscale(0.15); }}
    .arm-sdk-row {{ pointer-events: auto; opacity: 1; filter: none; }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
      pointer-events: none;
      box-shadow: none;
      transform: none;
    }}
    .sweep-wait {{ color: #d9b26a; }}
    .sweep-busy {{ color: #9fd0ff; }}
    .sweep-ready {{ color: #7dce8a; }}
    .jog-grid {{
      display: grid;
      grid-template-columns: 48px 1fr 1fr;
      gap: 6px;
      margin: 0 0 10px;
      align-items: center;
    }}
    .jog-label {{ font-size: 12px; color: #aaa; }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="video"><img src="/stream" /></div>
    <div class="panel">
      <h2>FK / Capture State</h2>
      <div class="camera-box">
        <div class="camera-now" id="camera-now">正在枚举 RealSense...</div>
        <div class="controls" style="margin-bottom:8px">
          <button class="mode" onclick="sendCommand('next_camera', this)">Next Camera</button>
        </div>
        <div class="camera-list" id="camera-list"></div>
      </div>
      {SWEEP_PANEL_HTML}
      <div class="controls">
        <button class="start" onclick="sendCommand('start_calib', this)">Start Calib</button>
        <button class="mode" onclick="sendCommand('switch_sdk', this)">Switch SDK Mode</button>
        <button class="test" onclick="sendCommand('test_move', this)">Test Move</button>
        <button class="stop" onclick="sendCommand('stop', this)">Stop</button>
        <button class="save" onclick="sendCommand('save', this)">Save / SPACE</button>
        <button class="solve" onclick="sendCommand('solve', this)">Solve / S</button>
        <button class="quit" onclick="sendCommand('quit', this)">Quit / Q</button>
      </div>
      <div class="camera-box" id="capture-box">
        <div class="camera-now" id="capture-now">Save：画面上要先有绿角点</div>
        <div class="camera-list" id="capture-path"></div>
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
        <button class="arm" onclick="sendCommand('arm_random_sweep', this)">Random Sweep 25</button>
        <button class="arm" onclick="sendCommand('arm_hold_current', this)">Hold Current</button>
        <button class="arm" onclick="sendCommand('arm_save_current', this)">Save Current</button>
        <button class="release" onclick="sendCommand('robot_default_pose', this)">Arm Default</button>
        <button class="release" onclick="sendCommand('arm_release', this)">Release Arm SDK</button>
      </div>
      <h2>Arm Presets</h2>
      <div class="hint">Preset 回到 URDF 零位 (0 rad)；关节微调会累积，默认保持不 release</div>
      <div class="controls" id="preset-buttons"></div>
      <h2>Right Arm Joints</h2>
      <div class="hint">Δ=相对当前角位移(rad)，Go=绝对目标(rad)，Rand=按 Δ 列幅度随机。Random Sweep 25=自动走 25 个上臂可达点，Stop 可中断</div>
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

    {SWEEP_PANEL_JS}

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

    function refreshCameraUi(data) {{
      const nowEl = document.getElementById('camera-now');
      const listEl = document.getElementById('camera-list');
      if (!nowEl || !listEl) return;
      const cam = (data && data.camera) || {{}};
      const serial = cam.serial || cam.selected_serial || '?';
      const model = cam.model || 'RealSense';
      const idx = Number(cam.device_index);
      const total = Number(cam.device_count);
      const slot = Number.isFinite(idx) && Number.isFinite(total) && total > 0
        ? `  (${{idx + 1}}/${{total}})`
        : '';
      nowEl.textContent = `当前相机: ${{model}}  SN=${{serial}}${{slot}}`;
      const devices = Array.isArray(cam.enumerated) ? cam.enumerated : [];
      if (!devices.length) {{
        listEl.textContent = '未枚举到 RealSense。检查 USB / pyrealsense2。';
        return;
      }}
      listEl.textContent = devices.map((dev, i) => {{
        const mark = (dev.serial || '') === serial ? '> ' : '  ';
        return `${{mark}}${{i + 1}}. ${{dev.model || 'RealSense'}}  SN=${{dev.serial || '?'}}`;
      }}).join('\\n');
    }}

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
      }} else if (warning.kind === 'capture_rejected_no_board') {{
        alert('本次 Save 没存上：没检出棋盘。\\n画面上要先看到绿角点，并核对网页里的 board 尺寸。');
      }} else if (warning.message) {{
        alert(warning.message);
      }}
    }}

    function refreshCaptureUi(data) {{
      const nowEl = document.getElementById('capture-now');
      const pathEl = document.getElementById('capture-path');
      if (!nowEl || !pathEl) return;
      const found = !!data.chessboard_detected;
      const saved = Number(data.saved_count || 0);
      const msg = data.last_message || '';
      nowEl.textContent = found
        ? `棋盘已检出  已存 ${{saved}} 张  ${{msg}}`
        : `棋盘未检出，Save 不会存图  已存 ${{saved}} 张  ${{msg}}`;
      nowEl.style.color = found ? '#7dce8a' : '#d9b26a';
      pathEl.textContent = data.session_dir
        ? ('保存目录: ' + data.session_dir)
        : '';
    }}

    async function refreshState() {{
      try {{
        const response = await fetch('/state', {{ cache: 'no-store' }});
        const data = await response.json();
        document.getElementById('state').textContent = JSON.stringify(data, null, 2);
        refreshCameraUi(data);
        refreshCaptureUi(data);
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
        refreshSweepUi(data.sweep || {{}});
      }} catch (error) {{
        document.getElementById('state').textContent = String(error);
      }}
    }}

    bindButtonPressFeedback();
    bindSweepJogControls();
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
                payload["sweep"] = self._sweep_public_unlocked()
            return jsonify(payload)

        @self._app.route("/sweep/heartbeat", methods=["POST"])
        def sweep_heartbeat() -> Response:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"ok": False, "error": "heartbeat must be an object"}), 400
            with self._lock:
                self._sweep_status = dict(payload)
                self._sweep_heartbeat_at = time.time()
            return jsonify({"ok": True, "sweep": self.sweep_public()})

        @self._app.route("/sweep/command", methods=["GET", "POST"])
        def sweep_command() -> Response:
            if request.method == "GET":
                with self._lock:
                    cmd = self._sweep_commands.pop(0) if self._sweep_commands else None
                return jsonify({"ok": True, "command": cmd})
            payload = request.get_json(silent=True) or {}
            cmd = normalize_sweep_command(payload.get("command"))
            if cmd is None:
                return jsonify({"ok": False, "error": "unsupported sweep command"}), 400
            with self._lock:
                if not self._sweep_connected_unlocked():
                    return jsonify({"ok": False, "error": "sweep client not connected"}), 409
                if not bool(self._sweep_status.get("accepts_input")):
                    return jsonify({"ok": False, "error": "sweep not accepting input yet"}), 409
                motion_commands = set(SWEEP_JOG_COMMANDS) | set(SWEEP_JOINT_COMMANDS)
                if cmd in motion_commands:
                    # A held web button may post faster than a quintic/service
                    # move can finish. Keep only the newest pending jog so the
                    # arm stops promptly when the user releases the button.
                    self._sweep_commands = [
                        pending
                        for pending in self._sweep_commands
                        if pending not in motion_commands
                    ]
                if len(self._sweep_commands) >= SWEEP_QUEUE_MAX:
                    return jsonify({"ok": False, "error": "sweep command queue full"}), 429
                self._sweep_commands.append(cmd)
                self._latest_state = dict(self._latest_state)
                self._latest_state["last_sweep_command"] = cmd
                self._latest_state["last_sweep_command_at"] = time.time()
            return jsonify({"ok": True, "command": cmd})

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

    def sweep_public(self) -> dict[str, Any]:
        with self._lock:
            return self._sweep_public_unlocked()

    def _sweep_connected_unlocked(self) -> bool:
        return self._sweep_heartbeat_at > 0 and (time.time() - self._sweep_heartbeat_at) < SWEEP_STALE_SEC

    def _sweep_public_unlocked(self) -> dict[str, Any]:
        connected = self._sweep_connected_unlocked()
        status = dict(self._sweep_status) if connected else {}

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return {
            "connected": connected,
            "accepts_input": bool(connected and status.get("accepts_input")),
            "phase": str(status.get("phase") or ""),
            "accepted": _as_int(status.get("accepted")),
            "target": _as_int(status.get("target")),
            "visits": _as_int(status.get("visits")),
            "step_mm": _as_float(status.get("step_mm"), 2.0),
            "z_step_mm": _as_float(status.get("z_step_mm"), 20.0),
            "rot_deg": _as_float(status.get("rot_deg"), 2.0),
            "joint_step_deg": _as_float(status.get("joint_step_deg"), 2.0),
            "xyz": status.get("xyz"),
            "rpy_deg": status.get("rpy_deg"),
            "joint_deg": status.get("joint_deg"),
            "message": str(status.get("message") or ""),
            "queued": len(self._sweep_commands),
        }

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
