# -*- coding: utf-8 -*-
"""Web buttons for dwell / EE jog. Terminal B heartbeats to Terminal A's debug page."""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from typing import Any, Callable, Optional


SWEEP_STALE_SEC = 2.5
SWEEP_QUEUE_MAX = 32

SWEEP_MENU_COMMANDS = frozenset(
    {
        "sweep_next",
        "sweep_skip",
        "sweep_quit",
        "sweep_cart",
        "sweep_stay",
    }
)
SWEEP_JOG_COMMANDS = {
    "sweep_x_plus": ("xyz", 0, 1.0),
    "sweep_x_minus": ("xyz", 0, -1.0),
    "sweep_y_plus": ("xyz", 1, 1.0),
    "sweep_y_minus": ("xyz", 1, -1.0),
    "sweep_z_plus": ("xyz", 2, 1.0),
    "sweep_z_minus": ("xyz", 2, -1.0),
    "sweep_roll_plus": ("rpy", 0, 1.0),
    "sweep_roll_minus": ("rpy", 0, -1.0),
    "sweep_pitch_plus": ("rpy", 1, 1.0),
    "sweep_pitch_minus": ("rpy", 1, -1.0),
    "sweep_yaw_plus": ("rpy", 2, 1.0),
    "sweep_yaw_minus": ("rpy", 2, -1.0),
}
RIGHT_ARM_JOINT_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
RIGHT_ARM_JOINT_LIMITS_RAD = (
    (-2.618, 1.833),
    (-2.494, 0.517),
    (-2.618, 2.618),
    (-0.986, 3.071),
    (-2.618, 2.618),
    (-0.576, 0.576),
    (-1.22, 1.22),
)
SWEEP_JOINT_COMMANDS = {
    f"sweep_joint_{index + 1}_{direction}": (index, 1.0 if direction == "plus" else -1.0)
    for index in range(7)
    for direction in ("plus", "minus")
}
SWEEP_META_COMMANDS = frozenset(
    {
        "sweep_step_halve",
        "sweep_step_double",
        "sweep_joint_step_halve",
        "sweep_joint_step_double",
        "sweep_follow",
    }
)
SWEEP_COMMANDS = (
    SWEEP_MENU_COMMANDS
    | set(SWEEP_JOG_COMMANDS)
    | set(SWEEP_JOINT_COMMANDS)
    | SWEEP_META_COMMANDS
)

KEY_TO_SWEEP = {
    "n": "sweep_next",
    "\r": "sweep_next",
    "\n": "sweep_next",
    "s": "sweep_skip",
    "q": "sweep_quit",
}

MENU_ACTIONS = {
    "sweep_next": "next",
    "sweep_skip": "skip",
    "sweep_quit": "quit",
    "sweep_cart": "stay",
    "sweep_stay": "stay",
}


def normalize_sweep_command(raw: Any) -> Optional[str]:
    cmd = str(raw or "").strip().lower()
    return cmd if cmd in SWEEP_COMMANDS else None


def command_from_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return KEY_TO_SWEEP.get(key.lower())


class SweepWebClient:
    """Background heartbeat + command poll against Terminal A's debug stream."""

    def __init__(self, base_url: str, timeout_s: float = 0.45) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "accepts_input": False,
            "phase": "starting",
            "accepted": 0,
            "target": 0,
            "visits": 0,
            "step_mm": 2.0,
            "z_step_mm": 20.0,
            "rot_deg": 2.0,
            "message": "",
        }
        self.ok = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.base_url or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def set_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._post_heartbeat()
            self._stop.wait(0.4)

    def _post_heartbeat(self) -> None:
        payload = self.snapshot()
        try:
            self._request("POST", "/sweep/heartbeat", payload)
            self.ok = True
        except Exception:
            self.ok = False

    def pop_command(self) -> Optional[str]:
        try:
            data = self._request("GET", "/sweep/command")
        except Exception:
            return None
        return normalize_sweep_command((data or {}).get("command"))

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)


def apply_sweep_command(
    command: str,
    *,
    controller: Any,
    ik: Any,
    arm: str,
    step_m: float,
    rot_deg: float,
    move_s: float,
    control_dt: float,
    log: Callable[[str], None] = print,
    z_step_m: float = 0.02,
    joint_step_deg: float = 2.0,
) -> tuple[Optional[str], float, float, float]:
    """Run one web/keyboard command. Menu actions return next/skip/quit/stay."""
    cmd = normalize_sweep_command(command)
    if cmd is None:
        return None, step_m, z_step_m, joint_step_deg
    if cmd in MENU_ACTIONS:
        return MENU_ACTIONS[cmd], step_m, z_step_m, joint_step_deg
    if cmd == "sweep_step_halve":
        step_m = max(0.002, step_m / 2.0)
        z_step_m = max(0.005, z_step_m / 2.0)
        log(f"[CART] 步长 XY {step_m * 1000:.1f} mm  Z {z_step_m * 1000:.1f} mm")
        return None, step_m, z_step_m, joint_step_deg
    if cmd == "sweep_step_double":
        step_m = min(0.02, step_m * 2.0)
        z_step_m = min(0.04, z_step_m * 2.0)
        log(f"[CART] 步长 XY {step_m * 1000:.1f} mm  Z {z_step_m * 1000:.1f} mm")
        return None, step_m, z_step_m, joint_step_deg
    if cmd == "sweep_joint_step_halve":
        joint_step_deg = max(0.5, joint_step_deg / 2.0)
        log(f"[JOINT] 右臂关节步长 {joint_step_deg:.1f}°")
        return None, step_m, z_step_m, joint_step_deg
    if cmd == "sweep_joint_step_double":
        joint_step_deg = min(10.0, joint_step_deg * 2.0)
        log(f"[JOINT] 右臂关节步长 {joint_step_deg:.1f}°")
        return None, step_m, z_step_m, joint_step_deg
    if controller is None:
        log("[JOINT] 微调需要可用的手臂控制器")
        return None, step_m, z_step_m, joint_step_deg

    joint_spec = SWEEP_JOINT_COMMANDS.get(cmd)
    if joint_spec is not None:
        index, sign = joint_spec
        q_rad = _commanded_or_measured(controller, arm)
        if q_rad is None or len(q_rad) != 7:
            log("[JOINT] 没有可用的右臂目标；请先执行平举")
            return None, step_m, z_step_m, joint_step_deg
        q_new = list(q_rad)
        lower, upper = RIGHT_ARM_JOINT_LIMITS_RAD[index]
        q_new[index] = max(
            lower,
            min(upper, q_new[index] + sign * math.radians(joint_step_deg)),
        )
        duration = max(0.35, min(0.9, 0.25 + joint_step_deg * 0.065))
        _move_arm_smooth(controller, arm, q_new, duration, control_dt)
        log(
            f"[JOINT] {RIGHT_ARM_JOINT_NAMES[index]} "
            f"{math.degrees(q_rad[index]):.1f}° → {math.degrees(q_new[index]):.1f}° "
            f"({duration:.2f}s quintic)"
        )
        return None, step_m, z_step_m, joint_step_deg

    if ik is None:
        log("[CART] 微调需要 --backend sdk 且能加载 H2.urdf / robot_kinematics")
        return None, step_m, z_step_m, joint_step_deg
    if cmd == "sweep_follow":
        q_rad = controller.measured_arm_rad(arm)
        if not q_rad:
            log("[CART] 当前后端没有实测关节反馈，继续沿用最后下发目标")
            return None, step_m, z_step_m, joint_step_deg
        xyz, rpy = ik.fk_xyz_rpy(q_rad)
        log(f"[CART] 跟随实测 xyz={[round(v, 4) for v in xyz]}")
        return None, step_m, z_step_m, joint_step_deg

    spec = SWEEP_JOG_COMMANDS.get(cmd)
    if spec is None:
        return None, step_m, z_step_m, joint_step_deg
    kind, axis, sign = spec
    dxyz = [0.0, 0.0, 0.0]
    drpy = [0.0, 0.0, 0.0]
    use_z = kind == "xyz" and axis == 2
    if kind == "xyz":
        dxyz[axis] = sign * (z_step_m if use_z else step_m)
    else:
        drpy[axis] = sign * math.radians(rot_deg)
    q_rad = _commanded_or_measured(controller, arm)
    if q_rad is None:
        log("[CART] 没有可用的手臂目标；请先执行平举")
        return None, step_m, z_step_m, joint_step_deg
    xyz0, _ = ik.fk_xyz_rpy(q_rad)
    q_new, ok, pos_err, ori_err = ik.apply_delta(q_rad, dxyz, drpy)
    xyz1, _ = ik.fk_xyz_rpy(q_new)
    lifted = abs(xyz1[2] - xyz0[2])
    if use_z and (not ok or lifted < 0.4 * abs(dxyz[2])):
        # H2 shoulder pitch: more negative = higher. 2 mm Cartesian is invisible.
        q_new = list(q_rad)
        q_new[0] -= sign * math.radians(6.0)
        q_new[0] = max(math.radians(-80.0), min(math.radians(-35.0), q_new[0]))
        ok = True
        log(
            f"[CART] Z 改用肩 pitch {math.degrees(q_rad[0]):.1f}° → "
            f"{math.degrees(q_new[0]):.1f}°"
        )
    if not ok:
        log(f"[CART] IK 拒绝 pos_err={pos_err * 1000:.1f}mm ori_err={ori_err:.3f}")
        return None, step_m, z_step_m, joint_step_deg
    _move_arm_smooth(
        controller,
        arm,
        q_new,
        0.55 if use_z else move_s,
        control_dt,
    )
    xyz, rpy = ik.fk_xyz_rpy(q_new)
    log(
        f"[CART] xyz={[round(v, 4) for v in xyz]} "
        f"rpy_deg={[round(math.degrees(v), 1) for v in rpy]} "
        f"q_deg={[round(math.degrees(v), 1) for v in q_new]}"
    )
    return None, step_m, z_step_m, joint_step_deg


def _commanded_or_measured(controller: Any, arm: str) -> Optional[list[float]]:
    q_rad = None
    if hasattr(controller, "commanded_arm_rad"):
        q_rad = controller.commanded_arm_rad(arm)
    if q_rad is None and hasattr(controller, "measured_arm_rad"):
        measured = controller.measured_arm_rad(arm)
        q_rad = measured or None
    return None if q_rad is None else [float(value) for value in q_rad]


def _move_arm_smooth(
    controller: Any,
    arm: str,
    target_rad: list[float],
    seconds: float,
    control_dt: float,
) -> None:
    if hasattr(controller, "play_quintic_arm_rad"):
        controller.play_quintic_arm_rad(
            arm,
            target_rad,
            seconds=seconds,
            dt=control_dt,
        )
        return
    controller.apply_arm_rad(
        arm,
        target_rad,
        seconds=seconds,
        dt=control_dt,
    )


def current_ee_status(controller: Any, ik: Any, arm: str) -> dict[str, Any]:
    if controller is None or ik is None:
        return {}
    try:
        q_rad = _commanded_or_measured(controller, arm)
        if q_rad is None:
            return {}
        xyz, rpy = ik.fk_xyz_rpy(q_rad)
    except Exception:
        return {}
    return {
        "xyz": [round(float(v), 4) for v in xyz],
        "rpy_deg": [round(math.degrees(float(v)), 2) for v in rpy],
        "joint_deg": [round(math.degrees(float(v)), 2) for v in q_rad],
    }
