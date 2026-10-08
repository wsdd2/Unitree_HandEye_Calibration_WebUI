# -*- coding: utf-8 -*-
"""Terminal XYZ/RPY jog for the H2 wrist camera, solved with IK."""
from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable


def _read_key_linux() -> str | None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            return None
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows() -> str | None:
    import msvcrt

    if not msvcrt.kbhit():
        time.sleep(0.05)
        return None
    raw = msvcrt.getwch()
    if raw in {"\x00", "\xe0"}:
        msvcrt.getwch()
        return None
    return raw


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    if sys.platform.startswith("win"):
        return _read_key_windows()
    return _read_key_linux()


def print_help(step_m: float, rot_deg: float) -> None:
    print(
        f"[CART] 末端相机  step={step_m * 1000:.1f}mm  rot={rot_deg:.1f}deg\n"
        "  W/S X+/-   A/D Y+/-   R/F Z+/-   (torso 系)\n"
        "  I/K roll+/-  J/L pitch+/-  U/O yaw+/-  (相机/工具系)\n"
        "  [ ] 步长   空格=跟当前实测    n=确认并下一个   s=再抽   q=结束扫掠   c=回到菜单"
    )


def run_ee_keyboard(
    *,
    controller: Any,
    ik: Any,
    arm: str,
    step_m: float = 0.002,
    rot_deg: float = 2.0,
    move_s: float = 0.35,
    control_dt: float = 0.02,
    on_hold: Callable[[], None] | None = None,
) -> str:
    q_rad = controller.measured_arm_rad(arm)
    xyz, rpy = ik.fk_xyz_rpy(q_rad)
    print(
        f"[CART] 当前末端 xyz={[round(v, 4) for v in xyz]} "
        f"rpy_deg={[round(math.degrees(v), 1) for v in rpy]}"
    )
    print_help(step_m, rot_deg)
    rot = math.radians(rot_deg)
    while True:
        if on_hold is not None:
            on_hold()
        key = read_key()
        if key is None:
            continue
        key = key.lower()
        dxyz = [0.0, 0.0, 0.0]
        drpy = [0.0, 0.0, 0.0]
        if key == "w":
            dxyz[0] = step_m
        elif key == "s":
            dxyz[0] = -step_m
        elif key == "a":
            dxyz[1] = step_m
        elif key == "d":
            dxyz[1] = -step_m
        elif key == "r":
            dxyz[2] = step_m
        elif key == "f":
            dxyz[2] = -step_m
        elif key == "i":
            drpy[0] = rot
        elif key == "k":
            drpy[0] = -rot
        elif key == "j":
            drpy[1] = rot
        elif key == "l":
            drpy[1] = -rot
        elif key == "u":
            drpy[2] = rot
        elif key == "o":
            drpy[2] = -rot
        elif key == "[":
            step_m = max(0.002, step_m / 2.0)
            print_help(step_m, math.degrees(rot))
            continue
        elif key == "]":
            step_m = min(0.02, step_m * 2.0)
            print_help(step_m, math.degrees(rot))
            continue
        elif key in {" ", "0"}:
            q_rad = controller.measured_arm_rad(arm)
            xyz, rpy = ik.fk_xyz_rpy(q_rad)
            print(f"[CART] reset xyz={[round(v, 4) for v in xyz]}")
            continue
        elif key in {"n", "\r", "\n"}:
            return "next"
        elif key == "s":
            return "skip"
        elif key in {"q", "\x03"}:
            return "quit"
        elif key == "c":
            return "stay"
        else:
            continue

        q_new, ok, pos_err, ori_err = ik.apply_delta(q_rad, dxyz, drpy)
        if not ok:
            print(f"[CART] IK 拒绝 pos_err={pos_err*1000:.1f}mm ori_err={ori_err:.3f}")
            continue
        controller.apply_arm_rad(arm, q_new, seconds=move_s, dt=control_dt)
        q_rad = list(q_new)
        xyz, rpy = ik.fk_xyz_rpy(q_rad)
        print(
            f"[CART] xyz={[round(v, 4) for v in xyz]} "
            f"rpy_deg={[round(math.degrees(v), 1) for v in rpy]} "
            f"q_deg={[round(math.degrees(v), 1) for v in q_rad]}"
        )
