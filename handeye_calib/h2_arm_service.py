# -*- coding: utf-8 -*-
"""H2 arm control through the existing ROS2 Trigger service node.

The service node remains the only ``rt/arm_sdk`` publisher.  Each target is
written to its install-space YAML and executed by the node's quintic joint
trajectory, which is smoother and safer than the interactive keyboard target
jumps.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence


SERVICE_JOINT = "/h2_arm/singlearmjoint"


def resolve_arm_config_dir() -> Path:
    override = os.environ.get("H2_ARM_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / "h2arm_control_ws/install/h2arm_control/share/h2arm_control/config"


def _fmt(values: Sequence[float]) -> str:
    return ", ".join(f"{float(value):.6f}" for value in values)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_joint_yaml(
    path: Path,
    *,
    arm: str,
    target_deg: Sequence[float],
    seconds: float,
    control_dt: float = 0.02,
    kp: float = 140.0,
    kd: float = 3.0,
) -> None:
    if arm not in {"left", "right"} or len(target_deg) != 7:
        raise ValueError("joint target requires left/right and seven angles")
    zero = [0.0] * 7
    left = list(target_deg) if arm == "left" else zero
    right = list(target_deg) if arm == "right" else zero
    text = (
        f'control_mode: "{arm}"\n'
        f"move_duration: {max(0.25, float(seconds)):.3f}\n"
        f"control_dt: {max(0.005, min(0.05, float(control_dt))):.3f}\n"
        f"arm_kp: {float(kp):.1f}\n"
        f"arm_kd: {float(kd):.1f}\n"
        f"left_target: [{_fmt(left)}]\n"
        f"right_target: [{_fmt(right)}]\n"
    )
    _atomic_write(path, text)


def ros_shell_prefix() -> str:
    domain = os.environ.get(
        "H2_ARM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "")
    ).strip()
    commands = []
    if domain:
        commands.append(f"export ROS_DOMAIN_ID={domain}")
    commands.append("source /opt/ros/humble/setup.bash")
    commands.append(
        'if [ -f "$HOME/h2arm_control_ws/install/setup.bash" ]; then '
        'source "$HOME/h2arm_control_ws/install/setup.bash"; fi'
    )
    return " && ".join(commands)


def _run_ros(script: str, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=max(3.0, float(timeout_s)),
        check=False,
    )


def call_trigger(service: str, timeout_s: float) -> tuple[bool, str]:
    prefix = ros_shell_prefix()
    try:
        listed = _run_ros(f"{prefix} && ros2 service list", 8.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"无法查询 ROS2 服务：{exc}"
    services = {line.strip() for line in listed.stdout.splitlines()}
    if service not in services:
        domain = os.environ.get(
            "H2_ARM_ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "0")
        )
        return False, (
            f"找不到 {service}（ROS_DOMAIN_ID={domain or '0'}）；"
            "请先启动 h2_arm_service_node，并退出阻塞中的 /h2_arm/keyboardmove"
        )
    try:
        result = _run_ros(
            f"{prefix} && ros2 service call {service} std_srvs/srv/Trigger",
            max(30.0, float(timeout_s)),
        )
    except subprocess.TimeoutExpired:
        return False, f"{service} 调用超时；不要继续堆叠网页指令"
    except OSError as exc:
        return False, f"无法调用 ROS2 服务：{exc}"
    output = f"{result.stdout}\n{result.stderr}".strip()
    compact = output.replace(" ", "").lower()
    ok = result.returncode == 0 and (
        "success=true" in compact or "success:true" in compact
    )
    detail = output.splitlines()[-1] if output else f"exit {result.returncode}"
    return ok, detail


class H2ArmServiceController:
    """Controller-shaped adapter for ``random_upper_arm_sweep.py``."""

    def __init__(
        self,
        config_dir: Path | None = None,
        *,
        service_joint: str = SERVICE_JOINT,
        caller: Callable[[str, float], tuple[bool, str]] | None = None,
        kp: float = 140.0,
        kd: float = 3.0,
    ) -> None:
        self.config_dir = (
            Path(config_dir) if config_dir is not None else resolve_arm_config_dir()
        )
        self.service_joint = str(service_joint)
        self._caller = caller or call_trigger
        self.kp = float(kp)
        self.kd = float(kd)
        self._commanded: dict[str, list[float]] = {}

    def commanded_arm_rad(self, arm: str) -> list[float] | None:
        target = self._commanded.get(arm)
        return None if target is None else list(target)

    def measured_arm_rad(self, arm: str) -> list[float]:
        # The Trigger API has no state response.  Callers should use the last
        # accepted command after the initial raise.
        _ = arm
        return []

    def measured_arm_deg(self, arm: str) -> list[float]:
        return [math.degrees(value) for value in self.measured_arm_rad(arm)]

    def ramp_arm(
        self,
        arm: str,
        target_deg: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        path = self.config_dir / "single" / "joint_point.yaml"
        write_joint_yaml(
            path,
            arm=arm,
            target_deg=target_deg,
            seconds=seconds,
            control_dt=dt,
            kp=self.kp,
            kd=self.kd,
        )
        ok, detail = self._caller(self.service_joint, float(seconds) + 20.0)
        if not ok:
            raise RuntimeError(detail)
        self._commanded[arm] = [
            math.radians(float(value)) for value in target_deg
        ]

    def apply_arm_rad(
        self,
        arm: str,
        target_rad: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        self.ramp_arm(
            arm,
            [math.degrees(float(value)) for value in target_rad],
            seconds,
            dt,
        )

    def play_quintic_arm_rad(
        self,
        arm: str,
        target_rad: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        # MoveSingleArmJoint performs the quintic interpolation in C++.
        self.apply_arm_rad(arm, target_rad, seconds, dt)

    def hold(self, seconds: float, dt: float) -> None:
        _ = dt
        if seconds > 0:
            import time

            time.sleep(float(seconds))

    def release(self, seconds: float = 0.0) -> None:
        # This process never owns arm_sdk.  The service node continues holding.
        _ = seconds
