# -*- coding: utf-8 -*-
"""H2 arm joint moves over Unitree DDS rt/arm_sdk. No ROS2."""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence


COMMON_SDK_ROOTS = (
    Path("/home/unitree/unitree_sdk2_python"),
    Path("/home/unitree/unitree_sdk2/python"),
    Path("/home/unitree/h2/unitree_sdk2_python"),
    Path("/opt/unitree_sdk2_python"),
)


def _sdk_search_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [
        here.parents[2] / "unitree_sdk2_python",
        here.parents[1] / "third_party" / "unitree_sdk2_python",
        *COMMON_SDK_ROOTS,
    ]
    seen: set[str] = set()
    ordered: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(root)
    return ordered


def _purge_module(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _cyclonedds_site_dirs() -> list[Path]:
    home = Path.home()
    dirs = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/local/lib/python3/dist-packages"),
        home / ".local" / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages",
        home / ".local" / "lib" / "python3.10" / "site-packages",
        home / "anaconda3" / "envs" / "unitree" / "lib" / "python3.10" / "site-packages",
    ]
    dirs.extend(sorted((home / ".local" / "lib").glob("python3.*/site-packages")))
    out: list[Path] = []
    seen: set[str] = set()
    for folder in dirs:
        key = str(folder)
        if key in seen or not folder.is_dir():
            continue
        pkg = folder / "cyclonedds"
        if pkg.is_dir() and list(pkg.glob("_clayer*")):
            seen.add(key)
            out.append(folder)
    return out


def _ensure_cyclonedds_clayer() -> None:
    try:
        import cyclonedds._clayer  # noqa: F401
        return
    except ImportError:
        pass
    for folder in _cyclonedds_site_dirs():
        path = str(folder)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        _purge_module("cyclonedds")
        try:
            import cyclonedds._clayer  # noqa: F401
            return
        except ImportError:
            continue


def ensure_unitree_sdk2py() -> None:
    _ensure_cyclonedds_clayer()
    try:
        import unitree_sdk2py  # noqa: F401
        return
    except ImportError:
        pass
    tried: list[str] = []
    for root in _sdk_search_roots():
        tried.append(str(root))
        if (root / "unitree_sdk2py").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    _ensure_cyclonedds_clayer()
    try:
        import unitree_sdk2py  # noqa: F401
    except ImportError as exc:
        existing = [path for path in tried if (Path(path) / "unitree_sdk2py").is_dir()]
        raise RuntimeError(
            "找不到 unitree_sdk2py。\n"
            f"  实际报错: {exc}\n"
            f"  已存在的 SDK 目录: {existing or '无'}\n"
            "  在 H2 上应使用:\n"
            "    export PYTHONPATH=<unitree_sdk2_python目录>:$PYTHONPATH\n"
            "  缺的是 cyclonedds._clayer 时，不要只加 SDK 目录。"
        ) from exc


WAIST_JOINTS = (12, 13, 14)
LEFT_ARM_JOINTS = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_JOINTS = (22, 23, 24, 25, 26, 27, 28)
WEIGHT_INDEX = 31
ALL_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
HOLD_JOINTS = ALL_ARM_JOINTS
ARM_JOINTS = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}
TRACK_RESYNC_RAD = math.radians(12.0)


class H2ArmSdkController:
    def __init__(
        self,
        iface: str = "eth0",
        domain_id: int = 0,
        kp: float = 80.0,
        kd: float = 1.5,
        gravity: bool = True,
        tau_scale: float = 1.0,
        wrist_payload_kg: float = 0.75,
        urdf: str = "",
    ) -> None:
        ensure_unitree_sdk2py()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.kp = float(kp)
        self.kd = float(kd)
        self._gravity = None
        self._format_tau = None
        if gravity:
            try:
                from .h2_gravity import H2GravityCompensator, format_tau_summary

                self._gravity = H2GravityCompensator(
                    urdf,
                    wrist_payload_kg=wrist_payload_kg,
                    payload_arm="right",
                    tau_scale=tau_scale,
                )
                self._format_tau = format_tau_summary
                print(
                    f"[SDK] 重力补偿已开 urdf={self._gravity.urdf} "
                    f"tau_scale={float(tau_scale):.2f} "
                    f"wrist_payload={float(wrist_payload_kg):.2f}kg"
                )
            except Exception as exc:
                print(f"[SDK][WARN] 重力补偿未启用，tau=0: {exc}")
        else:
            print("[SDK] 重力补偿关闭，tau=0")
        self._low_cmd_cls = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()
        self._cmd = self._low_cmd_cls()
        self._latest: Any = None
        self._last_q: dict[int, float] | None = None
        self._last_dq: dict[int, float] = {}
        self._weight = 0.0
        self._state_lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._servo_stop = threading.Event()
        self._servo_thread: threading.Thread | None = None
        self._servo_dt = 0.02
        self._loco: Any = None
        if iface:
            ChannelFactoryInitialize(int(domain_id), iface)
        else:
            ChannelFactoryInitialize(int(domain_id))
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_lowstate, 10)
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        print(f"[SDK] rt/arm_sdk iface={iface or '-'} domain={domain_id}")

    def _on_lowstate(self, msg: Any) -> None:
        self._latest = msg

    def wait_lowstate(self, timeout: float = 5.0) -> Any:
        deadline = time.time() + timeout
        while self._latest is None and time.time() < deadline:
            time.sleep(0.02)
        if self._latest is None:
            raise RuntimeError("等不到 rt/lowstate。检查 --iface，以及机器人是否已开机进运控。")
        return self._latest

    def _loco_client(self) -> Any:
        if self._loco is not None:
            return self._loco
        from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient

        client = LocoClient()
        client.SetTimeout(5.0)
        client.Init()
        self._loco = client
        return client

    def motion_mode(self) -> tuple[int, dict[str, Any] | None]:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        return client.CheckMode()

    def select_ai_mode(self) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        code, _ = client.SelectMode("ai")
        print(f"[SDK] SelectMode(ai) code={code}")
        time.sleep(2.0)

    def print_control_status(self) -> str:
        status, result = self.motion_mode()
        name = ""
        if isinstance(result, dict):
            name = str(result.get("name") or "")
        fsm_code = fsm_id = arm_code = arm_on = None
        try:
            loco = self._loco_client()
            fsm_code, fsm_id = loco.GetFsmId()
            arm_code, arm_on = loco.GetArmSdkStatus()
        except Exception as exc:
            print(f"[SDK] 读 Loco 状态失败: {exc}")
        print(
            f"[SDK] MotionSwitcher={status} {result}  "
            f"fsm=({fsm_code},{fsm_id}) arm_sdk=({arm_code},{arm_on})"
        )
        return name

    def ensure_ai_mode(self, select_ai: bool = False) -> None:
        name = self.print_control_status()
        if "ai" not in name.lower():
            if select_ai:
                self.select_ai_mode()
                name = self.print_control_status()
            if "ai" not in name.lower():
                raise RuntimeError(
                    f"当前不在 AI 运控模式（MotionSwitcher={name or '空'}）。"
                    "请用遥控器切到 AI 并站稳后再跑；不要先 ReleaseMode。"
                    "若确认要脚本切 AI，加 --select-ai。"
                )

    def enable_arm_sdk(self) -> None:
        try:
            client = self._loco_client()
        except ImportError as exc:
            print(f"[SDK] 没有 H2 LocoClient，跳过 EnableArmSDK: {exc}")
            return
        code = client.SetArmSdkStatus(True)
        status_code, enabled = client.GetArmSdkStatus()
        print(f"[SDK] SetArmSdkStatus(True) code={code}  GetArmSdkStatus=({status_code},{enabled})")
        if enabled in {False, 0, "0", None}:
            print("[SDK][WARN] ArmSDK 仍未打开。先确认已进 AI 并站稳，再重跑。")
        time.sleep(0.3)
        self.engage()

    def disable_arm_sdk(self) -> None:
        if self._loco is None:
            return
        try:
            code = self._loco.SetArmSdkStatus(False)
            print(f"[SDK] SetArmSdkStatus(False) code={code}")
        except Exception as exc:
            print(f"[SDK] DisableArmSDK 失败: {exc}")

    def release_motion_mode(self, retries: int = 3) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        status, result = client.CheckMode()
        print(f"[SDK] MotionSwitcher before: {status} {result}")
        if not isinstance(result, dict) or not result.get("name"):
            print("[SDK] motion mode already released")
            return
        for attempt in range(1, retries + 1):
            code, data = client.ReleaseMode()
            print(f"[SDK] ReleaseMode {attempt}: {code} {data}")
            time.sleep(1.0)
            status, result = client.CheckMode()
            print(f"[SDK] MotionSwitcher after: {status} {result}")
            if isinstance(result, dict) and not result.get("name"):
                return
        raise RuntimeError("MotionSwitcher 仍占用；arm_sdk 指令会被丢掉")

    def _measured_hold(self) -> dict[int, float]:
        msg = self.wait_lowstate(2.0)
        return {joint: float(msg.motor_state[joint].q) for joint in HOLD_JOINTS}

    def hold_positions(self) -> dict[int, float]:
        measured = self._measured_hold()
        with self._state_lock:
            last = dict(self._last_q) if self._last_q is not None else None
            servo_on = self._servo_thread is not None
        if last is None:
            return measured
        if servo_on:
            # Keep commanding the last pose. Following sag here is what makes the arm drop.
            return last
        for joint in HOLD_JOINTS:
            if abs(float(last[joint]) - measured[joint]) > TRACK_RESYNC_RAD:
                err = max(abs(math.degrees(float(last[j]) - measured[j])) for j in HOLD_JOINTS)
                print(f"[SDK][WARN] 指令和实测最大偏差 {err:.1f}°，按实测重新咬合")
                with self._state_lock:
                    self._last_q = None
                    self._last_dq = {}
                return measured
        return last

    def commanded_arm_rad(self, arm: str) -> list[float] | None:
        with self._state_lock:
            if self._last_q is None:
                return None
            return [float(self._last_q[joint]) for joint in ARM_JOINTS[arm]]

    def start_servo(self) -> None:
        if self._servo_thread is not None:
            return
        self._servo_stop.clear()
        self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
        self._servo_thread.start()
        print("[SDK] 50Hz 伺服已开：等到点「结束」之前会一直发 rt/arm_sdk，否则手臂会垂下")

    def stop_servo(self) -> None:
        self._servo_stop.set()
        thread = self._servo_thread
        self._servo_thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def _servo_loop(self) -> None:
        while not self._servo_stop.wait(self._servo_dt):
            with self._state_lock:
                command_q = dict(self._last_q) if self._last_q is not None else None
                weight = self._weight
                command_dq = dict(self._last_dq)
            if command_q is None:
                continue
            self._publish(command_q, weight, command_dq)

    def engage(self, seconds: float = 1.5, dt: float = 0.02) -> None:
        """Hold the current pose and ramp weight 0→1 so the next raise is accepted."""
        with self._state_lock:
            already_engaged = self._servo_thread is not None and self._weight >= 0.99
        if already_engaged:
            print("[SDK] arm_sdk 已咬合，跳过 weight 0→1（避免运动中掉臂）")
            return
        steps = max(1, int(math.ceil(max(seconds, dt) / dt)))
        print(f"[SDK] 当前姿态咬合 weight 0→1，{seconds:.1f}s")
        for step in range(1, steps + 1):
            self.write(self._measured_hold(), weight=step / steps)
            time.sleep(dt)
        self.write(self._measured_hold(), weight=1.0)
        self.start_servo()
        hold_q = self.hold_positions()
        extra = ""
        if self._gravity is not None and self._format_tau is not None:
            extra = "  tau_right " + self._format_tau(self._gravity.motor_tau(hold_q, self._latest))
        print(f"[SDK] engaged right={ [round(v, 1) for v in self.measured_arm_deg('right')] }{extra}")

    def write(
        self,
        command_q: dict[int, float],
        weight: float = 1.0,
        dq: dict[int, float] | None = None,
    ) -> None:
        payload = dict(command_q)
        velocities = {joint: 0.0 for joint in HOLD_JOINTS}
        if dq:
            for joint, value in dq.items():
                velocities[int(joint)] = float(value)
        with self._state_lock:
            self._last_q = payload
            self._last_dq = velocities
            self._weight = float(weight)
        self._publish(payload, float(weight), velocities)

    def _publish(
        self,
        command_q: dict[int, float],
        weight: float,
        dq: dict[int, float] | None = None,
    ) -> None:
        with self._pub_lock:
            cmd = self._cmd
            low = self._latest
            if hasattr(cmd, "mode_pr"):
                cmd.mode_pr = 0
            if low is not None and hasattr(cmd, "mode_machine") and hasattr(low, "mode_machine"):
                cmd.mode_machine = int(low.mode_machine)
            cmd.motor_cmd[WEIGHT_INDEX].q = float(weight)
            taus: dict[int, float] = {}
            if self._gravity is not None:
                try:
                    taus = self._gravity.motor_tau(command_q, low)
                except Exception as exc:
                    print(f"[SDK][WARN] 重力补偿计算失败，本帧 tau=0: {exc}")
            for joint in HOLD_JOINTS:
                motor = cmd.motor_cmd[joint]
                if hasattr(motor, "mode"):
                    motor.mode = 1
                motor.q = float(command_q[joint])
                motor.dq = float(dq.get(joint, 0.0)) if dq else 0.0
                motor.kp = self.kp
                motor.kd = self.kd
                motor.tau = float(taus.get(joint, 0.0))
            cmd.crc = self._crc.Crc(cmd)
            ok = self._pub.Write(cmd, 1.0)
        if not ok:
            print("[SDK][WARN] rt/arm_sdk Write 失败")

    def measured_arm_deg(self, arm: str) -> list[float]:
        return [math.degrees(value) for value in self.measured_arm_rad(arm)]

    def measured_arm_rad(self, arm: str) -> list[float]:
        msg = self.wait_lowstate(2.0)
        return [float(msg.motor_state[joint].q) for joint in ARM_JOINTS[arm]]

    def apply_arm_rad(
        self,
        arm: str,
        target_rad: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        joints = ARM_JOINTS[arm]
        if len(target_rad) != 7:
            raise ValueError("target_rad must have 7 values")
        start = self.hold_positions()
        target = dict(start)
        for joint, value in zip(joints, target_rad):
            target[joint] = float(value)
        steps = max(1, int(math.ceil(max(seconds, dt) / dt)))
        for step in range(1, steps + 1):
            ratio = step / steps
            command = {
                joint: start[joint] * (1.0 - ratio) + target[joint] * ratio
                for joint in HOLD_JOINTS
            }
            self.write(command, weight=1.0)
            time.sleep(dt)
        self.write(target, weight=1.0)

    def play_quintic_arm_rad(
        self,
        arm: str,
        target_rad: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        """Synchronized 5th-order joint move. Velocity and acceleration start and end at zero."""
        joints = ARM_JOINTS[arm]
        if len(target_rad) != 7:
            raise ValueError("target_rad must have 7 values")
        start = self.hold_positions()
        q0 = [float(start[joint]) for joint in joints]
        q1 = [float(value) for value in target_rad]
        steps = max(1, int(math.ceil(max(float(seconds), dt) / dt)))
        for step in range(1, steps + 1):
            phase = step / steps
            blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            command = dict(start)
            for joint, begin, end in zip(joints, q0, q1):
                command[joint] = begin + (end - begin) * blend
            self.write(command, weight=1.0)
            time.sleep(dt)
        final = dict(start)
        for joint, value in zip(joints, q1):
            final[joint] = value
        self.write(final, weight=1.0)

    def ramp_arm(
        self,
        arm: str,
        target_deg: Sequence[float],
        seconds: float,
        dt: float,
    ) -> None:
        joints = ARM_JOINTS[arm]
        if len(target_deg) != 7:
            raise ValueError("target_deg must have 7 values")
        start = self.hold_positions()
        target = dict(start)
        for joint, deg in zip(joints, target_deg):
            target[joint] = math.radians(float(deg))
        print(
            f"[SDK] {arm} now={[round(v, 1) for v in self.measured_arm_deg(arm)]} "
            f"cmd={[round(float(v), 1) for v in target_deg]}"
        )
        steps = max(1, int(math.ceil(max(seconds, dt) / dt)))
        for step in range(1, steps + 1):
            ratio = step / steps
            command = {
                joint: start[joint] * (1.0 - ratio) + target[joint] * ratio
                for joint in HOLD_JOINTS
            }
            self.write(command, weight=1.0)
            time.sleep(dt)
        self.write(target, weight=1.0)
        after = self.measured_arm_deg(arm)
        err = max(abs(a - float(b)) for a, b in zip(after, target_deg))
        print(f"[SDK] {arm} after={[round(v, 1) for v in after]} err={err:.1f}deg")
        if err > 15.0:
            print(
                "[SDK][WARN] 手臂暂未跟上指令；保持当前目标继续发 50Hz。"
                "不会在运动中重新执行 weight 0→1，以免突然下垂再抬起。"
            )

    def hold(self, seconds: float, dt: float) -> None:
        if self._servo_thread is not None:
            time.sleep(max(0.0, seconds))
            return
        q = self.hold_positions()
        steps = max(1, int(math.ceil(max(seconds, dt) / dt)))
        for _ in range(steps):
            self.write(q, weight=1.0)
            time.sleep(dt)

    def release(self, seconds: float = 0.5, dt: float = 0.02) -> None:
        q = self.hold_positions()
        self.stop_servo()
        steps = max(1, int(math.ceil(max(seconds, dt) / dt)))
        for step in range(1, steps + 1):
            weight = max(0.0, 1.0 - step / steps)
            self.write(q, weight=weight)
            time.sleep(dt)
        self.disable_arm_sdk()
        print("[SDK] arm_sdk weight released")
