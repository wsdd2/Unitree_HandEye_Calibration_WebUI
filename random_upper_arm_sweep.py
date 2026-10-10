# -*- coding: utf-8 -*-
"""Drive H2 upper-arm joint-space poses. Default path is Unitree DDS, no ROS2.

Default --backend sdk publishes rt/arm_sdk (same topic as official H2 arm examples).
--backend ros is optional and only works if someone later installs /h2_arm services.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from handeye_calib.sweep_web import (
    SweepWebClient,
    apply_sweep_command,
    command_from_key,
    current_ee_status,
)
from handeye_calib.upper_arm_sweep import (
    DEFAULT_SEED_DEG,
    DEFAULT_SPAN_DEG,
    ReachablePoseSampler,
    SweepPose,
    parse_seed_deg,
    render_multi_joint_yaml,
    render_single_joint_yaml,
    sample_reachable_poses,
    write_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_MULTI = PROJECT_ROOT / "data" / "h2_random_upper_arm_multi.yaml"
DEFAULT_ROBOT_CONFIG = Path("/home/unitree/h2/unitree_sdk2/example/h2/config")
DEFAULT_SINGLE_REL = Path("single/joint_point.yaml")
DEFAULT_MULTI_REL = Path("single/multi_joint_points.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动生成并执行 20~30 个 H2 上臂随机可达关节位姿（默认复用 h2_arm ROS2 运控服务）"
    )
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument(
        "--backend",
        choices=("sdk", "ros"),
        default="ros",
        help="ros=/h2_arm Trigger 五次轨迹（默认）；sdk=本进程直接发布 rt/arm_sdk（备用）",
    )
    parser.add_argument("--iface", default="eth0", help="DDS 网卡，H2 本机通常是 eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--service-kp", type=float, default=140.0, help="ROS 运控节点关节 KP")
    parser.add_argument("--service-kd", type=float, default=3.0, help="ROS 运控节点关节 KD")
    parser.add_argument(
        "--no-gravity",
        action="store_true",
        help="关闭 URDF 重力前馈，tau=0。默认开启，否则平举会靠 KP 硬顶，跟踪误差大",
    )
    parser.add_argument(
        "--tau-scale",
        type=float,
        default=1.0,
        help="重力前馈缩放。1=按 URDF 质量全补，0.8 略保守",
    )
    parser.add_argument(
        "--wrist-payload-kg",
        type=float,
        default=0.75,
        help="右腕负载：Dex1-1 夹爪 0.55 + 腕部相机约 0.20。只有相机用 0.20",
    )
    parser.add_argument(
        "--no-enable-loco",
        dest="enable_loco",
        action="store_false",
        help="不调用 H2 LocoClient.EnableArmSDK()",
    )
    parser.add_argument(
        "--select-ai",
        action="store_true",
        help="当前不是 AI 时尝试 SelectMode(ai)。默认只检查，不自动切模式",
    )
    parser.add_argument(
        "--release-motion",
        action="store_true",
        help="退出 AI 进调试模式。站立机默认不要开，rt/arm_sdk 在 AI 下才有效",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25,
        help="write/multi=预生成点数；dwell=按 n 确认的目标数。0 表示 dwell 无限抽到按 q",
    )
    parser.add_argument(
        "--seed-deg",
        type=float,
        nargs=7,
        metavar=("PITCH", "ROLL", "YAW", "ELBOW", "W_ROLL", "W_PITCH", "W_YAW"),
        help="种子位姿，度；默认用 h2_arm 单臂关节 YAML 的出厂右/左臂点",
    )
    parser.add_argument(
        "--span-deg",
        type=float,
        nargs=7,
        metavar=("PITCH", "ROLL", "YAW", "ELBOW", "W_ROLL", "W_PITCH", "W_YAW"),
        help="相对种子的采样半宽，度",
    )
    parser.add_argument("--min-separation-deg", type=float, default=4.0)
    parser.add_argument(
        "--include-wrist",
        action="store_true",
        help="默认只随机上臂 4 轴；加上此项才扰动手腕",
    )
    parser.add_argument("--no-return-seed", action="store_true", help="结束后不回到种子位姿")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    parser.add_argument("--move-duration", type=float, default=3.0, help="每段关节插值时间，秒")
    parser.add_argument("--raise-duration", type=float, default=8.0, help="启动时从垂臂抬到平举的时间，秒")
    parser.add_argument("--dwell-s", type=float, default=1.5, help="dwell 模式到位后最少停留秒数，方便采图")
    parser.add_argument(
        "--wait-enter",
        action="store_true",
        default=True,
        help="dwell 到位后等按键（默认）。n/回车=下一个，s=看不见棋盘再抽，q=结束",
    )
    parser.add_argument(
        "--auto-advance",
        dest="wait_enter",
        action="store_false",
        help="dwell 停稳后自动走下一点，不按键",
    )
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--urdf", default="", help="H2 URDF，键盘 IK 用。默认搜 /home/unitree/MscapeTech/urdf/H2.urdf")
    parser.add_argument("--ee-link", default="", help="末端相机/手 link，默认 right/left_wrist_yaw_link")
    parser.add_argument("--cart-step-mm", type=float, default=2.0, help="网页 X/Y 一步，毫米")
    parser.add_argument("--cart-z-mm", type=float, default=20.0, help="网页 Z+ / Z- 一步，毫米。2 mm 看不出抬手")
    parser.add_argument("--cart-rot-deg", type=float, default=2.0, help="网页/键盘旋转一步，度")
    parser.add_argument(
        "--web-url",
        default="http://127.0.0.1:8081",
        help="终端 A 调试页地址。按钮点到这里。终端 A 换端口时一起改",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="不连网页，只保留终端 n/s/q",
    )
    parser.add_argument(
        "--mode",
        choices=("write", "multi", "dwell"),
        default="write",
        help="write=只写 YAML；multi=连续走完全部点；dwell=逐点停，等按键或 --auto-advance",
    )
    parser.add_argument("--out-multi", default=str(DEFAULT_LOCAL_MULTI), help="本地 multi YAML 输出路径")
    parser.add_argument(
        "--config-dir",
        default="",
        help="仅 --backend ros 时用：h2_arm 配置目录",
    )
    parser.add_argument(
        "--install-robot-yaml",
        action="store_true",
        help="仅 --backend ros：把 YAML 写进 h2_arm 实际读取的文件名",
    )
    parser.add_argument("--service-multi", default="/h2_arm/singlearmmultijoints")
    parser.add_argument("--service-single", default="/h2_arm/singlearmjoint")
    parser.add_argument(
        "--confirm-robot-motion",
        action="store_true",
        help="确认允许动臂；没有此开关最多只写文件",
    )
    parser.add_argument("--timeout-s", type=float, default=25.0, help="仅 ros：单次 service call 超时")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "dwell":
        if args.count < 0 or args.count > 200:
            raise SystemExit("dwell 的 --count 用 0 表示无限，或 1~200 个确认点")
    elif args.count < 8 or args.count > 40:
        raise SystemExit("write/multi 的 --count 建议 20~30，允许范围 8~40")
    if args.move_duration <= 0 or args.raise_duration <= 0 or args.dwell_s < 0 or args.control_dt <= 0:
        raise SystemExit("--move-duration/--raise-duration/--control-dt 必须 > 0，--dwell-s 必须 >= 0")
    if args.min_separation_deg <= 0:
        raise SystemExit("--min-separation-deg 必须 > 0")
    if args.mode in {"multi", "dwell"} and not args.confirm_robot_motion:
        raise SystemExit(f"--mode {args.mode} 会动真机，必须同时加 --confirm-robot-motion")
    if args.backend == "ros" and args.install_robot_yaml and not args.config_dir:
        args.config_dir = str(DEFAULT_ROBOT_CONFIG)
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s 必须 > 0")
    if args.tau_scale < 0:
        raise SystemExit("--tau-scale 必须 >= 0")
    if args.wrist_payload_kg < 0:
        raise SystemExit("--wrist-payload-kg 必须 >= 0")
    if args.cart_z_mm <= 0:
        raise SystemExit("--cart-z-mm 必须 > 0")


def build_poses(args: argparse.Namespace) -> list[SweepPose]:
    import random

    seed = parse_seed_deg(args.seed_deg) if args.seed_deg else DEFAULT_SEED_DEG[args.arm]
    span = parse_seed_deg(args.span_deg) if args.span_deg else DEFAULT_SPAN_DEG
    rng = random.Random(args.seed)
    return sample_reachable_poses(
        args.count,
        arm=args.arm,
        seed_deg=seed,
        span_deg=span,
        min_separation_deg=args.min_separation_deg,
        upper_arm_only=not args.include_wrist,
        return_to_seed=not args.no_return_seed,
        rng=rng,
    )


def print_poses(poses: list[SweepPose]) -> None:
    print(f"[SWEEP] {len(poses)} poses (deg)  order={('pitch', 'roll', 'yaw', 'elbow', 'w_roll', 'w_pitch', 'w_yaw')}")
    for pose in poses:
        values = " ".join(f"{value:8.3f}" for value in pose.as_list())
        print(f"  {pose.name:10s} {values}")


def run_ros2(args_cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args_cmd,
            check=False,
            timeout=timeout_s,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("找不到 ros2。这台机器人请用默认 --backend sdk，不要走 ROS。") from exc
    except subprocess.TimeoutExpired as exc:
        tail = ""
        if exc.stdout:
            tail = str(exc.stdout)[-400:]
        raise RuntimeError(
            f"{' '.join(args_cmd)} 等了 {timeout_s:.0f}s 没有返回。"
            "当前机上没有 /h2_arm 服务。请改用 --backend sdk。"
            + (f"\n{tail}" if tail else "")
        ) from exc


def ensure_arm_service(service: str, timeout_s: float = 8.0) -> None:
    print(f"[ROS2] checking {service} (timeout {timeout_s:.0f}s)")
    listed = run_ros2(["ros2", "service", "list"], timeout_s)
    text = listed.stdout or ""
    if listed.returncode != 0:
        raise RuntimeError(f"ros2 service list 失败:\n{text}")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if service not in names:
        preview = "\n".join(name for name in names if "h2_arm" in name) or "(当前域里没有任何 /h2_arm/* )"
        raise RuntimeError(
            f"当前 ROS 域看不到 {service}。\n"
            f"看到的 h2_arm 服务:\n{preview}\n"
            "请先启动 h2_arm_service_node，或明确使用备用 --backend sdk。"
        )
    print(f"[ROS2] found {service}")


def call_trigger(service: str, timeout_s: float) -> None:
    cmd = ["ros2", "service", "call", service, "std_srvs/srv/Trigger"]
    print(f"[ROS2] {' '.join(cmd)}")
    print(f"[ROS2] 正在等服务返回，最多 {timeout_s:.0f}s；这段时间按 n/q 无效，卡住用 Ctrl+C")
    completed = run_ros2(cmd, timeout_s)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"{service} 失败:\n{(completed.stdout or '').strip() or completed.returncode}")
    out = completed.stdout or ""
    if "success=False" in out or "success: false" in out.lower():
        raise RuntimeError(f"{service} 返回 success=False\n{out}")


def install_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config_dir = Path(args.config_dir) if args.config_dir else None
    multi = Path(args.out_multi)
    single = PROJECT_ROOT / "data" / "h2_random_upper_arm_single.yaml"
    if args.backend == "ros" and args.install_robot_yaml and config_dir is not None:
        multi = config_dir / DEFAULT_MULTI_REL
        single = config_dir / DEFAULT_SINGLE_REL
    return multi, single


def connect_sdk(args: argparse.Namespace) -> Any:
    from handeye_calib.h2_arm_sdk import H2ArmSdkController

    controller = H2ArmSdkController(
        iface=args.iface,
        domain_id=args.domain_id,
        kp=args.kp,
        kd=args.kd,
        gravity=not args.no_gravity,
        tau_scale=args.tau_scale,
        wrist_payload_kg=args.wrist_payload_kg,
        urdf=args.urdf,
    )
    controller.wait_lowstate()
    if args.release_motion:
        controller.release_motion_mode()
    else:
        controller.ensure_ai_mode(select_ai=args.select_ai)
    if args.enable_loco:
        controller.enable_arm_sdk()
        controller.print_control_status()
    return controller


def connect_ros_service(args: argparse.Namespace) -> Any:
    from handeye_calib.h2_arm_service import H2ArmServiceController

    config_dir = Path(args.config_dir) if args.config_dir else None
    controller = H2ArmServiceController(
        config_dir=config_dir,
        service_joint=args.service_single,
        kp=args.service_kp,
        kd=args.service_kd,
    )
    print(
        f"[ROS] h2_arm_service_node config={controller.config_dir} "
        f"joint_service={controller.service_joint} "
        f"kp={args.service_kp:.1f} kd={args.service_kd:.1f}"
    )
    return controller


def run_write(args: argparse.Namespace, poses: list[SweepPose]) -> Path:
    multi_path, _ = install_paths(args)
    text = render_multi_joint_yaml(
        poses,
        arm=args.arm,
        move_duration=args.move_duration,
        control_dt=args.control_dt,
    )
    write_text(multi_path, text)
    print(f"[WRITE] {multi_path.resolve()}")
    return multi_path


def move_to_pose(
    args: argparse.Namespace,
    pose: SweepPose,
    single_path: Path,
    label: str,
    controller: Any | None,
    *,
    duration: float | None = None,
) -> None:
    seconds = args.move_duration if duration is None else duration
    print(f"[MOVE] {label} {pose.name} {pose.as_list()}  {seconds:.1f}s")
    if controller is not None:
        controller.ramp_arm(args.arm, pose.as_list(), seconds, args.control_dt)
        if args.dwell_s > 0:
            print(f"[DWELL] hold {args.dwell_s:.1f}s，等臂停稳")
            controller.hold(args.dwell_s, args.control_dt)
        return
    write_text(
        single_path,
        render_single_joint_yaml(
            pose,
            arm=args.arm,
            move_duration=seconds,
            control_dt=args.control_dt,
        ),
    )
    print(f"[ROS] {label} -> {single_path}")
    call_trigger(args.service_single, max(args.timeout_s, seconds + 20.0))
    if args.dwell_s > 0:
        print(f"[DWELL] hold {args.dwell_s:.1f}s，等臂停稳")
        time.sleep(args.dwell_s)


def run_multi(args: argparse.Namespace, poses: list[SweepPose]) -> None:
    run_write(args, poses)
    if args.backend == "sdk":
        controller = connect_sdk(args)
        try:
            for index, pose in enumerate(poses, start=1):
                move_to_pose(args, pose, Path("."), f"{index}/{len(poses)}", controller)
        finally:
            controller.release()
        return
    if not args.install_robot_yaml:
        print(
            "[WARN] 未加 --install-robot-yaml：服务仍会读机器人上的旧 multi_joint_points.yaml。"
        )
    ensure_arm_service(args.service_multi)
    timeout = max(args.timeout_s, (len(poses) + 1) * args.move_duration + 20.0)
    call_trigger(args.service_multi, timeout)


def make_sampler(args: argparse.Namespace) -> ReachablePoseSampler:
    import random

    seed = parse_seed_deg(args.seed_deg) if args.seed_deg else DEFAULT_SEED_DEG[args.arm]
    span = parse_seed_deg(args.span_deg) if args.span_deg else DEFAULT_SPAN_DEG
    return ReachablePoseSampler(
        arm=args.arm,
        seed_deg=seed,
        span_deg=span,
        min_separation_deg=args.min_separation_deg,
        upper_arm_only=not args.include_wrist,
        rng=random.Random(args.seed),
    )


def make_web_client(args: argparse.Namespace) -> SweepWebClient | None:
    if args.no_web or not str(args.web_url).strip():
        return None
    client = SweepWebClient(args.web_url)
    client.set_status(
        accepts_input=False,
        phase="starting",
        accepted=0,
        target=int(args.count),
        visits=0,
        step_mm=float(args.cart_step_mm),
        z_step_mm=float(args.cart_z_mm),
        rot_deg=float(args.cart_rot_deg),
        joint_step_deg=2.0,
        message="正在连接手臂",
    )
    client.start()
    print(f"[WEB] 扫掠按钮 -> {args.web_url}  到位后网页按钮才可点")
    return client


def publish_web_status(
    client: SweepWebClient | None,
    args: argparse.Namespace,
    *,
    accepts_input: bool,
    phase: str,
    accepted: int,
    visits: int,
    controller: Any,
    ik: Any,
    step_m: float,
    z_step_m: float = 0.02,
    joint_step_deg: float = 2.0,
    message: str = "",
) -> None:
    if client is None:
        return
    extra = current_ee_status(controller, ik, args.arm)
    client.set_status(
        accepts_input=accepts_input,
        phase=phase,
        accepted=accepted,
        target=int(args.count),
        visits=visits,
        step_mm=step_m * 1000.0,
        z_step_mm=z_step_m * 1000.0,
        rot_deg=float(args.cart_rot_deg),
        joint_step_deg=float(joint_step_deg),
        message=message,
        **extra,
    )


def wait_after_pose(
    args: argparse.Namespace,
    accepted: int,
    target: int,
    controller: Any,
    ik: Any,
    client: SweepWebClient | None,
    visits: int,
    step_m: float,
    z_step_m: float,
    joint_step_deg: float,
) -> tuple[str, float, float, float]:
    from handeye_calib.ee_keyboard import read_key

    goal = "∞" if target <= 0 else str(target)
    print(
        f"[DWELL] 已确认 {accepted}/{goal}。网页 Save 后点「下一个」。"
        " 再抽=看不见棋盘。XYZ/RPY 在网页微调。终端仍可按 n/s/q。"
    )
    publish_web_status(
        client,
        args,
        accepts_input=True,
        phase="dwell",
        accepted=accepted,
        visits=visits,
        controller=controller,
        ik=ik,
        step_m=step_m,
        z_step_m=z_step_m,
        joint_step_deg=joint_step_deg,
        message="等待网页按钮",
    )
    while True:
        cmd = client.pop_command() if client is not None else None
        if cmd is None:
            cmd = command_from_key(read_key())
        if cmd is None:
            time.sleep(0.03)
            continue
        action, step_m, z_step_m, joint_step_deg = apply_sweep_command(
            cmd,
            controller=controller,
            ik=ik,
            arm=args.arm,
            step_m=step_m,
            rot_deg=args.cart_rot_deg,
            move_s=0.35,
            control_dt=args.control_dt,
            z_step_m=z_step_m,
            joint_step_deg=joint_step_deg,
        )
        publish_web_status(
            client,
            args,
            accepts_input=True,
            phase="dwell",
            accepted=accepted,
            visits=visits,
            controller=controller,
            ik=ik,
            step_m=step_m,
            z_step_m=z_step_m,
            joint_step_deg=joint_step_deg,
            message="" if action is None else f"收到 {action}",
        )
        if action in {None, "stay"}:
            continue
        publish_web_status(
            client,
            args,
            accepts_input=False,
            phase="moving",
            accepted=accepted,
            visits=visits,
            controller=controller,
            ik=ik,
            step_m=step_m,
            z_step_m=z_step_m,
            joint_step_deg=joint_step_deg,
            message=f"执行 {action}",
        )
        return action, step_m, z_step_m, joint_step_deg


def run_dwell(args: argparse.Namespace) -> None:
    _, single_path = install_paths(args)
    client = make_web_client(args)
    controller = None
    accepted = 0
    visits = 0
    step_m = max(0.002, float(args.cart_step_mm) / 1000.0)
    z_step_m = max(0.005, float(args.cart_z_mm) / 1000.0)
    joint_step_deg = 2.0
    try:
        if args.backend == "sdk":
            controller = connect_sdk(args)
        else:
            controller = connect_ros_service(args)
        sampler = make_sampler(args)
        ik = None
        if controller is not None:
            try:
                from handeye_calib.h2_ee_ik import H2CameraIK

                ik = H2CameraIK(urdf_path=args.urdf, arm=args.arm, ee_link=args.ee_link)
            except Exception as exc:
                print(f"[CART][WARN] 网页 IK 不可用: {exc}")
        target = int(args.count)
        print(
            "[DWELL] 先平举，再在附近小范围随机。到位后用网页按钮：下一个 / 再抽 / XYZ-RPY / 结束。"
            + (" 终端也可按 n/s/q。" if args.wait_enter else " 已开 --auto-advance，停稳后自动计入并前进。")
        )
        raise_pose = sampler.seed_pose(name="raise")
        print(f"[SWEEP] 平举种子 {raise_pose.as_list()}")
        publish_web_status(
            client,
            args,
            accepts_input=False,
            phase="raising",
            accepted=accepted,
            visits=visits,
            controller=controller,
            ik=ik,
            step_m=step_m,
            z_step_m=z_step_m,
            joint_step_deg=joint_step_deg,
            message="正在平举",
        )
        move_to_pose(
            args,
            raise_pose,
            single_path,
            "raise",
            controller,
            duration=args.raise_duration,
        )
        if args.wait_enter:
            action, step_m, z_step_m, joint_step_deg = wait_after_pose(
                args,
                accepted,
                target,
                controller,
                ik,
                client,
                visits,
                step_m,
                z_step_m,
                joint_step_deg,
            )
            if action == "quit":
                print("[DWELL] 平举后结束")
                return
        while target <= 0 or accepted < target:
            pose = sampler.next_pose()
            visits += 1
            goal = "∞" if target <= 0 else str(target)
            publish_web_status(
                client,
                args,
                accepts_input=False,
                phase="moving",
                accepted=accepted,
                visits=visits,
                controller=controller,
                ik=ik,
                step_m=step_m,
                z_step_m=z_step_m,
                joint_step_deg=joint_step_deg,
                message=f"移动 visit {visits}",
            )
            move_to_pose(args, pose, single_path, f"visit {visits} accepted {accepted}/{goal}", controller)
            if args.wait_enter:
                action, step_m, z_step_m, joint_step_deg = wait_after_pose(
                    args,
                    accepted,
                    target,
                    controller,
                    ik,
                    client,
                    visits,
                    step_m,
                    z_step_m,
                    joint_step_deg,
                )
            else:
                action = "next"
            if action == "quit":
                break
            if action == "next":
                accepted += 1
        if not args.no_return_seed:
            publish_web_status(
                client,
                args,
                accepts_input=False,
                phase="returning",
                accepted=accepted,
                visits=visits,
                controller=controller,
                ik=ik,
                step_m=step_m,
                z_step_m=z_step_m,
                joint_step_deg=joint_step_deg,
                message="回到平举",
            )
            move_to_pose(args, sampler.seed_pose(name="raise"), single_path, "return", controller)
    finally:
        if client is not None:
            client.stop()
        if controller is not None:
            controller.release()
    print(f"[DWELL] done accepted={accepted} visits={visits}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    print(f"[SWEEP] backend={args.backend} mode={args.mode} arm={args.arm}")
    if args.mode == "dwell":
        run_dwell(args)
        return 0
    poses = build_poses(args)
    print_poses(poses)
    if args.mode == "write":
        run_write(args, poses)
        print(
            "只写了 YAML，没有动臂。采图优先复用 ROS 运控服务：\n"
            "  python3 random_upper_arm_sweep.py --backend ros --mode dwell "
            "--count 0 --confirm-robot-motion\n"
            "DDS 直控备用：\n"
            "  python3 random_upper_arm_sweep.py --backend sdk --mode dwell "
            "--count 0 --confirm-robot-motion --iface eth0\n"
            "连续走完不采图：\n"
            "  python3 random_upper_arm_sweep.py --backend sdk --mode multi "
            "--count 25 --confirm-robot-motion --auto-advance --iface eth0"
        )
        return 0
    run_multi(args, poses)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SWEEP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"[SWEEP][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
