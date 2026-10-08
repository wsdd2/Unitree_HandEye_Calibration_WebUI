# -*- coding: utf-8 -*-
"""Sample reachable H2 upper-arm joint poses for h2_arm joint-space motion.

h2_arm /h2_arm/singlearmjoint reads a fixed YAML (degrees, 7 joints):
  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
  wrist_roll, wrist_pitch, wrist_yaw
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence


JOINT_ORDER = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
UPPER_ARM_INDICES = (0, 1, 2, 3)
WRIST_INDICES = (4, 5, 6)

# H2 URDF right/left arm limits (rad), matching unitree_ros H2.urdf.
URDF_LIMITS_RAD = {
    "right": (
        (-2.618, 1.833),
        (-2.494, 0.517),
        (-2.618, 2.618),
        (-0.986, 3.071),
        (-2.618, 2.618),
        (-0.576, 0.576),
        (-1.22, 1.22),
    ),
    "left": (
        (-2.618, 1.833),
        (-0.517, 2.494),
        (-2.618, 2.618),
        (-0.986, 3.071),
        (-2.618, 2.618),
        (-0.576, 0.576),
        (-1.22, 1.22),
    ),
}

# Keep the wrist camera up: chest-front raise, a bit below the previous -60°.
# H2 shoulder pitch 0=垂臂，负向前抬；绝对值越小，手臂越低。
SAFE_BOX_DEG = {
    "right": (
        (-63.0, -47.0),
        (-22.0, -6.0),
        (-10.0, 10.0),
        (32.0, 50.0),
        (-105.0, -75.0),
        (-12.0, 12.0),
        (-12.0, 12.0),
    ),
    "left": (
        (-63.0, -47.0),
        (6.0, 22.0),
        (-10.0, 10.0),
        (32.0, 50.0),
        (75.0, 105.0),
        (-12.0, 12.0),
        (-12.0, 12.0),
    ),
}

# Slightly below previous chest raise. Wrist_roll -90°.
RAISE_POSE_DEG = {
    "right": (-55.0, -14.0, 0.0, 40.0, -90.0, 0.0, 0.0),
    "left": (-55.0, 14.0, 0.0, 40.0, 90.0, 0.0, 0.0),
}
DEFAULT_SEED_DEG = RAISE_POSE_DEG

# Tight neighborhood around 平举; wrist stays near the seed unless --include-wrist.
DEFAULT_SPAN_DEG = (6.0, 4.0, 5.0, 6.0, 4.0, 3.0, 3.0)


@dataclass(frozen=True)
class SweepPose:
    name: str
    deg: tuple[float, float, float, float, float, float, float]

    def as_list(self) -> list[float]:
        return [round(float(v), 4) for v in self.deg]


def deg_to_rad(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) * math.pi / 180.0 for v in values)


def rad_to_deg(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) * 180.0 / math.pi for v in values)


def _intersect_box(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    lo = max(float(a[0]), float(b[0]))
    hi = min(float(a[1]), float(b[1]))
    if lo > hi:
        raise ValueError(f"empty joint box after intersect: {a} ∩ {b}")
    return lo, hi


def reachable_box_deg(
    arm: str,
    *,
    margin_deg: float = 2.0,
    extra_box_deg: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[tuple[float, float], ...]:
    side = _arm_side(arm)
    urdf_deg = tuple(
        (math.degrees(lo) + margin_deg, math.degrees(hi) - margin_deg)
        for lo, hi in URDF_LIMITS_RAD[side]
    )
    boxes: list[Sequence[tuple[float, float]]] = [SAFE_BOX_DEG[side], urdf_deg]
    if extra_box_deg is not None:
        boxes.append(tuple((float(lo), float(hi)) for lo, hi in extra_box_deg))
    return _merge_boxes(boxes)


def _merge_boxes(boxes: Sequence[Sequence[tuple[float, float]]]) -> tuple[tuple[float, float], ...]:
    merged = list(boxes[0])
    for box in boxes[1:]:
        merged = [_intersect_box(a, b) for a, b in zip(merged, box)]
    return tuple(merged)


def _arm_side(arm: str) -> str:
    side = str(arm).strip().lower()
    if side not in URDF_LIMITS_RAD:
        raise ValueError(f"arm must be left or right, got {arm!r}")
    return side


def clamp_pose_deg(
    pose_deg: Sequence[float],
    box_deg: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    if len(pose_deg) != 7 or len(box_deg) != 7:
        raise ValueError("pose and box must have 7 joints")
    return tuple(min(max(float(value), lo), hi) for value, (lo, hi) in zip(pose_deg, box_deg))


def pose_l2_deg(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _sample_one(
    seed_deg: Sequence[float],
    box_deg: Sequence[tuple[float, float]],
    span_deg: Sequence[float],
    rng: random.Random,
    *,
    upper_arm_only: bool,
) -> tuple[float, ...]:
    values: list[float] = []
    for index, (seed, (lo, hi), span) in enumerate(zip(seed_deg, box_deg, span_deg)):
        if upper_arm_only and index in WRIST_INDICES:
            values.append(float(seed))
            continue
        lo_s = max(lo, float(seed) - float(span))
        hi_s = min(hi, float(seed) + float(span))
        if lo_s > hi_s:
            values.append(min(max(float(seed), lo), hi))
            continue
        values.append(rng.uniform(lo_s, hi_s))
    return clamp_pose_deg(values, box_deg)


class ReachablePoseSampler:
    """Keep drawing new reachable poses. The box is continuous, so this can run indefinitely."""

    def __init__(
        self,
        *,
        arm: str = "right",
        seed_deg: Optional[Sequence[float]] = None,
        span_deg: Sequence[float] = DEFAULT_SPAN_DEG,
        min_separation_deg: float = 12.0,
        upper_arm_only: bool = True,
        margin_deg: float = 2.0,
        recent_window: int = 40,
        rng: Optional[random.Random] = None,
    ) -> None:
        if len(span_deg) != 7:
            raise ValueError("span_deg must have 7 values")
        side = _arm_side(arm)
        self.arm = side
        self.box_deg = reachable_box_deg(side, margin_deg=margin_deg)
        self.seed_deg = clamp_pose_deg(seed_deg or DEFAULT_SEED_DEG[side], self.box_deg)
        self.span_deg = tuple(float(v) for v in span_deg)
        self.min_separation_deg = float(min_separation_deg)
        self.upper_arm_only = bool(upper_arm_only)
        self.rng = rng or random.Random()
        self._recent: Deque[tuple[float, ...]] = deque(maxlen=max(4, int(recent_window)))
        self.drawn = 0

    def seed_pose(self, name: str = "raise") -> SweepPose:
        return SweepPose(name=name, deg=tuple(self.seed_deg))

    def next_pose(self, *, name: Optional[str] = None, remember: bool = True) -> SweepPose:
        """Sample one new pose. remember=False is a dry look; dwell skip still remembers."""
        candidate = self._draw_candidate()
        if remember:
            self._recent.append(candidate)
            self.drawn += 1
        label = name or f"rand_{self.drawn:04d}"
        return SweepPose(name=label, deg=tuple(candidate))

    def _draw_candidate(self) -> tuple[float, ...]:
        sep = self.min_separation_deg
        for attempt in range(120):
            candidate = _sample_one(
                self.seed_deg,
                self.box_deg,
                self.span_deg,
                self.rng,
                upper_arm_only=self.upper_arm_only,
            )
            if self._recent and pose_l2_deg(candidate, self.seed_deg) < sep * 0.45:
                continue
            if any(pose_l2_deg(candidate, prev) < sep for prev in self._recent):
                if attempt > 80:
                    sep *= 0.85
                continue
            return candidate
        return _sample_one(
            self.seed_deg,
            self.box_deg,
            self.span_deg,
            self.rng,
            upper_arm_only=self.upper_arm_only,
        )


def sample_reachable_poses(
    count: int,
    *,
    arm: str = "right",
    seed_deg: Optional[Sequence[float]] = None,
    span_deg: Sequence[float] = DEFAULT_SPAN_DEG,
    min_separation_deg: float = 12.0,
    upper_arm_only: bool = True,
    return_to_seed: bool = True,
    margin_deg: float = 2.0,
    rng: Optional[random.Random] = None,
) -> list[SweepPose]:
    """Draw a finite batch of diverse poses inside the conservative reachable box."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if count > 60:
        raise ValueError("count must be <= 60")
    sampler = ReachablePoseSampler(
        arm=arm,
        seed_deg=seed_deg,
        span_deg=span_deg,
        min_separation_deg=min_separation_deg,
        upper_arm_only=upper_arm_only,
        margin_deg=margin_deg,
        recent_window=max(40, count),
        rng=rng,
    )
    poses = [sampler.seed_pose(name="raise")]
    poses.extend(sampler.next_pose() for _ in range(count))
    if return_to_seed:
        poses.append(sampler.seed_pose(name="raise"))
    return poses


def render_multi_joint_yaml(
    poses: Sequence[SweepPose],
    *,
    arm: str = "right",
    move_duration: float = 3.0,
    control_dt: float = 0.02,
    arm_kp: float = 140.0,
    arm_kd: float = 3.0,
) -> str:
    side = _arm_side(arm)
    lines = [
        f'control_mode: "{side}"',
        f"move_duration: {float(move_duration):.2f}",
        f"control_dt: {float(control_dt):.3f}",
        f"arm_kp: {float(arm_kp):.1f}",
        f"arm_kd: {float(arm_kd):.1f}",
        "waist_kp: 300.0",
        "waist_kd: 3.0",
        "waypoints:",
    ]
    for pose in poses:
        values = ", ".join(f"{value:.4f}" for value in pose.as_list())
        lines.append(f"  - [{values}]  # {pose.name}")
    lines.append("")
    return "\n".join(lines)


def render_single_joint_yaml(
    pose: SweepPose,
    *,
    arm: str = "right",
    move_duration: float = 2.5,
    control_dt: float = 0.02,
    arm_kp: float = 140.0,
    arm_kd: float = 3.0,
    other_arm_target_deg: Optional[Sequence[float]] = None,
) -> str:
    side = _arm_side(arm)
    other = "left" if side == "right" else "right"
    other_values = list(other_arm_target_deg or DEFAULT_SEED_DEG[other])
    if len(other_values) != 7:
        raise ValueError("other_arm_target_deg must have 7 joints")
    target = ", ".join(f"{value:.4f}" for value in pose.as_list())
    other_target = ", ".join(f"{float(value):.4f}" for value in other_values)
    return (
        f'control_mode: "{side}"\n'
        f"move_duration: {float(move_duration):.2f}\n"
        f"control_dt: {float(control_dt):.3f}\n"
        f"left_target: [{target if side == 'left' else other_target}]\n"
        f"right_target: [{target if side == 'right' else other_target}]\n"
        f"arm_kp: {float(arm_kp):.1f}\n"
        f"arm_kd: {float(arm_kd):.1f}\n"
        "waist_kp: 300.0\n"
        "waist_kd: 3.0\n"
    )


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_seed_deg(values: Iterable[float]) -> tuple[float, ...]:
    seed = tuple(float(v) for v in values)
    if len(seed) != 7:
        raise ValueError("seed must have 7 joint angles in degrees")
    return seed
