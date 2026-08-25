"""Timestamp ROS JointState messages for synchronized URDF FK capture."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, Optional

from unitree_sdk2_bridge import JointSample, TimedJointState


class ROSJointStateBridge:
    """Expose ROS ``sensor_msgs/JointState`` through the FK bridge interface."""

    state_source = "ros_joint_states"

    def __init__(self, topic: str = "/joint_states", history_size: int = 512) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError(
                "ROS JointState FK requires sourced ROS 2 and importable "
                "rclpy/sensor_msgs in the current Python environment."
            ) from exc

        self.state_topic = str(topic).strip()
        if not self.state_topic:
            raise ValueError("JointState topic must not be empty")
        self._rclpy = rclpy
        self._history: deque[TimedJointState] = deque(
            maxlen=max(2, int(history_size))
        )
        self._latest: Dict[str, JointSample] = {}
        self._latest_host_monotonic_sec: Optional[float] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node(
            f"handeye_joint_state_reader_{int(time.time() * 1000)}"
        )
        self._subscription = self._node.create_subscription(
            JointState,
            self.state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin,
            name="handeye-ros-joint-state",
            daemon=True,
        )
        self._thread.start()

    def _on_joint_state(self, msg: object) -> None:
        received_at = time.monotonic()
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        latest: Dict[str, JointSample] = {}
        for index, name in enumerate(names):
            if index >= len(positions):
                continue
            latest[str(name)] = JointSample(
                q=float(positions[index]),
                dq=float(velocities[index]) if index < len(velocities) else 0.0,
                tau_est=0.0,
            )
        if not latest:
            return
        sample = TimedJointState(received_at, latest)
        with self._lock:
            self._latest = latest
            self._latest_host_monotonic_sec = received_at
            self._history.append(sample)
        self._ready.set()

    def wait_for_state(
        self,
        timeout: float,
        *,
        max_age_sec: Optional[float] = None,
    ) -> Dict[str, JointSample]:
        if not self._ready.wait(timeout):
            raise TimeoutError(
                f"Timed out waiting for ROS JointState on {self.state_topic}"
            )
        state = self.latest_state()
        age = self.latest_state_age_sec()
        if max_age_sec is not None and (
            age is None or age > float(max_age_sec)
        ):
            raise TimeoutError(
                f"Latest ROS JointState is stale: age={age!r}s, "
                f"limit={float(max_age_sec):.3f}s"
            )
        return state

    def latest_state(self) -> Dict[str, JointSample]:
        with self._lock:
            return dict(self._latest)

    def latest_state_age_sec(self) -> Optional[float]:
        with self._lock:
            timestamp = self._latest_host_monotonic_sec
        return (
            None
            if timestamp is None
            else max(0.0, time.monotonic() - timestamp)
        )

    def nearest_state(
        self,
        host_monotonic_sec: float,
        *,
        max_delta_sec: Optional[float] = None,
        max_age_sec: Optional[float] = None,
    ) -> TimedJointState:
        with self._lock:
            history = list(self._history)
        if not history:
            raise TimeoutError("No ROS JointState history is available")
        target = float(host_monotonic_sec)
        nearest = min(
            history,
            key=lambda item: abs(item.host_monotonic_sec - target),
        )
        delta = abs(nearest.host_monotonic_sec - target)
        if max_delta_sec is not None and delta > float(max_delta_sec):
            raise TimeoutError(
                f"No synchronized ROS JointState: nearest delta={delta:.3f}s, "
                f"limit={float(max_delta_sec):.3f}s"
            )
        age = max(0.0, time.monotonic() - nearest.host_monotonic_sec)
        if max_age_sec is not None and age > float(max_age_sec):
            raise TimeoutError(
                f"Matched ROS JointState is stale: age={age:.3f}s, "
                f"limit={float(max_age_sec):.3f}s"
            )
        return TimedJointState(
            nearest.host_monotonic_sec,
            dict(nearest.joints),
        )
