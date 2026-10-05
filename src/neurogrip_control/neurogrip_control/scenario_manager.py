"""
Apply a hidden EUFS grip schedule with atomic set-and-readback confirmation.

The manager is scenario infrastructure, not an AI input. It publishes the
scalar oracle value only for the explicitly labelled C1 baseline. C0 and C2
must never subscribe to this topic or parameter events.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import rclpy
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64, Float64MultiArray, String

TRACKING_SCHEMA = 1


@dataclass(frozen=True)
class GripTarget:
    """One desired pair of front/rear grip scales."""

    front: float
    rear: float
    label: str


def valid_grip_target(front: float, rear: float) -> bool:
    """Return whether a requested EUFS grip pair is finite and in range."""
    return all(
        math.isfinite(value) and 0.35 <= value <= 1.30
        for value in (front, rear)
    )


def periodic_progress_delta(previous: float, current: float) -> float:
    """Return signed progress change across the periodic start line."""
    delta = float(current) - float(previous)
    if delta < -0.5:
        return delta + 1.0
    if delta > 0.5:
        return delta - 1.0
    return delta


class ScenarioManager(Node):
    """Apply initial and scheduled grip pairs without blocking the ROS executor."""

    def __init__(self):
        super().__init__("neurogrip_scenario_manager")
        self.declare_parameter("front_grip_scale", 1.0)
        self.declare_parameter("rear_grip_scale", 1.0)
        self.declare_parameter("transition_start_s", -1.0)
        self.declare_parameter("transition_start_progress", -1.0)
        self.declare_parameter("transition_front_grip_scale", 1.0)
        self.declare_parameter("transition_rear_grip_scale", 1.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("readback_tolerance", 1e-9)

        self.initial = GripTarget(
            float(self.get_parameter("front_grip_scale").value),
            float(self.get_parameter("rear_grip_scale").value),
            "initial",
        )
        self.transition = GripTarget(
            float(self.get_parameter("transition_front_grip_scale").value),
            float(self.get_parameter("transition_rear_grip_scale").value),
            "transition",
        )
        self.transition_start = float(self.get_parameter("transition_start_s").value)
        self.transition_start_progress = float(
            self.get_parameter("transition_start_progress").value
        )
        if not valid_grip_target(self.initial.front, self.initial.rear):
            raise ValueError(
                "initial grip scales must be finite values in [0.35, 1.30]"
            )
        if self.transition_start >= 0.0 and not valid_grip_target(
            self.transition.front, self.transition.rear
        ):
            raise ValueError(
                "transition grip scales must be finite values in [0.35, 1.30]"
            )
        if not -1.0 <= self.transition_start_progress < 1.0:
            raise ValueError("transition_start_progress must be -1 or in [0, 1)")

        self.readback_tolerance = float(self.get_parameter("readback_tolerance").value)
        self.oracle_publisher = self.create_publisher(Float64, "/neurogrip/oracle_mu", 10)
        status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(
            String, "/neurogrip/scenario_status", status_qos
        )
        self.set_client = self.create_client(
            SetParametersAtomically, "/eufs_sim2/set_parameters_atomically"
        )
        self.get_client = self.create_client(GetParameters, "/eufs_sim2/get_parameters")
        self.create_subscription(
            String,
            "/neurogrip/controller_status",
            self.on_controller_status,
            status_qos,
        )
        self.create_subscription(
            Float64MultiArray,
            "/neurogrip/tracking_state",
            self.on_tracking,
            50,
        )

        self.start_time_s: float | None = None
        self.latest_progress: float | None = None
        self.previous_progress: float | None = None
        self.progress_since_start = 0.0
        self.applied_front: float | None = None
        self.applied_rear: float | None = None
        self.initial_confirmed = False
        self.transition_confirmed = (
            self.transition_start < 0.0 and self.transition_start_progress < 0.0
        )
        self.request_in_flight = False
        self.pending_target: GripTarget | None = None
        self.pending_set_monotonic_ns: int | None = None
        self.pending_set_sim_s: float | None = None

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(
            "initial grip front=%.3f rear=%.3f; transition_start=%.3fs progress=%.3f"
            % (
                self.initial.front,
                self.initial.rear,
                self.transition_start,
                self.transition_start_progress,
            )
        )

    def now_s(self) -> float:
        """Return current simulation time in seconds."""
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def tick(self) -> None:
        """Publish the confirmed oracle value and advance the schedule once safe."""
        if not self.initial_confirmed:
            self.request_target(self.initial)
            return
        if (
            not self.transition_confirmed
            and self._transition_due()
        ):
            self.request_target(self.transition)
        if self.applied_front is not None and self.applied_rear is not None:
            message = Float64()
            message.data = 0.5 * (self.applied_front + self.applied_rear)
            self.oracle_publisher.publish(message)

    def on_controller_status(self, message: String) -> None:
        """Synchronise the schedule to the controller's explicit start event."""
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if payload.get("event") not in {"experiment_start", "excitation_start"}:
            return
        if self.start_time_s is None:
            self.start_time_s = float(payload.get("sim_time_s", self.now_s()))
            self.previous_progress = self.latest_progress
            self.progress_since_start = 0.0

    def on_tracking(self, message: Float64MultiArray) -> None:
        """Cache lap progress for progress-triggered grip changes."""
        values = list(message.data)
        if len(values) == 10 and int(values[0]) == TRACKING_SCHEMA:
            progress = float(values[9]) % 1.0
            if self.start_time_s is not None and self.previous_progress is not None:
                self.progress_since_start += periodic_progress_delta(
                    self.previous_progress, progress
                )
            self.previous_progress = progress
            self.latest_progress = progress

    def _transition_due(self) -> bool:
        """Return whether the configured post-start trigger has been reached."""
        if self.start_time_s is None:
            return False
        if self.transition_start_progress >= 0.0:
            return (
                self.latest_progress is not None
                and self.progress_since_start >= self.transition_start_progress
            )
        return (
            self.transition_start >= 0.0
            and self.now_s() - self.start_time_s >= self.transition_start
        )

    def request_target(self, target: GripTarget) -> None:
        """Start one atomic set request unless a set/readback transaction is active."""
        if self.request_in_flight or not self.set_client.service_is_ready():
            return
        request = SetParametersAtomically.Request()
        request.parameters = [
            self._parameter("dynamics.front_grip_scale", target.front),
            self._parameter("dynamics.rear_grip_scale", target.rear),
        ]
        self.request_in_flight = True
        self.pending_target = target
        self.pending_set_monotonic_ns = time.monotonic_ns()
        self.pending_set_sim_s = self.now_s()
        future = self.set_client.call_async(request)
        future.add_done_callback(self.on_set_done)

    def on_set_done(self, future) -> None:
        """Request a readback only after EUFS accepts the atomic update."""
        target = self.pending_target
        if target is None:
            self.request_in_flight = False
            return
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - ROS service diagnostics
            self.publish_status(target, False, "set_exception", str(error), None, None)
            self.clear_pending()
            return
        if response is None or not response.result.successful:
            reason = "empty_response" if response is None else response.result.reason
            self.publish_status(target, False, "set_rejected", reason, None, None)
            self.clear_pending()
            return
        if not self.get_client.service_is_ready():
            self.publish_status(
                target, False, "readback_unavailable", "service_not_ready", None, None
            )
            self.clear_pending()
            return
        request = GetParameters.Request()
        request.names = ["dynamics.front_grip_scale", "dynamics.rear_grip_scale"]
        readback_future = self.get_client.call_async(request)
        readback_future.add_done_callback(self.on_readback_done)

    def on_readback_done(self, future) -> None:
        """Confirm a schedule event only when both read values match the request."""
        target = self.pending_target
        if target is None:
            self.request_in_flight = False
            return
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - ROS service diagnostics
            self.publish_status(target, False, "readback_exception", str(error), None, None)
            self.clear_pending()
            return
        values = [] if response is None else response.values
        if len(values) != 2:
            self.publish_status(
                target, False, "readback_invalid", "expected_two_values", None, None
            )
            self.clear_pending()
            return
        front = float(values[0].double_value)
        rear = float(values[1].double_value)
        matches = (
            abs(front - target.front) <= self.readback_tolerance
            and abs(rear - target.rear) <= self.readback_tolerance
        )
        if matches:
            self.applied_front, self.applied_rear = front, rear
            if target.label == "initial":
                self.initial_confirmed = True
            else:
                self.transition_confirmed = True
            self.publish_status(target, True, "readback_confirmed", "", front, rear)
        else:
            self.publish_status(target, False, "readback_mismatch", "", front, rear)
        self.clear_pending()

    def clear_pending(self) -> None:
        """Release the asynchronous transaction so a bounded retry can occur."""
        self.request_in_flight = False
        self.pending_target = None
        self.pending_set_monotonic_ns = None
        self.pending_set_sim_s = None

    def publish_status(
        self,
        target: GripTarget,
        confirmed: bool,
        phase: str,
        reason: str,
        readback_front: float | None,
        readback_rear: float | None,
    ) -> None:
        """Publish complete timing and readback provenance for one transaction."""
        status = String()
        status.data = json.dumps(
            {
                "confirmed": confirmed,
                "label": target.label,
                "requested_front": target.front,
                "requested_rear": target.rear,
                "readback_front": readback_front,
                "readback_rear": readback_rear,
                "phase": phase,
                "reason": reason,
                "set_monotonic_ns": self.pending_set_monotonic_ns,
                "set_sim_s": self.pending_set_sim_s,
                "confirmed_monotonic_ns": time.monotonic_ns(),
                "confirmed_sim_s": self.now_s(),
            },
            sort_keys=True,
        )
        self.status_publisher.publish(status)
        self.get_logger().info(
            "grip %s confirmed=%s requested=(%.3f, %.3f) readback=(%s, %s) phase=%s"
            % (
                target.label,
                confirmed,
                target.front,
                target.rear,
                readback_front,
                readback_rear,
                phase,
            )
        )

    @staticmethod
    def _parameter(name: str, value: float) -> ParameterMsg:
        """Build one ROS double parameter message."""
        parameter = ParameterMsg()
        parameter.name = name
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
        )
        return parameter


def main(args=None):
    """Run the scenario manager."""
    rclpy.init(args=args)
    node = ScenarioManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
