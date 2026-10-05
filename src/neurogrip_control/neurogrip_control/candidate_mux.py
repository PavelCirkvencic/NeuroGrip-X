"""Fail-closed selector between Python and MATLAB steering candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

MATLAB_CANDIDATE_SCHEMA = 1
MATLAB_CANDIDATE_LENGTH = 5
SOURCE_IDS = {
    "safe_stop": 0.0,
    "python": 1.0,
    "matlab": 2.0,
    "python_fallback": 3.0,
    "python_warmup": 4.0,
}


@dataclass(frozen=True)
class MatlabCandidate:
    """Validated MATLAB steering result and its original control timestamp."""

    control_time_s: float
    steering_rad: float
    solve_time_ms: float


def parse_matlab_candidate(
    values: list[float] | tuple[float, ...],
    *,
    maximum_steering_rad: float = 0.37,
    solver_deadline_ms: float = 20.0,
) -> tuple[MatlabCandidate | None, str]:
    """Decode a candidate only when every safety invariant is satisfied."""
    numeric = [float(value) for value in values]
    if len(numeric) != MATLAB_CANDIDATE_LENGTH:
        return None, "invalid_length"
    if any(not math.isfinite(value) for value in numeric):
        return None, "non_finite"
    (
        schema,
        control_time_s,
        steering_rad,
        solution_valid,
        solve_time_ms,
    ) = numeric
    if schema != MATLAB_CANDIDATE_SCHEMA:
        return None, "unsupported_schema"
    if solution_valid not in (0.0, 1.0) or solution_valid == 0.0:
        return None, "invalid_solution"
    if control_time_s < 0.0:
        return None, "invalid_control_time"
    if abs(steering_rad) > maximum_steering_rad:
        return None, "steering_out_of_bounds"
    if not 0.0 <= solve_time_ms <= solver_deadline_ms:
        return None, "solver_deadline_miss"
    return MatlabCandidate(control_time_s, steering_rad, solve_time_ms), ""


def select_candidate_source(
    selected_source: str,
    *,
    python_fresh: bool,
    matlab_fresh: bool,
    matlab_ever_active: bool,
    fallback_to_python: bool,
) -> str:
    """Select one source with explicit warm-up and post-activation fallback."""
    if selected_source == "python":
        return "python" if python_fresh else "safe_stop"
    if selected_source != "matlab":
        raise ValueError(f"unsupported candidate source: {selected_source}")
    if matlab_fresh and python_fresh:
        return "matlab"
    if not matlab_ever_active and python_fresh:
        return "python_warmup"
    if matlab_ever_active and fallback_to_python and python_fresh:
        return "python_fallback"
    return "safe_stop"


def activation_delay_elapsed(
    now_ns: int,
    first_python_receipt_ns: int | None,
    activation_delay_s: float,
) -> bool:
    """Keep startup steering on Python until both controllers are settled."""
    if first_python_receipt_ns is None:
        return False
    return now_ns - first_python_receipt_ns >= int(activation_delay_s * 1e9)


class CandidateMux(Node):
    """Publish one guarded candidate topic while retaining Python fallback."""

    def __init__(self):
        super().__init__("neurogrip_candidate_mux")
        self.declare_parameter("selected_source", "python")
        self.declare_parameter(
            "python_topic", "/neurogrip/command_candidate/python"
        )
        self.declare_parameter(
            "matlab_topic", "/neurogrip/command_candidate/matlab_flat"
        )
        self.declare_parameter("output_topic", "/neurogrip/command_candidate")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("timeout_s", 0.08)
        self.declare_parameter("maximum_control_time_skew_s", 0.08)
        self.declare_parameter("maximum_steering_rad", 0.37)
        self.declare_parameter("solver_deadline_ms", 20.0)
        self.declare_parameter("fallback_to_python", True)
        self.declare_parameter("matlab_activation_samples", 10)
        self.declare_parameter("matlab_activation_delay_s", 3.0)

        self.selected_source = str(self.get_parameter("selected_source").value)
        if self.selected_source not in {"python", "matlab"}:
            raise ValueError("selected_source must be 'python' or 'matlab'")
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.maximum_control_time_skew_s = float(
            self.get_parameter("maximum_control_time_skew_s").value
        )
        self.maximum_steering_rad = float(
            self.get_parameter("maximum_steering_rad").value
        )
        self.solver_deadline_ms = float(
            self.get_parameter("solver_deadline_ms").value
        )
        self.fallback_to_python = bool(
            self.get_parameter("fallback_to_python").value
        )
        self.matlab_activation_samples = int(
            self.get_parameter("matlab_activation_samples").value
        )
        self.matlab_activation_delay_s = float(
            self.get_parameter("matlab_activation_delay_s").value
        )
        if min(
            self.timeout_s,
            self.maximum_control_time_skew_s,
            self.maximum_steering_rad,
            self.solver_deadline_ms,
        ) <= 0.0:
            raise ValueError("candidate mux safety limits must be positive")
        if self.matlab_activation_samples < 1:
            raise ValueError("matlab_activation_samples must be positive")
        if self.matlab_activation_delay_s < 0.0:
            raise ValueError("matlab_activation_delay_s cannot be negative")

        self.python_command: AckermannDriveStamped | None = None
        self.python_receipt_ns: int | None = None
        self.first_python_receipt_ns: int | None = None
        self.matlab_candidate: MatlabCandidate | None = None
        self.matlab_receipt_ns: int | None = None
        self.last_matlab_control_time_s = -math.inf
        self.matlab_ever_active = False
        self.matlab_valid_streak = 0
        self.fallback_count = 0
        self.reject_count = 0
        self.last_source = ""

        self.command_publisher = self.create_publisher(
            AckermannDriveStamped,
            str(self.get_parameter("output_topic").value),
            10,
        )
        self.status_publisher = self.create_publisher(
            Float64MultiArray, "/neurogrip/command_mux_status", 10
        )
        self.event_publisher = self.create_publisher(
            String, "/neurogrip/actuation_status", 10
        )
        self.create_subscription(
            AckermannDriveStamped,
            str(self.get_parameter("python_topic").value),
            self.on_python_command,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("matlab_topic").value),
            self.on_matlab_candidate,
            10,
        )
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / publish_rate_hz, self.publish_selected)

    def on_python_command(self, message: AckermannDriveStamped) -> None:
        """Cache the longitudinal authority and Python lateral fallback."""
        self.python_command = message
        self.python_receipt_ns = self.get_clock().now().nanoseconds
        if self.first_python_receipt_ns is None:
            self.first_python_receipt_ns = self.python_receipt_ns

    def on_matlab_candidate(self, message: Float64MultiArray) -> None:
        """Cache only a valid, timely and monotonically newer MATLAB result."""
        candidate, reason = parse_matlab_candidate(
            list(message.data),
            maximum_steering_rad=self.maximum_steering_rad,
            solver_deadline_ms=self.solver_deadline_ms,
        )
        old_or_missing = (
            candidate is None
            or candidate.control_time_s <= self.last_matlab_control_time_s
        )
        if old_or_missing:
            self.matlab_candidate = None
            self.matlab_receipt_ns = None
            self.matlab_valid_streak = 0
            self.reject_count += 1
            if not reason:
                reason = "non_monotonic_control_time"
            self.get_logger().warning(f"rejected MATLAB candidate: {reason}")
            return
        self.matlab_candidate = candidate
        self.matlab_receipt_ns = self.get_clock().now().nanoseconds
        self.last_matlab_control_time_s = candidate.control_time_s
        self.matlab_valid_streak += 1

    @staticmethod
    def copy_python_command(
        source: AckermannDriveStamped, steering_rad: float | None = None
    ) -> AckermannDriveStamped:
        """Copy all longitudinal fields and optionally replace steering."""
        output = AckermannDriveStamped()
        output.drive.steering_angle = source.drive.steering_angle
        if steering_rad is not None:
            output.drive.steering_angle = steering_rad
        output.drive.steering_angle_velocity = (
            source.drive.steering_angle_velocity
        )
        output.drive.speed = source.drive.speed
        output.drive.acceleration = source.drive.acceleration
        output.drive.jerk = source.drive.jerk
        return output

    def publish_selected(self) -> None:
        """Publish one candidate or safe stop when selection is invalid."""
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        now_s = now_ns / 1e9
        python_age_s = (
            math.inf
            if self.python_receipt_ns is None
            else max(0.0, (now_ns - self.python_receipt_ns) / 1e9)
        )
        matlab_age_s = (
            math.inf
            if self.matlab_receipt_ns is None
            else max(0.0, (now_ns - self.matlab_receipt_ns) / 1e9)
        )
        matlab_control_age_s = (
            math.inf
            if self.matlab_candidate is None
            else abs(now_s - self.matlab_candidate.control_time_s)
        )
        python_fresh = (
            self.python_command is not None
            and python_age_s <= self.timeout_s
        )
        matlab_fresh = (
            self.matlab_candidate is not None
            and matlab_age_s <= self.timeout_s
            and matlab_control_age_s <= self.maximum_control_time_skew_s
        )
        if not self.matlab_ever_active and not matlab_fresh:
            self.matlab_valid_streak = 0
        matlab_selectable = matlab_fresh and (
            self.matlab_ever_active
            or (
                self.matlab_valid_streak >= self.matlab_activation_samples
                and activation_delay_elapsed(
                    now_ns,
                    self.first_python_receipt_ns,
                    self.matlab_activation_delay_s,
                )
            )
        )
        source = select_candidate_source(
            self.selected_source,
            python_fresh=python_fresh,
            matlab_fresh=matlab_selectable,
            matlab_ever_active=self.matlab_ever_active,
            fallback_to_python=self.fallback_to_python,
        )
        if source == "matlab":
            assert self.python_command is not None
            assert self.matlab_candidate is not None
            command = self.copy_python_command(
                self.python_command, self.matlab_candidate.steering_rad
            )
            self.matlab_ever_active = True
        elif source in {"python", "python_warmup", "python_fallback"}:
            assert self.python_command is not None
            command = self.copy_python_command(self.python_command)
            entering_fallback = (
                source == "python_fallback"
                and self.last_source != "python_fallback"
            )
            if entering_fallback:
                self.fallback_count += 1
        else:
            command = AckermannDriveStamped()
        command.header.stamp = now.to_msg()
        self.command_publisher.publish(command)
        self.publish_status(
            source,
            now_s,
            python_age_s,
            matlab_age_s,
            matlab_control_age_s,
        )

    def publish_status(
        self,
        source: str,
        now_s: float,
        python_age_s: float,
        matlab_age_s: float,
        matlab_control_age_s: float,
    ) -> None:
        """Publish telemetry each tick and JSON only on source changes."""
        status = Float64MultiArray()
        status.data = [
            1.0,
            now_s,
            SOURCE_IDS[source],
            python_age_s if math.isfinite(python_age_s) else -1.0,
            matlab_age_s if math.isfinite(matlab_age_s) else -1.0,
            (
                matlab_control_age_s
                if math.isfinite(matlab_control_age_s)
                else -1.0
            ),
            float(self.fallback_count),
            float(self.reject_count),
        ]
        self.status_publisher.publish(status)
        if source == self.last_source:
            return
        event = String()
        event.data = json.dumps(
            {
                "event": "actuation_source_changed",
                "selected_source": self.selected_source,
                "active_source": source,
                "fallback_count": self.fallback_count,
                "reject_count": self.reject_count,
            },
            sort_keys=True,
        )
        self.event_publisher.publish(event)
        self.last_source = source


def main(args=None):
    """Run the fail-closed candidate mux."""
    rclpy.init(args=args)
    node = CandidateMux()
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
