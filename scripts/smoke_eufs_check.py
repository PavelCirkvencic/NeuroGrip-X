#!/usr/bin/env python3
"""
EUFS Sim 2 runtime smoke checker.

Run against a live ``eufs_backend.launch.py``.  Verifies telemetry rate,
dynamic response, applied-steering rate limiting and atomic runtime grip
parameters.  Writes a JSON summary and returns non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import WheelSpeedsStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.node import Node


class SmokeChecker(Node):
    """Drive the vehicle and verify the v2 contract."""

    def __init__(self):
        super().__init__("neurogrip_eufs_smoke")
        self.latest_odom: Odometry | None = None
        self.odom_count = 0
        self.steering_samples: list[float] = []
        self.publisher = self.create_publisher(AckermannDriveStamped, "/cmd", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(
            WheelSpeedsStamped, "/ros_can/wheel_speeds", self.on_wheel_speeds, 50
        )
        self.set_params = self.create_client(
            SetParametersAtomically, "/eufs_sim2/set_parameters_atomically"
        )
        self.get_params = self.create_client(GetParameters, "/eufs_sim2/get_parameters")

    def on_odom(self, message: Odometry) -> None:
        self.latest_odom = message
        self.odom_count += 1

    def on_wheel_speeds(self, message: WheelSpeedsStamped) -> None:
        self.steering_samples.append(float(message.speeds.steering))

    def spin_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def publish_command(self, acceleration: float, steering: float, duration_s: float) -> None:
        message = AckermannDriveStamped()
        message.drive.acceleration = float(acceleration)
        message.drive.steering_angle = float(steering)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_odom(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_odom is not None:
                return True
        return False

    def set_runtime_grip(self, front: float, rear: float) -> tuple[bool, str]:
        request = SetParametersAtomically.Request()
        request.parameters = [
            self._double_parameter("dynamics.front_grip_scale", front),
            self._double_parameter("dynamics.rear_grip_scale", rear),
        ]
        future = self.set_params.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None:
            return False, "no response"
        result = response.result
        return bool(result.successful), result.reason

    def read_grip(self) -> tuple[float, float]:
        request = GetParameters.Request()
        request.names = ["dynamics.front_grip_scale", "dynamics.rear_grip_scale"]
        future = self.get_params.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        values = future.result().values
        return float(values[0].double_value), float(values[1].double_value)

    @staticmethod
    def _double_parameter(name: str, value: float) -> ParameterMsg:
        parameter = ParameterMsg()
        parameter.name = name
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
        )
        return parameter


def parse_arguments() -> argparse.Namespace:
    """Parse output path and timing parameters."""
    parser = argparse.ArgumentParser(description="EUFS smoke checker")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run every smoke check and write a JSON summary."""
    arguments = parse_arguments()
    rclpy.init()
    node = SmokeChecker()
    failures = []
    summary: dict = {}

    try:
        if not node.wait_for_odom(60.0):
            failures.append("no /odom received")
        else:
            # 1. odometry rate
            node.odom_count = 0
            node.spin_for(3.0)
            odom_hz = node.odom_count / 3.0
            summary["odom_hz"] = odom_hz
            if not 47.0 <= odom_hz <= 53.0:
                failures.append(f"odom_hz out of range: {odom_hz}")

            # 2. dynamic response (4 s, accel 1.5, steering 0.1)
            node.publish_command(1.5, 0.1, 4.0)
            node.spin_for(0.2)
            twist = node.latest_odom.twist.twist
            summary["v_x"] = twist.linear.x
            summary["v_y"] = twist.linear.y
            summary["yaw_rate"] = twist.angular.z
            if twist.linear.x <= 3.0:
                failures.append(f"v_x too low: {twist.linear.x}")
            if abs(twist.linear.y) <= 0.02:
                failures.append(f"v_y degenerate: {twist.linear.y}")
            if abs(twist.angular.z) <= 0.05:
                failures.append(f"yaw rate too low: {twist.angular.z}")

            # 3. applied-steering rate limit (max_rate 0.39 rad/s)
            node.publish_command(0.0, 0.0, 1.2)
            node.steering_samples.clear()
            node.publish_command(0.0, 0.3, 0.12)
            node.publish_command(0.0, 0.0, 0.4)
            steering_max = max(node.steering_samples) if node.steering_samples else None
            summary["applied_steering_max"] = steering_max
            summary["applied_steering_samples"] = len(node.steering_samples)
            if steering_max is None or not 0.02 < steering_max < 0.20:
                failures.append(f"applied steering not rate-limited: {steering_max}")

            # 4. atomic runtime grip parameters
            ok, reason = node.set_runtime_grip(0.6, 0.9)
            summary["grip_set_ok"] = ok
            summary["grip_set_reason"] = reason
            if not ok:
                failures.append(f"grip set rejected: {reason}")
            else:
                front, rear = node.read_grip()
                summary["front_grip_scale"] = front
                summary["rear_grip_scale"] = rear
                if not (0.59 < front < 0.61 and 0.89 < rear < 0.91):
                    failures.append(f"grip readback wrong: {front}, {rear}")
            invalid_ok, invalid_reason = node.set_runtime_grip(2.0, 0.9)
            summary["invalid_grip_rejected"] = not invalid_ok
            summary["invalid_grip_reason"] = invalid_reason
            if invalid_ok:
                failures.append("invalid grip scale 2.0 was accepted")

            # 5. stop
            node.publish_command(0.0, 0.0, 0.5)
    finally:
        summary["failures"] = failures
        summary["passed"] = not failures
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()

    if failures:
        for failure in failures:
            print(f"SMOKE_FAIL: {failure}")
        return 1
    print("EUFS_CHECK_PASS")
    for key, value in summary.items():
        if key != "failures":
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
