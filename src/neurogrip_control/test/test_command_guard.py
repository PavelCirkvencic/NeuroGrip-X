"""
Unit tests for the command guard clamp and watchdog behaviour.

These tests construct the real ``CommandGuard`` node, but they never publish
over the network: they exercise the pure clamp/watchdog methods directly.  A
future controller that changes the guard limits will therefore fail here.
"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import Twist  # noqa: E402

from neurogrip_control.command_guard import CommandGuard  # noqa: E402


@pytest.fixture(scope="module")
def guard():
    """Create one command guard for the module and tear it down afterwards."""
    if not rclpy.ok():
        rclpy.init()
    node = CommandGuard()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_clamp_saturates_both_directions():
    """The static clamp keeps values inside the symmetric limit."""
    assert CommandGuard.clamp(5.0, 0.8) == 0.8
    assert CommandGuard.clamp(-5.0, 0.8) == -0.8
    assert CommandGuard.clamp(0.3, 0.8) == 0.3


def test_bounded_command_enforces_hard_limits(guard):
    """Raw over-limit commands are reduced to the guard limits."""
    raw = Twist()
    raw.linear.x = 100.0
    raw.angular.z = -100.0
    bounded = guard.bounded_command(raw)
    assert bounded.linear.x == guard.max_linear_x_mps
    assert bounded.angular.z == -guard.max_angular_z_rps


def test_watchdog_returns_zero_without_any_command(guard):
    """Before the first command the applied command must be zero."""
    guard.last_command_received_ns = None
    guard.last_command = Twist()
    guard.last_command.linear.x = 0.5
    safe = guard.safe_command(guard.get_clock().now().nanoseconds)
    assert safe.linear.x == 0.0
    assert safe.angular.z == 0.0


def test_watchdog_passes_fresh_command(guard):
    """A command inside the timeout window is applied unchanged."""
    raw = Twist()
    raw.linear.x = 0.6
    raw.angular.z = 0.2
    guard.command_callback(raw)
    safe = guard.safe_command(guard.get_clock().now().nanoseconds)
    assert safe.linear.x == pytest.approx(0.6)
    assert safe.angular.z == pytest.approx(0.2)


def test_watchdog_zeroes_stale_command(guard):
    """A command older than the timeout is replaced by zero."""
    raw = Twist()
    raw.linear.x = 0.6
    raw.angular.z = 0.2
    guard.command_callback(raw)
    now_ns = guard.get_clock().now().nanoseconds
    guard.last_command_received_ns = now_ns - int(1.0e9)
    safe = guard.safe_command(now_ns)
    assert safe.linear.x == 0.0
    assert safe.angular.z == 0.0


class _CapturingPublisher:
    """Minimal publisher double that records the last published message."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        """Record one published message."""
        self.messages.append(message)


def test_apply_actuator_hard_limits_after_gain(guard):
    """Gain is applied first, then the guard hard-clamps the result."""
    original_delay, original_gain = guard.actuator.delay_s, guard.actuator.gain
    try:
        guard.actuator.delay_s = 0.0
        guard.actuator.gain = 2.0
        raw = Twist()
        raw.linear.x = 0.7
        raw.angular.z = 0.4
        applied = guard.apply_actuator(0.0, raw)
        assert applied.linear.x == pytest.approx(0.7)
        assert applied.angular.z == pytest.approx(guard.max_angular_z_rps)
    finally:
        guard.actuator.delay_s = original_delay
        guard.actuator.gain = original_gain


def test_watchdog_bypasses_actuator_delay(guard):
    """A stale command stops immediately instead of waiting for the delay."""
    original_publisher = guard.command_publisher
    original_delay = guard.actuator.delay_s
    capturing = _CapturingPublisher()
    try:
        guard.actuator.delay_s = 0.2
        guard.command_callback(_twist(0.0, 0.3))
        now_ns = guard.get_clock().now().nanoseconds
        guard.last_command_received_ns = now_ns - int(1.0e9)
        guard.command_publisher = capturing
        guard.publish_safe_command()
        published = capturing.messages[-1]
        assert published.linear.x == 0.0
        assert published.angular.z == 0.0
        assert not guard.actuator.history_ready
    finally:
        guard.command_publisher = original_publisher
        guard.actuator.delay_s = original_delay
        guard.actuator.reset()


def _twist(linear_x: float, angular_z: float) -> Twist:
    """Build a Twist with the given linear x and angular z."""
    message = Twist()
    message.linear.x = linear_x
    message.angular.z = angular_z
    return message
