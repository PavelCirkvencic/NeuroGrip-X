"""Unit tests for scenario validation that need no ROS graph."""

import pytest

from neurogrip_control.scenario_manager import (
    periodic_progress_delta,
    valid_grip_target,
)


def test_grip_target_requires_finite_supported_scales():
    """The manager must reject impossible and non-finite grip requests."""
    assert valid_grip_target(0.35, 1.30)
    assert not valid_grip_target(0.34, 1.0)
    assert not valid_grip_target(1.0, 1.31)
    assert not valid_grip_target(float("nan"), 1.0)
    assert not valid_grip_target(1.0, float("inf"))


def test_progress_delta_wraps_forward_without_triggering_at_endpoint():
    """Progress-trigger schedules must measure distance after experiment start."""
    assert periodic_progress_delta(0.99, 0.01) == pytest.approx(0.02)
    assert periodic_progress_delta(0.01, 0.99) == pytest.approx(-0.02)
