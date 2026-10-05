"""Tests for the causal body-frame velocity estimator."""

from __future__ import annotations

import math

import pytest

from neurogrip_control.body_velocity import (
    CausalBodyVelocityEstimator,
    normalize_angle,
    yaw_from_quaternion,
)


def test_first_sample_is_undefined():
    """Velocity cannot be estimated from the first pose alone."""
    estimator = CausalBodyVelocityEstimator()
    assert estimator.update(0.0, 0.0, 0.0, 0.0) == (None, None)


def test_straight_forward_motion():
    """Moving along +x while facing +x yields forward body velocity only."""
    estimator = CausalBodyVelocityEstimator()
    estimator.update(0.0, 0.0, 0.0, 0.0)
    v_x, v_y = estimator.update(0.1, 0.1, 0.0, 0.0)
    assert v_x == pytest.approx(1.0)
    assert v_y == pytest.approx(0.0)


def test_lateral_motion_is_positive_to_the_left():
    """Moving along +y while facing +x yields positive lateral velocity."""
    estimator = CausalBodyVelocityEstimator()
    estimator.update(0.0, 0.0, 0.0, 0.0)
    v_x, v_y = estimator.update(0.1, 0.0, 0.1, 0.0)
    assert v_x == pytest.approx(0.0)
    assert v_y == pytest.approx(1.0)


def test_circular_motion_keeps_forward_velocity_constant():
    """A coordinated turn produces constant forward body velocity."""
    estimator = CausalBodyVelocityEstimator()
    radius_m = 5.0
    angular_speed = 0.4
    dt_s = 0.02
    for step in range(200):
        time_s = step * dt_s
        yaw = angular_speed * time_s
        x_m = radius_m * math.sin(yaw)
        y_m = radius_m - radius_m * math.cos(yaw)
        v_x, v_y = estimator.update(time_s, x_m, y_m, yaw)
    assert v_x == pytest.approx(radius_m * angular_speed, rel=1e-3)
    assert v_y == pytest.approx(0.0, abs=1e-3)


def test_yaw_wrap_does_not_spike_velocity():
    """Crossing yaw = +/-pi must not corrupt the body velocity."""
    estimator = CausalBodyVelocityEstimator()
    estimator.update(0.0, -1.00, 0.0, math.pi - 0.01)
    v_x, v_y = estimator.update(0.1, -1.10, 0.0, -math.pi + 0.01)
    assert v_x == pytest.approx(1.0, abs=0.02)
    assert abs(v_y) < 0.02


def test_timestamp_gap_resets_the_baseline():
    """A gap longer than max_gap_s invalidates this and only this sample."""
    estimator = CausalBodyVelocityEstimator(max_gap_s=0.05)
    estimator.update(0.0, 0.0, 0.0, 0.0)
    assert estimator.update(1.0, 1.0, 0.0, 0.0) == (None, None)
    # The sample after the gap re-establishes the baseline; a normal interval
    # then produces a valid backward difference again.
    v_x, v_y = estimator.update(1.02, 1.02, 0.0, 0.0)
    assert v_x == pytest.approx(1.0)
    assert v_y == pytest.approx(0.0)


def test_duplicate_timestamp_is_undefined():
    """Two samples with the same stamp cannot produce a velocity."""
    estimator = CausalBodyVelocityEstimator()
    estimator.update(0.0, 0.0, 0.0, 0.0)
    assert estimator.update(0.0, 1.0, 0.0, 0.0) == (None, None)


def test_reset_forgets_history():
    """After reset the estimator returns to the undefined state."""
    estimator = CausalBodyVelocityEstimator()
    estimator.update(0.0, 0.0, 0.0, 0.0)
    estimator.update(0.1, 0.1, 0.0, 0.0)
    estimator.reset()
    assert estimator.update(0.2, 0.2, 0.0, 0.0) == (None, None)


def test_yaw_from_quaternion_matches_rotation():
    """A yaw-only quaternion decodes back to the same angle."""
    yaw = 0.7
    quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    assert yaw_from_quaternion(*quaternion) == pytest.approx(yaw)


def test_normalize_angle_wraps_into_range():
    """Angle normalization keeps values inside (-pi, pi]."""
    assert normalize_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert normalize_angle(-3.0 * math.pi) == pytest.approx(math.pi)
