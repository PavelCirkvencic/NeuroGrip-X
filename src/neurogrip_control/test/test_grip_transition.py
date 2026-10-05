"""Tests for grip-transition scheduling, validation, retries and provenance."""

from __future__ import annotations

import pytest

from neurogrip_control.grip_transition import (
    WHEEL_AXLES,
    GripTransitionController,
    is_transition_due,
    parse_boolean_response,
    resolve_compliance,
    transition_due_time_s,
)

SCENARIO = {
    "front_lateral_compliance": 0.10,
    "rear_lateral_compliance": 0.20,
    "longitudinal_compliance": 0.02,
    "grip_transition": {"start_s": 12.0, "compliance_scale": 2.0},
}


def always_success(wheel, lateral, longitudinal):
    """Service double that confirms every wheel update."""
    return True


class FakeClock:
    """Deterministic clock that advances by a fixed step per call."""

    def __init__(self, start: float = 100.0, step: float = 0.01):
        self.current = start
        self.step = step

    def __call__(self) -> float:
        value = self.current
        self.current += self.step
        return value


def test_resolve_scales_all_wheels():
    """Every wheel link receives the scaled compliance for its axle."""
    targets = resolve_compliance(SCENARIO, SCENARIO["grip_transition"])
    assert targets["front_left_wheel"] == pytest.approx((0.20, 0.04))
    assert targets["rear_left_wheel"] == pytest.approx((0.40, 0.04))


def test_parse_boolean_response():
    """The Gazebo Boolean response is interpreted safely."""
    assert parse_boolean_response(0, "data: true") is True
    assert parse_boolean_response(0, "data: false") is False
    assert parse_boolean_response(1, "data: true") is False
    assert parse_boolean_response(0, "") is False
    assert parse_boolean_response(0, "some unrelated output") is False


def test_transition_due_time_is_relative_to_excitation_start():
    """start_s is measured from the excitation start, not Gazebo startup."""
    due = transition_due_time_s(100.0, SCENARIO["grip_transition"])
    assert due == pytest.approx(112.0)
    assert is_transition_due(111.9, 100.0, SCENARIO["grip_transition"]) is False
    assert is_transition_due(112.0, 100.0, SCENARIO["grip_transition"]) is True
    assert is_transition_due(200.0, None, SCENARIO["grip_transition"]) is False


def test_all_four_wheels_success_records_skew():
    """A full success reports applied=True and the inter-wheel skew."""
    clock = FakeClock()
    controller = GripTransitionController(SCENARIO, always_success, clock)
    provenance = controller.apply()
    assert provenance["applied"] is True
    assert provenance["max_skew_s"] == pytest.approx(0.03)
    for wheel in WHEEL_AXLES:
        assert provenance["wheels"][wheel]["success"] is True
        assert provenance["wheels"][wheel]["attempts"] == 1
    assert controller.applied is True


def test_one_false_response_triggers_retry_then_success():
    """A single false response is retried and can still succeed."""
    calls = {}

    def flaky(wheel, lateral, longitudinal):
        calls[wheel] = calls.get(wheel, 0) + 1
        return not (wheel == "front_left_wheel" and calls[wheel] == 1)

    controller = GripTransitionController(SCENARIO, flaky, FakeClock())
    provenance = controller.apply()
    assert provenance["applied"] is True
    assert provenance["wheels"]["front_left_wheel"]["attempts"] == 2
    assert provenance["wheels"]["rear_left_wheel"]["attempts"] == 1


def test_permanent_failure_marks_not_applied():
    """A wheel that never confirms leaves applied=False after retries."""
    controller = GripTransitionController(
        SCENARIO, lambda wheel, lateral, longitudinal: wheel != "rear_right_wheel", FakeClock()
    )
    provenance = controller.apply()
    assert provenance["applied"] is False
    assert provenance["detail"] == "failed_after_retries"
    assert provenance["wheels"]["rear_right_wheel"]["attempts"] == 3
    assert controller.applied is False


def test_applied_flag_is_idempotent():
    """A second apply call returns the original provenance unchanged."""
    controller = GripTransitionController(SCENARIO, always_success, FakeClock())
    first = controller.apply()
    second = controller.apply()
    assert first is second


def test_no_transition_is_a_no_op():
    """A scenario without a transition reports expected=False."""
    controller = GripTransitionController({}, always_success, FakeClock())
    provenance = controller.apply()
    assert provenance["expected"] is False
    assert provenance["applied"] is False
