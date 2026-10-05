"""Regression tests for safety-valid lap acceptance."""

import numpy as np
import pytest

from neurogrip_control.track_safety import SafetyLapGate, TrackEnvelope


def envelope() -> TrackEnvelope:
    """Return a uniform two-metre periodic track envelope."""
    return TrackEnvelope(
        progress=np.asarray([0.0, 0.5, 0.99]),
        left_width_m=np.asarray([2.0, 2.0, 2.0]),
        right_width_m=np.asarray([2.0, 2.0, 2.0]),
    )


def cross_start(gate: SafetyLapGate):
    """Cross the start line once to arm an experiment."""
    gate.update(0.95, 0.0, 0.0, 3.0)
    return gate.update(0.02, 0.0, 0.0, 3.0)


def test_first_crossing_starts_experiment_but_is_not_a_lap():
    """The launch-side start-line wrap must never count as completion."""
    gate = SafetyLapGate(envelope())
    update = cross_start(gate)
    assert update.event == "experiment_start"
    assert update.laps_completed == 0


def test_launch_crossing_arms_while_vehicle_is_still_accelerating():
    """A valid first crossing at low speed must not force an extra warm-up lap."""
    gate = SafetyLapGate(envelope())
    gate.update(0.95, 0.0, 0.0, 0.05)
    update = gate.update(0.02, 0.0, 0.0, 0.2)
    assert update.event == "experiment_start"


def test_full_forward_safe_cycle_is_required_for_lap_complete():
    """A second crossing after forward progress is a valid lap."""
    gate = SafetyLapGate(envelope())
    cross_start(gate)
    for progress in (0.25, 0.50, 0.75, 0.95):
        assert gate.update(progress, 0.1, 0.0, 4.0).event is None
    update = gate.update(0.02, 0.1, 0.0, 4.0)
    assert update.event == "lap_complete"
    assert update.laps_completed == 1


def test_body_clearance_uses_heading_and_rejects_off_track():
    """Rotated vehicle footprint must reduce margin and abort the run."""
    gate = SafetyLapGate(envelope(), vehicle_half_width_m=0.7, vehicle_half_length_m=1.3)
    cross_start(gate)
    update = gate.update(0.2, 1.4, 0.5, 4.0)
    assert update.boundary_margin_m < 0.0
    assert update.event == "safety_violation"
    assert update.reason == "track_boundary"


def test_collision_is_terminal_and_never_becomes_a_lap():
    """Any EUFS cone collision invalidates the run immediately."""
    gate = SafetyLapGate(envelope())
    cross_start(gate)
    update = gate.update(0.4, 0.0, 0.0, 4.0, collision_count=1)
    assert update.event == "safety_violation"
    assert update.reason == "cone_collision"
    assert update.laps_completed == 0


def test_wrong_way_heading_must_persist_before_terminal_event():
    """A short sample glitch is tolerated but a spin is rejected."""
    gate = SafetyLapGate(envelope(), heading_violation_samples=3)
    cross_start(gate)
    assert gate.update(0.20, 0.0, np.pi, 2.0).event is None
    assert gate.update(0.21, 0.0, np.pi, 2.0).event is None
    update = gate.update(0.22, 0.0, np.pi, 2.0)
    assert update.event == "safety_violation"
    assert update.reason == "wrong_way_or_spin"


def test_margin_selects_the_boundary_on_the_error_side():
    """ISO left-positive error must consume left, not right, width."""
    asymmetric = TrackEnvelope(
        progress=np.asarray([0.0, 0.5, 0.99]),
        left_width_m=np.asarray([1.5, 1.5, 1.5]),
        right_width_m=np.asarray([3.0, 3.0, 3.0]),
    )
    _, _, left_margin = asymmetric.boundary_margin(0.1, 0.5, 0.0, 0.7, 1.3)
    _, _, right_margin = asymmetric.boundary_margin(0.1, -0.5, 0.0, 0.7, 1.3)
    assert left_margin == pytest.approx(0.3)
    assert right_margin == pytest.approx(1.8)
