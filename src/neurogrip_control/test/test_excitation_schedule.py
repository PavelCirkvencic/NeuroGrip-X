"""Unit tests for the deterministic excitation schedule."""

from __future__ import annotations

import pytest

from neurogrip_control.excitation_schedule import (
    EXCITATION_PROFILES,
    ExcitationPhase,
    ExcitationSchedule,
)


def test_first_phase_starts_at_zero_elapsed():
    """The schedule begins at the first phase with zero phase elapsed."""
    schedule = ExcitationSchedule(EXCITATION_PROFILES["baseline_v1"])
    state = schedule.advance(0.0)
    assert state.phase_name == "settle"
    assert state.phase_index == 0
    assert state.phase_elapsed_s == pytest.approx(0.0)
    assert (state.linear_x_mps, state.angular_z_rps) == (0.0, 0.0)


def test_phase_boundary_enters_new_phase_at_zero_elapsed():
    """Crossing a boundary starts the new phase with phase_elapsed 0."""
    schedule = ExcitationSchedule(EXCITATION_PROFILES["baseline_v1"])
    schedule.advance(0.0)
    state = schedule.advance(3.0)
    assert state.phase_name == "straight_slow"
    assert state.entered_new_phase is True
    assert state.phase_elapsed_s == pytest.approx(0.0)
    assert state.linear_x_mps == pytest.approx(0.30)


def test_late_timer_skips_multiple_phases():
    """A late tick jumps straight to the phase containing elapsed time."""
    schedule = ExcitationSchedule(EXCITATION_PROFILES["baseline_v1"])
    schedule.advance(0.0)
    state = schedule.advance(9.0)  # settle 3 s + straight_slow 5 s -> left_constant
    assert state.phase_name == "left_constant"
    assert state.phase_elapsed_s == pytest.approx(1.0)
    assert state.entered_new_phase is True


def test_first_chirp_sample_is_zero():
    """The first steering-chirp sample uses phase_elapsed 0, so sin(0)=0."""
    schedule = ExcitationSchedule(EXCITATION_PROFILES["dynamic_v2"])
    state = schedule.advance(12.0)  # settle+straight_mid+preload+preload
    assert state.phase_name == "steering_chirp"
    assert state.phase_elapsed_s == pytest.approx(0.0)
    assert state.angular_z_rps == pytest.approx(0.0)
    assert state.linear_x_mps == pytest.approx(0.65)


def test_schedule_is_deterministic():
    """The same elapsed time produces the same command on two schedules."""
    first = ExcitationSchedule(EXCITATION_PROFILES["dynamic_v2"])
    second = ExcitationSchedule(EXCITATION_PROFILES["dynamic_v2"])
    for elapsed_s in (0.0, 5.0, 12.0, 12.5, 25.0, 30.5):
        assert first.advance(elapsed_s) == second.advance(elapsed_s)


def test_completion_is_signalled_once_and_zeroes_command():
    """Completion fires once and all later samples are a safe zero."""
    schedule = ExcitationSchedule(EXCITATION_PROFILES["baseline_v1"])
    state = schedule.advance(schedule.total_duration_s)
    assert state.finished is True
    assert state.completed_now is True
    assert state.phase_name == "complete"
    assert (state.linear_x_mps, state.angular_z_rps) == (0.0, 0.0)

    later = schedule.advance(schedule.total_duration_s + 5.0)
    assert later.finished is True
    assert later.completed_now is False
    assert (later.linear_x_mps, later.angular_z_rps) == (0.0, 0.0)


def test_empty_schedule_is_rejected():
    """A schedule without phases is a configuration error."""
    with pytest.raises(ValueError):
        ExcitationSchedule(())


def test_chirp_waveform_matches_formula():
    """The chirp waveform keeps the documented instantaneous frequency."""
    phase = ExcitationPhase(
        "chirp", 2.0, 0.5, 1.0, waveform="chirp", chirp_start_hz=0.0, chirp_end_hz=1.0
    )
    linear, angular = phase.command_at(0.0)
    assert linear == pytest.approx(0.5)
    assert angular == pytest.approx(0.0)
    # At t=1 s the chirp phase is 2*pi*(0 + 0.5*0.5*1^2) = pi/2 -> sin = 1.
    _, half = phase.command_at(1.0)
    assert half == pytest.approx(1.0)
