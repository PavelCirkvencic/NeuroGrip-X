"""Synthetic regression coverage for strict v2 episode metrics."""

import numpy as np
import pytest

from experiments_v2.evaluate_closed_loop import (
    metrics,
    settling_time,
    validate_c2_activation,
)
from experiments_v2.evaluate_survivability import terminal_summary


def episode():
    """Build a 50 Hz episode with a deliberately bad stopped tail."""
    time_s = np.arange(0.0, 6.02, 0.02)
    error = np.where(time_s < 5.0, 0.2, 9.0)
    return {
        "time_s": time_s,
        "e_y_m": error,
        "v_x_mps": np.ones_like(time_s) * 3.0,
        "boundary_margin_m": np.ones_like(time_s) * 0.5,
        "cone_collision_count": np.zeros_like(time_s),
        "controller_solution_valid": np.ones_like(time_s),
        "controller_diagnostics_age_s": np.zeros_like(time_s),
        "controller_solve_time_ms": np.ones_like(time_s),
        "controller_max_slack_m": np.zeros_like(time_s),
    }


def test_metrics_truncate_at_lap_event_not_end_of_csv():
    """A stale tail after completion must have zero influence on benchmark RMS."""
    events = [
        {"event": "experiment_start", "sim_time_s": 0.0},
        {"event": "lap_complete", "sim_time_s": 4.98, "safety_valid": True},
    ]
    result = metrics(episode(), events)
    assert result["completed_lap"] is True
    assert result["rms_lateral_error_m"] == pytest.approx(0.2)


def test_metrics_reject_progress_wrap_after_track_exit():
    """A completion event cannot conceal a negative body-clearance sample."""
    data = episode()
    data["boundary_margin_m"][100] = -0.01
    events = [
        {"event": "experiment_start", "sim_time_s": 0.0},
        {"event": "lap_complete", "sim_time_s": 4.98, "safety_valid": True},
    ]
    with pytest.raises(ValueError, match="track envelope"):
        metrics(data, events)


def test_metrics_reject_legacy_unsafe_completion_event():
    """Old progress-only lap events are not valid v3 evidence."""
    events = [
        {"event": "experiment_start", "sim_time_s": 0.0},
        {"event": "lap_complete", "sim_time_s": 4.98},
    ]
    with pytest.raises(ValueError, match="safety-valid"):
        metrics(episode(), events)


def test_metrics_distinguish_bounded_discovery_gap_from_invalid_qp():
    """The -1 missing sentinel is allowed only before first diagnostics."""
    data = episode()
    data["controller_solution_valid"][:20] = -1.0
    events = [
        {"event": "experiment_start", "sim_time_s": 0.0},
        {"event": "lap_complete", "sim_time_s": 4.98, "safety_valid": True},
    ]
    result = metrics(data, events)
    assert result["initial_diagnostics_missing_samples"] == 20
    assert result["initial_diagnostics_gap_s"] == pytest.approx(0.4)


def test_metrics_reject_missing_or_invalid_diagnostics_mid_lap():
    """Neither a missing diagnostics stream nor a true invalid QP is hidden."""
    events = [
        {"event": "experiment_start", "sim_time_s": 0.0},
        {"event": "lap_complete", "sim_time_s": 4.98, "safety_valid": True},
    ]
    missing = episode()
    missing["controller_solution_valid"][100] = -1.0
    with pytest.raises(ValueError, match="disappeared"):
        metrics(missing, events)
    invalid = episode()
    invalid["controller_solution_valid"][100] = 0.0
    with pytest.raises(ValueError, match="invalid solution"):
        metrics(invalid, events)


def test_settling_requires_a_full_one_second_window():
    """One isolated in-bound sample is not a settled controller."""
    time_s = np.arange(0.0, 2.02, 0.02)
    error = np.where(time_s < 1.0, 0.4, 0.2)
    assert settling_time(time_s, error) == pytest.approx(1.0)


def test_c2_activation_uses_the_first_valid_model_before_transition():
    """A later reactivation must not erase evidence of pre-transition C2 use."""
    events = [
        {"event": "c2_model_active", "sim_time_s": 3.0, "dynamics_schema": 3, "artifact_sha256": "abc"},
        {"event": "c2_model_active", "sim_time_s": 15.0, "dynamics_schema": 3, "artifact_sha256": "abc"},
        {"event": "lap_complete", "sim_time_s": 20.0, "safety_valid": True},
    ]
    result = validate_c2_activation(events, transition_time=12.0)
    assert result["c2_model_active_sim_s"] == pytest.approx(3.0)
    assert result["c2_active_fraction"] == pytest.approx(1.0)


def test_c2_activation_rejects_excessive_post_transition_fallback():
    """A nominal controller wearing a C2 label cannot pass the benchmark."""
    events = [
        {"event": "c2_model_active", "sim_time_s": 3.0, "dynamics_schema": 3, "artifact_sha256": "abc"},
        {"event": "c2_nominal_fallback", "sim_time_s": 12.5},
        {"event": "c2_model_active", "sim_time_s": 19.5, "dynamics_schema": 3, "artifact_sha256": "abc"},
        {"event": "lap_complete", "sim_time_s": 20.0, "safety_valid": True},
    ]
    with pytest.raises(ValueError, match="active fraction"):
        validate_c2_activation(events, transition_time=12.0)


def test_terminal_summary_never_turns_a_timeout_into_a_partial_lap_metric():
    """DNF output must expose progress rather than a deceptively partial RMS."""
    incomplete = {
        "time_s": np.asarray([10.0, 10.02]),
        "lap_progress": np.asarray([0.2, 0.37]),
        "e_y_m": np.asarray([0.1, -2.5]),
    }
    result = terminal_summary(incomplete, {"runner": {"termination_reason": "timeout"}})
    assert result == {
        "completed_lap": False,
        "termination_reason": "timeout",
        "final_recorded_time_s": pytest.approx(0.02),
        "final_lap_progress": pytest.approx(0.37),
        "final_abs_lateral_error_m": pytest.approx(2.5),
    }
