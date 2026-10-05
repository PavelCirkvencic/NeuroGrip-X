"""Pure safety-contract tests for the Python/MATLAB candidate mux."""

from __future__ import annotations

import math

import pytest

from neurogrip_control.candidate_mux import (
    MatlabCandidate,
    activation_delay_elapsed,
    parse_matlab_candidate,
    select_candidate_source,
)


def test_valid_matlab_candidate_is_decoded():
    """A finite in-budget schema-1 result is eligible for selection."""
    candidate, reason = parse_matlab_candidate([1, 12.5, -0.2, 1, 3.4])
    assert reason == ""
    assert candidate == MatlabCandidate(12.5, -0.2, 3.4)


@pytest.mark.parametrize(
    "values,reason",
    [
        ([1, 1, 0, 1], "invalid_length"),
        ([2, 1, 0, 1, 2], "unsupported_schema"),
        ([1, 1, math.nan, 1, 2], "non_finite"),
        ([1, 1, 0, 0, 2], "invalid_solution"),
        ([1, -1, 0, 1, 2], "invalid_control_time"),
        ([1, 1, 0.38, 1, 2], "steering_out_of_bounds"),
        ([1, 1, 0.1, 1, 20.1], "solver_deadline_miss"),
    ],
)
def test_invalid_matlab_candidate_fails_closed(values, reason):
    """Malformed, unsafe or late MATLAB results never enter the mux."""
    candidate, actual_reason = parse_matlab_candidate(values)
    assert candidate is None
    assert actual_reason == reason


def test_matlab_selection_waits_for_first_valid_candidate():
    """Startup explicitly identifies the excitation-producing warm-up."""
    source = select_candidate_source(
        "matlab",
        python_fresh=True,
        matlab_fresh=False,
        matlab_ever_active=False,
        fallback_to_python=True,
    )
    assert source == "python_warmup"


def test_matlab_selection_stops_without_warmup_authority():
    """Missing Python longitudinal authority always means safe stop."""
    source = select_candidate_source(
        "matlab",
        python_fresh=False,
        matlab_fresh=False,
        matlab_ever_active=False,
        fallback_to_python=True,
    )
    assert source == "safe_stop"


def test_matlab_selection_falls_back_only_after_activation():
    """A later MATLAB dropout uses the explicitly permitted Python fallback."""
    assert select_candidate_source(
        "matlab",
        python_fresh=True,
        matlab_fresh=True,
        matlab_ever_active=False,
        fallback_to_python=True,
    ) == "matlab"
    assert select_candidate_source(
        "matlab",
        python_fresh=True,
        matlab_fresh=False,
        matlab_ever_active=True,
        fallback_to_python=True,
    ) == "python_fallback"


def test_no_fresh_source_means_safe_stop():
    """The selector never reuses a stale command."""
    assert select_candidate_source(
        "matlab",
        python_fresh=False,
        matlab_fresh=False,
        matlab_ever_active=True,
        fallback_to_python=True,
    ) == "safe_stop"


def test_matlab_activation_delay_is_inclusive_and_requires_python():
    """MATLAB cannot take over during the initial path-tracking transient."""
    assert not activation_delay_elapsed(4_000_000_000, None, 3.0)
    assert not activation_delay_elapsed(3_999_999_999, 1_000_000_000, 3.0)
    assert activation_delay_elapsed(4_000_000_000, 1_000_000_000, 3.0)
