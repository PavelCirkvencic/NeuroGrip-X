"""Regression tests for the telemetry replay's same-distance interval."""

import numpy as np
import pytest

from experiments_v2.render_telemetry_video import (
    Episode,
    build_lap_timing,
    same_distance_interval,
)


def linear_episode(label: str, duration_s: float) -> Episode:
    """Create one full constant-progress lap with a finish-line wrap."""
    time_s = np.linspace(0.0, duration_s, 101)
    progress = np.mod(np.linspace(0.0, 1.0, 101), 1.0)
    zeros = np.zeros_like(time_s)
    ones = np.ones_like(time_s)
    return Episode(
        label=label,
        time_s=time_s,
        progress=progress,
        lateral_error_m=zeros,
        heading_error_rad=zeros,
        speed_mps=ones,
        front_grip=ones,
        rear_grip=ones,
    )


def test_same_distance_interval_uses_passage_times_not_position_delta():
    """An 8 s lap must lead a 10 s lap by 0.8 s at 40% distance."""
    standard = linear_episode("Standard", 10.0)
    neurogrip = linear_episode("NeuroGripX", 8.0)
    standard_timing = build_lap_timing(standard)
    neurogrip_timing = build_lap_timing(neurogrip)

    left_leads, gap_s = same_distance_interval(
        standard,
        neurogrip,
        standard_timing,
        neurogrip_timing,
        4.0,
    )

    assert left_leads is False
    assert gap_s == pytest.approx(0.8)


def test_same_distance_interval_reaches_exact_finish_gap():
    """The finish-line interval must equal the measured lap-time difference."""
    standard = linear_episode("Standard", 10.0)
    neurogrip = linear_episode("NeuroGripX", 8.0)
    left_leads, gap_s = same_distance_interval(
        standard,
        neurogrip,
        build_lap_timing(standard),
        build_lap_timing(neurogrip),
        10.0,
    )

    assert left_leads is False
    assert gap_s == pytest.approx(2.0)
