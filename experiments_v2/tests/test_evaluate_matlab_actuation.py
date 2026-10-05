"""Tests for the MATLAB lateral actuator-in-loop evidence gate."""

from __future__ import annotations

import csv
import json

import pytest

from experiments_v2.evaluate_matlab_actuation import summarize, validate


def write_csv(path, fields, rows) -> None:
    """Write a compact test fixture."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(tmp_path):
    """Build one valid early-takeover, no-fallback lap report."""
    episode = tmp_path / "episode.csv"
    events = tmp_path / "events.jsonl"
    metadata = tmp_path / "metadata.json"
    candidate = tmp_path / "candidate.csv"
    mux = tmp_path / "mux.csv"
    source = tmp_path / "source.m"
    episode_fields = [
        "time_s",
        "lap_progress",
        "boundary_margin_m",
        "e_y_m",
        "cone_collision_count",
    ]
    write_csv(
        episode,
        episode_fields,
        [
            dict(zip(episode_fields, [1, 0.0, 1.0, 0.1, 0])),
            dict(zip(episode_fields, [2, 0.1, 0.8, -0.2, 0])),
            dict(zip(episode_fields, [10, 1.0, 0.7, 0.1, 0])),
        ],
    )
    event_rows = [
        {"event": "experiment_start", "sim_time_s": 1.0},
        {"event": "lap_complete", "sim_time_s": 10.0, "safety_valid": True},
    ]
    events.write_text(
        "\n".join(json.dumps(row) for row in event_rows) + "\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps({"runner": {"termination_reason": "lap_complete"}}),
        encoding="utf-8",
    )
    candidate_fields = [
        "time_s",
        "steering_rad",
        "solution_valid",
        "solve_time_ms",
    ]
    write_csv(
        candidate,
        candidate_fields,
        [
            dict(zip(candidate_fields, [1.5, 0.1, 1, 2.0])),
            dict(zip(candidate_fields, [10.0, -0.1, 1, 3.0])),
        ],
    )
    mux_fields = [
        "schema_version",
        "time_s",
        "source_id",
        "fallback_count",
        "reject_count",
    ]
    write_csv(
        mux,
        mux_fields,
        [
            dict(zip(mux_fields, [1, 1.0, 4, 0, 1])),
            dict(zip(mux_fields, [1, 1.5, 2, 0, 1])),
            dict(zip(mux_fields, [1, 10.0, 2, 0, 1])),
            dict(zip(mux_fields, [1, 10.1, 3, 1, 1])),
        ],
    )
    source.write_text("% source\n", encoding="utf-8")
    return summarize(episode, events, metadata, candidate, mux, [source])


def test_development_gate_excludes_shutdown_fallback(tmp_path):
    """A post-lap fallback is reported but does not fail the lap."""
    report = make_report(tmp_path)
    validate(report)
    assert report["post_lap_fallbacks_during_shutdown"] == 1
    assert report["new_fallbacks_after_takeover"] == 0
    assert report["active_deadline_misses"] == 0
    assert report["performance_claim"] is False


def test_development_gate_rejects_active_source_change(tmp_path):
    """Any fallback inside the measured lap invalidates actuator evidence."""
    report = make_report(tmp_path)
    report["active_non_matlab_mux_samples"] = 1
    with pytest.raises(ValueError, match="source changed"):
        validate(report)
