"""Tests for the live MATLAB exact-shadow evidence evaluator."""

from __future__ import annotations

import csv
import json

import pytest

from experiments_v2.evaluate_matlab_exact_shadow import load_trace, summarize


def write_trace(path) -> None:
    """Write a minimal valid trace with two strictly ordered decisions."""
    fields = [
        "time_s",
        "matlab_steering_rad",
        "python_steering_rad",
        "abs_error_rad",
        "solve_time_ms",
        "deadline_miss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [1.0, 0.1, 0.1001, 0.0001, 2.0, 0])))
        writer.writerow(dict(zip(fields, [1.02, -0.2, -0.2, 0.0, 3.0, 0])))


def test_summary_is_hash_bound_and_reports_lap(tmp_path):
    """The compact report preserves timing, parity and provenance truth."""
    trace = tmp_path / "shadow.csv"
    metadata = tmp_path / "episode.metadata.json"
    source = tmp_path / "solver.m"
    write_trace(trace)
    metadata.write_text(
        json.dumps({"runner": {"termination_reason": "lap_complete"}}),
        encoding="utf-8",
    )
    source.write_text("% solver\n", encoding="utf-8")
    report = summarize(trace, metadata, [source])
    assert report["samples"] == 2
    assert report["episode_termination"] == "lap_complete"
    assert report["deadline_misses"] == 0
    assert report["max_abs_error_rad"] == pytest.approx(1e-4)
    assert len(report["capture_sha256"]) == 64


def test_inconsistent_error_column_is_rejected(tmp_path):
    """A manually altered error column cannot pass the evidence parser."""
    trace = tmp_path / "shadow.csv"
    write_trace(trace)
    text = trace.read_text(encoding="utf-8").replace("0.0001", "0.5", 1)
    trace.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_trace(trace)
