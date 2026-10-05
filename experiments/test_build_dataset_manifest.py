"""Tests for catalog admission rules (completion and grip provenance)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_dataset_manifest import load_run_records


def write_run(
    processed_dir: Path,
    stem: str,
    scenario: dict,
    termination_reason: str = "excitation_complete",
    grip_transition_status=None,
) -> None:
    """Write one minimal processed transition dataset with provenance."""
    (processed_dir / f"{stem}_transitions.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    pd.DataFrame({"time_s": [0.0, 0.02]}).to_parquet(
        processed_dir / f"{stem}_transitions.parquet", index=False
    )
    provenance = {
        "git_commit": "abc123",
        "excitation_profile": "dynamic_v2",
        "termination_reason": termination_reason,
        "grip_transition_status": grip_transition_status,
        "scenario_manifest": {
            "scenario_id": scenario.get("scenario_id", "scenario_x"),
            "sha256": "deadbeef",
            "scenario": scenario,
        },
    }
    metadata = {
        "transitions": 2,
        "source_csv": f"{processed_dir}/{stem}.csv",
        "source_run_provenance": provenance,
    }
    (processed_dir / f"{stem}_transitions.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_clean_run_is_admitted(tmp_path):
    """A completed run with a valid manifest enters the catalog."""
    write_run(tmp_path, "run_a", {"profile": "nominal", "seed": 1})
    records, untraceable, incomplete, transition = load_run_records(tmp_path)
    assert [record["run_id"] for record in records] == ["run_a"]
    assert not incomplete and not transition


def test_unconfirmed_grip_transition_is_rejected(tmp_path):
    """A transitioning scenario without a confirmed status is skipped."""
    scenario = {
        "profile": "transitioning_grip",
        "seed": 1,
        "grip_transition": {"start_s": 12.0, "compliance_scale": 2.0},
    }
    write_run(tmp_path, "run_missing", scenario)
    write_run(
        tmp_path,
        "run_failed",
        scenario,
        grip_transition_status={"expected": True, "applied": False},
    )
    write_run(
        tmp_path,
        "run_ok",
        scenario,
        grip_transition_status={"expected": True, "applied": True},
    )
    records, _, _, transition = load_run_records(tmp_path)
    assert [record["run_id"] for record in records] == ["run_ok"]
    assert set(transition) == {
        "run_failed_transitions.metadata.json",
        "run_missing_transitions.metadata.json",
    }


def test_incomplete_run_is_rejected(tmp_path):
    """A run that did not complete excitation never enters the catalog."""
    write_run(
        tmp_path, "run_partial", {"profile": "nominal", "seed": 2}, "interrupted"
    )
    write_run(tmp_path, "run_good", {"profile": "nominal", "seed": 3})
    records, _, incomplete, _ = load_run_records(tmp_path)
    assert [record["run_id"] for record in records] == ["run_good"]
    assert len(incomplete) == 1
