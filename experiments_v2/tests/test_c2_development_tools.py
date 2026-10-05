"""Regression tests for the pre-MATLAB C2 matrix and visual replay helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments_v2.render_telemetry_video import Episode, reconstruct_pose
from experiments_v2.run_identification_matrix import (
    complete_episode,
    scenario_ids_from_manifest,
)
from experiments_v2.run_paired_holdout import terminal_episode


def test_matrix_manifest_keeps_all_identification_splits_disjoint(tmp_path: Path):
    """A duplicate scenario cannot silently leak between train and validation."""
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "splits": {
            "train": ["a"], "validation": ["b"], "calibration": ["c"],
        },
    }))
    assert scenario_ids_from_manifest(manifest) == ["a", "b", "c"]
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "splits": {
            "train": ["a"], "validation": ["a"], "calibration": ["c"],
        },
    }))
    with pytest.raises(ValueError, match="more than one"):
        scenario_ids_from_manifest(manifest)


def test_matrix_resume_requires_identity_and_explicit_success(tmp_path: Path):
    """An interrupted or mismatched run can never be incorrectly skipped."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "episode.csv").write_text("header\n")
    (run / "episode.events.jsonl").write_text("{}\n")
    metadata = {
        "scenario_id": "scenario", "seed": 7, "controller": "EXCITATION",
        "runner": {"termination_reason": "timeout"},
    }
    (run / "episode.metadata.json").write_text(json.dumps(metadata))
    assert not complete_episode(run, "scenario", 7)
    metadata["runner"]["termination_reason"] = "excitation_complete"
    (run / "episode.metadata.json").write_text(json.dumps(metadata))
    assert complete_episode(run, "scenario", 7)
    assert not complete_episode(run, "scenario", 8)


def test_holdout_resume_preserves_a_timeout_but_rejects_failure(tmp_path: Path):
    """A valid baseline timeout must not be rerun until it looks more favourable."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "episode.csv").write_text("header\n")
    (run / "episode.events.jsonl").write_text("{}\n")
    metadata = {
        "scenario_id": "holdout", "seed": 301, "controller": "C1_ORACLE_MU",
        "runner": {"termination_reason": "timeout"},
    }
    (run / "episode.metadata.json").write_text(json.dumps(metadata))
    assert terminal_episode(run, "holdout", "C1_ORACLE_MU", 301)
    metadata["runner"]["termination_reason"] = "process_failure"
    (run / "episode.metadata.json").write_text(json.dumps(metadata))
    assert not terminal_episode(run, "holdout", "C1_ORACLE_MU", 301)


def test_telemetry_replay_reconstructs_reference_pose_at_zero_error():
    """The replay remains tied to recorded track progress, not decoration."""
    episode = Episode(
        label="test", time_s=np.asarray([0.0, 0.02]),
        progress=np.asarray([0.0, 0.5]), lateral_error_m=np.zeros(2),
        heading_error_rad=np.zeros(2), speed_mps=np.ones(2),
        front_grip=np.ones(2), rear_grip=np.ones(2),
    )
    track = {
        "x": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "y": np.asarray([5.0, 5.0, 5.0, 5.0]),
        "yaw": np.zeros(4),
    }
    x, y, yaw = reconstruct_pose(track, episode)
    assert x == pytest.approx([1.0, 3.0])
    assert y == pytest.approx([5.0, 5.0])
    assert yaw == pytest.approx([0.0, 0.0])
