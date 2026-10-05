"""Tests for the deterministic EUFS track builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from build_track import (
    _orient_cycle,
    align_reference_start,
    build_centerline,
    build_track,
    load_eufs_csv,
    resample_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_ROOT = REPO_ROOT / "external" / "eufs_ws" / "src" / "map_lib" / "maps"
SMALL_TRACK = MAP_ROOT / "tracks" / "small_track.csv"
TRACKDRIVE = MAP_ROOT / "competitions" / "FSUK" / "2023" / "trackdrive.csv"

pytestmark = pytest.mark.skipif(
    not SMALL_TRACK.is_file(),
    reason="EUFS map_lib checkout is not present (run scripts/bootstrap_eufs.sh)",
)


def test_small_track_cycle_has_67_midpoints():
    """The small track yields one cycle of 67 cross-colour midpoints."""
    cones, _ = load_eufs_csv(SMALL_TRACK)
    cycle = build_centerline(cones)
    assert cycle.shape == (67, 2)


def test_trackdrive_cycle_has_196_midpoints():
    """The FSUK trackdrive map yields one cycle of 196 midpoints."""
    if not TRACKDRIVE.is_file():
        pytest.skip("trackdrive.csv not present")
    cones, _ = load_eufs_csv(TRACKDRIVE)
    cycle = build_centerline(cones)
    assert cycle.shape == (196, 2)


def test_reference_is_finite_and_monotonic():
    """Arc length is strictly increasing and every array is finite."""
    cones, car_start = load_eufs_csv(SMALL_TRACK)
    cycle = build_centerline(cones)
    reference = resample_reference(cycle, 0.20)
    assert np.all(np.diff(reference["s"]) > 0.0)
    for key in ("s", "x", "y", "yaw", "curvature"):
        assert np.all(np.isfinite(reference[key])), key


def test_closure_error_is_small():
    """A periodic reference closes on itself within 0.25 m."""
    cones, _ = load_eufs_csv(SMALL_TRACK)
    cycle = build_centerline(cones)
    reference = resample_reference(cycle, 0.20)
    closure = np.hypot(
        reference["x"][-1] - reference["x"][0],
        reference["y"][-1] - reference["y"][0],
    )
    assert closure < 0.25


def test_reference_curvature_is_physically_trackable():
    """Smoothing must remove interpolation spikes beyond steering authority."""
    cones, car_start = load_eufs_csv(SMALL_TRACK)
    cycle = _orient_cycle(build_centerline(cones, np.zeros(2)), car_start[2])
    reference = resample_reference(cycle, 0.20)
    reference = align_reference_start(reference, cycle, 0.20)
    assert np.max(np.abs(reference["curvature"])) < 0.30
    assert abs(reference["yaw"][0] - car_start[2]) < 0.30


def test_npz_and_mat_agree(tmp_path):
    """The .npz and .mat references agree within 1e-10."""
    manifest = build_track(SMALL_TRACK, "agree_test", tmp_path, 0.20)
    npz = np.load(manifest["outputs"]["npz"])
    struct = loadmat(manifest["outputs"]["mat"])["reference"][0, 0]
    for key in ("s", "x", "y", "yaw", "curvature", "left_width", "right_width"):
        mat_values = np.asarray(struct[key]).ravel()
        assert np.allclose(npz[key], mat_values, atol=1e-10), key


def test_manifest_records_hashes(tmp_path):
    """The manifest carries input and output hashes for provenance."""
    manifest = build_track(SMALL_TRACK, "hash_test", tmp_path, 0.20)
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["npz_sha256"]) == 64
    assert manifest["centerline_midpoints"] == 67
