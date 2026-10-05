"""Contract tests for causal periodic track projection."""

import numpy as np

from neurogrip_control.track_projection import TrackProjector


def projector() -> TrackProjector:
    """Return a compact square-like cyclic reference for geometry tests."""
    return TrackProjector(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        yaw=np.array([0.0, 0.0, np.pi / 2.0, np.pi]),
        curvature=np.zeros(4),
        s=np.array([0.0, 1.0, 2.0, 3.0]),
    )


def test_lateral_error_has_iso_8855_left_positive_sign():
    """A point left of an eastward reference heading must have positive error."""
    result = projector().project(0.5, 0.2, 0.0, None)
    assert result["e_y_m"] > 0.0


def test_teleport_falls_back_from_local_window_to_global_search():
    """A reset pose cannot remain pinned to an old local track neighbourhood."""
    result = projector().project(1.0, 1.0, np.pi / 2.0, 0)
    assert result["index"] == 2


def test_npz_loader_removes_duplicate_periodic_endpoint(tmp_path):
    """Start position must never alternate between progress zero and one."""
    path = tmp_path / "track.npz"
    np.savez(
        path,
        x=np.asarray([0.0, 1.0, 0.0]),
        y=np.asarray([0.0, 0.0, 0.0]),
        yaw=np.zeros(3),
        curvature=np.zeros(3),
        s=np.asarray([0.0, 1.0, 2.0]),
        length=np.asarray([2.0]),
    )
    loaded = TrackProjector.from_npz(str(path))
    assert len(loaded.s) == 2
    assert loaded.length_m == 2.0
    assert loaded.project(0.0, 0.0, 0.0, None)["lap_progress"] == 0.0
