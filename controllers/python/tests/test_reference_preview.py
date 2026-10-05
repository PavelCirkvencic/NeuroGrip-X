"""Tests for the non-myopic MPC reference preview."""

import numpy as np
import pytest

from reference_preview import ReferencePreview


def test_preview_advances_by_speed_and_wraps_periodically():
    """Future samples must follow arc length rather than repeat the current turn."""
    reference = ReferencePreview(
        s_m=np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        length_m=10.0,
        curvature_1pm=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 0.0]),
        left_width_m=np.ones(6) * 2.0,
        right_width_m=np.ones(6) * 2.5,
    )
    curvature, left, right = reference.horizon(0.9, 5.0, 0.2, 3, 0.7)
    assert curvature == pytest.approx([0.0, 0.1, 0.2])
    assert left == pytest.approx([1.3, 1.3, 1.3])
    assert right == pytest.approx([1.8, 1.8, 1.8])


def test_preview_never_returns_a_negative_corridor():
    """Malformed narrow geometry cannot invert QP bounds."""
    reference = ReferencePreview(
        s_m=np.asarray([0.0, 0.5, 1.0]),
        length_m=1.0,
        curvature_1pm=np.zeros(3),
        left_width_m=np.ones(3) * 0.4,
        right_width_m=np.ones(3) * 0.5,
    )
    _, left, right = reference.horizon(0.0, 1.0, 0.1, 2, 0.7)
    assert np.all(left == 0.05)
    assert np.all(right == 0.05)
