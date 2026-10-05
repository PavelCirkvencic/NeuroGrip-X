"""Tests for the uncertainty, conformal and OOD helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neurogrip.uncertainty import (
    conformal_quantile,
    empirical_coverage,
    mahalanobis_scores,
    normalized_scores,
    per_horizon_sigma,
)


def test_per_horizon_sigma_has_floor():
    """Sigma is positive even for perfectly repeating residuals."""
    predicted = np.zeros((4, 3, 3))
    target = np.zeros((4, 3, 3))
    sigma = per_horizon_sigma(predicted, target)
    assert sigma.shape == (3, 3)
    assert np.all(sigma > 0.0)


def test_conformal_quantile_is_monotone_in_coverage():
    """Higher coverage cannot reduce the calibrated radius."""
    scores = np.linspace(0.0, 1.0, 101)
    radius_90 = conformal_quantile(scores, 0.10)
    radius_50 = conformal_quantile(scores, 0.50)
    assert radius_90 >= radius_50


def test_conformal_quantile_rejects_empty_input():
    """Calibration without scores is a configuration error."""
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)


def test_empirical_coverage_bounds():
    """A huge radius covers everything; a zero radius covers nothing."""
    rng = np.random.default_rng(0)
    predicted = rng.normal(size=(50, 4, 3))
    target = rng.normal(size=(50, 4, 3))
    sigma = np.ones((4, 3))
    full_coverage, _ = empirical_coverage(predicted, target, sigma, radius=1e6)
    no_coverage, _ = empirical_coverage(predicted, target, sigma, radius=0.0)
    assert full_coverage == 1.0
    assert no_coverage == 0.0


def test_normalized_scores_use_worst_horizon():
    """The conformal score is the maximum normalized residual."""
    predicted = np.zeros((1, 2, 1))
    target = np.array([[[1.0], [0.5]]])
    sigma = np.ones((2, 1))
    score = normalized_scores(predicted, target, sigma)
    assert score[0] == pytest.approx(1.0)


def test_mahalanobis_grows_with_distance():
    """Points farther from the train mean have larger OOD scores."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(200, 4))
    near = train.mean(axis=0, keepdims=True)
    far = train.mean(axis=0, keepdims=True) + 5.0
    distances = mahalanobis_scores(train, np.vstack([near, far]))
    assert distances[1] > distances[0]
