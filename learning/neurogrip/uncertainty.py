"""Uncertainty, conformal calibration and OOD helpers.

The design follows the project plan: raw per-horizon residual statistics give
an uncertainty scale, split conformal calibration turns it into a calibrated
radius on a dedicated calibration split, and a context-embedding Mahalanobis
distance provides an out-of-distribution signal.  None of these can increase
aggressiveness; they only widen intervals or trigger fallback downstream.
"""

from __future__ import annotations

import numpy as np


def per_horizon_sigma(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return residual standard deviation per horizon and state component.

    ``predicted`` and ``target`` are ``[N, K, state_dim]``.  The result is
    ``[K, state_dim]`` and is used as the uncertainty scale in the conformal
    score.  A small floor prevents division by zero.
    """
    residual = predicted - target
    sigma = residual.std(axis=0)
    return np.maximum(sigma, 1e-9)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample split-conformal quantile for coverage 1-alpha.

    Uses the standard ``ceil((n + 1) * (1 - alpha)) / n`` order statistic so the
    coverage guarantee holds for exchangeable data.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot calibrate conformal radius from no scores.")
    n = scores.size
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def normalized_scores(predicted: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the per-window conformal score ``max_h |err| / sigma``."""
    error = np.abs(predicted - target) / sigma[None, :, :]
    return error.max(axis=(1, 2))


def empirical_coverage(
    predicted: np.ndarray, target: np.ndarray, sigma: np.ndarray, radius: float
) -> tuple[float, float]:
    """Return (all-step coverage, mean interval width) for a radius."""
    error = np.abs(predicted - target)
    within = error <= radius * sigma[None, :, :]
    coverage = float(within.all(axis=(1, 2)).mean())
    width = float((2.0 * radius * sigma).mean())
    return coverage, width


def mahalanobis_scores(train_embeddings: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray:
    """Return Mahalanobis distance of each query embedding from train stats."""
    train_embeddings = np.asarray(train_embeddings, dtype=float)
    query_embeddings = np.asarray(query_embeddings, dtype=float)
    mean = train_embeddings.mean(axis=0)
    covariance = np.cov(train_embeddings, rowvar=False)
    inverse = np.linalg.pinv(covariance)
    delta = query_embeddings - mean
    return np.sqrt(np.einsum("nd,de,ne->n", delta, inverse, delta))
