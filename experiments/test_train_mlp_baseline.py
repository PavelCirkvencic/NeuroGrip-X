"""Correctness tests for the multi-step rollout evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_mlp_baseline import rollout_rmse

HORIZONS = [1, 2, 3]


def synthetic_frame(samples: int = 6) -> pd.DataFrame:
    """Build a tiny transition frame with known linear dynamics."""
    index = np.arange(samples, dtype=float)
    return pd.DataFrame(
        {
            "time_s": index * 0.02,
            "dt_s": np.full(samples, 0.02),
            "v_x_t_mps": index,
            "yaw_rate_t_rps": np.zeros(samples),
            "cmd_linear_x_t_mps": np.ones(samples),
            "cmd_angular_z_t_rps": np.zeros(samples),
            "v_x_t1_mps": index + 1.0,
            "yaw_rate_t1_rps": np.zeros(samples),
        }
    )


def test_rollout_is_exact_for_a_matching_model():
    """A model that matches the true transition has zero rollout error."""

    def exact_step(states, commands):
        output = states.copy()
        output[:, 0] += commands[:, 0]
        return output

    result = rollout_rmse({"test": [synthetic_frame()]}, exact_step, HORIZONS)
    for horizon in HORIZONS:
        assert result["test"][horizon]["v_x_t1_mps"] == pytest.approx(0.0, abs=1e-12)


def test_rollout_error_grows_for_a_zero_model():
    """A model predicting zero state accumulates growing rollout error."""

    def zero_step(states, commands):
        return np.zeros_like(states)

    result = rollout_rmse({"test": [synthetic_frame()]}, zero_step, HORIZONS)
    first = result["test"][1]["v_x_t1_mps"]
    third = result["test"][3]["v_x_t1_mps"]
    assert first > 0.0
    assert third >= first
