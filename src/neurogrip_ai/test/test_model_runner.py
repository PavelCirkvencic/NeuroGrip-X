"""Unit tests for the ROS-free learned-dynamics runner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurogrip_ai.model_runner import TransitionModelRunner, build_model


def make_linear_bundle() -> dict:
    """Return a bundle whose model computes known linear combinations."""
    model = build_model([], 4, 2)
    linear = model[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 0] = 1.0
        linear.weight[0, 2] = 1.0  # v_x + linear command
        linear.weight[1, 1] = 1.0  # yaw rate
    return {
        "state_dict": model.state_dict(),
        "hidden_sizes": [],
        "feature_columns": [
            "v_x_t_mps",
            "yaw_rate_t_rps",
            "cmd_linear_x_t_mps",
            "cmd_angular_z_t_rps",
        ],
        "target_columns": ["v_x_t1_mps", "yaw_rate_t1_rps"],
        "feature_mean": [0.0] * 4,
        "feature_scale": [1.0] * 4,
        "target_mean": [0.0] * 2,
        "target_scale": [1.0] * 2,
    }


def test_predict_applies_known_linear_map():
    """The runner reproduces the checkpoint's linear output."""
    runner = TransitionModelRunner(make_linear_bundle())
    prediction = runner.predict(np.array([2.0, 3.0, 4.0, 5.0]))
    assert prediction == pytest.approx([6.0, 3.0])


def test_predict_state_matches_predict():
    """The convenience wrapper matches the raw predict call."""
    runner = TransitionModelRunner(make_linear_bundle())
    expected = runner.predict(np.array([1.0, 2.0, 3.0, 4.0]))
    actual = runner.predict_state(1.0, 2.0, 3.0, 4.0)
    assert actual == pytest.approx(tuple(expected))


def test_from_checkpoint_round_trip(tmp_path):
    """A checkpoint written to disk loads back with identical behavior."""
    checkpoint = tmp_path / "bundle.pt"
    torch.save(make_linear_bundle(), checkpoint)
    runner = TransitionModelRunner.from_checkpoint(checkpoint)
    assert runner.predict(np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx([2.0, 0.0])
