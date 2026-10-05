"""Unit tests for the ROS-free contextual Koopman runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from neurogrip_ai.contextual_runner import ContextualModelRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
LEARNING_ROOT = REPO_ROOT / "learning"
if str(LEARNING_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNING_ROOT))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.model import NeuroGripModel  # noqa: E402

CONFIG = {
    "state_dim": 2,
    "input_dim": 2,
    "history_steps": 3,
    "context_dim": 4,
    "latent_extra": 4,
    "rank": 2,
    "hidden_dim": 8,
    "rollout_steps": 1,
    "dropout": 0.0,
}


def make_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny N1 checkpoint with the real state/feature schema."""
    model = NeuroGripModel(NeuroGripConfig(**CONFIG), context_mode="encoder")
    checkpoint = tmp_path / "n1.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": CONFIG,
            "context_mode": "encoder",
            "statistics": {
                "state_mean": np.zeros(2).tolist(),
                "state_scale": np.ones(2).tolist(),
                "cmd_mean": np.zeros(2).tolist(),
                "cmd_scale": np.ones(2).tolist(),
            },
        },
        checkpoint,
    )
    return checkpoint


def test_runner_requires_full_history(tmp_path):
    """Prediction is undefined until a full causal window exists."""
    runner = ContextualModelRunner(make_checkpoint(tmp_path), LEARNING_ROOT)
    assert runner.ready() is False
    assert runner.predict() is None
    for _ in range(runner.config.history_steps):
        runner.append(0.3, 0.1, 0.4, 0.0)
    assert runner.ready() is True
    result = runner.predict()
    assert result is not None
    assert result["predicted_state"].shape == (2,)
    assert result["context"].shape == (runner.config.context_dim,)
    assert result["a_matrix"].shape == (
        runner.config.latent_dim,
        runner.config.latent_dim,
    )
    assert result["b_matrix"].shape == (runner.config.latent_dim, runner.config.input_dim)


def test_runner_predicts_after_history(tmp_path):
    """The runner produces a finite prediction once warmed up."""
    runner = ContextualModelRunner(make_checkpoint(tmp_path), LEARNING_ROOT)
    for _ in range(runner.config.history_steps):
        runner.append(0.5, 0.0, 0.5, 0.0)
    result = runner.predict()
    assert np.all(np.isfinite(result["predicted_state"]))
    assert np.isfinite(result["spectral_radius"])
