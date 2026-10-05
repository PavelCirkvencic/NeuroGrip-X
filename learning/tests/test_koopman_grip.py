"""Tests for the grip-aware Koopman extension."""

import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.koopman_grip import GripAwareKoopmanModel


def make_model() -> GripAwareKoopmanModel:
    """Return a compact deterministic test model."""
    config = NeuroGripConfig(
        state_dim=2,
        input_dim=1,
        history_steps=25,
        context_dim=8,
        latent_extra=4,
        rank=3,
        hidden_dim=16,
        rollout_steps=5,
        history_feature_dim=7,
    )
    return GripAwareKoopmanModel(config)


def test_grip_head_is_bounded_and_nominal_at_initialisation():
    """The untrained head starts at one and can never leave simulator bounds."""
    model = make_model()
    history = torch.randn(6, 25, 7) * 100.0
    grip = model.grip_from_context(model.encode_context(history))
    assert torch.all(grip >= 0.35)
    assert torch.all(grip <= 1.30)
    assert torch.allclose(grip, torch.ones_like(grip), atol=1e-6)


def test_forward_retains_physical_state_and_returns_two_axles():
    """Grip supervision does not change the lifted physical contract."""
    model = make_model()
    output = model(
        torch.zeros(2, 25, 7),
        torch.zeros(2, 2),
        torch.zeros(2, 5),
        torch.eye(2).repeat(2, 1, 1),
        torch.zeros(2, 2, 1),
    )
    assert output.state_rollout.shape == (2, 5, 2)
    assert output.grip_estimate.shape == (2, 2)
    assert torch.isfinite(output.state_rollout).all()
