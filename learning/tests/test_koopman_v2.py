"""Tests for the deployable physics-informed contextual Koopman model."""

import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.koopman_v2 import PhysicalKoopmanModel


def config() -> NeuroGripConfig:
    """Return a small v2 contract for fast deterministic tests."""
    return NeuroGripConfig(
        state_dim=2,
        input_dim=1,
        history_steps=25,
        context_dim=4,
        latent_extra=3,
        rank=2,
        hidden_dim=12,
        rollout_steps=20,
        history_feature_dim=7,
    )


def test_physical_coordinates_are_preserved_in_lift():
    """The first Koopman coordinates must equal the SI-unit physical state."""
    model = PhysicalKoopmanModel(config())
    state = torch.tensor([[0.2, -0.1]])
    context = torch.zeros((1, 4))
    lifted = model.lift_state(state, context)
    assert torch.equal(lifted[:, :2], state)


def test_zero_initialized_operator_is_nominal_and_stable():
    """Before training, the model must exactly reproduce the physical prior."""
    model = PhysicalKoopmanModel(config())
    history = torch.zeros((1, 25, 7))
    state = torch.tensor([[0.1, -0.2]])
    command = torch.tensor([[[0.05]]])
    nominal_a = torch.tensor([[[0.7, 0.1], [0.0, 0.6]]])
    nominal_b = torch.tensor([[[0.2], [0.3]]])
    output = model(history, state, command, nominal_a, nominal_b)
    expected = torch.bmm(nominal_a, state.unsqueeze(-1)).squeeze(-1)
    expected += nominal_b[:, :, 0] * command[:, 0, 0:1]
    assert torch.allclose(output.state_rollout[:, 0], expected, atol=1e-7)
    radius = torch.linalg.eigvals(output.koopman[0]).abs().max()
    assert radius < 1.0


def test_physical_jacobians_have_controller_shape():
    """Runtime linearization must emit physical A[2,2] and B[2,1]."""
    model = PhysicalKoopmanModel(config())
    matrix_a, matrix_b, context, prediction = model.physical_jacobians(
        torch.zeros((1, 25, 7)),
        torch.tensor([[0.1, 0.2]]),
        torch.tensor([0.03]),
        torch.tensor([[[0.8, 0.0], [0.1, 0.7]]]),
        torch.tensor([[[0.2], [0.3]]]),
    )
    assert matrix_a.shape == (2, 2)
    assert matrix_b.shape == (2, 1)
    assert context.shape == (4,)
    assert prediction.shape == (2,)
