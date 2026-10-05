"""Contract tests for the deployable physical context-residual model."""

import numpy as np
import torch

from neurogrip_ai.physics_residual import (
    PhysicsResidualConfig,
    PhysicsResidualNet,
    physical_prediction,
    stable_enough,
)


def test_zero_initial_residual_preserves_nominal_physical_model():
    """Untrained output must be nominal A/B, not an arbitrary learned matrix."""
    model = PhysicsResidualNet(PhysicsResidualConfig())
    history = torch.zeros((2, 25, 7))
    nominal_a = torch.eye(2).repeat(2, 1, 1)
    nominal_b = torch.ones((2, 2, 1))
    actual_a, actual_b, context = model(history, nominal_a, nominal_b)
    assert context.shape == (2, 8)
    assert torch.allclose(actual_a, nominal_a)
    assert torch.allclose(actual_b, nominal_b)


def test_physical_prediction_and_health_contract():
    """A/B have the documented dimensions and unstable matrices are rejected."""
    predicted = physical_prediction(
        torch.eye(2).unsqueeze(0), torch.ones((1, 2, 1)),
        torch.zeros((1, 2)), torch.tensor([0.1]),
    )
    assert torch.allclose(predicted, torch.tensor([[0.1, 0.1]]))
    assert stable_enough(np.eye(2))
    assert not stable_enough(np.array([[1.2, 0.0], [0.0, 1.0]]))
