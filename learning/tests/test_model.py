"""Invariant tests for the contextual Koopman model."""

from __future__ import annotations

import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.model import NeuroGripModel


def make_config() -> NeuroGripConfig:
    """Small deterministic configuration for fast tests."""
    return NeuroGripConfig(
        history_steps=5,
        rollout_steps=4,
        context_dim=6,
        latent_extra=9,
        rank=4,
        hidden_dim=16,
    )


def test_lifted_state_starts_with_physical_state():
    """ADR-004: the first three latent components are the physical state."""
    torch.manual_seed(0)
    config = make_config()
    model = NeuroGripModel(config)
    history = torch.randn(3, config.history_steps, config.feature_dim)
    state = torch.randn(3, config.state_dim)
    commands = torch.randn(3, config.rollout_steps, config.input_dim)
    output = model(history, state, commands)
    assert torch.allclose(output.lifted_state[:, : config.state_dim], state)


def test_output_shapes_are_consistent():
    """A/B/bias/rollout shapes follow the configuration."""
    torch.manual_seed(0)
    config = make_config()
    model = NeuroGripModel(config)
    history = torch.randn(2, config.history_steps, config.feature_dim)
    state = torch.randn(2, config.state_dim)
    commands = torch.randn(2, config.rollout_steps, config.input_dim)
    output = model(history, state, commands)
    assert output.context.shape == (2, config.context_dim)
    assert output.A.shape == (2, config.latent_dim, config.latent_dim)
    assert output.B.shape == (2, config.latent_dim, config.input_dim)
    assert output.bias.shape == (2, config.latent_dim)
    assert output.rollout.shape == (2, config.rollout_steps, config.state_dim)


def test_rollout_first_step_matches_manual_operator():
    """The first rollout step equals the manual affine operator application."""
    torch.manual_seed(0)
    config = make_config()
    model = NeuroGripModel(config)
    history = torch.randn(2, config.history_steps, config.feature_dim)
    state = torch.randn(2, config.state_dim)
    commands = torch.randn(2, config.rollout_steps, config.input_dim)
    output = model(history, state, commands)
    manual = (
        torch.bmm(output.A, output.lifted_state.unsqueeze(-1)).squeeze(-1)
        + torch.bmm(output.B, commands[:, 0].unsqueeze(-1)).squeeze(-1)
        + output.bias
    )
    assert torch.allclose(output.rollout[:, 0], manual[:, : config.state_dim], atol=1e-5)


def test_global_ablation_ignores_history():
    """B1-K uses one shared context regardless of the input window."""
    torch.manual_seed(0)
    config = make_config()
    model = NeuroGripModel(config, context_mode="learned_global")
    history_a = torch.randn(1, config.history_steps, config.feature_dim)
    history_b = torch.randn(1, config.history_steps, config.feature_dim)
    context_a = model.encode_context(history_a)
    context_b = model.encode_context(history_b)
    assert torch.allclose(context_a, context_b)


def test_context_depends_only_on_history():
    """The context must not depend on future commands (causality)."""
    torch.manual_seed(0)
    config = make_config()
    model = NeuroGripModel(config)
    history = torch.randn(1, config.history_steps, config.feature_dim)
    context = model.encode_context(history)
    mutated = history.clone()
    mutated[:, 0, :] += 1.0
    assert not torch.allclose(context, model.encode_context(mutated))
