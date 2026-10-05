"""NeuroGrip-X contextual Koopman model (ROS-free PyTorch).

Structure follows the project plan: a causal context encoder summarises the
recent telemetry window, a lifting encoder appends learned observables to the
physical state, and a context-conditioned low-rank operator produces the local
linear predictor ``z_{k+1} = A(c) z_k + B(c) u_k + b(c)``.  The first
``state_dim`` latent components are always copied verbatim from the physical
state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurogrip.config import NeuroGripConfig


@dataclass
class NeuroGripOutput:
    """Bundle of model outputs for one batch."""

    context: torch.Tensor
    lifted_state: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    bias: torch.Tensor
    aux: torch.Tensor
    rollout: torch.Tensor


class CausalContextEncoder(torch.nn.Module):
    """Summarise a past-only window into a small context vector."""

    def __init__(self, feature_dim: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.gru = torch.nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return the context from the last valid hidden state of the window."""
        encoded, _ = self.gru(history)
        return self.head(encoded[:, -1])


class LiftingEncoder(torch.nn.Module):
    """Append learned observables to the physical state, conditioned on context."""

    def __init__(self, config: NeuroGripConfig):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.state_dim + config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.latent_extra),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return the learned latent features for each batch element."""
        return self.net(torch.cat([state, context], dim=-1))


class LowRankContextOperator(torch.nn.Module):
    """Context-conditioned A/B via a low-rank correction around a shared base."""

    def __init__(self, config: NeuroGripConfig):
        super().__init__()
        self.rank = config.rank
        self.latent_dim = config.latent_dim
        self.A0 = torch.nn.Parameter(torch.eye(config.latent_dim))
        self.B0 = torch.nn.Parameter(torch.zeros(config.latent_dim, config.input_dim))
        self.Ua = torch.nn.Parameter(torch.randn(config.latent_dim, config.rank) * 0.01)
        self.Va = torch.nn.Parameter(torch.randn(config.latent_dim, config.rank) * 0.01)
        self.Ub = torch.nn.Parameter(torch.randn(config.latent_dim, config.rank) * 0.01)
        self.Vb = torch.nn.Parameter(torch.randn(config.input_dim, config.rank) * 0.01)
        self.coefficient_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, 2 * config.rank),
        )
        self.bias_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (A, B, bias) for each batch element."""
        coefficients = self.coefficient_head(context)
        a_coefficients = coefficients[:, : self.rank]
        b_coefficients = coefficients[:, self.rank :]
        a_delta = torch.einsum("ir,br,jr->bij", self.Ua, a_coefficients, self.Va)
        b_delta = torch.einsum("ir,br,jr->bij", self.Ub, b_coefficients, self.Vb)
        return self.A0 + a_delta, self.B0 + b_delta, self.bias_head(context)


class PhysicsAuxiliaryHead(torch.nn.Module):
    """Auxiliary head predicting a normalised acceleration proxy.

    It is intentionally *not* advertised as calibrated tire force; it exists to
    give the model an interpretable auxiliary target and to regularise the
    latent dynamics.  A real force head requires the nominal vehicle parameters.
    """

    def __init__(self, config: NeuroGripConfig):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                config.state_dim + config.input_dim + config.context_dim,
                config.hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.state_dim),
        )

    def forward(
        self, state: torch.Tensor, command: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Return the auxiliary acceleration proxy for each batch element."""
        return self.net(torch.cat([state, command, context], dim=-1))


class NeuroGripModel(torch.nn.Module):
    """Context-conditioned Koopman state-transition model."""

    def __init__(self, config: NeuroGripConfig, context_mode: str = "encoder"):
        super().__init__()
        if context_mode not in ("encoder", "learned_global"):
            raise ValueError(f"Unsupported context_mode: {context_mode}")
        self.config = config
        self.context_mode = context_mode
        self.context_encoder = CausalContextEncoder(
            config.feature_dim, config.hidden_dim, config.context_dim
        )
        self.lifting_encoder = LiftingEncoder(config)
        self.operator = LowRankContextOperator(config)
        self.auxiliary_head = PhysicsAuxiliaryHead(config)
        if context_mode == "learned_global":
            self.global_context = torch.nn.Parameter(torch.zeros(config.context_dim))
        else:
            self.global_context = None

    def encode_context(self, history: torch.Tensor) -> torch.Tensor:
        """Return the causal context for the window (or the global ablation)."""
        if self.context_mode == "learned_global":
            return self.global_context.expand(history.shape[0], -1)
        return self.context_encoder(history)

    def forward(
        self,
        history: torch.Tensor,
        current_state: torch.Tensor,
        future_commands: torch.Tensor,
    ) -> NeuroGripOutput:
        """Roll the lifted model forward over the provided command sequence.

        ``history`` is ``[batch, H, feature_dim]`` and must contain only past
        samples.  ``future_commands`` is ``[batch, K, input_dim]``.  The returned
        rollout uses predicted states and the given (open-loop) commands.
        """
        context = self.encode_context(history)
        learned_features = self.lifting_encoder(current_state, context)
        lifted_state = torch.cat([current_state, learned_features], dim=-1)
        matrix_a, matrix_b, bias = self.operator(context)
        aux = self.auxiliary_head(current_state, future_commands[:, 0], context)

        states = []
        latent = lifted_state
        for step in range(future_commands.shape[1]):
            command = future_commands[:, step]
            latent = (
                torch.bmm(matrix_a, latent.unsqueeze(-1)).squeeze(-1)
                + torch.bmm(matrix_b, command.unsqueeze(-1)).squeeze(-1)
                + bias
            )
            states.append(latent[:, : self.config.state_dim])

        return NeuroGripOutput(
            context=context,
            lifted_state=lifted_state,
            A=matrix_a,
            B=matrix_b,
            bias=bias,
            aux=aux,
            rollout=torch.stack(states, dim=1),
        )
