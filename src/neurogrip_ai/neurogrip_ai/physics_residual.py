"""Deployable physical context-conditioned lateral model used by C2.

The network sees a causal 0.5 s telemetry window and predicts bounded residuals
around a discrete physical bicycle model.  Unlike the legacy Koopman model, its
public output is explicitly ``A[2,2]`` and ``B[2,1]`` for ``[v_y, r]`` in SI
units, suitable for the C2 MPC interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

FEATURE_NAMES = (
    "v_x_mps",
    "v_y_mps",
    "yaw_rate_rps",
    "steering_applied_rad",
    "acceleration_command_mps2",
    "a_x_mps2",
    "a_y_mps2",
)
STATE_NAMES = ("v_y_mps", "yaw_rate_rps")


@dataclass(frozen=True)
class PhysicsResidualConfig:
    """Architecture and physical output bounds stored with every checkpoint."""

    history_steps: int = 25
    feature_dim: int = 7
    hidden_dim: int = 48
    context_dim: int = 8
    residual_a_limit: float = 0.15
    residual_b_limit: float = 0.35

    def as_dict(self) -> dict:
        """Return a JSON/torch-serialisable config representation."""
        return asdict(self)


class PhysicsResidualNet(torch.nn.Module):
    """GRU context encoder plus bounded physical A/B residual head."""

    def __init__(self, config: PhysicsResidualConfig):
        super().__init__()
        self.config = config
        self.encoder = torch.nn.GRU(
            config.feature_dim, config.hidden_dim, batch_first=True
        )
        self.context_head = torch.nn.Sequential(
            torch.nn.LayerNorm(config.hidden_dim),
            torch.nn.Linear(config.hidden_dim, config.context_dim),
        )
        self.residual_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, 6),
        )
        # Starting at exact nominal physics makes an interrupted/short training
        # run conservative rather than producing arbitrary dynamics.
        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        normalized_history: torch.Tensor,
        nominal_a: torch.Tensor,
        nominal_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return physical ``(A, B, context)`` for a batch of causal windows."""
        _, hidden = self.encoder(normalized_history)
        context = self.context_head(hidden[-1])
        residual = torch.tanh(self.residual_head(context))
        delta_a = residual[:, :4].reshape(-1, 2, 2) * self.config.residual_a_limit
        delta_b = residual[:, 4:].reshape(-1, 2, 1) * self.config.residual_b_limit
        return nominal_a + delta_a, nominal_b + delta_b, context


def physical_prediction(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    state: torch.Tensor,
    steering: torch.Tensor,
) -> torch.Tensor:
    """Predict next physical state from A/B, current [v_y,r] and steering."""
    return (
        torch.bmm(matrix_a, state.unsqueeze(-1)).squeeze(-1)
        + torch.bmm(matrix_b, steering.reshape(-1, 1, 1)).squeeze(-1)
    )


def stable_enough(matrix_a: np.ndarray, maximum_radius: float = 1.10) -> bool:
    """Return whether a finite local discrete model stays inside health bounds."""
    matrix_a = np.asarray(matrix_a, dtype=float)
    if matrix_a.shape != (2, 2) or not np.all(np.isfinite(matrix_a)):
        return False
    return float(np.max(np.abs(np.linalg.eigvals(matrix_a)))) <= maximum_radius
