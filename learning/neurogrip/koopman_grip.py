"""Grip-aware contextual neural Koopman model for limit handling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.koopman_v2 import PhysicalKoopmanModel, PhysicalKoopmanOutput


@dataclass
class GripAwareKoopmanOutput(PhysicalKoopmanOutput):
    """Lifted dynamics plus causal front/rear grip estimates."""

    grip_estimate: torch.Tensor


class GripAwareKoopmanModel(PhysicalKoopmanModel):
    """Koopman predictor with an auxiliary per-axle grip-identification head."""

    def __init__(self, config: NeuroGripConfig):
        super().__init__(config)
        self.grip_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, 2),
        )
        torch.nn.init.zeros_(self.grip_head[-1].weight)
        # Initialise both axles at nominal grip scale 1.0 inside [0.35, 1.30].
        nominal_fraction = (1.0 - 0.35) / (1.30 - 0.35)
        initial_logit = torch.logit(torch.tensor(nominal_fraction)).item()
        torch.nn.init.constant_(self.grip_head[-1].bias, initial_logit)

    def grip_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """Map latent context to bounded front/rear grip scales."""
        return 0.35 + 0.95 * torch.sigmoid(self.grip_head(context))

    def forward(
        self,
        normalized_history: torch.Tensor,
        current_state: torch.Tensor,
        future_commands: torch.Tensor,
        nominal_a: torch.Tensor,
        nominal_b: torch.Tensor,
    ) -> GripAwareKoopmanOutput:
        """Predict lifted dynamics and grip from the same causal context."""
        base = super().forward(
            normalized_history,
            current_state,
            future_commands,
            nominal_a,
            nominal_b,
        )
        return GripAwareKoopmanOutput(
            **base.__dict__,
            grip_estimate=self.grip_from_context(base.context),
        )
