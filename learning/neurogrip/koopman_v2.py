"""Physics-informed contextual Koopman model for NeuroGrip-X v2.

The physical state remains in the first lifted coordinates while learned
observables make the predictor expressive.  A causal context encoder conditions
a bounded low-rank correction around the discrete bicycle-model prior:

    z = [x, phi(x, c)]
    z_next = K(c) z + B(c) delta + b(c)
    x_next = z_next[:2]

This is a genuine lifted linear evolution model, unlike the earlier direct
physical A/B residual network retained as an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.model import CausalContextEncoder


@dataclass
class PhysicalKoopmanOutput:
    """All tensors needed for rollout loss, calibration and deployment."""

    context: torch.Tensor
    lifted_state: torch.Tensor
    koopman: torch.Tensor
    input_matrix: torch.Tensor
    bias: torch.Tensor
    latent_rollout: torch.Tensor
    state_rollout: torch.Tensor


class PhysicalKoopmanModel(torch.nn.Module):
    """Causal neural lift plus context-conditioned low-rank Koopman operator."""

    def __init__(
        self,
        config: NeuroGripConfig,
        operator_residual_limit: float = 0.15,
        input_residual_limit: float = 0.35,
        bias_limit: float = 0.05,
    ):
        super().__init__()
        if config.state_dim != 2 or config.input_dim != 1:
            raise ValueError("physical Koopman v2 requires state_dim=2 and input_dim=1")
        self.config = config
        self.operator_residual_limit = float(operator_residual_limit)
        self.input_residual_limit = float(input_residual_limit)
        self.bias_limit = float(bias_limit)
        self.context_encoder = CausalContextEncoder(
            config.feature_dim, config.hidden_dim, config.context_dim
        )
        self.lift = torch.nn.Sequential(
            torch.nn.Linear(config.state_dim + config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.latent_extra),
            torch.nn.Tanh(),
        )
        self.operator_left = torch.nn.Parameter(
            torch.randn(config.latent_dim, config.rank) * 0.01
        )
        self.operator_right = torch.nn.Parameter(
            torch.randn(config.latent_dim, config.rank) * 0.01
        )
        self.operator_coefficients = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.rank),
        )
        self.input_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.bias_head = torch.nn.Sequential(
            torch.nn.Linear(config.context_dim, config.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_dim, config.latent_dim),
        )
        # An untrained checkpoint is exactly the stable nominal physical prior.
        for head in (
            self.operator_coefficients[-1],
            self.input_head[-1],
            self.bias_head[-1],
        ):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def encode_context(self, normalized_history: torch.Tensor) -> torch.Tensor:
        """Encode a past-only normalized telemetry window."""
        return self.context_encoder(normalized_history)

    def lift_state(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Copy physical coordinates and append bounded learned observables."""
        observables = self.lift(torch.cat((state, context), dim=-1))
        return torch.cat((state, observables), dim=-1)

    def local_operator(
        self,
        context: torch.Tensor,
        nominal_a: torch.Tensor,
        nominal_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bounded K/B/b around the per-sample physical prior."""
        batch = context.shape[0]
        latent_dim = self.config.latent_dim
        base = torch.zeros(
            (batch, latent_dim, latent_dim),
            dtype=context.dtype,
            device=context.device,
        )
        base[:, :2, :2] = nominal_a
        if latent_dim > 2:
            identity = torch.eye(
                latent_dim - 2, dtype=context.dtype, device=context.device
            )
            base[:, 2:, 2:] = 0.85 * identity.unsqueeze(0)
        coefficients = torch.tanh(self.operator_coefficients(context))
        correction = torch.einsum(
            "ir,br,jr->bij",
            self.operator_left,
            coefficients,
            self.operator_right,
        )
        correction = self.operator_residual_limit * torch.tanh(correction)
        koopman = base + correction

        input_base = torch.zeros(
            (batch, latent_dim, 1), dtype=context.dtype, device=context.device
        )
        input_base[:, :2, :] = nominal_b
        input_delta = self.input_residual_limit * torch.tanh(
            self.input_head(context)
        ).unsqueeze(-1)
        bias = self.bias_limit * torch.tanh(self.bias_head(context))
        return koopman, input_base + input_delta, bias

    def forward(
        self,
        normalized_history: torch.Tensor,
        current_state: torch.Tensor,
        future_commands: torch.Tensor,
        nominal_a: torch.Tensor,
        nominal_b: torch.Tensor,
    ) -> PhysicalKoopmanOutput:
        """Roll the fixed local lifted operator over an open-loop command sequence."""
        context = self.encode_context(normalized_history)
        lifted = self.lift_state(current_state, context)
        koopman, input_matrix, bias = self.local_operator(
            context, nominal_a, nominal_b
        )
        latent_states = []
        latent = lifted
        for step in range(future_commands.shape[1]):
            command = future_commands[:, step].reshape(-1, 1, 1)
            latent = (
                torch.bmm(koopman, latent.unsqueeze(-1)).squeeze(-1)
                + torch.bmm(input_matrix, command).squeeze(-1)
                + bias
            )
            latent_states.append(latent)
        latent_rollout = torch.stack(latent_states, dim=1)
        return PhysicalKoopmanOutput(
            context=context,
            lifted_state=lifted,
            koopman=koopman,
            input_matrix=input_matrix,
            bias=bias,
            latent_rollout=latent_rollout,
            state_rollout=latent_rollout[:, :, :2],
        )

    def physical_jacobians(
        self,
        normalized_history: torch.Tensor,
        state: torch.Tensor,
        steering: torch.Tensor,
        nominal_a: torch.Tensor,
        nominal_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiate the one-step lifted predictor in physical coordinates."""
        if state.shape[0] != 1:
            raise ValueError("runtime physical_jacobians expects one sample")
        state_for_grad = state.detach().clone().requires_grad_(True)
        output = self(
            normalized_history,
            state_for_grad,
            steering.reshape(1, 1, 1),
            nominal_a,
            nominal_b,
        )
        prediction = output.state_rollout[:, 0]
        rows = []
        for index in range(2):
            gradient = torch.autograd.grad(
                prediction[0, index],
                state_for_grad,
                retain_graph=index == 0,
                create_graph=False,
            )[0]
            rows.append(gradient[0])
        matrix_a = torch.stack(rows, dim=0)
        matrix_b = output.input_matrix[0, :2, :]
        return matrix_a, matrix_b, output.context[0], prediction[0]
