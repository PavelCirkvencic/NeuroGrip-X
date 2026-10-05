"""
ROS-free runtime for the contextual Koopman (N1) checkpoint.

The model definition lives in the repository ``learning/neurogrip`` package;
the runner adds that root to ``sys.path`` (documented workspace layout) and
keeps a strictly causal history buffer so the context encoder sees exactly the
same kind of window it was trained on.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

STATE_ORDER = ["v_x_t_mps", "yaw_rate_t_rps", "cmd_linear_x_t_mps", "cmd_angular_z_t_rps"]


class ContextualModelRunner:
    """Load an N1 checkpoint and predict the next modeled state."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_root: Path,
        device: str = "cpu",
        history_steps: int | None = None,
    ):
        root = str(Path(model_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from neurogrip.config import NeuroGripConfig  # noqa: E402
        from neurogrip.model import NeuroGripModel  # noqa: E402

        bundle = torch.load(
            Path(checkpoint_path).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        self.device = torch.device(device)
        self.config = NeuroGripConfig(**bundle["config"])
        if history_steps is not None:
            self.config = NeuroGripConfig(
                **{**bundle["config"], "history_steps": history_steps}
            )
        self.model = NeuroGripModel(
            self.config, context_mode=bundle["context_mode"]
        ).to(self.device)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.context_mode = bundle["context_mode"]
        statistics = bundle["statistics"]
        self.state_mean = np.asarray(statistics["state_mean"])
        self.state_scale = np.asarray(statistics["state_scale"])
        self.cmd_mean = np.asarray(statistics["cmd_mean"])
        self.cmd_scale = np.asarray(statistics["cmd_scale"])
        self.history = deque(maxlen=self.config.history_steps)
        self.last_prediction = None

    def ready(self) -> bool:
        """Return whether a full causal history window is available."""
        return len(self.history) >= self.config.history_steps

    def append(
        self,
        v_x_mps: float,
        yaw_rate_rps: float,
        linear_cmd: float,
        angular_cmd: float,
    ) -> None:
        """Append one causal feature sample to the history buffer."""
        self.history.append(
            np.array([v_x_mps, yaw_rate_rps, linear_cmd, angular_cmd], dtype=np.float64)
        )

    def predict(self) -> dict | None:
        """Return the next-state prediction and operator, or None if not ready."""
        if not self.ready():
            return None
        history = np.stack(self.history)
        state = history[-1, : self.config.state_dim]
        history_mean = np.concatenate([self.state_mean, self.cmd_mean])
        history_scale = np.concatenate([self.state_scale, self.cmd_scale])
        command = history[-1, self.config.state_dim:]
        with torch.no_grad():
            output = self.model(
                torch.tensor(
                    ((history - history_mean) / history_scale)[None, :, :],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.tensor(
                    ((state - self.state_mean) / self.state_scale)[None, :],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.tensor(
                    ((command - self.cmd_mean) / self.cmd_scale)[None, None, :],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        predicted = output.rollout[0, 0].detach().cpu().numpy()
        predicted_physical = predicted * self.state_scale + self.state_mean
        matrix_a = output.A[0].detach().cpu().numpy()
        result = {
            "predicted_state": predicted_physical,
            "context": output.context[0].detach().cpu().numpy(),
            "a_matrix": matrix_a,
            "b_matrix": output.B[0].detach().cpu().numpy(),
            "bias": output.bias[0].detach().cpu().numpy(),
            "spectral_radius": float(np.abs(np.linalg.eigvals(matrix_a)).max()),
        }
        self.last_prediction = result
        return result
