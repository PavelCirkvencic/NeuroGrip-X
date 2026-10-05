"""
ROS-free inference wrapper for the fixed MLP dynamics baseline.

The ROS node and any offline check share this class, so the same normalization
and matrix conventions are exercised in tests without starting ROS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# Modeled state is [v_x, yaw_rate]; lateral velocity is excluded because the
# low-speed kinematic plant makes it degenerate (see docs/limitations.md).
FEATURE_COLUMNS = [
    "v_x_t_mps",
    "yaw_rate_t_rps",
    "cmd_linear_x_t_mps",
    "cmd_angular_z_t_rps",
]
TARGET_COLUMNS = ["v_x_t1_mps", "yaw_rate_t1_rps"]


def build_model(hidden_sizes: list[int], input_dim: int, output_dim: int):
    """Rebuild the SiLU MLP used during training."""
    layers: list[torch.nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.append(torch.nn.Linear(previous, width))
        layers.append(torch.nn.SiLU())
        previous = width
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


class TransitionModelRunner:
    """Load a checkpoint bundle and predict the next body-frame state."""

    def __init__(self, bundle: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        self.target_columns = bundle.get("target_columns", TARGET_COLUMNS)
        self.feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
        self.feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
        self.target_mean = np.asarray(bundle["target_mean"], dtype=np.float64)
        self.target_scale = np.asarray(bundle["target_scale"], dtype=np.float64)
        self.model = build_model(
            list(bundle["hidden_sizes"]),
            len(self.feature_columns),
            len(self.target_columns),
        )
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu"):
        """Load a checkpoint written by ``train_mlp_baseline.py``."""
        bundle = torch.load(
            Path(checkpoint_path).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        return cls(bundle, device=device)

    def predict(self, feature: np.ndarray) -> np.ndarray:
        """Return the predicted next state for one 5-element feature vector."""
        normalized = (np.asarray(feature, dtype=np.float64) - self.feature_mean) / (
            self.feature_scale
        )
        with torch.no_grad():
            prediction = self.model(
                torch.tensor(normalized[None, :], dtype=torch.float32, device=self.device)
            ).cpu().numpy()[0]
        return prediction * self.target_scale + self.target_mean

    def predict_state(
        self,
        v_x_mps: float,
        yaw_rate_rps: float,
        linear_x_command_mps: float,
        angular_z_command_rps: float,
    ) -> tuple[float, float]:
        """Return the predicted next (v_x, yaw_rate) in physical units."""
        feature = np.array(
            [
                v_x_mps,
                yaw_rate_rps,
                linear_x_command_mps,
                angular_z_command_rps,
            ],
            dtype=np.float64,
        )
        prediction = self.predict(feature)
        return float(prediction[0]), float(prediction[1])
