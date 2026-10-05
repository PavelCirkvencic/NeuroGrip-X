"""Configuration objects for the NeuroGrip-X contextual Koopman model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NeuroGripConfig:
    """Model and data dimensions shared by training and evaluation."""

    state_dim: int = 2
    input_dim: int = 2
    history_steps: int = 25
    context_dim: int = 6
    latent_extra: int = 9
    rank: int = 4
    hidden_dim: int = 64
    rollout_steps: int = 20
    dropout: float = 0.0
    history_feature_dim: int | None = None

    @property
    def latent_dim(self) -> int:
        """Full lifted state dimension: physical state plus learned features."""
        return self.state_dim + self.latent_extra

    @property
    def feature_dim(self) -> int:
        """Per-sample context feature: state plus command."""
        if self.history_feature_dim is not None:
            return self.history_feature_dim
        return self.state_dim + self.input_dim

    def as_dict(self) -> dict:
        """Serializable representation for manifests."""
        return asdict(self)
