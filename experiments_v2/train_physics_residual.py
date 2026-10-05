#!/usr/bin/env python3
"""Train the deployable 3-seed physical NeuroGrip-X ensemble from v2 episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import cont2discrete

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/neurogrip_ai"))

from neurogrip_ai.physics_residual import (  # noqa: E402
    FEATURE_NAMES,
    PhysicsResidualConfig,
    PhysicsResidualNet,
    physical_prediction,
)

HORIZONS = (1, 5, 10, 20)
SEEDS = (42, 43, 44)
FIELD_MAP = {
    "v_x_mps": "v_x_mps",
    "v_y_mps": "v_y_mps",
    "yaw_rate_rps": "yaw_rate_rps",
    "steering_applied_rad": "steering_applied_rad",
    "acceleration_command_mps2": "acceleration_command_mps2",
    "a_x_mps2": "a_x_mps2",
    "a_y_mps2": "a_y_mps2",
}


@dataclass
class Windows:
    """Dense causal windows from a disjoint scenario split."""

    history: np.ndarray
    state: np.ndarray
    commands: np.ndarray
    targets: np.ndarray
    nominal_a: np.ndarray
    nominal_b: np.ndarray
    scenario_ids: list[str]


def sha256_path(path: Path) -> str:
    """Hash an immutable artifact or data file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Parse explicit data splits and conservative training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def nominal_matrices(vx_mps: float, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """Create the exact ZOH physical baseline for one longitudinal speed."""
    vx = max(float(vx_mps), 1.0)
    mass = float(fit["mass_kg"])
    inertia = float(fit.get("yaw_inertia_kgm2", 172.44))
    wheelbase = float(fit["wheelbase_m"])
    front_fraction = float(fit["front_fraction"])
    front, rear = float(fit["front_cornering_stiffness_n_rad"]), float(fit["rear_cornering_stiffness_n_rad"])
    lf, lr = front_fraction * wheelbase, (1.0 - front_fraction) * wheelbase
    a = np.array([
        [-(front + rear) / (mass * vx), -vx - (lf * front - lr * rear) / (mass * vx)],
        [-(lf * front - lr * rear) / (inertia * vx), -(lf**2 * front + lr**2 * rear) / (inertia * vx)],
    ])
    b = np.array([[front / mass], [lf * front / inertia]])
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), 0.02)
    return np.asarray(ad), np.asarray(bd)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read only complete finite v2 rows; malformed data must not enter training."""
    values = {name: [] for name in ["time_s", *FIELD_MAP.values()]}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                for name in values:
                    values[name].append(float(row[name]))
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"invalid training telemetry in {path}") from error
    result = {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"non-finite training telemetry in {path}")
    dt = np.diff(result["time_s"])
    if len(dt) < 100 or np.any(dt <= 0.0) or not 49.0 <= 1.0 / np.median(dt) <= 51.0:
        raise ValueError(f"{path} does not provide continuous 50 Hz telemetry")
    return result


def make_windows(episode: dict[str, np.ndarray], scenario_id: str, fit: dict) -> Windows:
    """Turn one complete episode into 25-sample causal, 20-step target windows."""
    features = np.column_stack([episode[FIELD_MAP[name]] for name in FEATURE_NAMES])
    states = features[:, 1:3]
    steering = features[:, 3]
    histories, current, commands, targets, matrices_a, matrices_b = [], [], [], [], [], []
    last_index = len(features) - max(HORIZONS)
    for index in range(24, last_index):
        histories.append(features[index - 24 : index + 1])
        current.append(states[index])
        commands.append(steering[index : index + max(HORIZONS)])
        targets.append(np.stack([states[index + horizon] for horizon in HORIZONS]))
        matrix_a, matrix_b = nominal_matrices(float(features[index, 0]), fit)
        matrices_a.append(matrix_a)
        matrices_b.append(matrix_b)
    if not histories:
        raise ValueError(f"{scenario_id} is too short for 25+20 sample windows")
    count = len(histories)
    return Windows(
        np.asarray(histories, dtype=np.float32), np.asarray(current, dtype=np.float32),
        np.asarray(commands, dtype=np.float32), np.asarray(targets, dtype=np.float32),
        np.asarray(matrices_a, dtype=np.float32), np.asarray(matrices_b, dtype=np.float32),
        [scenario_id] * count,
    )


def concatenate(items: list[Windows]) -> Windows:
    """Concatenate separate scenario windows while retaining scenario identities."""
    if not items:
        raise ValueError("split contains no valid episodes")
    return Windows(
        *(np.concatenate([getattr(item, name) for item in items], axis=0) for name in (
            "history", "state", "commands", "targets", "nominal_a", "nominal_b"
        )),
        sum((item.scenario_ids for item in items), []),
    )


def load_splits(run_root: Path, manifest_path: Path, fit: dict) -> dict[str, Windows]:
    """Load disjoint split assignments by full scenario ID, never by window."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", manifest)
    required = {"train", "validation", "calibration"}
    if not required <= splits.keys():
        raise ValueError("split manifest requires train, validation and calibration")
    memberships = [scenario for names in splits.values() for scenario in names]
    if len(memberships) != len(set(memberships)):
        raise ValueError("a scenario ID appears in more than one split")
    discovered: dict[str, tuple[Path, dict]] = {}
    for csv_path in run_root.glob("*/episode.csv"):
        metadata_path = csv_path.with_suffix(".metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            discovered[metadata.get("scenario_id")] = (csv_path, metadata)
    result = {}
    for split, scenario_ids in splits.items():
        items = []
        for scenario_id in scenario_ids:
            if scenario_id not in discovered:
                raise ValueError(f"split {split}: missing run for {scenario_id}")
            path, metadata = discovered[scenario_id]
            if metadata.get("runner", {}).get("termination_reason") not in {
                "lap_complete", "excitation_complete"
            }:
                raise ValueError(f"split {split}: incomplete run {scenario_id}")
            items.append(make_windows(read_csv(path), scenario_id, fit))
        result[split] = concatenate(items)
    return result


def standardization(train: Windows) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature statistics from train windows only."""
    mean = train.history.reshape(-1, train.history.shape[-1]).mean(axis=0)
    scale = train.history.reshape(-1, train.history.shape[-1]).std(axis=0) + 1e-6
    return mean.astype(np.float32), scale.astype(np.float32)


def tensors(batch: Windows, mean: np.ndarray, scale: np.ndarray, device: torch.device):
    """Convert one split to device tensors with train-only standardisation."""
    return tuple(
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (
            (batch.history - mean) / scale, batch.state, batch.commands, batch.targets,
            batch.nominal_a, batch.nominal_b,
        )
    )


def rollout_loss(model, values, indices=None):
    """Compute physical one/multi-step loss plus a modest spectral health penalty."""
    history, state, commands, targets, nominal_a, nominal_b = values
    if indices is not None:
        history, state, commands, targets, nominal_a, nominal_b = (
            value[indices] for value in values
        )
    matrix_a, matrix_b, context = model(history, nominal_a, nominal_b)
    predicted = state
    losses = []
    command_step = 0
    for index, horizon in enumerate(HORIZONS):
        previous_horizon = 0 if index == 0 else HORIZONS[index - 1]
        for _ in range(horizon - previous_horizon):
            predicted = physical_prediction(
                matrix_a, matrix_b, predicted, commands[:, command_step]
            )
            command_step += 1
        losses.append(torch.mean((predicted - targets[:, index]) ** 2))
    radius = torch.linalg.eigvals(matrix_a).abs().amax(dim=1)
    penalty = torch.relu(radius - 1.10).pow(2).mean()
    return sum(losses) / len(losses) + 0.1 * penalty, context, matrix_a


def evaluate(model, values) -> tuple[float, np.ndarray, np.ndarray]:
    """Return validation loss plus all contexts/matrices for calibration reporting."""
    model.eval()
    with torch.no_grad():
        loss, context, matrices = rollout_loss(model, values)
    return float(loss.cpu()), context.cpu().numpy(), matrices.cpu().numpy()


def train_member(seed: int, train_values, validation_values, arguments, config, device):
    """Train one independently seeded member; early stopping sees validation only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PhysicsResidualNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-5)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    count = train_values[0].shape[0]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        for start in range(0, count, arguments.batch_size):
            indices = torch.randperm(count, generator=generator, device=device)[start:start + arguments.batch_size]
            loss, _, _ = rollout_loss(model, train_values, indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss, _, _ = evaluate(model, validation_values)
        if validation_loss < best:
            best, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "validation_loss": best}


def main() -> int:
    """Train, calibrate and write a hash-linked physical ensemble manifest."""
    arguments = parse_arguments()
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fit_path = arguments.nominal_fit.expanduser().resolve()
    fit = json.loads(fit_path.read_text())
    splits = load_splits(arguments.run_root.expanduser().resolve(), arguments.split_manifest, fit)
    mean, scale = standardization(splits["train"])
    device = torch.device(arguments.device)
    config = PhysicsResidualConfig()
    train_values = tensors(splits["train"], mean, scale, device)
    validation_values = tensors(splits["validation"], mean, scale, device)
    calibration_values = tensors(splits["calibration"], mean, scale, device)
    members, reports = [], []
    train_context_members, calibration_context_members, calibration_predictions = [], [], []
    for seed in SEEDS:
        model, report = train_member(seed, train_values, validation_values, arguments, config, device)
        with torch.no_grad():
            _, _, train_context = model(
                train_values[0], train_values[4], train_values[5]
            )
            matrix_a, matrix_b, calibration_context = model(
                calibration_values[0], calibration_values[4], calibration_values[5]
            )
            prediction = physical_prediction(
                matrix_a, matrix_b, calibration_values[1], calibration_values[2][:, 0]
            )
        train_context_members.append(train_context.cpu().numpy())
        calibration_context_members.append(calibration_context.cpu().numpy())
        calibration_predictions.append(prediction.cpu().numpy())
        checkpoint = output / f"member_seed{seed}.pt"
        torch.save(
            {"config": config.as_dict(), "state_dict": model.cpu().state_dict(), "feature_mean": mean, "feature_scale": scale,
             "nominal_fit_sha256": sha256_path(fit_path), "seed": seed}, checkpoint
        )
        members.append({"seed": seed, "path": checkpoint.name, "sha256": sha256_path(checkpoint)})
        reports.append(report)
    train_contexts = np.mean(np.stack(train_context_members, axis=0), axis=0)
    calibration_contexts = np.mean(np.stack(calibration_context_members, axis=0), axis=0)
    ensemble_prediction = np.mean(np.stack(calibration_predictions, axis=0), axis=0)
    errors = np.linalg.norm(
        ensemble_prediction - calibration_values[3][:, 0].cpu().numpy(), axis=1
    )
    train_context_mean = train_contexts.mean(axis=0)
    covariance = np.cov(train_contexts, rowvar=False)
    covariance = 0.9 * covariance + 0.1 * np.eye(covariance.shape[0]) * np.trace(covariance) / covariance.shape[0]
    calibration = {
        "conformal_radius": float(np.quantile(errors, 0.90)),
        "context_mean": train_context_mean.tolist(),
        "context_covariance_inverse": np.linalg.pinv(covariance).tolist(),
        "ood_threshold": float(np.quantile(np.sum((calibration_contexts - train_context_mean) @ np.linalg.pinv(covariance) * (calibration_contexts - train_context_mean), axis=1), 0.99)),
    }
    dataset_hash = hashlib.sha256(arguments.split_manifest.expanduser().read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "architecture": config.as_dict(), "feature_names": FEATURE_NAMES,
                "state_names": ["v_y_mps", "yaw_rate_rps"], "sample_time_s": 0.02,
                "nominal_fit_sha256": sha256_path(fit_path), "dataset_split_manifest_sha256": dataset_hash,
                "members": members, "calibration": calibration, "reports": reports}
    manifest_path = output / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PHYSICS_ENSEMBLE_TRAINED manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
