#!/usr/bin/env python3
"""Train a fixed neural state-transition baseline for NeuroGrip-X.

The model uses exactly the Ridge feature and target schema and is trained only
on scenario-grouped catalog splits.  It reports one-step and 5/20-step
open-loop rollout error and compares directly against Ridge and a persistence
baseline on the frozen test scenario(s).

This is deliberately a small fixed MLP (no context encoder, no adaptation).
If it does not beat or complement Ridge, the plant/data/targets must be
diagnosed before adding complexity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_linear_baseline import (
    CATALOG_SPLITS,
    EVALUATION_SPLITS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_scenario_splits,
    metrics_by_target,
)

STATE_COLUMNS = ["v_x_t_mps", "yaw_rate_t_rps"]
COMMAND_COLUMNS = ["cmd_linear_x_t_mps", "cmd_angular_z_t_rps"]


def parse_arguments() -> argparse.Namespace:
    """Parse paths, architecture and reproducibility options."""
    parser = argparse.ArgumentParser(
        description="Train a fixed MLP vehicle-dynamics baseline."
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Scenario-grouped dataset_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/models/scenario_mlp_latest"),
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=[1, 5, 20])
    return parser.parse_args()


def git_commit() -> str:
    """Return the current Git commit, or 'unknown' outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def resolve_device() -> torch.device:
    """Prefer CUDA when present, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch for reproducible runs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(input_dim: int, hidden_sizes: list[int], output_dim: int, seed: int):
    """Build a small SiLU MLP with deterministic initialization."""
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.append(torch.nn.Linear(previous, width))
        layers.append(torch.nn.SiLU())
        previous = width
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


def train_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    arguments: argparse.Namespace,
    device: torch.device,
):
    """Fit the MLP with Adam and mean-squared error."""
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    dataset_size = x_tensor.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(arguments.seed)

    x_val_tensor = torch.tensor(x_validation, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_validation, dtype=torch.float32, device=device)
    history = []
    for epoch in range(arguments.epochs):
        model.train()
        permutation = torch.randperm(dataset_size, generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, dataset_size, arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            prediction = model(x_tensor[indices])
            loss = loss_fn(prediction, y_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": epoch_loss / max(batches, 1),
                "validation_mse": validation_loss,
            }
        )
    return history


def make_mlp_batch_step(model, scaler_x: StandardScaler, scaler_y: StandardScaler, device):
    """Return a batch function mapping (state, command) to predicted next state."""
    x_mean = scaler_x.mean_.astype(np.float64)
    x_scale = scaler_x.scale_.astype(np.float64)
    y_mean = scaler_y.mean_.astype(np.float64)
    y_scale = scaler_y.scale_.astype(np.float64)

    def step_batch(states: np.ndarray, commands: np.ndarray) -> np.ndarray:
        features = np.concatenate([states, commands], axis=1)
        normalized = (features - x_mean) / x_scale
        with torch.no_grad():
            predicted = model(
                torch.tensor(normalized, dtype=torch.float32, device=device)
            ).cpu().numpy()
        return predicted * y_scale + y_mean

    return step_batch


def make_ridge_batch_step(model: Pipeline):
    """Return an equivalent batch step function for the Ridge pipeline."""

    def step_batch(states: np.ndarray, commands: np.ndarray) -> np.ndarray:
        features = np.concatenate([states, commands], axis=1)
        return model.predict(features)

    return step_batch


def make_persistence_batch_step():
    """Return a baseline that predicts the current state is unchanged."""

    def step_batch(states: np.ndarray, commands: np.ndarray) -> np.ndarray:
        return states

    return step_batch


def contiguous_links(dataframe: pd.DataFrame) -> np.ndarray:
    """Mark rows whose next row is temporally adjacent in this filtered set.

    The stored ``dt_s`` describes the original sampling interval.  After
    quality filtering the *actual* gap between two surviving rows can be
    larger, so contiguity is computed from timestamp differences instead.
    """
    time_s = dataframe["time_s"].to_numpy(dtype=float)
    next_gap = np.full(len(time_s), np.inf)
    next_gap[:-1] = time_s[1:] - time_s[:-1]
    median_dt = float(np.median(time_s[1:] - time_s[:-1])) if len(time_s) > 1 else 0.02
    return np.abs(next_gap - median_dt) <= 0.005


def load_run_frames(manifest_path: Path) -> dict[str, list[pd.DataFrame]]:
    """Return per-split, per-run time-sorted frames from the catalog.

    Rollouts must stay inside a single run: concatenating scenarios and sorting
    by simulation time would interleave two runs that share time stamps and
    create false temporal links.
    """
    manifest = json.loads(Path(manifest_path).expanduser().resolve().read_text())
    grouped: dict[str, list[pd.DataFrame]] = {split: [] for split in CATALOG_SPLITS}
    for run in manifest["runs"]:
        frame = pd.read_parquet(run["parquet_path"]).sort_values("time_s")
        grouped[run["split"]].append(frame.reset_index(drop=True))
    return grouped


def _rollout_one_frame(frame: pd.DataFrame, step_batch, horizons: list[int]) -> dict:
    """Return per-horizon squared errors for one temporally contiguous run."""
    frame = frame.sort_values("time_s").reset_index(drop=True)
    sample_count = len(frame)
    states = frame[STATE_COLUMNS].to_numpy(dtype=np.float64)
    commands = frame[COMMAND_COLUMNS].to_numpy(dtype=np.float64)
    next_actual = frame[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    links = contiguous_links(frame)

    max_horizon = max(horizons)
    active = np.flatnonzero(links)
    predicted_state = states[active].copy()
    squared_error = {horizon: [] for horizon in horizons}
    for step in range(1, max_horizon + 1):
        predicted_next = step_batch(predicted_state, commands[active])
        if step in squared_error:
            squared_error[step].append(predicted_next - next_actual[active])

        advanced = active + 1
        in_bounds = advanced < sample_count
        advanced = advanced[in_bounds]
        predicted_next = predicted_next[in_bounds]
        if advanced.size:
            keep = links[advanced]
            active = advanced[keep]
            predicted_state = predicted_next[keep]
        else:
            active = advanced
            predicted_state = predicted_next
        if active.size == 0:
            break
    return squared_error


def rollout_rmse(
    run_frames_by_split: dict[str, list[pd.DataFrame]],
    step_batch,
    horizons: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute open-loop multi-step RMSE per split and horizon.

    At each step the model consumes its own predicted physical state and the
    *logged* command, then compares against the recorded next state.  Only
    temporally contiguous chains inside one run are used, so scenarios are
    never interleaved.
    """
    results: dict[str, dict[int, dict[str, float]]] = {}
    for split, frames in run_frames_by_split.items():
        accumulated = {horizon: [] for horizon in horizons}
        for frame in frames:
            per_frame = _rollout_one_frame(frame, step_batch, horizons)
            for horizon in horizons:
                accumulated[horizon].extend(per_frame[horizon])
        split_result = {}
        for horizon in horizons:
            if not accumulated[horizon]:
                continue
            stacked = np.concatenate(accumulated[horizon], axis=0)
            split_result[horizon] = {
                target: float(np.sqrt(np.mean(stacked[:, index] ** 2)))
                for index, target in enumerate(TARGET_COLUMNS)
            }
        results[split] = split_result
    return results


def one_step_predictions(step_batch, frame: pd.DataFrame) -> np.ndarray:
    """Return one-step predictions in physical units for a split frame."""
    states = frame[STATE_COLUMNS].to_numpy(dtype=np.float64)
    commands = frame[COMMAND_COLUMNS].to_numpy(dtype=np.float64)
    return step_batch(states, commands)


def save_rollout_plot(rollouts: dict, horizons: list[int], output_path: Path) -> None:
    """Save yaw-rate rollout RMSE versus horizon for every compared model."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for name in ("persistence", "ridge", "mlp"):
        for split in ("validation", "test"):
            split_horizons = rollouts[name].get(split, {})
            available = sorted(h for h in split_horizons if h in horizons)
            if not available:
                continue
            values = [split_horizons[h]["yaw_rate_t1_rps"] for h in available]
            style = "-" if split == "test" else "--"
            axis.plot(available, values, style, marker="o", label=f"{name} ({split})")
    axis.set_xlabel("rollout horizon [steps]")
    axis.set_ylabel("yaw-rate RMSE [rad/s]")
    axis.set_title("Fixed baseline open-loop rollout error")
    axis.set_xscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Fit the MLP baseline, evaluate rollouts and store a training manifest."""
    arguments = parse_arguments()
    seed_everything(arguments.seed)
    device = resolve_device()

    split_frames, split_provenance = load_scenario_splits(arguments.dataset_manifest)
    catalog_path = arguments.dataset_manifest.expanduser().resolve()
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()

    train_frame = split_frames["train"]
    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_frame[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    x_validation = split_frames["validation"][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_validation = split_frames["validation"][TARGET_COLUMNS].to_numpy(dtype=np.float64)

    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)
    normalized_x_train = scaler_x.transform(x_train)
    normalized_y_train = scaler_y.transform(y_train)
    normalized_x_validation = scaler_x.transform(x_validation)
    normalized_y_validation = scaler_y.transform(y_validation)

    model = build_model(
        len(FEATURE_COLUMNS), arguments.hidden_sizes, len(TARGET_COLUMNS), arguments.seed
    )
    history = train_model(
        model,
        normalized_x_train,
        normalized_y_train,
        normalized_x_validation,
        normalized_y_validation,
        arguments,
        device,
    )

    # Ridge trained on the exact same split for an apples-to-apples comparison.
    ridge = Pipeline([("standardize", StandardScaler()), ("ridge", Ridge(alpha=1e-3))])
    ridge.fit(x_train, y_train)

    steps = {
        "mlp": make_mlp_batch_step(model, scaler_x, scaler_y, device),
        "ridge": make_ridge_batch_step(ridge),
        "persistence": make_persistence_batch_step(),
    }

    evaluation_sets = {
        split: frame
        for split, frame in split_frames.items()
        if split in EVALUATION_SPLITS and not frame.empty
    }
    run_frames = {
        split: frames
        for split, frames in load_run_frames(arguments.dataset_manifest).items()
        if split in EVALUATION_SPLITS
    }
    one_step = {}
    rollouts = {}
    for name, step_batch in steps.items():
        one_step[name] = {
            split: {
                "rows": int(len(frame)),
                "metrics": metrics_by_target(
                    frame[TARGET_COLUMNS].to_numpy(dtype=np.float64),
                    one_step_predictions(step_batch, frame),
                ),
            }
            for split, frame in evaluation_sets.items()
        }
        rollouts[name] = rollout_rmse(run_frames, step_batch, arguments.rollout_horizons)

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mlp_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_sizes": arguments.hidden_sizes,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "feature_mean": scaler_x.mean_.tolist(),
            "feature_scale": scaler_x.scale_.tolist(),
            "target_mean": scaler_y.mean_.tolist(),
            "target_scale": scaler_y.scale_.tolist(),
        },
        checkpoint_path,
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "fixed_mlp_state_transition",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "split_schema_version": split_provenance.get("split_schema_version"),
        "git_commit": git_commit(),
        "device": str(device),
        "seed": arguments.seed,
        "hyperparameters": {
            "hidden_sizes": arguments.hidden_sizes,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "training_rows": int(len(x_train)),
        "normalization": {
            "feature_mean": scaler_x.mean_.tolist(),
            "feature_scale": scaler_x.scale_.tolist(),
            "target_mean": scaler_y.mean_.tolist(),
            "target_scale": scaler_y.scale_.tolist(),
        },
        "scenario_split": split_provenance,
        "training_history": history,
        "one_step": one_step,
        "rollout_rmse": rollouts,
        "rollout_horizons": arguments.rollout_horizons,
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    save_rollout_plot(
        rollouts, arguments.rollout_horizons, output_dir / "rollout_rmse.png"
    )

    print(f"Device: {device}, training rows: {len(x_train)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")
    for name in ("persistence", "ridge", "mlp"):
        print(f"[{name}] one-step yaw-rate RMSE:")
        for split, result in one_step[name].items():
            metrics = result["metrics"]["yaw_rate_t1_rps"]
            print(f"    {split}: RMSE={metrics['rmse']:.6f} R²={metrics['r2']:.4f}")
    print("20-step rollout yaw-rate RMSE:")
    for name in ("persistence", "ridge", "mlp"):
        for split, horizons in rollouts[name].items():
            if 20 in horizons:
                print(f"    [{name}] {split}: {horizons[20]['yaw_rate_t1_rps']:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"MLP training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
