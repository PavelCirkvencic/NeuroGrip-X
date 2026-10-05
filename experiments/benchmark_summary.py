#!/usr/bin/env python3
"""Consolidated final benchmark with provenance validation.

Loads the trained Ridge, MLP and contextual Koopman artifacts, verifies that
they were trained on exactly this catalog (SHA-256, split schema and
feature/target schema), then evaluates development-test and the sealed OOD test
for one-step and open-loop rollout metrics.

Any provenance mismatch aborts with a non-zero status; it never silently mixes
metrics from different catalogs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_linear_baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    git_commit,
    metrics_by_target,
)
from train_mlp_baseline import (
    _rollout_one_frame,
    make_persistence_batch_step,
    make_ridge_batch_step,
)

LEARNING_ROOT = Path(__file__).resolve().parents[1] / "learning"
if str(LEARNING_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNING_ROOT))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.data import build_split_windows, load_catalog, run_frames_by_split  # noqa: E402
from neurogrip.model import NeuroGripModel  # noqa: E402

REPORT_SPLITS = ("development_test", "sealed_test")
HORIZONS = [1, 5, 20]


def parse_arguments() -> argparse.Namespace:
    """Parse artifact locations and output directory."""
    parser = argparse.ArgumentParser(description="Final NeuroGrip-X benchmark.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--ridge-metrics", type=Path, required=True)
    parser.add_argument("--mlp-manifest", type=Path, required=True)
    parser.add_argument("--neurogrip-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON artifact, failing clearly if it is missing."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_provenance(catalog_path, catalog, ridge_metrics, mlp, neurogrip) -> list[str]:
    """Return a list of provenance problems (empty means consistent)."""
    problems = []
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    split_schema = catalog.get("split_schema_version")

    ridge_split = ridge_metrics.get("scenario_split") or {}
    if ridge_split.get("dataset_manifest_sha256") != catalog_sha:
        problems.append("Ridge metrics were trained on a different catalog SHA.")
    if ridge_metrics.get("feature_columns") != FEATURE_COLUMNS:
        problems.append("Ridge feature schema mismatch.")
    if ridge_metrics.get("target_columns") != TARGET_COLUMNS:
        problems.append("Ridge target schema mismatch.")

    if mlp.get("catalog_sha256") != catalog_sha:
        problems.append("MLP manifest catalog SHA mismatch.")
    if mlp.get("split_schema_version") != split_schema:
        problems.append("MLP split schema mismatch.")
    if mlp.get("feature_columns") != FEATURE_COLUMNS:
        problems.append("MLP feature schema mismatch.")
    if mlp.get("target_columns") != TARGET_COLUMNS:
        problems.append("MLP target schema mismatch.")

    if neurogrip.get("catalog_sha256") != catalog_sha:
        problems.append("NeuroGrip manifest catalog SHA mismatch.")
    if neurogrip.get("split_schema_version") != split_schema:
        problems.append("NeuroGrip split schema mismatch.")
    if int(neurogrip.get("config", {}).get("state_dim", -1)) != len(
        ["v_x_t_mps", "yaw_rate_t_rps"]
    ):
        problems.append("NeuroGrip state schema mismatch.")
    if neurogrip.get("sealed_test_used"):
        problems.append("NeuroGrip training used the sealed test unexpectedly.")
    return problems


def one_step_metrics(step_batch, frame: pd.DataFrame) -> dict:
    """Return one-step metrics in physical units for a split frame."""
    states = frame[["v_x_t_mps", "yaw_rate_t_rps"]].to_numpy(dtype=float)
    commands = frame[["cmd_linear_x_t_mps", "cmd_angular_z_t_rps"]].to_numpy(dtype=float)
    prediction = step_batch(states, commands)
    return metrics_by_target(
        frame[TARGET_COLUMNS].to_numpy(dtype=float), prediction
    )


def aggregate_scenario_rmse(scenario_errors: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """Pool every run of a scenario, then return one yaw RMSE per scenario.

    Aggregation happens *before* the scenario-macro average, and multiple runs
    of the same scenario are pooled instead of overwriting one another.  This
    is the rule that prevents mixing window-micro and scenario-macro metrics.
    """
    per_scenario = {}
    for scenario, error_arrays in scenario_errors.items():
        if not error_arrays:
            continue
        stacked = np.concatenate(error_arrays, axis=0)
        if stacked.size == 0:
            continue
        per_scenario[scenario] = float(np.sqrt(np.mean(stacked[:, 1] ** 2)))
    return per_scenario


def macro_average_scenarios(per_scenario: dict[str, float]) -> float:
    """Return the equal-weight mean over scenarios (never over runs)."""
    if not per_scenario:
        return float("nan")
    return float(np.mean(list(per_scenario.values())))


def per_scenario_rollout(run_frames, step_batch, horizons) -> dict:
    """Return per-scenario rollout RMSE for the final horizon and macro stats."""
    horizon = max(horizons)
    scenario_errors: dict[str, list[np.ndarray]] = {}
    for frame in run_frames:
        errors = _rollout_one_frame(frame, step_batch, [horizon])[horizon]
        if not errors:
            continue
        scenario = frame.attrs.get("scenario_id", "unknown")
        scenario_errors.setdefault(scenario, []).extend(errors)
    per_scenario = aggregate_scenario_rmse(scenario_errors)
    macro = macro_average_scenarios(per_scenario)
    return {
        "macro_yaw_rmse": macro,
        "per_scenario": per_scenario,
        "scenarios": len(per_scenario),
        "runs": len(run_frames),
    }


def evaluate_neurogrip_variant(model_path, config, batch, device) -> dict:
    """Evaluate one contextual Koopman checkpoint on a window batch."""
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    model = NeuroGripModel(config, context_mode=bundle["context_mode"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    statistics = {key: np.asarray(value) for key, value in bundle["statistics"].items()}
    state_mean, state_scale = statistics["state_mean"], statistics["state_scale"]
    cmd_mean, cmd_scale = statistics["cmd_mean"], statistics["cmd_scale"]
    history_mean = np.concatenate([state_mean, cmd_mean])
    history_scale = np.concatenate([state_scale, cmd_scale])
    with torch.no_grad():
        output = model(
            torch.tensor(
                (batch.history - history_mean) / history_scale,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                (batch.current_state - state_mean) / state_scale,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                (batch.future_commands - cmd_mean) / cmd_scale,
                dtype=torch.float32,
                device=device,
            ),
        )
    predicted = output.rollout.detach().cpu().numpy() * state_scale + state_mean
    scenario_ids = np.array(batch.scenario_id)
    per_scenario = {}
    for scenario in sorted(set(scenario_ids.tolist())):
        mask = scenario_ids == scenario
        error = predicted[mask, -1, 1] - batch.target_states[mask, -1, 1]
        per_scenario[scenario] = float(np.sqrt(np.mean(error**2)))
    metrics = {}
    for horizon in HORIZONS:
        error = predicted[:, horizon - 1, 1] - batch.target_states[:, horizon - 1, 1]
        metrics[str(horizon)] = float(np.sqrt(np.mean(error**2)))
    spectral = np.abs(np.linalg.eigvals(output.A.detach().cpu().numpy())).max(axis=1)
    return {
        "rollout_yaw_rmse": metrics,
        "macro_yaw_rmse": float(np.mean(list(per_scenario.values()))),
        "per_scenario": per_scenario,
        "max_spectral_radius": float(np.max(spectral)),
    }


def write_plot(rows: list[dict], output_path: Path) -> None:
    """Save a bar chart of the sealed-test 20-step rollout comparison."""
    sealed = [row for row in rows if row["sealed_test_20"] is not None]
    labels = [row["model"] for row in sealed]
    values = [row["sealed_test_20"] for row in sealed]
    if not values:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color="#3b6ea5")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("20-step yaw-rate RMSE [rad/s]")
    axis.set_title("Sealed OOD test open-loop rollout error (lower is better)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Validate provenance, evaluate all models and write benchmark artifacts."""
    arguments = parse_arguments()
    catalog_path = arguments.dataset_manifest.expanduser().resolve()
    catalog = load_json(catalog_path)
    ridge_metrics = load_json(arguments.ridge_metrics)
    mlp = load_json(arguments.mlp_manifest)
    neurogrip = load_json(arguments.neurogrip_manifest)

    problems = validate_provenance(catalog_path, catalog, ridge_metrics, mlp, neurogrip)
    if problems:
        for problem in problems:
            print(f"PROVENANCE ERROR: {problem}", file=sys.stderr)
        return 1

    def run_frames_with_ids(split: str) -> list[pd.DataFrame]:
        frames = []
        for run in catalog["runs"]:
            if run["split"] != split:
                continue
            frame = (
                pd.read_parquet(run["parquet_path"])
                .sort_values("time_s")
                .reset_index(drop=True)
            )
            frame.attrs["scenario_id"] = run["scenario_id"]
            frames.append(frame)
        return frames

    split_frames = {split: run_frames_with_ids(split) for split in REPORT_SPLITS}

    ridge = joblib.load(Path(ridge_metrics["model_path"]))
    ridge_step = make_ridge_batch_step(ridge)
    persistence_step = make_persistence_batch_step()

    from neurogrip_ai.model_runner import TransitionModelRunner  # noqa: E402

    mlp_runner = TransitionModelRunner.from_checkpoint(mlp["checkpoint_path"])

    def mlp_step(states, commands):
        features = np.concatenate([states, commands], axis=1)
        return np.stack([mlp_runner.predict(row) for row in features])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NeuroGripConfig(**neurogrip["config"])
    grouped = run_frames_by_split(catalog)
    window_batches = build_split_windows(grouped, config)

    neurogrip_models = {}
    for variant in ("n1", "b1_k"):
        path = Path(arguments.neurogrip_manifest).parent / f"{variant}_model.pt"
        neurogrip_models[variant] = {
            split: evaluate_neurogrip_variant(
                path, config, window_batches[split], device
            )
            for split in REPORT_SPLITS
        }

    rows = []
    for name, step in (
        ("persistence", persistence_step),
        ("Ridge", ridge_step),
        ("fixed MLP", mlp_step),
    ):
        row = {
            "model": name,
            "development_test_20": per_scenario_rollout(
                split_frames["development_test"], step, HORIZONS
            )["macro_yaw_rmse"],
            "sealed_test_20": per_scenario_rollout(
                split_frames["sealed_test"], step, HORIZONS
            )["macro_yaw_rmse"],
            "one_step_development_test": one_step_metrics(
                step,
                pd.concat(split_frames["development_test"], ignore_index=True),
            ),
            "one_step_sealed_test": one_step_metrics(
                step, pd.concat(split_frames["sealed_test"], ignore_index=True)
            ),
        }
        rows.append(row)

    for name, key in (("B1-K (global Koopman)", "b1_k"), ("N1 (context Koopman)", "n1")):
        results = neurogrip_models[key]
        rows.append(
            {
                "model": name,
                "development_test_20": results["development_test"]["rollout_yaw_rmse"]["20"],
                "sealed_test_20": results["sealed_test"]["rollout_yaw_rmse"]["20"],
                "sealed_rollout": results["sealed_test"]["rollout_yaw_rmse"],
                "sealed_macro_yaw_rmse": results["sealed_test"]["macro_yaw_rmse"],
                "sealed_per_scenario": results["sealed_test"]["per_scenario"],
                "development_per_scenario": results["development_test"]["per_scenario"],
                "max_spectral_radius": results["sealed_test"]["max_spectral_radius"],
                "one_step_development_test": None,
                "one_step_sealed_test": None,
            }
        )

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "backend": "legacy_ackermann_v0",
        "legacy": True,
        "metric_protocol": "scenario_macro_pooled_rollout_20",
        "note": (
            "Legacy Ackermann v0 simulator. Do not use these rows in the v2 "
            "EUFS hero table; see experiments_v2/evaluate_closed_loop.py."
        ),
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "report_splits": list(REPORT_SPLITS),
        "rows": rows,
        "sealed_test_declaration": catalog.get("sealed_test"),
        "calibration": (
            load_json(arguments.calibration) if arguments.calibration else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    write_plot(rows, output_dir / "sealed_rollout_comparison.png")

    lines = [
        "# NeuroGrip-X final benchmark",
        "",
        "Macro (equal weight per scenario) 20-step yaw-rate RMSE [rad/s].",
        "",
        "| Model | development test | sealed OOD test |",
        "|---|---:|---:|",
    ]
    for row in rows:
        dev = row["development_test_20"]
        sealed = row["sealed_test_20"]
        lines.append(
            f"| {row['model']} | {'-' if dev is None else f'{dev:.6f}'} "
            f"| {'-' if sealed is None else f'{sealed:.6f}'} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Sealed OOD test 20-step macro yaw-rate RMSE [rad/s]:")
    for row in rows:
        value = row["sealed_test_20"]
        print(f"  {row['model']:26s} {'-' if value is None else f'{value:.6f}'}")
    print(f"Summary: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
