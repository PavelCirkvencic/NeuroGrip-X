#!/usr/bin/env python3
"""Build a coverage report and acceptance gate from the dataset catalog.

The report is driven exclusively by ``dataset_manifest.json`` and the Parquet
files it points at.  It never globs raw recordings, and it never feeds hidden
scenario parameters to any model: simulator labels appear only as an explicitly
labelled *offline evaluation* table.

Run it after every batch of recordings to confirm that the frozen benchmark is
getting more diverse instead of merely larger.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configured steering-chirp band of the ``dynamic_v2`` excitation profile.
# Kept here as documentation; the empirical duration is measured from data.
CHIRP_BAND_HZ = (0.10, 0.55)
SIGNAL_COLUMNS = {
    "v_x": "v_x_t_mps",
    "yaw_rate": "yaw_rate_t_rps",
    "cmd_linear_x": "cmd_linear_x_t_mps",
    "cmd_angular_z": "cmd_angular_z_t_rps",
    "cmd_age_s": "cmd_age_s",
}
# States that the learned models actually consume.  Lateral body velocity is
# intentionally *not* here: it is estimated and stored for analysis but is not
# a modeled state (see docs/limitations.md).
MODELED_STATE_COLUMNS = ("v_x_t_mps", "yaw_rate_t_rps")
SPLITS = ("train", "validation", "calibration", "development_test", "sealed_test")
MIN_SCENARIOS_PER_SPLIT = 2


def parse_arguments() -> argparse.Namespace:
    """Parse catalog, output and acceptance-threshold options."""
    parser = argparse.ArgumentParser(
        description="Report NeuroGrip-X dataset coverage and run acceptance checks."
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/processed/dataset_manifest.json"),
        help="Scenario-grouped dataset catalog.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/coverage"),
        help="Directory for the JSON summary, CSV and plots.",
    )
    parser.add_argument("--min-episodes", type=int, default=12)
    parser.add_argument("--min-profiles", type=int, default=3)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument(
        "--min-state-std",
        type=float,
        default=1e-3,
        help="Minimum standard deviation for a modeled state within each split.",
    )
    parser.add_argument(
        "--allow-empty-test",
        action="store_true",
        help="Do not fail acceptance when the locked test split is empty.",
    )
    return parser.parse_args()


def load_catalog(manifest_path: Path) -> dict:
    """Load and minimally validate the dataset catalog."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    try:
        catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON dataset manifest: {manifest_path}") from error
    if catalog.get("split_unit") != "scenario_id":
        raise ValueError("Catalog must use scenario_id as its split unit.")
    if not isinstance(catalog.get("runs"), list) or not catalog["runs"]:
        raise ValueError("Catalog has no runs.")
    return catalog


def describe(values: pd.Series) -> dict[str, float]:
    """Return robust distribution statistics for one signal."""
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def chirp_statistics(dataframe: pd.DataFrame) -> dict[str, float]:
    """Measure observed steering-chirp duration and average frequency."""
    if "phase" not in dataframe.columns:
        return {"seconds": 0.0, "zero_crossings": 0, "mean_hz": 0.0}
    chirp = dataframe.loc[dataframe["phase"] == "steering_chirp", "cmd_angular_z_t_rps"]
    if chirp.empty:
        return {"seconds": 0.0, "zero_crossings": 0, "mean_hz": 0.0}
    dt_s = float(dataframe["dt_s"].median()) if "dt_s" in dataframe else 0.02
    seconds = float(len(chirp) * dt_s)
    sign = np.sign(chirp.to_numpy(dtype=float))
    sign = sign[sign != 0.0]
    zero_crossings = int(np.sum(np.abs(np.diff(sign)) > 0.0)) if sign.size else 0
    mean_hz = zero_crossings / (2.0 * seconds) if seconds > 0 else 0.0
    return {
        "seconds": seconds,
        "zero_crossings": zero_crossings,
        "mean_hz": float(mean_hz),
    }


def collect_split_data(catalog: dict) -> dict:
    """Read catalog Parquet files and aggregate per-split statistics."""
    per_split = defaultdict(list)
    run_rows = []
    for run in catalog["runs"]:
        split = run.get("split")
        parquet_path = Path(run["parquet_path"])
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing Parquet: {parquet_path}")
        dataframe = pd.read_parquet(parquet_path)
        per_split[split].append((run, dataframe))
        transition = (run.get("scenario") or {}).get("grip_transition") or {}
        run_rows.append(
            {
                "run_id": run.get("run_id"),
                "split": split,
                "scenario_id": run.get("scenario_id"),
                "excitation_profile": run.get("excitation_profile"),
                "termination_reason": run.get("termination_reason"),
                "transition_start_s": transition.get("start_s"),
                "transition_compliance_scale": transition.get("compliance_scale"),
                "rows": len(dataframe),
                **{f"chirp_{key}": value for key, value in chirp_statistics(dataframe).items()},
            }
        )

    split_summary = {}
    for split in SPLITS:
        entries = per_split.get(split, [])
        frames = [frame for _, frame in entries]
        scenario_ids = sorted({run.get("scenario_id") for run, _ in entries})
        seeds = sorted({int(run["scenario"]["seed"]) for run, _ in entries if "scenario" in run})
        profiles = sorted({run["scenario"]["profile"] for run, _ in entries if "scenario" in run})
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        signals = {
            name: describe(combined[column])
            for name, column in SIGNAL_COLUMNS.items()
            if not combined.empty and column in combined.columns
        }
        chirp_seconds = sum(
            chirp_statistics(frame)["seconds"] for frame in frames
        )
        modeled_state_std = {
            column: (
                float(combined[column].std())
                if not combined.empty and column in combined.columns
                else 0.0
            )
            for column in MODELED_STATE_COLUMNS
        }
        qa = {}
        if not combined.empty and "dt_s" in combined.columns:
            dt_values = combined["dt_s"].to_numpy(dtype=float)
            median_dt = float(np.median(dt_values))
            qa["dt_median_s"] = median_dt
            qa["max_dt_deviation_s"] = float(np.max(np.abs(dt_values - median_dt)))
        if not combined.empty and "cmd_age_s" in combined.columns:
            qa["max_cmd_age_s"] = float(combined["cmd_age_s"].max())
        if not combined.empty and "odom_imu_dt_s" in combined.columns:
            qa["max_abs_odom_imu_dt_s"] = float(
                combined["odom_imu_dt_s"].abs().max()
            )
        split_summary[split] = {
            "runs": len(entries),
            "scenario_ids": scenario_ids,
            "seeds": seeds,
            "profiles": profiles,
            "rows": int(len(combined)),
            "chirp_seconds": float(chirp_seconds),
            "signals": signals,
            "modeled_state_std": modeled_state_std,
            "qa": qa,
        }
    return {"split_summary": split_summary, "run_rows": run_rows}


def offline_metadata_table(catalog: dict) -> pd.DataFrame:
    """Build an offline-only scenario parameter table for provenance review."""
    rows = []
    for run in catalog["runs"]:
        scenario = run.get("scenario", {})
        transition = scenario.get("grip_transition") or {}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "scenario_id": run.get("scenario_id"),
                "split": run.get("split"),
                "profile": scenario.get("profile"),
                "seed": scenario.get("seed"),
                "friction_scale": scenario.get("friction_scale"),
                "front_lateral_compliance": scenario.get("front_lateral_compliance"),
                "rear_lateral_compliance": scenario.get("rear_lateral_compliance"),
                "longitudinal_compliance": scenario.get("longitudinal_compliance"),
                "mass_scale": scenario.get("mass_scale"),
                "cg_shift_m": scenario.get("cg_shift_m"),
                "steering_delay_s": scenario.get("steering_delay_s"),
                "steering_gain": scenario.get("steering_gain"),
                "transition_start_s": transition.get("start_s"),
                "transition_compliance_scale": transition.get("compliance_scale"),
                "excitation_profile": run.get("excitation_profile"),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "scenario_id", "run_id"])


def run_acceptance_checks(catalog: dict, split_summary: dict, arguments) -> list[dict]:
    """Evaluate the Phase 1 gate criteria from the coverage data."""
    records = catalog["runs"]
    episode_count = len(records)
    all_profiles = {run["scenario"]["profile"] for run in records if "scenario" in run}
    all_seeds = {int(run["scenario"]["seed"]) for run in records if "scenario" in run}
    invalid_runs = [
        run.get("run_id")
        for run in records
        if run.get("termination_reason") != "excitation_complete"
    ]
    split_scenario_sets = {
        split: set(split_summary[split]["scenario_ids"]) for split in SPLITS
    }
    development_scenarios = split_scenario_sets["development_test"]
    sealed_scenarios = split_scenario_sets["sealed_test"]

    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add(
        "minimum_episodes",
        episode_count >= arguments.min_episodes,
        f"{episode_count} traceable episodes (>= {arguments.min_episodes})",
    )
    add(
        "multiple_seeds",
        len(all_seeds) >= arguments.min_seeds,
        f"{len(all_seeds)} distinct seeds (>= {arguments.min_seeds})",
    )
    add(
        "minimum_profiles",
        len(all_profiles) >= arguments.min_profiles,
        f"{len(all_profiles)} profiles: {sorted(all_profiles)} (>= {arguments.min_profiles})",
    )
    add(
        "no_invalid_runs",
        not invalid_runs,
        "no incomplete runs in catalog" if not invalid_runs else f"invalid: {invalid_runs}",
    )
    overlap = set()
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            overlap |= split_scenario_sets[first] & split_scenario_sets[second]
    add(
        "disjoint_scenario_splits",
        not overlap,
        "all five splits share no scenario_id"
        if not overlap
        else f"overlapping scenario_ids: {sorted(overlap)}",
    )
    add(
        "sealed_test_declared",
        bool(sealed_scenarios) or arguments.allow_empty_test,
        f"sealed scenarios: {sorted(sealed_scenarios)}",
    )
    add(
        "development_test_separate",
        bool(development_scenarios) or arguments.allow_empty_test,
        f"development scenarios: {sorted(development_scenarios)}",
    )
    add(
        "all_splits_populated",
        all(split_summary[split]["runs"] > 0 for split in SPLITS),
        "every split has at least one run",
    )
    for split in SPLITS:
        if split_summary[split]["runs"] == 0:
            continue
        state_std = split_summary[split].get("modeled_state_std", {})
        degenerate = {
            name: std
            for name, std in state_std.items()
            if std < arguments.min_state_std
        }
        add(
            f"state_not_degenerate_{split}",
            not degenerate,
            (
                ", ".join(f"{name}={std:.2e}" for name, std in state_std.items())
                if not degenerate
                else "near-zero variance: "
                + ", ".join(f"{name}={std:.2e}" for name, std in degenerate.items())
            ),
        )

    required_minimums = {
        "train": 3,
        "validation": 1,
        "calibration": 1,
        "development_test": MIN_SCENARIOS_PER_SPLIT,
        "sealed_test": MIN_SCENARIOS_PER_SPLIT,
    }
    for split, minimum in required_minimums.items():
        count = len(split_summary[split]["scenario_ids"])
        add(
            f"min_scenarios_{split}",
            count >= minimum,
            f"{count} scenarios (>= {minimum})",
        )

    sealed_profiles = set(split_summary["sealed_test"]["profiles"])
    add(
        "sealed_regime_representation",
        len(sealed_profiles) >= 3,
        f"sealed profiles: {sorted(sealed_profiles)} (>= 3)",
    )

    unapplied = [
        run.get("run_id")
        for run in records
        if isinstance((run.get("scenario") or {}).get("grip_transition"), dict)
        and run.get("grip_transition_applied") is not True
    ]
    add(
        "transition_applied_in_catalog",
        not unapplied,
        "every transitioning run has confirmed application"
        if not unapplied
        else f"unconfirmed: {unapplied}",
    )

    missing_provenance = [
        run.get("run_id")
        for run in records
        if not run.get("dataset_schema_version")
        or not run.get("scenario_manifest_sha256")
    ]
    catalog_has_schema = bool(catalog.get("schema_version")) and bool(
        catalog.get("split_schema_version")
    )
    add(
        "artifact_schema_and_hash",
        catalog_has_schema and not missing_provenance,
        (
            "catalog and every run carry schema version and manifest hash"
            if catalog_has_schema and not missing_provenance
            else f"missing provenance: {missing_provenance}"
        ),
    )

    for split in SPLITS:
        qa = split_summary[split].get("qa", {})
        if not qa:
            continue
        ok = (
            qa.get("max_dt_deviation_s", 0.0) <= 0.005
            and qa.get("max_cmd_age_s", 0.0) <= 0.25
            and qa.get("max_abs_odom_imu_dt_s", 0.0) <= 0.05
        )
        add(
            f"qa_thresholds_{split}",
            ok,
            (
                f"dt_dev={qa.get('max_dt_deviation_s', float('nan')):.4f}s, "
                f"cmd_age={qa.get('max_cmd_age_s', float('nan')):.4f}s, "
                f"align={qa.get('max_abs_odom_imu_dt_s', float('nan')):.4f}s"
            ),
        )
    return checks


def _histogram_panel(axis, split_frames: dict, column: str, title: str, xlabel: str) -> None:
    """Plot per-split histograms for one signal on a shared axis."""
    for split in SPLITS:
        if column not in split_frames[split].columns or split_frames[split].empty:
            continue
        axis.hist(
            split_frames[split][column].to_numpy(dtype=float),
            bins=40,
            alpha=0.5,
            density=True,
            label=split,
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("density")
    axis.grid(True, alpha=0.3)
    axis.legend()


def make_plots(catalog: dict, output_dir: Path) -> None:
    """Save per-split distribution and overview figures."""
    frames = {
        split: pd.concat(
            [
                pd.read_parquet(run["parquet_path"])
                for run in catalog["runs"]
                if run["split"] == split
            ],
            ignore_index=True,
        )
        if any(run["split"] == split for run in catalog["runs"])
        else pd.DataFrame()
        for split in SPLITS
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    counts = {
        split: sum(1 for run in catalog["runs"] if run["split"] == split)
        for split in SPLITS
    }
    axis.bar(list(counts.keys()), list(counts.values()), color="#3b6ea5")
    for index, (split, value) in enumerate(counts.items()):
        axis.text(index, value, str(value), ha="center", va="bottom")
    axis.set_title("Traceable runs per split")
    axis.set_ylabel("runs")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "split_overview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    _histogram_panel(axes[0, 0], frames, "v_x_t_mps", "Longitudinal speed", "v_x [m/s]")
    _histogram_panel(axes[0, 1], frames, "yaw_rate_t_rps", "Yaw rate", "yaw rate [rad/s]")
    _histogram_panel(
        axes[1, 0], frames, "cmd_angular_z_t_rps", "Steering command", "angular z [rad/s]"
    )
    _histogram_panel(axes[1, 1], frames, "cmd_age_s", "Command age", "cmd age [s]")
    figure.suptitle("NeuroGrip-X per-split excitation coverage")
    figure.tight_layout()
    figure.savefig(output_dir / "coverage_distributions.png", dpi=160)
    plt.close(figure)

    chirp_seconds = []
    labels = []
    for run in sorted(catalog["runs"], key=lambda entry: (entry["split"], entry["scenario_id"])):
        dataframe = pd.read_parquet(run["parquet_path"])
        stats = chirp_statistics(dataframe)
        if stats["seconds"] <= 0:
            continue
        chirp_seconds.append(stats["seconds"])
        labels.append(f"{run['scenario_id']}\n({run['split']})")
    if chirp_seconds:
        figure, axis = plt.subplots(figsize=(max(7, len(labels) * 0.6), 5))
        axis.bar(range(len(labels)), chirp_seconds, color="#a5533b")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_ylabel("steering_chirp duration [s]")
        axis.set_title(
            f"Observed chirp duration (configured band {CHIRP_BAND_HZ[0]}-{CHIRP_BAND_HZ[1]} Hz)"
        )
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / "chirp_coverage.png", dpi=160)
        plt.close(figure)


def main() -> int:
    """Generate the coverage report and enforce acceptance criteria."""
    arguments = parse_arguments()
    catalog = load_catalog(arguments.dataset_manifest)
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_split_data(catalog)
    split_summary = collected["split_summary"]
    checks = run_acceptance_checks(catalog, split_summary, arguments)
    passed = all(check["passed"] for check in checks)

    metadata_table = offline_metadata_table(catalog)
    metadata_path = output_dir / "scenario_metadata_offline.csv"
    metadata_table.to_csv(metadata_path, index=False)

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str(arguments.dataset_manifest.expanduser().resolve()),
        "split_unit": "scenario_id",
        "total_episodes": len(catalog["runs"]),
        "skipped_untraceable": catalog.get("skipped_untraceable_metadata", []),
        "skipped_incomplete": catalog.get("skipped_incomplete_runs", []),
        "chirp_band_hz": list(CHIRP_BAND_HZ),
        "split_summary": split_summary,
        "runs": collected["run_rows"],
        "acceptance": {"passed": passed, "checks": checks},
    }
    summary_path = output_dir / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_plots(catalog, output_dir)

    print(f"Catalog: {summary['dataset_manifest']}")
    print(f"Total traceable episodes: {summary['total_episodes']}")
    for split in SPLITS:
        data = split_summary[split]
        print(
            f"  {split:10s}: {data['runs']} runs, {len(data['scenario_ids'])} scenarios, "
            f"{data['rows']} rows, chirp={data['chirp_seconds']:.1f}s"
        )
        v_x = data["signals"].get("v_x", {})
        yaw = data["signals"].get("yaw_rate", {})
        if v_x:
            print(
                f"      v_x [{v_x['min']:.3f}, {v_x['max']:.3f}] m/s, "
                f"yaw_rate [{yaw['min']:.3f}, {yaw['max']:.3f}] rad/s"
            )
        state_std = data.get("modeled_state_std", {})
        if state_std:
            print(
                "      modeled-state std: "
                + ", ".join(f"{name}={value:.2e}" for name, value in state_std.items())
            )
    print("Acceptance:")
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")
    print(f"Summary: {summary_path}")
    print(f"Offline scenario metadata: {metadata_path}")
    print(f"Plots: {output_dir}")

    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Coverage report failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
