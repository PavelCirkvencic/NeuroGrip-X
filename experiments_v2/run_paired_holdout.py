#!/usr/bin/env python3
"""Run one hash-frozen controller holdout matrix without selection by result.

The manifest is deliberately more restrictive than the development runner:
every scenario YAML and the C2 ensemble manifest are hash-pinned before a
single ROS episode starts. Schema 1 retains the historical C0/C1/C2 matrix;
schema 2 runs the historical publication comparison; schema 3 runs the
grip-aware Koopman C0/C2 comparison; schema 4 additionally freezes each
controller's Python/MATLAB actuation source. A natural
controller timeout is a terminal result, not a reason to re-run that
controller. The command owns only one ROS graph at a time by delegating every
episode to :mod:`run_episode`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scenario_builder_v2 import REPO_ROOT, load_and_validate_scenario

CONTROLLERS = ("C0_FIXED", "C1_ORACLE_MU", "C2_NEUROGRIP")
PUBLICATION_CONTROLLERS = ("C0_FIXED", "C2_NEUROGRIP")
TERMINAL_OUTCOMES = {"lap_complete", "timeout"}


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading an arbitrary artifact at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    """Parse a frozen manifest and locations for new, non-overwriting output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--real-time-timeout-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def development_members(path: Path) -> set[str]:
    """Load the prior train/validation/calibration IDs to prevent data leakage."""
    document = json.loads(path.read_text(encoding="utf-8"))
    splits = document.get("splits")
    if document.get("schema_version") not in (1, 2) or not isinstance(splits, dict):
        raise ValueError("development split manifest must have schema_version 1/2 and splits")
    members = [item for values in splits.values() for item in values]
    if not all(isinstance(item, str) and item for item in members):
        raise ValueError("development split members must be non-empty strings")
    if len(members) != len(set(members)):
        raise ValueError("development split manifest contains duplicate scenario IDs")
    return set(members)


def resolve_repo_file(value: str, label: str) -> Path:
    """Resolve a manifest's repository-relative path and reject traversal."""
    candidate = (REPO_ROOT / value).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def load_holdout_plan(path: Path) -> dict[str, Any]:
    """Validate all immutable inputs before starting any ROS process."""
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "purpose", "development_split_manifest",
        "target_speed_mps", "controllers", "c2_ensemble_manifest",
        "c2_ensemble_manifest_sha256", "scenarios",
    }
    missing = required - document.keys()
    schema_version = document.get("schema_version")
    if schema_version not in (1, 2, 3, 4) or missing:
        raise ValueError(f"holdout manifest schema invalid; missing {sorted(missing)}")
    expected_controllers = CONTROLLERS if schema_version == 1 else PUBLICATION_CONTROLLERS
    if tuple(document["controllers"]) != expected_controllers:
        raise ValueError(
            f"controllers must be exactly {list(expected_controllers)} in that order"
        )
    target_speed = float(document["target_speed_mps"])
    maximum_target_speed = 16.0 if schema_version in (3, 4) else 8.0
    if not 1.0 <= target_speed <= maximum_target_speed:
        raise ValueError(f"target_speed_mps must be in [1.0, {maximum_target_speed}]")
    development_path = resolve_repo_file(
        document["development_split_manifest"], "development_split_manifest"
    )
    development_ids = development_members(development_path)
    ensemble_path = resolve_repo_file(document["c2_ensemble_manifest"], "c2_ensemble_manifest")
    if sha256_file(ensemble_path) != document["c2_ensemble_manifest_sha256"]:
        raise ValueError("C2 ensemble manifest hash does not match the frozen holdout plan")
    try:
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("C2 ensemble manifest is not valid JSON") from error
    required_ensemble_schema = schema_version if schema_version in (1, 2) else 3
    if (
        ensemble.get("schema_version") != required_ensemble_schema
        or len(ensemble.get("members", [])) != 3
    ):
        raise ValueError(
            f"C2 ensemble must be a valid three-member schema {required_ensemble_schema} ensemble"
        )
    c2_adaptive_speed_max = float(document.get("c2_adaptive_speed_max_mps", target_speed))
    c2_model_blend = float(document.get("c2_model_blend", 1.0))
    if schema_version in (2, 3, 4) and not (
        target_speed <= c2_adaptive_speed_max <= maximum_target_speed
        and 0.0 <= c2_model_blend <= 1.0
    ):
        raise ValueError("invalid C2 speed envelope or model blend")
    target_speed_min = float(document.get("target_speed_min_mps", 1.2))
    fixed_mu = float(document.get("fixed_mu", 1.0))
    scalar_utilisation = float(document.get("scalar_mu_utilisation", 0.78))
    neurogrip_utilisation = float(document.get("neurogrip_utilisation", 0.95))
    grip_margin_scale = float(document.get("grip_error_margin_scale", 0.50))
    startup_speed_cap = float(document.get("startup_speed_cap_mps", target_speed))
    startup_speed_cap_end = float(
        document.get("startup_speed_cap_end_progress", 0.0)
    )
    if schema_version in (3, 4) and not (
        0.0 < target_speed_min <= target_speed
        and 0.35 <= fixed_mu <= 1.30
        and 0.0 < scalar_utilisation <= 1.0
        and 0.0 < neurogrip_utilisation <= 1.0
        and 0.0 <= grip_margin_scale <= 1.0
        and target_speed_min <= startup_speed_cap <= target_speed
        and 0.0 <= startup_speed_cap_end <= 0.25
    ):
        raise ValueError("invalid schema-3 physical speed-profile policy")

    actuation_sources = document.get(
        "actuation_sources",
        {controller: "python" for controller in expected_controllers},
    )
    if (
        not isinstance(actuation_sources, dict)
        or set(actuation_sources) != set(expected_controllers)
        or any(
            source not in {"python", "matlab"}
            for source in actuation_sources.values()
        )
    ):
        raise ValueError("actuation_sources must map every controller to python/matlab")
    if schema_version == 4 and actuation_sources != {
        "C0_FIXED": "python",
        "C2_NEUROGRIP": "matlab",
    }:
        raise ValueError("schema-4 holdout requires C0=python and C2=matlab")

    scenarios = document["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) < 3:
        raise ValueError("holdout requires at least three independently seeded scenarios")
    planned = []
    identifiers, seeds = set(), set()
    for entry in scenarios:
        if not isinstance(entry, dict):
            raise ValueError("each holdout scenario must be an object")
        scenario_id, yaml_rel, yaml_hash = (
            entry.get("scenario_id"), entry.get("yaml"), entry.get("yaml_sha256")
        )
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("each holdout scenario needs a non-empty scenario_id")
        if (
            not isinstance(yaml_rel, str)
            or not isinstance(yaml_hash, str)
            or len(yaml_hash) != 64
        ):
            raise ValueError(f"{scenario_id}: yaml and 64-character yaml_sha256 are required")
        if scenario_id in identifiers:
            raise ValueError(f"duplicate final-holdout scenario_id: {scenario_id}")
        if scenario_id in development_ids:
            raise ValueError(f"final holdout leaks development scenario: {scenario_id}")
        scenario_path = resolve_repo_file(yaml_rel, f"{scenario_id}.yaml")
        if sha256_file(scenario_path) != yaml_hash:
            raise ValueError(
                f"{scenario_id}: scenario YAML hash does not match frozen holdout plan"
            )
        scenario = load_and_validate_scenario(scenario_path)
        if scenario["scenario_id"] != scenario_id:
            raise ValueError(f"{scenario_id}: YAML scenario_id does not match manifest")
        if scenario["seed"] in seeds:
            raise ValueError(f"{scenario_id}: final-holdout seed is not independent")
        identifiers.add(scenario_id)
        seeds.add(scenario["seed"])
        planned.append((scenario_id, scenario_path, scenario))
    return {
        "target_speed_mps": target_speed,
        "c2_adaptive_speed_max_mps": c2_adaptive_speed_max,
        "c2_model_blend": c2_model_blend,
        "target_speed_min_mps": target_speed_min,
        "fixed_mu": fixed_mu,
        "scalar_mu_utilisation": scalar_utilisation,
        "neurogrip_utilisation": neurogrip_utilisation,
        "grip_error_margin_scale": grip_margin_scale,
        "startup_speed_cap_mps": startup_speed_cap,
        "startup_speed_cap_end_progress": startup_speed_cap_end,
        "controllers": expected_controllers,
        "actuation_sources": actuation_sources,
        "ensemble_path": ensemble_path,
        "planned": planned,
    }


def terminal_episode(
    run_directory: Path,
    scenario_id: str,
    controller: str,
    seed: int,
    actuation_source: str = "python",
) -> bool:
    """Allow resume only for an identity-matching terminal run with full evidence."""
    metadata_path = run_directory / "episode.metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("scenario_id") == scenario_id
        and metadata.get("controller") == controller
        and metadata.get("seed") == seed
        and metadata.get("provenance", {}).get("actuation_source", "python")
        == actuation_source
        and metadata.get("runner", {}).get("termination_reason") in TERMINAL_OUTCOMES
        and (run_directory / "episode.csv").is_file()
        and (run_directory / "episode.events.jsonl").is_file()
    )


def main() -> int:
    """Freeze-validate, then execute the entire paired matrix sequentially."""
    arguments = parse_arguments()
    plan = load_holdout_plan(arguments.holdout_manifest.expanduser().resolve())
    run_root = arguments.run_root.expanduser().resolve()
    actions = []
    for scenario_id, path, scenario in plan["planned"]:
        for controller in plan["controllers"]:
            actuation_source = plan["actuation_sources"][controller]
            run_id = f"{scenario_id}__{controller}__seed{scenario['seed']:04d}"
            run_directory = run_root / run_id
            if arguments.resume and terminal_episode(
                run_directory,
                scenario_id,
                controller,
                scenario["seed"],
                actuation_source,
            ):
                actions.append(
                    ("skip", scenario_id, path, scenario, controller, actuation_source)
                )
            elif run_directory.exists():
                raise FileExistsError(
                    f"{run_directory} is incomplete or --resume was not specified; "
                    "inspect it first"
                )
            else:
                actions.append(
                    ("run", scenario_id, path, scenario, controller, actuation_source)
                )

    for action, scenario_id, path, scenario, controller, actuation_source in actions:
        print(
            f"FINAL_HOLDOUT_{action.upper()} scenario={scenario_id} "
            f"controller={controller} actuation={actuation_source}"
        )
        if action == "skip" or arguments.dry_run:
            continue
        command = [
            sys.executable, "experiments_v2/run_episode.py", "--scenario", str(path),
            "--controller", controller, "--seed", str(scenario["seed"]),
            "--target-speed-mps", str(plan["target_speed_mps"]), "--run-root", str(run_root),
            "--target-speed-min-mps", str(plan["target_speed_min_mps"]),
            "--fixed-mu", str(plan["fixed_mu"]),
            "--scalar-mu-utilisation", str(plan["scalar_mu_utilisation"]),
            "--neurogrip-utilisation", str(plan["neurogrip_utilisation"]),
            "--grip-error-margin-scale", str(plan["grip_error_margin_scale"]),
            "--startup-speed-cap-mps", str(plan["startup_speed_cap_mps"]),
            "--startup-speed-cap-end-progress",
            str(plan["startup_speed_cap_end_progress"]),
            "--real-time-timeout-s", str(arguments.real_time_timeout_s),
            "--actuation-source", actuation_source,
        ]
        if controller == "C2_NEUROGRIP":
            command.extend(
                [
                    "--c2-ensemble-manifest",
                    str(plan["ensemble_path"]),
                    "--c2-adaptive-speed-max-mps",
                    str(plan["c2_adaptive_speed_max_mps"]),
                    "--c2-model-blend",
                    str(plan["c2_model_blend"]),
                ]
            )
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"FINAL_HOLDOUT_MATRIX_COMPLETE total={len(actions)} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
