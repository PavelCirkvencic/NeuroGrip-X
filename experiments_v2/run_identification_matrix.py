#!/usr/bin/env python3
"""Run a disjoint NeuroGrip C2 identification matrix safely and resumably.

The script deliberately runs one ROS graph at a time.  It discovers scenario
YAML files by their immutable ``scenario_id`` in a split manifest and delegates
each episode to :mod:`run_episode`, the project's only ROS process authority.
No completed episode is overwritten; ``--resume`` skips only an episode with a
matching identity and an explicit successful terminal event.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scenario_builder_v2 import REPO_ROOT, load_and_validate_scenario

SUCCESS_TERMINATIONS = {"lap_complete", "excitation_complete"}


def parse_arguments() -> argparse.Namespace:
    """Parse the fixed identification controller and artifact locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path, default=Path("config/scenarios"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--real-time-timeout-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scenario_ids_from_manifest(path: Path) -> list[str]:
    """Return ordered, unique IDs from valid train/validation/calibration splits."""
    document = json.loads(path.read_text(encoding="utf-8"))
    splits = document.get("splits")
    if document.get("schema_version") != 1 or not isinstance(splits, dict):
        raise ValueError("split manifest must have schema_version 1 and a splits object")
    required = ("train", "validation", "calibration")
    if any(name not in splits or not isinstance(splits[name], list) for name in required):
        raise ValueError("split manifest needs train, validation and calibration arrays")
    identifiers = [identifier for name in required for identifier in splits[name]]
    if not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise ValueError("split scenario IDs must be non-empty strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("a scenario ID appears in more than one identification split")
    return identifiers


def complete_episode(run_directory: Path, scenario_id: str, seed: int) -> bool:
    """Accept a resume skip only for an identity-matching successful run."""
    metadata_path = run_directory / "episode.metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("scenario_id") == scenario_id
        and metadata.get("seed") == seed
        and metadata.get("controller") == "EXCITATION"
        and metadata.get("runner", {}).get("termination_reason") in SUCCESS_TERMINATIONS
        and (run_directory / "episode.csv").is_file()
        and (run_directory / "episode.events.jsonl").is_file()
    )


def planned_scenarios(manifest_path: Path, scenario_dir: Path) -> list[tuple[str, Path, dict]]:
    """Resolve and validate every YAML before any expensive ROS process starts."""
    planned = []
    for scenario_id in scenario_ids_from_manifest(manifest_path):
        scenario_path = scenario_dir / f"{scenario_id}.yaml"
        if not scenario_path.is_file():
            raise FileNotFoundError(f"no scenario YAML for split member {scenario_id}: {scenario_path}")
        scenario = load_and_validate_scenario(scenario_path)
        if scenario["scenario_id"] != scenario_id:
            raise ValueError(f"{scenario_path}: scenario_id must match its split manifest entry")
        planned.append((scenario_id, scenario_path, scenario))
    return planned


def main() -> int:
    """Validate first, then execute or resume the complete identification matrix."""
    arguments = parse_arguments()
    manifest = arguments.split_manifest.expanduser().resolve()
    scenario_dir = arguments.scenario_dir.expanduser().resolve()
    run_root = arguments.run_root.expanduser().resolve()
    planned = planned_scenarios(manifest, scenario_dir)
    actions = []
    for scenario_id, path, scenario in planned:
        run_id = f"{scenario_id}__EXCITATION__seed{scenario['seed']:04d}"
        run_directory = run_root / run_id
        if arguments.resume and complete_episode(run_directory, scenario_id, scenario["seed"]):
            actions.append(("skip", scenario_id, path, scenario))
        elif run_directory.exists():
            raise FileExistsError(
                f"{run_directory} is incomplete or --resume was not requested; inspect it before rerunning"
            )
        else:
            actions.append(("run", scenario_id, path, scenario))
    for action, scenario_id, path, scenario in actions:
        print(f"MATRIX_{action.upper()} scenario={scenario_id} seed={scenario['seed']}")
        if action == "skip" or arguments.dry_run:
            continue
        command = [
            sys.executable, "experiments_v2/run_episode.py", "--scenario", str(path),
            "--controller", "EXCITATION", "--seed", str(scenario["seed"]),
            "--run-root", str(run_root), "--real-time-timeout-s", str(arguments.real_time_timeout_s),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"IDENTIFICATION_MATRIX_COMPLETE total={len(planned)} run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
