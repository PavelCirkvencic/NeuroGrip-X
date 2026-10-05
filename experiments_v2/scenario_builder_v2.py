#!/usr/bin/env python3
"""Validate a v2 scenario and create its immutable EUFS core configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = (
    REPO_ROOT
    / "external/eufs_ws/src/vehicle_models/config/DynamicBicycle/ads-dv-calculated.yaml"
)
REQUIRED_TOP_LEVEL = {
    "schema_version", "scenario_id", "seed", "track", "base_vehicle", "actuator",
    "grip_schedule", "sensors",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_positive(mapping: dict[str, Any], name: str) -> float:
    """Read a required finite positive scalar from a user scenario mapping."""
    try:
        value = float(mapping[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not value > 0.0 or value == float("inf"):
        raise ValueError(f"{name} must be a finite positive number")
    return value


def load_and_validate_scenario(path: Path) -> dict[str, Any]:
    """Load the strict, portable scenario schema used by the only v2 runner."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("scenario root must be a YAML mapping")
    missing = REQUIRED_TOP_LEVEL - document.keys()
    if missing:
        raise ValueError(f"scenario is missing required fields: {sorted(missing)}")
    if document["schema_version"] != 1:
        raise ValueError("only scenario schema_version 1 is supported")
    if not isinstance(document["scenario_id"], str) or not document["scenario_id"]:
        raise ValueError("scenario_id must be a non-empty string")
    if not isinstance(document["seed"], int) or document["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if document["track"] not in {"small_track", "trackdrive"}:
        raise ValueError("track must be small_track or trackdrive")
    base = document["base_vehicle"]
    actuator = document["actuator"]
    schedule = document["grip_schedule"]
    if not isinstance(base, dict) or not isinstance(actuator, dict) or not isinstance(schedule, list):
        raise ValueError("base_vehicle, actuator and grip_schedule have invalid types")
    for name in ("mass_scale", "inertia_scale", "pacejka_A_scale", "pacejka_B_scale", "pacejka_C_scale"):
        require_finite_positive(base, name)
    for name in ("steering_gain",):
        require_finite_positive(actuator, name)
    delay = float(actuator.get("steering_delay_s", -1.0))
    if not 0.0 <= delay <= 0.25:
        raise ValueError("actuator.steering_delay_s must be in [0, 0.25]")
    if not schedule or float(schedule[0].get("start_s", -1.0)) != 0.0:
        raise ValueError("grip_schedule must begin with start_s: 0.0")
    previous_start = -1.0
    for index, entry in enumerate(schedule):
        if not isinstance(entry, dict):
            raise ValueError("each grip_schedule entry must be a mapping")
        start_s = float(entry.get("start_s", -1.0))
        front = float(entry.get("front_scale", float("nan")))
        rear = float(entry.get("rear_scale", float("nan")))
        if not start_s >= previous_start or not start_s >= 0.0:
            raise ValueError("grip schedule start times must be non-decreasing and non-negative")
        if not (0.35 <= front <= 1.30 and 0.35 <= rear <= 1.30):
            raise ValueError("grip scales must be inside [0.35, 1.30]")
        if "start_progress" in entry:
            start_progress = float(entry["start_progress"])
            if index == 0 or not 0.0 <= start_progress < 1.0:
                raise ValueError(
                    "start_progress is allowed only on transitions and must be in [0, 1)"
                )
        previous_start = start_s
    return document


def build_core_config(scenario: dict[str, Any], source: Path, destination: Path) -> dict[str, Any]:
    """Apply per-episode immutable vehicle scales to the pinned EUFS YAML."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_document = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = scenario["base_vehicle"]
    source_document["inertia"]["m"] *= float(base["mass_scale"])
    source_document["inertia"]["I_z"] *= float(base["inertia_scale"])
    for coefficient in ("A", "B", "C"):
        source_document["tyre"][coefficient] *= float(base[f"pacejka_{coefficient}_scale"])
    for field, value in (
        ("inertia.m", source_document["inertia"]["m"]),
        ("inertia.I_z", source_document["inertia"]["I_z"]),
        ("tyre.A", source_document["tyre"]["A"]),
        ("tyre.B", source_document["tyre"]["B"]),
        ("tyre.C", source_document["tyre"]["C"]),
    ):
        if not isinstance(value, (int, float)) or not value > 0.0:
            raise ValueError(f"generated {field} is not finite and positive")
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(yaml.safe_dump(source_document, sort_keys=False), encoding="utf-8")
    return {
        "scenario_sha256": sha256_file_from_object(scenario),
        "core_source": str(source),
        "core_source_sha256": sha256_file(source),
        "generated_core": str(destination),
        "generated_core_sha256": sha256_file(destination),
    }


def sha256_file_from_object(value: dict[str, Any]) -> str:
    """Hash a scenario canonically, independent of YAML formatting/comments."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    """Create one generated core YAML and JSON provenance manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--core-source", type=Path, default=DEFAULT_CORE)
    arguments = parser.parse_args()
    scenario = load_and_validate_scenario(arguments.scenario.expanduser().resolve())
    output = arguments.output_dir.expanduser().resolve()
    metadata = build_core_config(scenario, arguments.core_source, output / "generated_core.yaml")
    metadata["scenario"] = scenario
    (output / "scenario_provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
