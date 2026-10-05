"""Build a leakage-safe NeuroGrip-X dataset catalog from processed runs.

Splits are assigned to complete ``scenario_id`` groups, never individual rows
or windows.  The catalog distinguishes five roles:

* ``train``             -- model fitting;
* ``validation``        -- early stopping / architecture selection;
* ``calibration``       -- conformal scale and radius fitting;
* ``development_test``  -- previously reviewed test scenarios (tuning allowed);
* ``sealed_test``       -- final OOD test, only evaluated in the final benchmark.

``sealed_test`` membership comes from a versioned declaration file (default
``config/sealed_test.json``) that records when and at which commit the test was
sealed.  Sealed scenarios are never used for fitting or tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CATALOG_SPLITS = (
    "train",
    "validation",
    "calibration",
    "development_test",
    "sealed_test",
)
LEARNABLE_SPLITS = ("train", "validation", "calibration")
LEGACY_SPLIT_NAMES = {"test": "development_test"}
CATALOG_SCHEMA_VERSION = 2


def parse_arguments() -> argparse.Namespace:
    """Parse catalog locations and split settings."""
    parser = argparse.ArgumentParser(
        description="Create a scenario-grouped NeuroGrip-X dataset manifest."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing *_transitions.metadata.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dataset_manifest.json"),
        help="Output JSON catalog path.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Seed used only for deterministic scenario-group ordering.",
    )
    parser.add_argument(
        "--previous-manifest",
        type=Path,
        help=(
            "Optional earlier catalog whose assignments remain frozen. New "
            "scenarios are added without moving existing scenario roles."
        ),
    )
    parser.add_argument(
        "--sealed-test",
        type=Path,
        default=Path("config/sealed_test.json"),
        help="Versioned sealed-test declaration; sealed scenarios are held out.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    return parser.parse_args()


def validate_fractions(arguments: argparse.Namespace) -> None:
    """Require a valid learnable-split definition."""
    fractions = (
        arguments.train_fraction,
        arguments.validation_fraction,
        arguments.calibration_fraction,
    )
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("All split fractions must be positive.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Learnable split fractions must sum to exactly 1.0.")


def load_sealed_declaration(declaration_path: Path | None) -> dict | None:
    """Load the versioned sealed-test declaration, if present."""
    if declaration_path is None:
        return None
    path = declaration_path.expanduser().resolve()
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid sealed-test declaration: {path}") from error
    scenario_ids = document.get("scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids:
        raise ValueError("Sealed-test declaration needs a non-empty scenario_ids list.")
    return {
        "path": str(path),
        "scenario_ids": sorted(set(scenario_ids)),
        "sealed_at_utc": document.get("sealed_at_utc"),
        "sealed_git_commit": document.get("sealed_git_commit"),
        "description": document.get("description", ""),
    }


def load_run_records(
    processed_dir: Path,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Load only traceable, cleanly completed datasets and their scenarios."""
    records = []
    skipped_untraceable = []
    skipped_incomplete = []
    skipped_unapplied_transition = []
    for metadata_path in sorted(processed_dir.glob("*_transitions.metadata.json")):
        parquet_path = metadata_path.with_name(
            metadata_path.name.replace(".metadata.json", ".parquet")
        )
        if not parquet_path.is_file():
            raise ValueError(f"Missing Parquet dataset for {metadata_path.name}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON: {metadata_path}") from error

        provenance = metadata.get("source_run_provenance")
        manifest = provenance.get("scenario_manifest") if isinstance(provenance, dict) else None
        required = {"scenario_id", "sha256", "scenario"}
        if not isinstance(manifest, dict) or required.difference(manifest):
            # Historical debugging runs remain on disk but are never silently
            # admitted into a benchmark catalog.
            skipped_untraceable.append(metadata_path.name)
            continue

        termination_reason = provenance.get("termination_reason", "unknown")
        if termination_reason != "excitation_complete":
            # A manually interrupted or topology-invalid run is never a
            # benchmark episode, even if it carries a scenario manifest.
            skipped_incomplete.append(f"{metadata_path.name} ({termination_reason})")
            continue

        scenario = manifest["scenario"] if isinstance(manifest["scenario"], dict) else {}
        grip_transition = scenario.get("grip_transition")
        grip_transition_applied = None
        if isinstance(grip_transition, dict):
            status = provenance.get("grip_transition_status")
            if not (isinstance(status, dict) and status.get("applied") is True):
                # An expected grip transition that was never confirmed makes the
                # episode unusable for the benchmark, no matter what the
                # excitation termination reason says.
                skipped_unapplied_transition.append(metadata_path.name)
                continue
            grip_transition_applied = True

        records.append(
            {
                "run_id": Path(metadata["source_csv"]).stem,
                "parquet_path": str(parquet_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "dataset_schema_version": metadata.get("schema_version"),
                "rows": int(metadata.get("transitions", 0)),
                "git_commit": provenance.get("git_commit", "unknown"),
                "excitation_profile": provenance.get("excitation_profile", "unknown"),
                "termination_reason": termination_reason,
                "grip_transition_applied": grip_transition_applied,
                "scenario_id": manifest["scenario_id"],
                "scenario_manifest_sha256": manifest["sha256"],
                "scenario": manifest["scenario"],
            }
        )

    if not records:
        raise ValueError(f"No processed transition metadata found in {processed_dir}")
    return (
        records,
        skipped_untraceable,
        skipped_incomplete,
        skipped_unapplied_transition,
    )


def _ranked(scenario_ids, split_seed: int) -> list[str]:
    """Rank scenario IDs by a deterministic hash."""
    return sorted(
        scenario_ids,
        key=lambda scenario_id: hashlib.sha256(
            f"{split_seed}:{scenario_id}".encode("utf-8")
        ).hexdigest(),
    )


def assign_scenario_splits(
    scenario_ids: set[str],
    sealed_ids: set[str],
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    """Assign whole scenarios to deterministic learnable splits from scratch."""
    assignments = {}
    for scenario_id in sealed_ids & scenario_ids:
        assignments[scenario_id] = "sealed_test"
    remaining = _ranked(scenario_ids - set(assignments), split_seed)
    if len(remaining) < 3:
        raise ValueError(
            "At least three non-sealed scenarios are required for a leakage-safe "
            "train/validation/calibration split."
        )
    total = len(remaining)
    validation_count = max(1, round(total * validation_fraction))
    train_count = max(1, round(total * train_fraction))
    calibration_count = total - train_count - validation_count
    if calibration_count < 1:
        if train_count > 1:
            train_count -= 1
        else:
            validation_count -= 1
        calibration_count = 1
    for scenario_id in remaining[:train_count]:
        assignments[scenario_id] = "train"
    for scenario_id in remaining[train_count : train_count + validation_count]:
        assignments[scenario_id] = "validation"
    for scenario_id in remaining[train_count + validation_count :]:
        assignments[scenario_id] = "calibration"
    return assignments


def load_frozen_assignments(
    previous_manifest_path: Path, scenario_ids: set[str]
) -> dict[str, str]:
    """Read reusable assignments, migrating legacy split names."""
    previous_manifest_path = previous_manifest_path.expanduser().resolve()
    if not previous_manifest_path.is_file():
        raise FileNotFoundError(
            f"Previous dataset manifest does not exist: {previous_manifest_path}"
        )
    try:
        previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {previous_manifest_path}") from error

    previous_splits = previous.get("scenario_splits")
    if not isinstance(previous_splits, dict):
        raise ValueError("Previous dataset manifest has no scenario_splits mapping.")

    assignments = {}
    for split_name, split_ids in previous_splits.items():
        split = LEGACY_SPLIT_NAMES.get(split_name, split_name)
        if split not in CATALOG_SPLITS:
            continue
        if not isinstance(split_ids, list):
            raise ValueError(f"Previous {split_name} split must be a list.")
        for scenario_id in split_ids:
            if scenario_id in assignments:
                raise ValueError(
                    f"Scenario {scenario_id} appears in multiple previous splits."
                )
            if scenario_id in scenario_ids:
                assignments[scenario_id] = split

    # One-time migration: the legacy catalog had no calibration split, so move
    # one validation scenario (deterministically the last) to calibration.
    if "calibration" not in assignments.values():
        validation_ids = sorted(
            scenario_id
            for scenario_id, split in assignments.items()
            if split == "validation"
        )
        if len(validation_ids) > 1:
            moved = validation_ids[-1]
            assignments[moved] = "calibration"
    return assignments


def assign_new_scenarios_with_frozen_splits(
    scenario_ids: set[str],
    frozen_assignments: dict[str, str],
    sealed_ids: set[str],
    split_seed: int,
    fractions: dict[str, float],
) -> dict[str, str]:
    """Add new scenarios while preserving all earlier assignments."""
    assignments = dict(frozen_assignments)
    # Sealed membership always wins over any earlier role.
    for scenario_id in sealed_ids & scenario_ids:
        assignments[scenario_id] = "sealed_test"

    new_ids = _ranked(scenario_ids - set(assignments), split_seed)
    for scenario_id in new_ids:
        counts = {
            split: sum(value == split for value in assignments.values())
            for split in LEARNABLE_SPLITS
        }
        total = sum(counts.values()) + 1
        split = max(
            LEARNABLE_SPLITS,
            key=lambda candidate: (
                fractions[candidate] * total - counts[candidate],
                -LEARNABLE_SPLITS.index(candidate),
            ),
        )
        assignments[scenario_id] = split
    return assignments


def main() -> int:
    """Create and save a catalog with leakage-safe scenario group splits."""
    arguments = parse_arguments()
    try:
        validate_fractions(arguments)
        processed_dir = arguments.processed_dir.expanduser().resolve()
        (
            records,
            skipped_untraceable,
            skipped_incomplete,
            skipped_unapplied_transition,
        ) = load_run_records(processed_dir)
        scenario_ids = {record["scenario_id"] for record in records}

        sealed = load_sealed_declaration(arguments.sealed_test)
        sealed_ids = set(sealed["scenario_ids"]) if sealed else set()
        missing_sealed = sealed_ids - scenario_ids
        if missing_sealed:
            raise ValueError(
                f"Sealed scenarios have no processed runs yet: {sorted(missing_sealed)}"
            )

        fractions = {
            "train": arguments.train_fraction,
            "validation": arguments.validation_fraction,
            "calibration": arguments.calibration_fraction,
        }
        if arguments.previous_manifest:
            frozen_assignments = load_frozen_assignments(
                arguments.previous_manifest, scenario_ids
            )
            assignments = assign_new_scenarios_with_frozen_splits(
                scenario_ids,
                frozen_assignments,
                sealed_ids,
                arguments.split_seed,
                fractions,
            )
            assignment_mode = "frozen_previous_manifest"
            previous_manifest_path = str(
                arguments.previous_manifest.expanduser().resolve()
            )
        else:
            assignments = assign_scenario_splits(
                scenario_ids,
                sealed_ids,
                arguments.split_seed,
                arguments.train_fraction,
                arguments.validation_fraction,
            )
            assignment_mode = "fresh_deterministic"
            previous_manifest_path = None
    except (OSError, ValueError) as error:
        print(f"Dataset manifest failed: {error}", file=sys.stderr)
        return 1

    for record in records:
        record["split"] = assignments[record["scenario_id"]]

    scenario_splits = {split: [] for split in CATALOG_SPLITS}
    for scenario_id, split in assignments.items():
        scenario_splits[split].append(scenario_id)
    for split_ids in scenario_splits.values():
        split_ids.sort()

    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "split_schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_unit": "scenario_id",
        "split_seed": arguments.split_seed,
        "assignment_mode": assignment_mode,
        "previous_manifest_path": previous_manifest_path,
        "fractions": fractions,
        "sealed_test": sealed,
        "scenario_splits": scenario_splits,
        "runs": records,
        "skipped_untraceable_metadata": skipped_untraceable,
        "skipped_incomplete_runs": skipped_incomplete,
        "skipped_unapplied_transition_runs": skipped_unapplied_transition,
    }
    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    print(f"Catalog: {output_path}")
    print(f"Traceable runs: {len(records)}")
    if skipped_untraceable:
        print(f"Skipped untraceable debugging runs: {len(skipped_untraceable)}")
    if skipped_incomplete:
        print(f"Skipped incomplete/invalid runs: {len(skipped_incomplete)}")
    if skipped_unapplied_transition:
        print(
            "Skipped runs with unconfirmed grip transitions: "
            f"{len(skipped_unapplied_transition)}"
        )
    for split, split_ids in scenario_splits.items():
        print(f"{split}: {len(split_ids)} scenario(s), {split_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
