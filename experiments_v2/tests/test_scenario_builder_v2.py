"""Regression tests for immutable EUFS v2 scenario materialisation."""

from pathlib import Path

import pytest
import yaml

from experiments_v2.scenario_builder_v2 import build_core_config, load_and_validate_scenario


def scenario() -> dict:
    """Return a smallest valid scenario used by the pure builder tests."""
    return {
        "schema_version": 1,
        "scenario_id": "test",
        "seed": 1,
        "track": "small_track",
        "base_vehicle": {
            "mass_scale": 1.1, "inertia_scale": 1.2, "pacejka_A_scale": 0.9,
            "pacejka_B_scale": 1.0, "pacejka_C_scale": 1.05,
        },
        "actuator": {"steering_gain": 1.0, "steering_delay_s": 0.0},
        "grip_schedule": [{"start_s": 0.0, "front_scale": 1.0, "rear_scale": 1.0}],
        "sensors": {"imu_noise_profile": "nominal_v1"},
    }


def test_build_core_config_scales_only_requested_physical_parameters(tmp_path: Path):
    """Mass, inertia and all Pacejka scales must be visible in the generated YAML."""
    source = tmp_path / "base.yaml"
    source.write_text(
        yaml.safe_dump({"inertia": {"m": 300.0, "I_z": 100.0}, "tyre": {"A": 1.0, "B": 2.0, "C": 3.0}})
    )
    metadata = build_core_config(scenario(), source, tmp_path / "run" / "core.yaml")
    generated = yaml.safe_load(Path(metadata["generated_core"]).read_text())
    assert generated["inertia"] == {"m": 330.0, "I_z": 120.0}
    assert generated["tyre"] == pytest.approx({"A": 0.9, "B": 2.0, "C": 3.15})
    assert len(metadata["generated_core_sha256"]) == 64


def test_scenario_requires_initial_grip_event(tmp_path: Path):
    """A transition-only scenario is ambiguous and must fail before a run starts."""
    bad = scenario()
    bad["grip_schedule"][0]["start_s"] = 1.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="begin"):
        load_and_validate_scenario(path)


def test_transition_can_be_synchronised_to_lap_progress(tmp_path: Path):
    """A transition may use deterministic post-start progress as its trigger."""
    document = scenario()
    document["grip_schedule"].append(
        {
            "start_s": 999.0,
            "start_progress": 0.35,
            "front_scale": 0.6,
            "rear_scale": 0.9,
        }
    )
    path = tmp_path / "progress.yaml"
    path.write_text(yaml.safe_dump(document))
    loaded = load_and_validate_scenario(path)
    assert loaded["grip_schedule"][1]["start_progress"] == pytest.approx(0.35)
