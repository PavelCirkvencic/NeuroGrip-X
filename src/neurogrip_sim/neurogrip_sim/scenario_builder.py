"""
Generate deterministic Gazebo worlds and manifests for NeuroGrip-X runs.

The scenario label is intentionally *not* sent on ROS topics.  It is stored in
the run manifest only, so a learned model must infer the dynamic context from
the state and command history rather than reading a ground-truth regime label.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory


WHEEL_AXLES = {
    "front_left_wheel": "front",
    "front_right_wheel": "front",
    "rear_left_wheel": "rear",
    "rear_right_wheel": "rear",
}
BASE_CONTACT_FRICTION = 0.5


def parse_arguments() -> argparse.Namespace:
    """Parse the requested profile, deterministic seed and output location."""
    parser = argparse.ArgumentParser(
        description="Generate a seeded NeuroGrip-X Gazebo scenario world."
    )
    parser.add_argument("profile", help="Profile name from scenario_profiles.yaml.")
    parser.add_argument(
        "--seed", type=int, required=True, help="Deterministic scenario seed."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated_worlds"),
        help="Directory for the generated SDF and YAML manifest.",
    )
    return parser.parse_args()


def package_share_path(*parts: str) -> Path:
    """Return an installed NeuroGrip-X package asset path."""
    return Path(get_package_share_directory("neurogrip_sim"), *parts)


def load_profiles() -> dict[str, dict[str, Any]]:
    """Load and minimally validate version-controlled scenario profiles."""
    profiles_path = package_share_path("config", "scenario_profiles.yaml")
    with profiles_path.open(encoding="utf-8") as profiles_file:
        document = yaml.safe_load(profiles_file)
    profiles = document.get("profiles", {}) if isinstance(document, dict) else {}
    if not profiles:
        raise ValueError(f"No profiles defined in {profiles_path}")
    return profiles


def sample_range(generator: random.Random, values: Any, field: str) -> float:
    """Sample one scalar from an inclusive two-value profile range."""
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{field} must be a two-value YAML list.")
    lower, upper = (float(values[0]), float(values[1]))
    if lower < 0.0 or upper < lower:
        raise ValueError(f"{field} must satisfy 0 <= lower <= upper.")
    return generator.uniform(lower, upper)


def sample_optional_range(
    generator: random.Random, values: Any, field: str, default: float
) -> float:
    """
    Sample an optional range without perturbing profiles that omit it.

    Historical profiles define only the four grip parameters.  Skipping the
    RNG call entirely for omitted fields keeps those profiles byte-for-byte
    reproducible while new profiles can opt into extra regimes.
    """
    if values is None:
        return default
    return sample_range(generator, values, field)


def sample_signed_range(generator: random.Random, values: Any, field: str) -> float:
    """Sample a two-value range that may contain negative values."""
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{field} must be a two-value YAML list.")
    lower, upper = (float(values[0]), float(values[1]))
    if upper < lower:
        raise ValueError(f"{field} must satisfy lower <= upper.")
    return generator.uniform(lower, upper)


def replace_tag_in_block(block: str, tag: str, value: float) -> str:
    """Replace one XML tag value in a known wheel block."""
    expression = rf"(<{tag}>)[^<]*(</{tag}>)"
    replacement, replacements = re.subn(
        expression,
        lambda match: f"{match.group(1)}{value:.8f}{match.group(2)}",
        block,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Could not find exactly one <{tag}> in wheel block.")
    return replacement


def replace_wheel_plugin_block(
    sdf_text: str, wheel_name: str, lateral: float, longitudinal: float
) -> str:
    """Update WheelSlip parameters for one named wheel plugin entry."""
    expression = rf'(<wheel link_name="{re.escape(wheel_name)}">.*?</wheel>)'
    match = re.search(expression, sdf_text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"WheelSlip entry missing for {wheel_name}.")
    updated = replace_tag_in_block(match.group(1), "slip_compliance_lateral", lateral)
    updated = replace_tag_in_block(
        updated, "slip_compliance_longitudinal", longitudinal
    )
    return sdf_text[:match.start()] + updated + sdf_text[match.end():]


def replace_wheel_contact_friction(sdf_text: str, wheel_name: str, friction: float) -> str:
    """Update the primary ODE contact friction for one wheel link."""
    expression = (
        rf"(<link name=['\"]{re.escape(wheel_name)}['\"]>.*?"
        rf"<collision name=['\"]collision['\"]>.*?<ode>.*?<mu>)"
        rf"[^<]*(</mu>)"
    )
    replacement, replacements = re.subn(
        expression,
        lambda match: f"{match.group(1)}{friction:.8f}{match.group(2)}",
        sdf_text,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError(f"Could not find contact friction for {wheel_name}.")
    return replacement


def sample_grip_transition(
    generator: random.Random, specification: Any
) -> dict[str, float] | None:
    """
    Sample an optional mid-episode global grip transition.

    The transition is stored only in the manifest.  A scheduler node applies
    it through the Gazebo wheel-slip service, so the learned model observes it
    exclusively as a change in the vehicle's sensor response.
    """
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise ValueError("grip_transition must be a YAML mapping.")
    return {
        "start_s": sample_range(
            generator, specification.get("start_s_range"), "start_s_range"
        ),
        "compliance_scale": sample_range(
            generator,
            specification.get("compliance_scale_range"),
            "compliance_scale_range",
        ),
    }


def sample_imu_noise(
    generator: random.Random, specification: Any
) -> dict[str, Any] | None:
    """
    Sample an optional IMU noise/dropout regime for the manifest.

    Supports the modern per-unit schema (``angular_velocity_stddev_rps`` /
    ``linear_acceleration_stddev_mps2``) and the legacy isotropic ``stddev``
    used by already-recorded scenarios.  Omitting a key consumes no RNG, so
    historical profiles stay byte-for-byte reproducible.
    """
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise ValueError("imu_noise must be a YAML mapping.")

    noise: dict[str, Any] = {}
    if specification.get("stddev_range") is not None:
        noise["stddev"] = sample_range(
            generator, specification["stddev_range"], "stddev_range"
        )
    if specification.get("angular_velocity_stddev_rps_range") is not None:
        noise["angular_velocity_stddev_rps"] = sample_range(
            generator,
            specification["angular_velocity_stddev_rps_range"],
            "angular_velocity_stddev_rps_range",
        )
    if specification.get("linear_acceleration_stddev_mps2_range") is not None:
        noise["linear_acceleration_stddev_mps2"] = sample_range(
            generator,
            specification["linear_acceleration_stddev_mps2_range"],
            "linear_acceleration_stddev_mps2_range",
        )
    if specification.get("dropout_probability_range") is not None:
        noise["dropout_probability"] = sample_range(
            generator,
            specification["dropout_probability_range"],
            "dropout_probability_range",
        )
    for key, field in (
        ("angular_velocity_bias_rps_range", "angular_velocity_bias_rps"),
        ("linear_acceleration_bias_mps2_range", "linear_acceleration_bias_mps2"),
    ):
        if specification.get(key) is not None:
            noise[field] = [
                sample_signed_range(generator, specification[key], key)
                for _ in range(3)
            ]
    for key, field in (
        ("angular_velocity_random_walk_rps_range", "angular_velocity_random_walk_rps"),
        (
            "linear_acceleration_random_walk_mps2_range",
            "linear_acceleration_random_walk_mps2",
        ),
    ):
        if specification.get(key) is not None:
            noise[field] = sample_range(generator, specification[key], key)
    return noise


def apply_chassis_mass_and_cg(
    sdf_text: str, mass_scale: float, cg_shift_m: float
) -> str:
    """Scale the chassis mass/inertia and offset the centre of gravity."""
    if abs(mass_scale - 1.0) < 1e-12 and abs(cg_shift_m) < 1e-12:
        return sdf_text

    link_match = re.search(
        r"<link name=['\"]chassis['\"]>.*?</link>", sdf_text, flags=re.DOTALL
    )
    if link_match is None:
        raise ValueError("chassis link not found.")
    link_block = link_match.group(0)
    inertial_match = re.search(r"<inertial>.*?</inertial>", link_block, flags=re.DOTALL)
    if inertial_match is None:
        raise ValueError("chassis inertial block not found.")
    inertial = inertial_match.group(0)

    mass_pattern = re.compile(r"(<mass>)([^<]*)(</mass>)")
    inertial = mass_pattern.sub(
        lambda match: (
            f"{match.group(1)}{float(match.group(2)) * mass_scale:.8f}{match.group(3)}"
        ),
        inertial,
        count=1,
    )

    for tag in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        tag_pattern = re.compile(rf"(<{tag}>)([^<]*)(</{tag}>)")
        inertial = tag_pattern.sub(
            lambda match: (
                f"{match.group(1)}{float(match.group(2)) * mass_scale:.8f}{match.group(3)}"
            ),
            inertial,
            count=1,
        )

    if abs(cg_shift_m) > 1e-12:
        inertial = inertial.replace(
            "<inertial>",
            f"<inertial>\n          <pose>{cg_shift_m:.8f} 0 0 0 0 0</pose>",
            1,
        )

    updated_link = (
        link_block[:inertial_match.start()] + inertial + link_block[inertial_match.end():]
    )
    return sdf_text[:link_match.start()] + updated_link + sdf_text[link_match.end():]


def scale_wheel_normal_force(sdf_text: str, mass_scale: float) -> str:
    """Scale the WheelSlip assumed wheel load to stay consistent with mass."""
    if abs(mass_scale - 1.0) < 1e-12:
        return sdf_text
    pattern = re.compile(r"(<wheel_normal_force>)([^<]*)(</wheel_normal_force>)")
    return pattern.sub(
        lambda match: (
            f"{match.group(1)}{float(match.group(2)) * mass_scale:.8f}{match.group(3)}"
        ),
        sdf_text,
    )


def build_scenario(profile_name: str, profile: dict[str, Any], seed: int) -> dict[str, Any]:
    """Sample all simulator parameters deterministically from one profile."""
    generator = random.Random(seed)
    scenario = {
        "profile": profile_name,
        "seed": seed,
        "description": str(profile.get("description", "")),
        "friction_scale": sample_range(
            generator, profile.get("friction_scale_range"), "friction_scale_range"
        ),
        "front_lateral_compliance": sample_range(
            generator,
            profile.get("front_lateral_compliance_range"),
            "front_lateral_compliance_range",
        ),
        "rear_lateral_compliance": sample_range(
            generator,
            profile.get("rear_lateral_compliance_range"),
            "rear_lateral_compliance_range",
        ),
        "longitudinal_compliance": sample_range(
            generator,
            profile.get("longitudinal_compliance_range"),
            "longitudinal_compliance_range",
        ),
        # Regime parameters below are opt-in per profile.  They are sampled
        # *after* the historical four so existing profiles stay reproducible.
        "steering_delay_s": sample_optional_range(
            generator,
            profile.get("steering_delay_s_range"),
            "steering_delay_s_range",
            0.0,
        ),
        "steering_gain": sample_optional_range(
            generator,
            profile.get("steering_gain_range"),
            "steering_gain_range",
            1.0,
        ),
        "mass_scale": sample_optional_range(
            generator,
            profile.get("mass_scale_range"),
            "mass_scale_range",
            1.0,
        ),
        "cg_shift_m": (
            sample_signed_range(generator, profile["cg_shift_m_range"], "cg_shift_m_range")
            if profile.get("cg_shift_m_range") is not None
            else 0.0
        ),
        "grip_transition": sample_grip_transition(
            generator, profile.get("grip_transition")
        ),
        "imu_noise": sample_imu_noise(generator, profile.get("imu_noise")),
    }
    return scenario


def render_world(scenario: dict[str, Any]) -> str:
    """Render a scenario-specific SDF without changing the source template."""
    template_path = package_share_path("worlds", "neurogrip_ackermann.sdf")
    sdf_text = template_path.read_text(encoding="utf-8")
    friction = BASE_CONTACT_FRICTION * float(scenario["friction_scale"])
    for wheel_name, axle in WHEEL_AXLES.items():
        lateral = float(scenario[f"{axle}_lateral_compliance"])
        longitudinal = float(scenario["longitudinal_compliance"])
        sdf_text = replace_wheel_plugin_block(
            sdf_text, wheel_name, lateral, longitudinal
        )
        sdf_text = replace_wheel_contact_friction(sdf_text, wheel_name, friction)

    mass_scale = float(scenario.get("mass_scale", 1.0))
    cg_shift_m = float(scenario.get("cg_shift_m", 0.0))
    sdf_text = apply_chassis_mass_and_cg(sdf_text, mass_scale, cg_shift_m)
    sdf_text = scale_wheel_normal_force(sdf_text, mass_scale)
    return sdf_text


def write_scenario(scenario: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write generated SDF and traceability manifest to an ignored directory."""
    scenario_id = f"{scenario['profile']}_seed{scenario['seed']:04d}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    world_path = output_dir / f"{scenario_id}.sdf"
    manifest_path = output_dir / f"{scenario_id}.manifest.yaml"
    world_path.write_text(render_world(scenario), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "generator_version": 2,
        "scenario_id": scenario_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_path": str(world_path),
        "scenario": scenario,
        "label_visibility": "manifest_only_not_published_to_ros",
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        yaml.safe_dump(manifest, manifest_file, sort_keys=False)
    return world_path, manifest_path


def main() -> int:
    """Build one requested scenario and report launch-ready output paths."""
    arguments = parse_arguments()
    try:
        profiles = load_profiles()
        if arguments.profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown profile '{arguments.profile}'. Available: {available}."
            )
        scenario = build_scenario(arguments.profile, profiles[arguments.profile], arguments.seed)
        world_path, manifest_path = write_scenario(scenario, arguments.output_dir)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Scenario build failed: {error}", file=sys.stderr)
        return 1

    print(f"Scenario: {scenario['profile']} (seed {scenario['seed']})")
    print(f"World: {world_path}")
    print(f"Manifest: {manifest_path}")
    print("Launch with: ros2 launch neurogrip_bringup vehicle_sim.launch.py \\")
    print(f"  world_path:=\"{world_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
