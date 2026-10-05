"""Unit tests for deterministic scenario sampling and SDF rendering."""

from __future__ import annotations

import pytest

from neurogrip_sim import scenario_builder as sb

SYNTHETIC_CHASSIS = (
    "<link name='chassis'>\n"
    "  <inertial>\n"
    "    <mass>1.14395</mass>\n"
    "    <inertia><ixx>0.126164</ixx><iyy>0.416519</iyy><izz>0.481014</izz>"
    "<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>\n"
    "  </inertial>\n"
    "</link>\n"
)


def test_build_scenario_is_deterministic():
    """The same profile and seed must produce identical parameters."""
    profiles = sb.load_profiles()
    first = sb.build_scenario("nominal", profiles["nominal"], 7)
    second = sb.build_scenario("nominal", profiles["nominal"], 7)
    assert first == second


def test_base_profile_values_are_frozen():
    """Historical profiles keep the exact values used by recorded runs."""
    profiles = sb.load_profiles()
    nominal = sb.build_scenario("nominal", profiles["nominal"], 7)
    assert nominal["friction_scale"] == pytest.approx(0.9823832764833162)
    assert nominal["front_lateral_compliance"] == pytest.approx(0.006508491739245019)

    front_limited = sb.build_scenario("front_limited", profiles["front_limited"], 42)
    assert front_limited["friction_scale"] == pytest.approx(0.836731215814946)
    # Opt-in regimes must default to identity for historical profiles.
    assert nominal["steering_delay_s"] == 0.0
    assert nominal["steering_gain"] == 1.0
    assert nominal["mass_scale"] == 1.0
    assert nominal["cg_shift_m"] == 0.0


def test_actuated_profile_samples_visible_parameters():
    """Actuated profiles expose hidden actuator parameters in the manifest."""
    profiles = sb.load_profiles()
    scenario = sb.build_scenario("actuated_grip", profiles["actuated_grip"], 21)
    assert 0.02 <= scenario["steering_delay_s"] <= 0.08
    assert 0.85 <= scenario["steering_gain"] <= 1.15


def test_base_profiles_have_no_grip_transition():
    """Historical profiles must not gain an unlabelled transition."""
    profiles = sb.load_profiles()
    scenario = sb.build_scenario("low_grip", profiles["low_grip"], 7)
    assert scenario["grip_transition"] is None


def test_transitioning_profile_samples_a_transition():
    """Transition profiles expose a deterministic scheduled grip drop."""
    profiles = sb.load_profiles()
    first = sb.build_scenario("transitioning_grip", profiles["transitioning_grip"], 41)
    second = sb.build_scenario("transitioning_grip", profiles["transitioning_grip"], 41)
    assert first["grip_transition"] == second["grip_transition"]
    assert 12.0 <= first["grip_transition"]["start_s"] <= 20.0
    assert 1.5 <= first["grip_transition"]["compliance_scale"] <= 3.0


def test_noisy_profile_samples_imu_noise():
    """Noisy profiles expose a deterministic IMU noise regime."""
    profiles = sb.load_profiles()
    first = sb.build_scenario("noisy_sensors", profiles["noisy_sensors"], 51)
    second = sb.build_scenario("noisy_sensors", profiles["noisy_sensors"], 51)
    assert first["imu_noise"] == second["imu_noise"]
    assert 0.02 <= first["imu_noise"]["stddev"] <= 0.08
    assert 0.0 <= first["imu_noise"]["dropout_probability"] <= 0.05


def test_imu_noise_v2_samples_per_unit_scales():
    """The v2 profile samples angular and linear scales independently."""
    profiles = sb.load_profiles()
    first = sb.build_scenario("imu_noise_v2", profiles["imu_noise_v2"], 61)
    second = sb.build_scenario("imu_noise_v2", profiles["imu_noise_v2"], 61)
    assert first["imu_noise"] == second["imu_noise"]
    noise = first["imu_noise"]
    assert 0.01 <= noise["angular_velocity_stddev_rps"] <= 0.05
    assert 0.10 <= noise["linear_acceleration_stddev_mps2"] <= 0.50
    assert len(noise["angular_velocity_bias_rps"]) == 3


def test_legacy_noisy_profile_keeps_isotropic_schema():
    """The legacy profile still emits a single isotropic stddev."""
    profiles = sb.load_profiles()
    scenario = sb.build_scenario("noisy_sensors", profiles["noisy_sensors"], 51)
    assert "stddev" in scenario["imu_noise"]
    assert "angular_velocity_stddev_rps" not in scenario["imu_noise"]


def test_base_profiles_have_no_imu_noise():
    """Historical profiles must not gain a sensor-noise regime."""
    profiles = sb.load_profiles()
    scenario = sb.build_scenario("nominal", profiles["nominal"], 11)
    assert scenario["imu_noise"] is None


def test_mass_scale_changes_only_chassis_values():
    """Mass scaling multiplies chassis mass and inertia."""
    scaled = sb.apply_chassis_mass_and_cg(SYNTHETIC_CHASSIS, 2.0, 0.0)
    assert "<mass>2.28790000</mass>" in scaled
    assert "<ixx>0.25232800</ixx>" in scaled


def test_cg_shift_inserts_inertial_pose():
    """A non-zero CG shift adds an inertial pose offset."""
    shifted = sb.apply_chassis_mass_and_cg(SYNTHETIC_CHASSIS, 1.0, 0.05)
    assert "<pose>0.05000000 0 0 0 0 0</pose>" in shifted


def test_identity_mass_and_cg_is_unchanged():
    """Identity parameters must not alter a single SDF byte."""
    assert sb.apply_chassis_mass_and_cg(SYNTHETIC_CHASSIS, 1.0, 0.0) == SYNTHETIC_CHASSIS


def test_wheel_normal_force_tracks_mass_scale():
    """The WheelSlip normal force scales with the mass to stay consistent."""
    text = "<wheel_normal_force>22.5</wheel_normal_force>"
    scaled = sb.scale_wheel_normal_force(text, 1.1)
    assert "<wheel_normal_force>24.75000000</wheel_normal_force>" in scaled
