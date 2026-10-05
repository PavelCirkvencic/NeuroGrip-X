"""Tests for deterministic per-episode simulator configuration."""

from __future__ import annotations

import yaml

from experiments_v2.run_episode import build_seeded_plugin_config


def test_seeded_plugin_config_uses_independent_reproducible_sensor_seeds(tmp_path):
    """A scenario seed reaches both EUFS noisy sensor plugins explicitly."""
    output = build_seeded_plugin_config(923, tmp_path / "plugin.yaml")
    document = yaml.safe_load(output.read_text(encoding="utf-8"))
    plugins = document["eufs_sim2"]["ros__parameters"]["plugin"]
    assert plugins["wheel_speed_plugin"]["noise_seed"] == 923
    assert plugins["imu_plugin"]["noise_seed"] == 100_923
