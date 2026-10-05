"""
Cross-package serialization test for the NeuroGrip-X interfaces.

These messages are the wire contract between the AI node, the Python/Simulink
controllers and the loggers, so an explicit CDR round trip guards the schema.
"""

from __future__ import annotations

import pytest
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message, serialize_message

from neurogrip_interfaces.msg import DynamicsModel, SafetyStatus, VehicleState


def round_trip(message):
    """Serialize and deserialize one message, returning the copy."""
    return deserialize_message(serialize_message(message), type(message))


def test_vehicle_state_round_trip():
    """VehicleState v2 survives a full CDR round trip."""
    message = VehicleState()
    message.header.stamp = Time(sec=12, nanosec=345)
    message.header.frame_id = "base_footprint"
    message.v_x_mps = 5.4
    message.v_y_mps = 0.21
    message.yaw_rate_rps = 0.35
    message.steering_applied_rad = 0.0468
    message.acceleration_command_mps2 = 1.5
    message.source = 0
    copy = round_trip(message)
    assert copy.header.frame_id == "base_footprint"
    assert copy.header.stamp.sec == 12
    assert copy.v_y_mps == pytest.approx(0.21)
    assert copy.steering_applied_rad == pytest.approx(0.0468)
    assert copy.source == 0


def test_dynamics_model_round_trip():
    """Grip-aware DynamicsModel physical matrices survive serialization."""
    message = DynamicsModel()
    message.header.frame_id = "base_footprint"
    message.schema_version = 4
    message.artifact_sha256 = "deadbeef"
    message.sample_time_s = 0.02
    message.state_dim = 2
    message.input_dim = 1
    message.state_names = ["v_y_mps", "yaw_rate_rps"]
    message.input_names = ["steering_angle_rad"]
    message.a_matrix = [0.9, 0.01, 0.02, 0.95]
    message.b_matrix = [0.3, 1.1]
    message.context = [0.1, 0.2]
    message.ensemble_std = 0.01
    message.conformal_radius = 0.5
    message.ood_score = 1.2
    message.latency_ms = 1.7
    message.estimated_front_grip = 0.82
    message.estimated_rear_grip = 0.67
    message.front_grip_std = 0.02
    message.rear_grip_std = 0.03
    message.front_grip_error_q90 = 0.06
    message.rear_grip_error_q90 = 0.13
    message.valid = True
    copy = round_trip(message)
    assert copy.schema_version == 4
    assert copy.sample_time_s == pytest.approx(0.02)
    assert list(copy.state_names) == ["v_y_mps", "yaw_rate_rps"]
    assert list(copy.a_matrix) == pytest.approx([0.9, 0.01, 0.02, 0.95])
    assert list(copy.b_matrix) == pytest.approx([0.3, 1.1])
    assert copy.estimated_front_grip == pytest.approx(0.82)
    assert copy.estimated_rear_grip == pytest.approx(0.67)
    assert copy.rear_grip_error_q90 == pytest.approx(0.13)
    assert copy.valid is True


def test_safety_status_round_trip():
    """SafetyStatus reports the same health fields after a round trip."""
    message = SafetyStatus()
    message.healthy = False
    message.state = "out_of_distribution"
    message.detail = "context distance above threshold"
    message.uncertainty_radius = 3.5
    message.ood_score = 9.9
    copy = round_trip(message)
    assert copy.healthy is False
    assert copy.state == "out_of_distribution"
    assert copy.ood_score == pytest.approx(9.9)
