"""Contract tests for the command-mux evidence recorder."""

import rclpy

from experiments_v2.record_command_mux_status import (
    CommandMuxStatusRecorder,
    STATUS_FIELDS,
    STATUS_LENGTH,
    STATUS_SCHEMA,
)


def test_mux_status_schema_is_fixed_and_unique():
    """CSV order must remain synchronized with the ROS numeric frame."""
    assert STATUS_SCHEMA == 1
    assert STATUS_LENGTH == len(STATUS_FIELDS) == 8
    assert len(set(STATUS_FIELDS)) == STATUS_LENGTH
    assert STATUS_FIELDS[2] == "source_id"
    assert STATUS_FIELDS[-2:] == ["fallback_count", "reject_count"]


def test_recorder_does_not_collide_with_rclpy_node_properties(tmp_path):
    """Construction must not overwrite the reserved Node.handle property."""
    if not rclpy.ok():
        rclpy.init()
    node = CommandMuxStatusRecorder(tmp_path / "mux.csv")
    try:
        assert node.rows == 0
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
