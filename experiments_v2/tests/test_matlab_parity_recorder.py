"""Contract tests for the read-only MATLAB parity capture."""

from experiments_v2.record_matlab_parity import (
    FRAME_FIELDS,
    FRAME_LENGTH,
    FRAME_SCHEMA,
    matrix_fields,
)


def test_matlab_parity_recorder_has_fixed_unique_schema():
    """CSV order must stay identical to the controller's 144-value frame."""
    assert FRAME_SCHEMA == 3
    assert FRAME_LENGTH == len(FRAME_FIELDS) == 155
    assert len(set(FRAME_FIELDS)) == FRAME_LENGTH
    assert FRAME_FIELDS[:3] == [
        "schema_version",
        "control_time_s",
        "sample_time_s",
    ]
    assert FRAME_FIELDS[-2:] == ["first_move_rad", "solution_valid"]
    assert matrix_fields("a", 2, 2) == ["a_11", "a_12", "a_21", "a_22"]
