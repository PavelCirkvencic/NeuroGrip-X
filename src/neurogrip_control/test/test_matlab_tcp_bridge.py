"""Protocol tests for the transport-only MATLAB TCP bridge."""

from __future__ import annotations

import math
import socket

import pytest

from neurogrip_control.matlab_tcp_bridge import (
    CONTROL_FRAME_LENGTH,
    decode_candidate,
    encode_control_frame,
)


def test_control_frame_round_trip_preserves_all_values():
    """The text boundary retains all 125 finite double values."""
    values = [1.0, *[index / 17.0 for index in range(CONTROL_FRAME_LENGTH - 1)]]
    decoded = [float(field) for field in encode_control_frame(values).split(b",")]
    assert decoded == values


@pytest.mark.parametrize(
    "values",
    (
        [1.0] * (CONTROL_FRAME_LENGTH - 1),
        [2.0] + [1.0] * (CONTROL_FRAME_LENGTH - 1),
        [1.0, math.nan] + [1.0] * (CONTROL_FRAME_LENGTH - 2),
    ),
)
def test_invalid_control_frame_fails_before_transport(values):
    """Malformed controller input never reaches MATLAB."""
    with pytest.raises(ValueError):
        encode_control_frame(values)


def test_candidate_decoder_accepts_transport_contract():
    """One complete MATLAB response decodes without changing its values."""
    assert decode_candidate(b"1,12.5,-0.123,1,3.75\n") == [
        1.0,
        12.5,
        -0.123,
        1.0,
        3.75,
    ]


@pytest.mark.parametrize(
    "line",
    (b"1,2,3,4", b"not,numeric,data,at,all", b"1,2,nan,1,4"),
)
def test_invalid_candidate_is_not_published(line):
    """Malformed responses fail before the existing safety mux."""
    with pytest.raises(ValueError):
        decode_candidate(line)


def test_one_in_flight_protocol_has_no_unbounded_request_queue():
    """A replacement frame is sent only after the current response arrives."""
    left, right = socket.socketpair()
    try:
        first = encode_control_frame(
            [1.0, *[1.0] * (CONTROL_FRAME_LENGTH - 1)]
        )
        latest = encode_control_frame(
            [1.0, *[2.0] * (CONTROL_FRAME_LENGTH - 1)]
        )
        left.sendall(first)
        assert right.recv(len(first)) == first
        left.sendall(latest)
        assert right.recv(len(latest)) == latest
    finally:
        left.close()
        right.close()
