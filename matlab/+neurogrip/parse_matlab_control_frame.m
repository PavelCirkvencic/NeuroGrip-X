function frame = parse_matlab_control_frame(values)
%PARSE_MATLAB_CONTROL_FRAME Decode the continuous 125-value QP contract.

values = double(values(:).');
frame = struct('valid', false, 'reason', 'unparsed');
if numel(values) ~= 125
    frame.reason = 'invalid_control_frame_length';
    return;
end
if any(~isfinite(values))
    frame.reason = 'non_finite_control_frame';
    return;
end
if values(1) ~= 1.0
    frame.reason = 'unsupported_control_frame_schema';
    return;
end
if abs(values(3) - 0.02) > 1e-6 || values(4) <= 0.0
    frame.reason = 'invalid_control_frame_timing';
    return;
end
if ~ismember(values(125), [0.0, 1.0])
    frame.reason = 'invalid_control_solution_flag';
    return;
end

frame.schema_version = uint32(values(1));
frame.control_time_s = values(2);
frame.sample_time_s = values(3);
frame.v_x_mps = values(4);
frame.previous_steering_rad = values(5);
frame.state = values(6:9).';
frame.a_tracking = reshape(values(10:25), 4, 4).';
frame.b_tracking = reshape(values(26:33), 2, 4).';
frame.curvature_preview = values(34:63).';
frame.left_corridor_preview = values(64:93).';
frame.right_corridor_preview = values(94:123).';
frame.first_move_rad = values(124);
frame.solution_valid = logical(values(125));
frame.valid = true;
frame.reason = '';
end
