function frame = parse_matlab_parity_frame(values)
%PARSE_MATLAB_PARITY_FRAME Decode the exact read-only C2 decision snapshot.

values = double(values(:).');
frame = struct('valid', false, 'reason', 'unparsed');
if numel(values) ~= 155
    frame.reason = 'invalid_parity_frame_length';
    return;
end
if any(~isfinite(values))
    frame.reason = 'non_finite_parity_frame';
    return;
end
if values(1) ~= 3.0
    frame.reason = 'unsupported_parity_frame_schema';
    return;
end
if abs(values(3) - 0.02) > 1e-6
    frame.reason = 'unexpected_parity_sample_time';
    return;
end
if ~ismember(values(155), [0.0, 1.0])
    frame.reason = 'invalid_parity_solution_flag';
    return;
end

frame.schema_version = uint32(values(1));
frame.control_time_s = values(2);
frame.sample_time_s = values(3);
frame.raw_grip = values(4:5).';
frame.raw_grip_std = values(6:7).';
frame.raw_grip_error_q90 = values(8:9).';
frame.filtered_grip = values(10:11).';
frame.held_grip_std = values(12:13).';
frame.held_grip_error_q90 = values(14:15).';
frame.conservative_grip = values(16:17).';
frame.effective_grip = values(18);
frame.profile_utilisation = values(19);
frame.grip_error_margin_scale = values(20);
frame.target_speed_mps = values(21);
frame.v_x_mps = values(22);
frame.lap_progress = values(23);
frame.maximum_speed_mps = values(24);
frame.minimum_speed_mps = values(25);
frame.acceleration_limit_mps2 = values(26);
frame.braking_limit_mps2 = values(27);
frame.model_blend = values(28);
frame.previous_steering_rad = values(29);
frame.state = values(30:33).';
frame.raw_a_lateral = reshape(values(34:37), 2, 2).';
frame.raw_b_lateral = values(38:39).';
frame.a_tracking = reshape(values(40:55), 4, 4).';
frame.b_tracking = reshape(values(56:63), 2, 4).';
frame.curvature_preview = values(64:93).';
frame.left_corridor_preview = values(94:123).';
frame.right_corridor_preview = values(124:153).';
frame.first_move_rad = values(154);
frame.solution_valid = logical(values(155));
frame.valid = true;
frame.reason = '';
end
