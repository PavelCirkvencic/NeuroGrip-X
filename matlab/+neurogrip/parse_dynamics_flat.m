function packet = parse_dynamics_flat(values)
%PARSE_DYNAMICS_FLAT Validate the ROS-standard C2 dynamics compatibility topic.
%
% `/neurogrip/dynamics_flat` is deliberately a std_msgs/Float64MultiArray
% mirror of DynamicsModel.msg. It permits MATLAB ROS 2 integration without
% hand-generated custom MATLAB ROS messages. Wire order (Python row-major):
% [schema, Ts, valid, A11, A12, A21, A22, B1, B2,
%  ensemble_std, conformal_radius, ood_score, latency_ms,
%  estimated_front_grip, estimated_rear_grip,
%  front_grip_std, rear_grip_std,
%  front_grip_error_q90, rear_grip_error_q90].
%
% Schema 2/3 legacy packets contain only the first 13 values. Schema 4 must
% contain all 19 values; silently filling missing grip fields would make an
% adaptive speed policy appear valid while actually running on defaults.

values = double(values(:).');
packet = struct( ...
    'valid', false, ...
    'reason', 'unparsed', ...
    'schema_version', uint32(0), ...
    'sample_time_s', NaN, ...
    'a_lateral', nan(2, 2), ...
    'b_lateral', nan(2, 1), ...
    'ensemble_std', NaN, ...
    'conformal_radius', NaN, ...
    'ood_score', NaN, ...
    'latency_ms', NaN, ...
    'estimated_front_grip', NaN, ...
    'estimated_rear_grip', NaN, ...
    'front_grip_std', NaN, ...
    'rear_grip_std', NaN, ...
    'front_grip_error_q90', NaN, ...
    'rear_grip_error_q90', NaN ...
);

if numel(values) < 1 || ~ismember(values(1), [2.0, 3.0, 4.0])
    packet.reason = 'unsupported_dynamics_schema';
    return;
end
schemaVersion = values(1);
expectedLength = 13;
if schemaVersion == 4.0
    expectedLength = 19;
end
if numel(values) ~= expectedLength
    packet.reason = 'invalid_dynamics_length';
    return;
end
if any(~isfinite(values))
    packet.reason = 'non_finite_dynamics';
    return;
end
if abs(values(2) - 0.02) > 1e-6
    packet.reason = 'unexpected_dynamics_sample_time';
    return;
end
if values(3) ~= 1.0
    packet.reason = 'dynamics_marked_invalid';
    return;
end

% Python NumPy ravel is row-major; transpose MATLAB's column-major reshape.
packet.a_lateral = reshape(values(4:7), 2, 2).';
packet.b_lateral = values(8:9).';
packet.schema_version = uint32(schemaVersion);
packet.sample_time_s = values(2);
packet.ensemble_std = values(10);
packet.conformal_radius = values(11);
packet.ood_score = values(12);
packet.latency_ms = values(13);
if schemaVersion == 4.0
    packet.estimated_front_grip = values(14);
    packet.estimated_rear_grip = values(15);
    packet.front_grip_std = values(16);
    packet.rear_grip_std = values(17);
    packet.front_grip_error_q90 = values(18);
    packet.rear_grip_error_q90 = values(19);
    gripValues = values(14:19);
    if packet.estimated_front_grip < 0.35 || packet.estimated_front_grip > 1.30 || ...
            packet.estimated_rear_grip < 0.35 || packet.estimated_rear_grip > 1.30 || ...
            any(gripValues(3:6) < 0.0)
        packet.reason = 'invalid_schema4_grip_fields';
        return;
    end
end
if max(abs(eig(packet.a_lateral))) > 1.10 + 1e-12
    packet.reason = 'unstable_dynamics_matrix';
    return;
end
packet.valid = true;
packet.reason = '';
end
