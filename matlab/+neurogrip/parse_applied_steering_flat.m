function packet = parse_applied_steering_flat(values)
%PARSE_APPLIED_STEERING_FLAT Validate measured-wheel telemetry for MATLAB.
%
% Wire order: [schema, sim_time_s, measured_steering_rad].  This packet is
% produced only by the read-only wheel telemetry bridge; it is used as the
% previous manipulated variable in a shadow calculation and never commands a
% vehicle.

values = double(values(:).');
packet = struct( ...
    'valid', false, ...
    'reason', 'unparsed', ...
    'time_s', NaN, ...
    'steering_rad', NaN ...
);
if numel(values) ~= 3
    packet.reason = 'invalid_applied_steering_length';
    return;
end
if any(~isfinite(values))
    packet.reason = 'non_finite_applied_steering';
    return;
end
if values(1) ~= 1.0
    packet.reason = 'unsupported_applied_steering_schema';
    return;
end
if values(2) < 0.0
    packet.reason = 'invalid_applied_steering_time';
    return;
end
if abs(values(3)) > 0.37 + 1e-8
    packet.reason = 'applied_steering_out_of_bounds';
    return;
end
packet.time_s = values(2);
packet.steering_rad = values(3);
packet.valid = true;
packet.reason = '';
end
