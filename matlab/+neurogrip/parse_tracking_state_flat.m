function state = parse_tracking_state_flat(values)
%PARSE_TRACKING_STATE_FLAT Validate the 10-value Python/EUFS tracking schema.
%
% Wire order: [schema, sim_time, e_y, e_psi, v_y, yaw_rate, v_x,
% curvature, d_kappa, lap_progress].  Only this shared schema provides the
% speed needed to lift a C2 lateral matrix into the Adaptive MPC plant.

values = double(values(:).');
state = struct( ...
    'valid', false, ...
    'reason', 'unparsed', ...
    'time_s', NaN, ...
    'e_y_m', NaN, ...
    'e_psi_rad', NaN, ...
    'v_y_mps', NaN, ...
    'yaw_rate_rps', NaN, ...
    'v_x_mps', NaN, ...
    'curvature_1pm', NaN, ...
    'd_kappa_rad_s', NaN, ...
    'lap_progress', NaN ...
);

if numel(values) ~= 10
    state.reason = 'invalid_tracking_length';
    return;
end
if any(~isfinite(values))
    state.reason = 'non_finite_tracking';
    return;
end
if values(1) ~= 1.0
    state.reason = 'unsupported_tracking_schema';
    return;
end
if abs(values(9) - values(7) * values(8)) > 1e-9
    state.reason = 'inconsistent_tracking_d_kappa';
    return;
end
if values(10) < -1e-9 || values(10) > 1.0 + 1e-9
    state.reason = 'invalid_lap_progress';
    return;
end

state.time_s = values(2);
state.e_y_m = values(3);
state.e_psi_rad = values(4);
state.v_y_mps = values(5);
state.yaw_rate_rps = values(6);
state.v_x_mps = max(values(7), 1.0);
state.curvature_1pm = values(8);
state.d_kappa_rad_s = values(9);
state.lap_progress = values(10);
state.valid = true;
state.reason = '';
end
