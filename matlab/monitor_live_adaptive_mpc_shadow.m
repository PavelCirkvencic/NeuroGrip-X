function result = monitor_live_adaptive_mpc_shadow(durationS, minimumValidMoves, outputCsvPath)
%MONITOR_LIVE_ADAPTIVE_MPC_SHADOW Run MATLAB Adaptive MPC without publishing.
%
% This is an end-to-end safety gate: C2 ROS matrices -> validation -> physical
% tracking plant -> mpcmoveAdaptive. It deliberately creates no publisher and
% must never be confused with an actuator-in-the-loop controller.

arguments
    durationS (1, 1) double {mustBePositive, mustBeFinite} = 10.0
    minimumValidMoves (1, 1) double {mustBeInteger, mustBePositive} = 20
    outputCsvPath (1, :) char = ''
end

thisFile = mfilename('fullpath');
matlabRoot = fileparts(thisFile);
repositoryRoot = fileparts(matlabRoot);
fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
adapter = neurogrip.Ros2AdaptiveModelAdapter( ...
    '/neurogrip_matlab_adaptive_shadow', 0.20, true);
fileHandle = open_shadow_csv(outputCsvPath);
cleanup = onCleanup(@() release_resources(adapter, fileHandle)); %#ok<NASGU>
controller = [];
controllerState = [];
lastSimTimeS = -Inf;
moves = [];
lastReason = 'no_packets';
clock = tic;
while toc(clock) < durationS
    update = adapter.poll();
    if ~update.valid
        lastReason = update.reason;
        pause(0.01);
        continue;
    end
    if update.tracking.time_s <= lastSimTimeS
        pause(0.005);
        continue;
    end
    if isempty(controller)
        [controller, ~] = neurogrip.create_c0_mpc( ...
            update.tracking.v_x_mps, fit, update.dynamics.sample_time_s);
        controller = neurogrip.configure_shadow_mpc(controller);
        controllerState = mpcstate(controller);
    end
    onlinePlant = neurogrip.online_mpc_plant_from_update(update);
    measuredOutput = [ ...
        update.tracking.e_y_m, ...
        update.tracking.e_psi_rad, ...
        update.tracking.v_y_mps, ...
        update.tracking.yaw_rate_rps ...
    ];
    % The shared tracking contract measures all four controller states and the
    % online model output is identity.  Re-anchor the estimator to that
    % physical state every sample: otherwise an observer that is disconnected
    % from the vehicle can integrate its own shadow commands into a fictitious
    % trajectory and create artificial steering saturation.
    controllerState.Plant = measuredOutput(:);
    % Python C2 applies its delta-rate constraint against patched EUFS wheel
    % telemetry, not against a prior requested command.  The MATLAB shadow
    % must do the same or its optimization problem is physically different.
    controllerState.LastMove = update.applied_steering.steering_rad;
    [move, info] = mpcmoveAdaptive(controller, controllerState, onlinePlant, [], ...
        measuredOutput, zeros(1, 4), update.tracking.d_kappa_rad_s);
    assert(isfinite(move) && strcmp(info.QPCode, 'feasible'), ...
        'Adaptive MPC shadow failed to return a feasible finite steering move.');
    assert(move >= controller.MV.Min - 1e-12 && move <= controller.MV.Max + 1e-12, ...
        'Adaptive MPC shadow returned a steering move outside its magnitude bound.');
    moves(end + 1, 1) = move; %#ok<AGROW>
    if fileHandle >= 0
        fprintf(fileHandle, ['%.9f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,' ...
            '%.17g,%.17g\n'], ...
            update.tracking.time_s, move, update.applied_steering.steering_rad, ...
            update.tracking.e_y_m, ...
            update.tracking.e_psi_rad, update.tracking.v_y_mps, ...
            update.tracking.yaw_rate_rps, update.tracking.v_x_mps, ...
            update.tracking.d_kappa_rad_s, ...
            max(abs(eig(update.dynamics.a_lateral))));
    end
    lastSimTimeS = update.tracking.time_s;
    pause(0.005);
end
result = struct( ...
    'duration_s', durationS, ...
    'valid_moves', numel(moves), ...
    'max_abs_shadow_steering_rad', max_or_nan(abs(moves)), ...
    'last_invalid_reason', lastReason, ...
    'shadow_csv_path', outputCsvPath ...
);
assert(result.valid_moves >= minimumValidMoves, ...
    ['MATLAB Adaptive MPC shadow produced only %d valid moves (need %d); ' ...
    'last reason: %s.'], result.valid_moves, minimumValidMoves, lastReason);
fprintf(['LIVE_ADAPTIVE_MPC_SHADOW_PASS duration_s=%.2f valid_moves=%d ' ...
    'max_abs_steering_rad=%.6f\n'], result.duration_s, result.valid_moves, ...
    result.max_abs_shadow_steering_rad);
end

function fileHandle = open_shadow_csv(outputCsvPath)
fileHandle = -1;
if isempty(outputCsvPath)
    return;
end
parentDirectory = fileparts(outputCsvPath);
if ~isempty(parentDirectory) && ~isfolder(parentDirectory)
    mkdir(parentDirectory);
end
[fileHandle, errorMessage] = fopen(outputCsvPath, 'w');
assert(fileHandle >= 0, 'Could not open shadow CSV %s: %s', ...
    outputCsvPath, errorMessage);
fprintf(fileHandle, ['time_s,shadow_steering_rad,applied_steering_rad,e_y_m,e_psi_rad,v_y_mps,' ...
    'yaw_rate_rps,v_x_mps,d_kappa_rad_s,lateral_spectral_radius\n']);
end

function release_resources(adapter, fileHandle)
if fileHandle >= 0
    fclose(fileHandle);
end
delete(adapter);
end

function value = max_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end
