function result = monitor_live_python_equivalent_shadow( ...
    durationS, minimumValidMoves, outputCsvPath)
%MONITOR_LIVE_PYTHON_EQUIVALENT_SHADOW Recompute live C2 MPC read-only.
%
% This monitor consumes the self-contained schema-3 parity frame, solves the
% release Python MPC equations independently with MATLAB quadprog and compares
% the first move. It creates no ROS publisher and has no actuator authority.

arguments
    durationS (1, 1) double {mustBePositive, mustBeFinite} = 10.0
    minimumValidMoves (1, 1) double {mustBeInteger, mustBePositive} = 20
    outputCsvPath (1, :) char = ''
end

adapter = neurogrip.Ros2MatlabParityAdapter( ...
    '/neurogrip_matlab_exact_shadow', 0.20);
fileHandle = open_shadow_csv(outputCsvPath);
cleanup = onCleanup(@() release_resources(adapter, fileHandle));
lastControlTimeS = -Inf;
validMoves = 0;
deadlineMisses = 0;
signMismatches = 0;
errorsRad = [];
solveTimesMs = [];
lastReason = 'no_packets';
clock = tic;
while toc(clock) < durationS
    frame = adapter.poll();
    if ~frame.valid
        lastReason = frame.reason;
        pause(0.005);
        continue;
    end
    if frame.control_time_s <= lastControlTimeS
        pause(0.002);
        continue;
    end
    lastControlTimeS = frame.control_time_s;
    if ~frame.solution_valid
        lastReason = 'python_solution_invalid';
        continue;
    end

    solveClock = tic;
    solution = neurogrip.python_equivalent_mpc_move( ...
        frame.state, frame.curvature_preview, frame.v_x_mps, ...
        frame.a_tracking, frame.b_tracking, frame.sample_time_s, ...
        frame.previous_steering_rad, frame.left_corridor_preview, ...
        frame.right_corridor_preview, 'mpc-active-set');
    solveTimeMs = 1000.0 * toc(solveClock);
    errorRad = solution.first_move_rad - frame.first_move_rad;
    validMoves = validMoves + 1;
    errorsRad(end + 1, 1) = errorRad; %#ok<AGROW>
    solveTimesMs(end + 1, 1) = solveTimeMs; %#ok<AGROW>
    deadlineMisses = deadlineMisses + double( ...
        solveTimeMs > 1000.0 * frame.sample_time_s);
    if abs(solution.first_move_rad) > 1e-5 && ...
            abs(frame.first_move_rad) > 1e-5 && ...
            sign(solution.first_move_rad) ~= sign(frame.first_move_rad)
        signMismatches = signMismatches + 1;
    end
    if fileHandle >= 0
        fprintf(fileHandle, '%.9f,%.17g,%.17g,%.17g,%.9f,%d\n', ...
            frame.control_time_s, solution.first_move_rad, ...
            frame.first_move_rad, abs(errorRad), solveTimeMs, ...
            solveTimeMs > 1000.0 * frame.sample_time_s);
    end
end

result = struct( ...
    'duration_s', durationS, ...
    'valid_moves', validMoves, ...
    'median_abs_error_rad', median_or_nan(abs(errorsRad)), ...
    'max_abs_error_rad', max_or_nan(abs(errorsRad)), ...
    'median_solve_time_ms', median_or_nan(solveTimesMs), ...
    'p95_solve_time_ms', percentile_or_nan(solveTimesMs, 95), ...
    'deadline_misses', deadlineMisses, ...
    'sign_mismatches', signMismatches, ...
    'last_invalid_reason', lastReason, ...
    'shadow_csv_path', outputCsvPath ...
);
assert(result.valid_moves >= minimumValidMoves, ...
    ['MATLAB exact shadow produced only %d valid moves (need %d); ' ...
    'last reason: %s.'], result.valid_moves, minimumValidMoves, lastReason);
assert(result.max_abs_error_rad < 1e-3 && result.sign_mismatches == 0, ...
    ['MATLAB exact shadow diverged from the release MPC: max error %.6g rad, ' ...
    'sign mismatches %d.'], result.max_abs_error_rad, result.sign_mismatches);
fprintf(['LIVE_PYTHON_EQUIVALENT_SHADOW_PASS duration_s=%.2f valid_moves=%d ' ...
    'median_error_rad=%.3e max_error_rad=%.3e median_solve_ms=%.3f ' ...
    'p95_solve_ms=%.3f deadline_misses=%d\n'], ...
    result.duration_s, result.valid_moves, result.median_abs_error_rad, ...
    result.max_abs_error_rad, result.median_solve_time_ms, ...
    result.p95_solve_time_ms, result.deadline_misses);
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
assert(fileHandle >= 0, 'Could not open exact shadow CSV %s: %s', ...
    outputCsvPath, errorMessage);
fprintf(fileHandle, ['time_s,matlab_steering_rad,python_steering_rad,' ...
    'abs_error_rad,solve_time_ms,deadline_miss\n']);
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

function value = median_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = median(values);
end
end

function value = percentile_or_nan(values, percentile)
if isempty(values)
    value = NaN;
else
    value = prctile(values, percentile);
end
end
