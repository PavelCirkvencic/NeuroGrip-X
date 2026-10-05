function result = run_live_matlab_candidate( ...
    maximumDurationS, minimumPublished, outputCsvPath, warmupSolverCalls)
%RUN_LIVE_MATLAB_CANDIDATE Publish lateral candidates, never `/cmd`.
%
% Longitudinal control remains with the Python C2 process. The ROS candidate
% mux accepts this five-value schema only while every packet is fresh, valid,
% in bounds and solved inside the 20 ms controller deadline.

arguments
    maximumDurationS (1, 1) double {mustBePositive, mustBeFinite} = 60.0
    minimumPublished (1, 1) double {mustBeInteger, mustBePositive} = 100
    outputCsvPath (1, :) char = ''
    warmupSolverCalls (1, 1) double {mustBeInteger, mustBeNonnegative} = 20
end

nodeSuffix = char(string(randi([100000, 999999])));
adapter = neurogrip.Ros2MatlabControlAdapter( ...
    ['/neurogrip_matlab_actuation_input_', nodeSuffix], 0.20);
publisherNode = ros2node(['/neurogrip_matlab_candidate_', nodeSuffix]);
publisher = ros2publisher(publisherNode, ...
    '/neurogrip/command_candidate/matlab_flat', ...
    'std_msgs/Float64MultiArray');
fileHandle = open_candidate_csv(outputCsvPath);
cleanup = onCleanup(@() release_resources( ...
    adapter, publisher, publisherNode, fileHandle));
lastControlTimeS = -Inf;
lastNewPacketWallS = 0.0;
published = 0;
invalidSolutions = 0;
deadlineMisses = 0;
solveTimesMs = [];
warmupSolveTimesMs = [];
solverWarmedUp = false;
clock = tic;
while toc(clock) < maximumDurationS
    frame = adapter.poll();
    if ~frame.valid || frame.control_time_s <= lastControlTimeS
        if published > 0 && toc(clock) - lastNewPacketWallS > 2.0
            break;
        end
        pause(0.002);
        continue;
    end
    lastControlTimeS = frame.control_time_s;
    lastNewPacketWallS = toc(clock);
    if ~solverWarmedUp
        assert(frame.solution_valid, ...
            'Cannot warm MATLAB solver from an invalid control frame.');
        for warmupIndex = 1:warmupSolverCalls
            solveClock = tic;
            neurogrip.python_equivalent_mpc_move( ...
                frame.state, frame.curvature_preview, frame.v_x_mps, ...
                frame.a_tracking, frame.b_tracking, frame.sample_time_s, ...
                frame.previous_steering_rad, frame.left_corridor_preview, ...
                frame.right_corridor_preview, 'mpc-active-set');
            warmupSolveTimesMs(end + 1, 1) = 1000.0 * toc(solveClock); %#ok<AGROW>
        end
        solverWarmedUp = true;
        fprintf(['MATLAB_SOLVER_WARMUP_COMPLETE calls=%d first_ms=%.3f ' ...
            'last_ms=%.3f\n'], warmupSolverCalls, ...
            first_or_nan(warmupSolveTimesMs), last_or_nan(warmupSolveTimesMs));
        continue;
    end
    valid = frame.solution_valid;
    steeringRad = 0.0;
    solveTimeMs = 0.0;
    if valid
        try
            solveClock = tic;
            solution = neurogrip.python_equivalent_mpc_move( ...
                frame.state, frame.curvature_preview, frame.v_x_mps, ...
                frame.a_tracking, frame.b_tracking, frame.sample_time_s, ...
                frame.previous_steering_rad, frame.left_corridor_preview, ...
                frame.right_corridor_preview, 'mpc-active-set');
            solveTimeMs = 1000.0 * toc(solveClock);
            steeringRad = solution.first_move_rad;
            valid = isfinite(steeringRad) && abs(steeringRad) <= 0.37 && ...
                solveTimeMs <= 20.0;
        catch
            valid = false;
        end
    end
    if ~valid
        invalidSolutions = invalidSolutions + 1;
    end
    deadlineMisses = deadlineMisses + double(solveTimeMs > 20.0);
    message = ros2message(publisher);
    message.data = [1.0, frame.control_time_s, steeringRad, ...
        double(valid), solveTimeMs];
    send(publisher, message);
    published = published + 1;
    solveTimesMs(end + 1, 1) = solveTimeMs; %#ok<AGROW>
    if fileHandle >= 0
        fprintf(fileHandle, '%.9f,%.17g,%d,%.9f\n', ...
            frame.control_time_s, steeringRad, valid, solveTimeMs);
    end
end

result = struct( ...
    'published', published, ...
    'invalid_solutions', invalidSolutions, ...
    'deadline_misses', deadlineMisses, ...
    'warmup_solver_calls', warmupSolverCalls, ...
    'warmup_max_solve_time_ms', max_or_nan(warmupSolveTimesMs), ...
    'median_solve_time_ms', median_or_nan(solveTimesMs), ...
    'p95_solve_time_ms', percentile_or_nan(solveTimesMs, 95), ...
    'max_solve_time_ms', max_or_nan(solveTimesMs), ...
    'output_csv_path', outputCsvPath ...
);
validPublished = result.published - result.invalid_solutions;
assert(validPublished >= minimumPublished, ...
    'MATLAB candidate published only %d valid packets (need %d).', ...
    validPublished, minimumPublished);
fprintf(['LIVE_MATLAB_CANDIDATE_COMPLETE published=%d valid=%d ' ...
    'median_solve_ms=%.3f p95_solve_ms=%.3f max_solve_ms=%.3f ' ...
    'deadline_misses=%d\n'], ...
    result.published, validPublished, result.median_solve_time_ms, ...
    result.p95_solve_time_ms, result.max_solve_time_ms, ...
    result.deadline_misses);
end

function value = first_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = values(1);
end
end

function value = last_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = values(end);
end
end

function fileHandle = open_candidate_csv(outputCsvPath)
fileHandle = -1;
if isempty(outputCsvPath)
    return;
end
parentDirectory = fileparts(outputCsvPath);
if ~isempty(parentDirectory) && ~isfolder(parentDirectory)
    mkdir(parentDirectory);
end
[fileHandle, errorMessage] = fopen(outputCsvPath, 'w');
assert(fileHandle >= 0, 'Could not open MATLAB candidate CSV %s: %s', ...
    outputCsvPath, errorMessage);
fprintf(fileHandle, 'time_s,steering_rad,solution_valid,solve_time_ms\n');
end

function release_resources(adapter, publisher, publisherNode, fileHandle) %#ok<INUSD>
if fileHandle >= 0
    fclose(fileHandle);
end
delete(adapter);
clear publisher publisherNode
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

function value = max_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end
