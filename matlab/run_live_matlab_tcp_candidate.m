function result = run_live_matlab_tcp_candidate( ...
    tcpPort, maximumDurationS, minimumPublished, outputCsvPath, ...
    readyFilePath, warmupSolverCalls)
%RUN_LIVE_MATLAB_TCP_CANDIDATE Solve live QPs over a local text protocol.
%
% The TCP boundary prevents MATLAB's newer bundled ROS 2 runtime from joining
% the Humble DDS graph. A transport-only Python node converts the same 125- and
% 5-value contracts to/from ROS; safety selection remains in candidate_mux.

arguments
    tcpPort (1, 1) double {mustBeInteger, mustBePositive} = 55980
    maximumDurationS (1, 1) double {mustBePositive, mustBeFinite} = 180.0
    minimumPublished (1, 1) double {mustBeInteger, mustBePositive} = 100
    outputCsvPath (1, :) char = ''
    readyFilePath (1, :) char = ''
    warmupSolverCalls (1, 1) double {mustBeInteger, mustBeNonnegative} = 20
end

server = javaObject('java.net.ServerSocket', tcpPort);
server.setReuseAddress(true);
write_ready_file(readyFilePath, tcpPort);
fileHandle = open_candidate_csv(outputCsvPath);
client = [];
reader = [];
writer = [];
cleanup = onCleanup(@() release_resources( ...
    server, client, reader, writer, fileHandle, readyFilePath));
fprintf('MATLAB_TCP_CANDIDATE_READY port=%d\n', tcpPort);
client = server.accept();
client.setTcpNoDelay(true);
client.setSoTimeout(250);
reader = javaObject('java.io.BufferedReader', javaObject( ...
    'java.io.InputStreamReader', client.getInputStream()));
writer = javaObject('java.io.PrintWriter', client.getOutputStream(), true);

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
    try
        line = reader.readLine();
    catch error
        if contains(error.message, 'timed out', 'IgnoreCase', true)
            if published > 0 && toc(clock) - lastNewPacketWallS > 2.0
                break;
            end
            continue;
        end
        rethrow(error);
    end
    if isempty(line)
        break;
    end
    while reader.ready()
        newerLine = reader.readLine();
        if isempty(newerLine)
            break;
        end
        line = newerLine;
    end
    values = sscanf(char(line), '%f,').';
    frame = neurogrip.parse_matlab_control_frame(values);
    if ~frame.valid || frame.control_time_s <= lastControlTimeS
        continue;
    end
    lastControlTimeS = frame.control_time_s;
    lastNewPacketWallS = toc(clock);
    if ~solverWarmedUp
        assert(frame.solution_valid, ...
            'Cannot warm MATLAB solver from an invalid control frame.');
        for warmupIndex = 1:warmupSolverCalls
            solveClock = tic;
            solve_frame(frame);
            warmupSolveTimesMs(warmupIndex, 1) = ...
                1000.0 * toc(solveClock); %#ok<AGROW>
        end
        solverWarmedUp = true;
        fprintf(['MATLAB_TCP_SOLVER_WARMUP_COMPLETE calls=%d ' ...
            'first_ms=%.3f last_ms=%.3f\n'], warmupSolverCalls, ...
            first_or_nan(warmupSolveTimesMs), last_or_nan(warmupSolveTimesMs));
    end

    valid = frame.solution_valid;
    steeringRad = 0.0;
    solveTimeMs = 0.0;
    if valid
        try
            solveClock = tic;
            solution = solve_frame(frame);
            solveTimeMs = 1000.0 * toc(solveClock);
            steeringRad = solution.first_move_rad;
            valid = isfinite(steeringRad) && abs(steeringRad) <= 0.37 && ...
                solveTimeMs <= 20.0;
        catch
            valid = false;
        end
    end
    invalidSolutions = invalidSolutions + double(~valid);
    deadlineMisses = deadlineMisses + double(solveTimeMs > 20.0);
    response = sprintf('1,%.9f,%.17g,%d,%.9f', ...
        frame.control_time_s, steeringRad, valid, solveTimeMs);
    writer.println(response);
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
    'MATLAB TCP candidate published only %d valid packets (need %d).', ...
    validPublished, minimumPublished);
fprintf(['LIVE_MATLAB_TCP_CANDIDATE_COMPLETE published=%d valid=%d ' ...
    'median_solve_ms=%.3f p95_solve_ms=%.3f max_solve_ms=%.3f ' ...
    'deadline_misses=%d\n'], result.published, validPublished, ...
    result.median_solve_time_ms, result.p95_solve_time_ms, ...
    result.max_solve_time_ms, result.deadline_misses);
end

function solution = solve_frame(frame)
solution = neurogrip.latency_compensated_mpc_move( ...
    frame, 'mpc-active-set');
end

function write_ready_file(path, tcpPort)
if isempty(path)
    return;
end
parentDirectory = fileparts(path);
if ~isempty(parentDirectory) && ~isfolder(parentDirectory)
    mkdir(parentDirectory);
end
[handle, errorMessage] = fopen(path, 'w');
assert(handle >= 0, 'Could not create ready file %s: %s', path, errorMessage);
fprintf(handle, '%d\n', tcpPort);
fclose(handle);
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

function release_resources( ...
    server, client, reader, writer, fileHandle, readyFilePath)
if fileHandle >= 0
    fclose(fileHandle);
end
try
    if ~isempty(reader)
        reader.close();
    end
    if ~isempty(writer)
        writer.close();
    end
    if ~isempty(client)
        client.close();
    end
    server.close();
catch
end
if ~isempty(readyFilePath) && isfile(readyFilePath)
    delete(readyFilePath);
end
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
