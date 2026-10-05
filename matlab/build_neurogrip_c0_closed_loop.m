function modelPath = build_neurogrip_c0_closed_loop(outputDirectory, vxMps)
%BUILD_NEUROGRIP_C0_CLOSED_LOOP Generate an executable nominal-MPC Simulink model.
%
% The model has the same four physical tracking states, steering/rate limits,
% prediction horizon and measured curvature disturbance as the MATLAB C0 MPC
% object. It is intentionally a self-contained fixed-speed closed-loop smoke
% model; ROS 2 bus I/O and C2 online A/B updates are later integration layers.

arguments
    outputDirectory (1, :) char {mustBeNonzeroLengthText} = ''
    vxMps (1, 1) double {mustBeFinite, mustBePositive} = 4.5
end

thisFile = mfilename('fullpath');
matlabRoot = fileparts(thisFile);
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
if isempty(outputDirectory)
    outputDirectory = fullfile(matlabRoot, 'models');
end
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end

fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
[controller, ~] = neurogrip.create_c0_mpc(vxMps, fit, 0.02);
[aTracking, bTracking] = neurogrip.build_tracking_plant(vxMps, fit, 0.02);
modelName = 'neurogrip_c0_closed_loop';
modelPath = fullfile(outputDirectory, [modelName, '.slx']);

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
cleanup = onCleanup(@() close_loaded_model(modelName));
set_param(modelName, ...
    'Solver', 'FixedStepDiscrete', ...
    'FixedStep', '0.02', ...
    'StopTime', '0.4', ...
    'ReturnWorkspaceOutputs', 'on');
workspace = get_param(modelName, 'ModelWorkspace');
assignin(workspace, 'c0Mpc', controller);

add_block('mpclib/MPC Controller', [modelName, '/C0 fixed MPC'], ...
    'mpcobj', 'c0Mpc', 'Position', [375, 85, 525, 185]);
add_block('simulink/Discrete/Discrete State-Space', [modelName, '/Python-compatible tracking plant'], ...
    'A', mat2str(aTracking, 17), ...
    'B', mat2str(bTracking, 17), ...
    'C', 'eye(4)', ...
    'D', 'zeros(4, 2)', ...
    'X0', '[0.20; 0.05; 0.00; 0.00]', ...
    'SampleTime', '0.02', ...
    'Position', [690, 80, 850, 190]);
add_block('simulink/Signal Routing/Mux', [modelName, '/Plant input mux'], ...
    'Inputs', '2', 'Position', [595, 91, 620, 139]);
add_block('simulink/Sources/Constant', [modelName, '/Zero reference'], ...
    'Value', '[0; 0; 0; 0]', 'Position', [85, 145, 180, 175]);
add_block('simulink/Sources/Constant', [modelName, '/Zero curvature disturbance'], ...
    'Value', '0', 'Position', [85, 240, 180, 270]);
add_block('simulink/Sinks/To Workspace', [modelName, '/Tracking state log'], ...
    'VariableName', 'tracking_state', 'SaveFormat', 'Structure With Time', ...
    'Position', [930, 105, 1045, 135]);
add_block('simulink/Sinks/To Workspace', [modelName, '/Steering log'], ...
    'VariableName', 'steering_move', 'SaveFormat', 'Structure With Time', ...
    'Position', [555, 35, 665, 65]);

add_line(modelName, 'Zero reference/1', 'C0 fixed MPC/2', 'autorouting', 'on');
add_line(modelName, 'Zero curvature disturbance/1', 'C0 fixed MPC/3', 'autorouting', 'on');
add_line(modelName, 'C0 fixed MPC/1', 'Plant input mux/1', 'autorouting', 'on');
add_line(modelName, 'Zero curvature disturbance/1', 'Plant input mux/2', 'autorouting', 'on');
add_line(modelName, 'Plant input mux/1', 'Python-compatible tracking plant/1', 'autorouting', 'on');
add_line(modelName, 'Python-compatible tracking plant/1', 'C0 fixed MPC/1', 'autorouting', 'on');
add_line(modelName, 'Python-compatible tracking plant/1', 'Tracking state log/1', 'autorouting', 'on');
add_line(modelName, 'C0 fixed MPC/1', 'Steering log/1', 'autorouting', 'on');

save_system(modelName, modelPath);
end

function close_loaded_model(modelName)
% Close only the temporary programmatic model; never close a user's model.
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
end
