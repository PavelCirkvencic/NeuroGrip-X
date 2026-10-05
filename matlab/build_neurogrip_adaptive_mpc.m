function modelPath = build_neurogrip_adaptive_mpc(outputDirectory, vxMps)
%BUILD_NEUROGRIP_ADAPTIVE_MPC Generate a headless Adaptive MPC bus prototype.
%
% This is deliberately a static-nominal online-model smoke model: the model
% bus is structurally identical to the future ROS C2 update, but its matrices
% are the nominal physical model.  It validates the hard Simulink integration
% point before ROS 2 transport is added.  It is not an adaptive-performance
% result and does not claim equality to the custom Python OSQP controller.

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
onlineModel = neurogrip.make_adaptive_model_bus(aTracking, bTracking);
modelName = 'neurogrip_adaptive_mpc_closed_loop';
modelPath = fullfile(outputDirectory, [modelName, '.slx']);

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
cleanup = onCleanup(@() close_loaded_model(modelName)); %#ok<NASGU>
set_param(modelName, ...
    'Solver', 'FixedStepDiscrete', ...
    'FixedStep', '0.02', ...
    'StopTime', '0.4', ...
    'ReturnWorkspaceOutputs', 'on');
workspace = get_param(modelName, 'ModelWorkspace');
assignin(workspace, 'adaptiveMpc', controller);
assignin(workspace, 'onlineA', onlineModel.A);
assignin(workspace, 'onlineB', onlineModel.B);
assignin(workspace, 'onlineC', onlineModel.C);
assignin(workspace, 'onlineD', onlineModel.D);
assignin(workspace, 'onlineX', onlineModel.X);
assignin(workspace, 'onlineY', onlineModel.Y);
assignin(workspace, 'onlineU', onlineModel.U);
assignin(workspace, 'onlineDX', onlineModel.DX);

add_block('mpclib/Adaptive MPC Controller', [modelName, '/Adaptive MPC'], ...
    'mpcobj', 'adaptiveMpc', ...
    'isltv_plant', 'on', ...
    'Position', [505, 100, 655, 205]);
add_block('simulink/Discrete/Discrete State-Space', [modelName, '/Tracking plant'], ...
    'A', mat2str(aTracking, 17), ...
    'B', mat2str(bTracking, 17), ...
    'C', 'eye(4)', ...
    'D', 'zeros(4, 2)', ...
    'X0', '[0.20; 0.05; 0.00; 0.00]', ...
    'SampleTime', '0.02', ...
    'Position', [805, 95, 965, 205]);
add_block('simulink/Signal Routing/Mux', [modelName, '/Plant input mux'], ...
    'Inputs', '2', 'Position', [720, 106, 745, 154]);
add_block('simulink/Sources/Constant', [modelName, '/Zero reference'], ...
    'Value', '[0; 0; 0; 0]', 'Position', [280, 200, 375, 230]);
add_block('simulink/Sources/Constant', [modelName, '/Zero curvature disturbance'], ...
    'Value', '0', 'Position', [280, 275, 375, 305]);
add_block('simulink/Sinks/To Workspace', [modelName, '/Tracking state log'], ...
    'VariableName', 'adaptive_tracking_state', 'SaveFormat', 'Structure With Time', ...
    'Position', [1035, 120, 1165, 150]);
add_block('simulink/Sinks/To Workspace', [modelName, '/Steering log'], ...
    'VariableName', 'adaptive_steering_move', 'SaveFormat', 'Structure With Time', ...
    'Position', [675, 45, 795, 75]);

% The eight named constant signals form the documented online-model bus.
fieldNames = {'A', 'B', 'C', 'D', 'X', 'Y', 'U', 'DX'};
valueNames = {'onlineA', 'onlineB', 'onlineC', 'onlineD', ...
    'onlineX', 'onlineY', 'onlineU', 'onlineDX'};
for index = 1:numel(fieldNames)
    y = 25 + (index - 1) * 48;
    blockName = [modelName, '/Online ', fieldNames{index}];
    add_block('simulink/Sources/Constant', blockName, ...
        'Value', valueNames{index}, 'Position', [35, y, 125, y + 25]);
end
add_block('simulink/Signal Routing/Bus Creator', [modelName, '/Online model bus'], ...
    'Inputs', '8', 'Position', [300, 45, 330, 165]);

for index = 1:numel(fieldNames)
    lineHandle = add_line(modelName, ['Online ', fieldNames{index}, '/1'], ...
        ['Online model bus/', num2str(index)], 'autorouting', 'on');
    set_param(lineHandle, 'Name', fieldNames{index});
end
add_line(modelName, 'Online model bus/1', 'Adaptive MPC/1', 'autorouting', 'on');
add_line(modelName, 'Tracking plant/1', 'Adaptive MPC/2', 'autorouting', 'on');
add_line(modelName, 'Zero reference/1', 'Adaptive MPC/3', 'autorouting', 'on');
add_line(modelName, 'Zero curvature disturbance/1', 'Adaptive MPC/4', 'autorouting', 'on');
add_line(modelName, 'Adaptive MPC/1', 'Plant input mux/1', 'autorouting', 'on');
add_line(modelName, 'Zero curvature disturbance/1', 'Plant input mux/2', 'autorouting', 'on');
add_line(modelName, 'Plant input mux/1', 'Tracking plant/1', 'autorouting', 'on');
add_line(modelName, 'Tracking plant/1', 'Tracking state log/1', 'autorouting', 'on');
add_line(modelName, 'Adaptive MPC/1', 'Steering log/1', 'autorouting', 'on');

save_system(modelName, modelPath);
end

function close_loaded_model(modelName)
% Close only the temporary generated model; never close a user-opened model.
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
end
