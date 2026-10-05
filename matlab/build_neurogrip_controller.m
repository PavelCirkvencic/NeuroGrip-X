function modelPath = build_neurogrip_controller(outputDirectory, vxMps)
%BUILD_NEUROGRIP_CONTROLLER Generate the nominal Simulink lateral-plant model.
%
% The generated `.slx` is intentionally small and deterministic: a fixed 4.5
% m/s C0 nominal *plant* driven by one steering input. It is the verified
% Simulink substrate for the next C0 MPC wrapper, not a decorative car model
% and not yet the full closed-loop Adaptive MPC architecture.

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
[~, ~, aDiscrete, bDiscrete] = neurogrip.nominal_lateral_matrices(vxMps, fit, 0.02);
modelName = 'neurogrip_c0_nominal_lateral_plant';
modelPath = fullfile(outputDirectory, [modelName, '.slx']);

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
cleanup = onCleanup(@() close_loaded_model(modelName));
set_param(modelName, ...
    'Solver', 'FixedStepDiscrete', ...
    'FixedStep', '0.02', ...
    'StopTime', '0.2', ...
    'ReturnWorkspaceOutputs', 'on');

add_block('simulink/Sources/Constant', [modelName, '/Steering command'], ...
    'Value', '0.05', 'Position', [45, 85, 110, 115]);
add_block('simulink/Discrete/Discrete State-Space', [modelName, '/C0 nominal lateral plant'], ...
    'A', mat2str(aDiscrete, 17), ...
    'B', mat2str(bDiscrete, 17), ...
    'C', '[1 0; 0 1]', ...
    'D', '[0; 0]', ...
    'X0', '[0; 0]', ...
    'SampleTime', '0.02', ...
    'Position', [190, 60, 350, 140]);
add_block('simulink/Sinks/To Workspace', [modelName, '/Lateral state log'], ...
    'VariableName', 'lateral_state', ...
    'SaveFormat', 'Structure With Time', ...
    'Position', [425, 82, 535, 118]);
add_line(modelName, 'Steering command/1', 'C0 nominal lateral plant/1', 'autorouting', 'on');
add_line(modelName, 'C0 nominal lateral plant/1', 'Lateral state log/1', 'autorouting', 'on');

save_system(modelName, modelPath);
end

function close_loaded_model(modelName)
% Close only the temporary programmatic model; never close a user's model.
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
end
