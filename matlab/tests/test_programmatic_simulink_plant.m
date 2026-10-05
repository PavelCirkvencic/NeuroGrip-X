function result = test_programmatic_simulink_plant()
%TEST_PROGRAMMATIC_SIMULINK_PLANT Build and simulate the nominal `.slx` model.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
modelDirectory = test_model_directory();
modelPath = build_neurogrip_controller(modelDirectory, 4.5);
assert(isfile(modelPath), 'Programmatic Simulink build did not create an .slx file.');

[~, modelName] = fileparts(modelPath);
output = sim(modelPath, 'StopTime', '0.2', 'ReturnWorkspaceOutputs', 'on');
recorded = output.get('lateral_state');
states = recorded.signals.values;
time = recorded.time;
assert(size(states, 2) == 2 && numel(time) >= 3, ...
    'Simulink lateral-state log has an unexpected dimension.');
assert(all(isfinite(states), 'all'), 'Simulink lateral-state log has non-finite values.');

fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
[~, ~, aDiscrete, bDiscrete] = neurogrip.nominal_lateral_matrices(4.5, fit, 0.02);
expected = zeros(size(states));
for index = 2:size(states, 1)
    previous = reshape(expected(index - 1, :), 2, 1);
    expected(index, :) = reshape(aDiscrete * previous + bDiscrete * 0.05, 1, 2);
end
maxError = max(abs(states - expected), [], 'all');
assert(maxError < 5e-12, ...
    'Simulink nominal lateral plant does not match the MATLAB ZOH recurrence.');
result = struct('model_path', modelPath, 'max_state_error', maxError);
fprintf('PROGRAMMATIC_SIMULINK_PLANT_PASS max_state_error=%.3e model=%s\n', ...
    result.max_state_error, result.model_path);
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
end
