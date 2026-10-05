function result = test_shadow_mpc_configuration()
%TEST_SHADOW_MPC_CONFIGURATION Verify measured-state-only shadow setup.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
[controller, plant] = neurogrip.create_c0_mpc(4.5, fit, 0.02);
controller = neurogrip.configure_shadow_mpc(controller);
[outputDisturbance, outputChannels] = getoutdist(controller);
assert(order(outputDisturbance) == 0 && isempty(outputChannels), ...
    'Shadow MPC retained a dynamic output-disturbance model.');
assert(strcmp(controller.EstGain, 'custom'), ...
    'Shadow MPC did not require the caller to provide measured state.');
controllerState = mpcstate(controller);
assert(isequal(size(controllerState.Plant), [4, 1]), ...
    'Shadow MPC does not expose the four measured tracking states.');
assert(isempty(controllerState.Disturbance), ...
    'Shadow MPC retained a nonphysical output-disturbance estimate.');
measuredOutput = [0.005; 0.0; 0.0; 0.0];
controllerState.Plant = measuredOutput;
controllerState.LastMove = 0.0;
[move, info] = mpcmoveAdaptive(controller, controllerState, plant, [], ...
    measuredOutput, zeros(1, 4), 0.0);
assert(isfinite(move) && strcmp(info.QPCode, 'feasible'), ...
    'Measured-state shadow MPC failed to compute a feasible move.');
result = struct('move_rad', move, 'estimator_state_dimension', numel(controllerState.Plant));
fprintf('SHADOW_MPC_CONFIGURATION_PASS move=%.6f states=%d\n', ...
    result.move_rad, result.estimator_state_dimension);
end
