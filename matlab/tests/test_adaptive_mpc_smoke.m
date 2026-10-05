function result = test_adaptive_mpc_smoke()
%TEST_ADAPTIVE_MPC_SMOKE Exercise mpcmoveAdaptive with a real updated plant.
%
% The test intentionally uses the nominal plant as its update. Its purpose is
% to prove that this installed MATLAB version accepts the online state-space
% model and returns a bounded finite move; it is not an adaptive-performance
% benchmark.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
[controller, plant] = neurogrip.create_c0_mpc(4.5, fit, 0.02);

state = mpcstate(controller);
measuredOutputs = [0.20, 0.05, 0.0, 0.0];
reference = zeros(1, 4);
measuredDisturbance = 0.0;
[move, info] = mpcmoveAdaptive(controller, state, plant, [], ...
    measuredOutputs, reference, measuredDisturbance);

assert(isscalar(move) && isfinite(move), ...
    'mpcmoveAdaptive did not return one finite steering move.');
assert(move >= controller.MV.Min - 1e-12 && move <= controller.MV.Max + 1e-12, ...
    'mpcmoveAdaptive violated the steering bound.');
assert(isfield(info, 'QPCode') && strcmp(info.QPCode, 'feasible'), ...
    'mpcmoveAdaptive reported an infeasible QP.');
result = struct('first_move_rad', move, 'qp_code', info.QPCode);
fprintf('ADAPTIVE_MPC_SMOKE_PASS move_rad=%.9f qp_code=%s\n', ...
    result.first_move_rad, result.qp_code);
end
