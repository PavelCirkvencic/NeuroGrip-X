function result = test_online_mpc_plant_from_update()
%TEST_ONLINE_MPC_PLANT_FROM_UPDATE Verify C2 packet -> MPC model -> QP move.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
dynamics = [2, 0.02, 1, 0.90, 0.01, 0.02, 0.95, 0.30, 1.10, 0.01, 0.20, 1.20, 1.70];
tracking = [1, 42, 0.20, 0.05, 0.10, 0.40, 4.5, 0.08, 0.36, 0.60];
update = neurogrip.prepare_adaptive_model_update(dynamics, tracking);
plant = neurogrip.online_mpc_plant_from_update(update);
assert(max(abs(plant.A - update.a_tracking), [], 'all') < 1e-14, ...
    'Online MPC plant A differs from the validated physical C2 update.');
assert(max(abs(plant.B - update.b_tracking), [], 'all') < 1e-14, ...
    'Online MPC plant B differs from the validated physical C2 update.');
fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
[controller, ~] = neurogrip.create_c0_mpc(4.5, fit, 0.02);
controllerState = mpcstate(controller);
controllerState.Plant = [0.20; 0.05; 0.10; 0.40];
assert(max(abs(controllerState.Plant - [0.20; 0.05; 0.10; 0.40])) < 1e-14, ...
    'The measured physical tracking state cannot anchor the MPC estimator.');
[move, info] = mpcmoveAdaptive(controller, controllerState, plant, [], ...
    [0.20, 0.05, 0.10, 0.40], zeros(1, 4), 0.36);
assert(isfinite(move) && strcmp(info.QPCode, 'feasible'), ...
    'C2-derived online plant did not produce a feasible Adaptive MPC move.');
assert(move >= controller.MV.Min - 1e-12 && move <= controller.MV.Max + 1e-12, ...
    'C2-derived online plant violated the steering magnitude bound.');
result = struct('first_move_rad', move, 'qp_code', info.QPCode);
fprintf('ONLINE_MPC_PLANT_FROM_UPDATE_PASS move_rad=%.9f qp_code=%s\n', ...
    result.first_move_rad, result.qp_code);
end
