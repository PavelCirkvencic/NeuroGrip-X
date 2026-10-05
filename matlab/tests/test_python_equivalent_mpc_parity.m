function result = test_python_equivalent_mpc_parity()
%TEST_PYTHON_EQUIVALENT_MPC_PARITY Compare exact MATLAB QP against Python OSQP.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);
fit = neurogrip.load_nominal_fit(fullfile(repositoryRoot, 'runs', ...
    'eufs_v1', 'models', 'nominal_fit.json'));
golden = jsondecode(fileread(fullfile(matlabRoot, 'data', ...
    'python_c0_first_move_golden_v1.json')));
assert(golden.schema_version == 2 && golden.steering_rate_limit_rad_s == 2.5, ...
    'C0 golden artifact does not describe the current release constraints.');

% The release OSQP uses 2e-3 primal/dual tolerances for 50 Hz execution.
% MATLAB quadprog solves the same QP more tightly, so compare within the
% measured controller-solver tolerance instead of requiring bit equality.
tolerance = 5e-5;
errors = zeros(numel(golden.vectors), 1);
for index = 1:numel(golden.vectors)
    vector = golden.vectors(index);
    [aTracking, bTracking] = neurogrip.build_tracking_plant( ...
        vector.vx_mps, fit, golden.sample_time_s);
    x0 = reshape(vector.state, 4, 1);
    solution = neurogrip.python_equivalent_mpc_move( ...
        x0, repmat(vector.curvature_1pm, golden.mpc_horizon, 1), vector.vx_mps, ...
        aTracking, bTracking, golden.sample_time_s, vector.previous_steering_rad);
    errors(index) = abs(solution.first_move_rad - vector.python_first_move_rad);
end
maxError = max(errors);
assert(maxError < tolerance, ...
    'Python-equivalent MATLAB QP first move differs from OSQP golden vectors.');
result = struct('vectors_checked', numel(errors), 'max_first_move_error_rad', maxError);
fprintf('PYTHON_EQUIVALENT_MPC_PARITY_PASS vectors=%d max_first_move_error=%.3e\n', ...
    result.vectors_checked, result.max_first_move_error_rad);
end
