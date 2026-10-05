function result = test_nominal_zoh_parity()
%TEST_NOMINAL_ZOH_PARITY Verify MATLAB matrices against Python golden vectors.
%
% This is the first MATLAB gate. It does not compare controllers yet: it proves
% that both languages use the identical physical [v_y, yaw_rate] model before
% an Adaptive MPC or Simulink wrapper is allowed to consume it.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
repositoryRoot = fileparts(matlabRoot);
addpath(matlabRoot);

fitPath = fullfile(repositoryRoot, 'runs', 'eufs_v1', 'models', ...
    'nominal_fit.json');
goldenPath = fullfile(matlabRoot, 'data', 'nominal_zoh_golden_v1.json');
fit = neurogrip.load_nominal_fit(fitPath);
golden = jsondecode(fileread(goldenPath));

if golden.schema_version ~= 1 || golden.sample_time_s ~= 0.02
    error('neurogrip:InvalidGoldenVectors', 'Unsupported nominal ZOH golden-vector schema.');
end

tolerance = 5e-12;
maxContinuousError = 0.0;
maxDiscreteError = 0.0;
for index = 1:numel(golden.vectors)
    expected = golden.vectors(index);
    [actualAC, actualBC, actualAD, actualBD] = neurogrip.nominal_lateral_matrices( ...
        expected.vx_mps, fit, golden.sample_time_s);
    expectedAC = reshape(expected.a_cont, 2, 2);
    expectedBC = reshape(expected.b_cont, 2, 1);
    expectedAD = reshape(expected.a_discrete, 2, 2);
    expectedBD = reshape(expected.b_discrete, 2, 1);
    maxContinuousError = max(maxContinuousError, max(abs([ ...
        actualAC(:) - expectedAC(:); actualBC(:) - expectedBC(:) ...
    ])));
    maxDiscreteError = max(maxDiscreteError, max(abs([ ...
        actualAD(:) - expectedAD(:); actualBD(:) - expectedBD(:) ...
    ])));
end

assert(maxContinuousError < tolerance, ...
    'MATLAB continuous bicycle matrices differ from Python golden vectors.');
assert(maxDiscreteError < tolerance, ...
    'MATLAB ZOH bicycle matrices differ from Python golden vectors.');
result = struct( ...
    'vectors_checked', numel(golden.vectors), ...
    'max_continuous_error', maxContinuousError, ...
    'max_discrete_error', maxDiscreteError, ...
    'nominal_fit_sha256', golden.nominal_fit_sha256 ...
);
fprintf(['NOMINAL_ZOH_PARITY_PASS vectors=%d max_continuous_error=%.3e ' ...
    'max_discrete_error=%.3e fit_sha256=%s\n'], ...
    result.vectors_checked, result.max_continuous_error, ...
    result.max_discrete_error, result.nominal_fit_sha256);
end
