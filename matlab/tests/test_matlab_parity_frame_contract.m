function result = test_matlab_parity_frame_contract()
%TEST_MATLAB_PARITY_FRAME_CONTRACT Detect ordering and reshape regressions.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
addpath(matlabRoot);
values = zeros(1, 155);
values(1:3) = [3.0, 42.0, 0.02];
values(4:29) = 1:26;
values(30:33) = [0.1, 0.2, 0.3, 0.4];
values(34:37) = [1.0, 2.0, 3.0, 4.0];
values(38:39) = [5.0, 6.0];
values(40:55) = 1:16;
values(56:63) = 21:28;
values(64:93) = linspace(-0.1, 0.1, 30);
values(94:123) = linspace(1.0, 2.0, 30);
values(124:153) = linspace(1.5, 2.5, 30);
values(154:155) = [-0.04, 1.0];
frame = neurogrip.parse_matlab_parity_frame(values);
assert(frame.valid, 'A valid MATLAB parity frame was rejected.');
assert(isequal(frame.raw_a_lateral, [1.0, 2.0; 3.0, 4.0]), ...
    'Raw lateral A row-major order was decoded incorrectly.');
assert(isequal(frame.a_tracking, reshape(1:16, 4, 4).'), ...
    'Tracking A row-major order was decoded incorrectly.');
assert(isequal(frame.b_tracking, reshape(21:28, 2, 4).'), ...
    'Tracking B row-major order was decoded incorrectly.');
assert(numel(frame.curvature_preview) == 30 && ...
    numel(frame.left_corridor_preview) == 30 && ...
    numel(frame.right_corridor_preview) == 30, ...
    'Parity preview arrays have incorrect dimensions.');
assert(frame.solution_valid && frame.first_move_rad == -0.04, ...
    'Parity controller output fields were decoded incorrectly.');
invalid = neurogrip.parse_matlab_parity_frame(values(1:end - 1));
assert(~invalid.valid && strcmp(invalid.reason, 'invalid_parity_frame_length'), ...
    'A truncated parity frame was not rejected.');
result = struct('frame_length', numel(values), 'preview_length', 30);
fprintf('MATLAB_PARITY_FRAME_CONTRACT_PASS fields=%d preview=%d\n', ...
    result.frame_length, result.preview_length);
end
