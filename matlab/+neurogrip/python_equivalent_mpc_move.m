function result = python_equivalent_mpc_move(x0, curvaturePreview, vxMps, ...
    aTracking, bTracking, sampleTimeS, previousSteeringRad, ...
    leftCorridorPreview, rightCorridorPreview, solverAlgorithm)
%PYTHON_EQUIVALENT_MPC_MOVE Solve the Python OSQP lateral QP in MATLAB.
%
% This function is intentionally an equation-level translation of
% `controllers/python/mpc_qp.py`. It is the mathematical parity reference for
% MATLAB; the `mpcmoveAdaptive` object is tested separately because its default
% state estimator and cost parameterisation are not the Python OSQP solver.

arguments
    x0 (4, 1) double {mustBeFinite}
    curvaturePreview (:, 1) double {mustBeFinite}
    vxMps (1, 1) double {mustBeFinite, mustBePositive}
    aTracking (4, 4) double {mustBeFinite}
    bTracking (4, 2) double {mustBeFinite}
    sampleTimeS (1, 1) double {mustBePositive, mustBeFinite}
    previousSteeringRad (1, 1) double {mustBeFinite}
    leftCorridorPreview (:, 1) double {mustBeFinite} = double.empty(0, 1)
    rightCorridorPreview (:, 1) double {mustBeFinite} = double.empty(0, 1)
    solverAlgorithm (1, :) char {mustBeMember(solverAlgorithm, ...
        {'interior-point-convex', 'active-set', 'mpc-active-set'})} = ...
        'interior-point-convex'
end

horizon = numel(curvaturePreview);
deltaMax = 0.37;
deltaRateMax = 2.5;
steeringWeight = 20.0;
steeringRateWeight = 20.0;
slackWeight = 1000.0;
stateWeight = diag([40.0, 60.0, 5.0, 10.0]);

if horizon < 1
    error('neurogrip:InvalidMpcHorizon', ...
        'Curvature preview must contain at least one horizon sample.');
end
if isempty(leftCorridorPreview)
    leftCorridorPreview = repmat(1.5, horizon, 1);
end
if isempty(rightCorridorPreview)
    rightCorridorPreview = repmat(1.5, horizon, 1);
end
if numel(leftCorridorPreview) ~= horizon || ...
        numel(rightCorridorPreview) ~= horizon || ...
        any(leftCorridorPreview <= 0.0) || any(rightCorridorPreview <= 0.0)
    error('neurogrip:InvalidMpcCorridorPreview', ...
        'Left/right corridor previews must match the MPC horizon and be positive.');
end
disturbance = vxMps * curvaturePreview;
[fMatrix, gMatrix, hMatrix] = prediction_matrices(aTracking, bTracking, horizon);
stateCost = kron(eye(horizon), stateWeight);
freeResponse = fMatrix * x0 + hMatrix * disturbance;
hessianU = gMatrix' * stateCost * gMatrix;
rateMatrix = eye(horizon) - diag(ones(horizon - 1, 1), -1);
previousVector = zeros(horizon, 1);
previousVector(1) = previousSteeringRad;
hessianU = hessianU + steeringRateWeight * (rateMatrix' * rateMatrix);
hessian = blkdiag(hessianU + steeringWeight * eye(horizon), ...
    slackWeight * eye(horizon));
% Floating-point matrix products can differ from their transpose at ~1e-15.
% Make the symmetry required by quadprog explicit without changing the QP.
hessian = 0.5 * (hessian + hessian.');
gradient = [gMatrix' * stateCost * freeResponse - ...
    steeringRateWeight * (rateMatrix' * previousVector); zeros(horizon, 1)];

eRows = gMatrix(1:4:end, :);
eFree = freeResponse(1:4:end);
upperE = [eRows, -eye(horizon)];
lowerE = [-eRows, -eye(horizon)];
steeringRows = [eye(horizon), zeros(horizon)];
rateRows = [rateMatrix, zeros(horizon)];
rateLimit = deltaRateMax * sampleTimeS;
rateUpper = repmat(rateLimit, horizon, 1);
rateUpper(1) = previousSteeringRad + rateLimit;
rateLower = repmat(-rateLimit, horizon, 1);
rateLower(1) = previousSteeringRad - rateLimit;

% Each row is exactly the finite upper half of the Python OSQP bound form.
inequalityMatrix = [ ...
    upperE; ...
    lowerE; ...
    steeringRows; ...
    -steeringRows; ...
    rateRows; ...
    -rateRows; ...
    [zeros(horizon), -eye(horizon)] ...
];
inequalityBound = [ ...
    leftCorridorPreview - eFree; ...
    rightCorridorPreview + eFree; ...
    deltaMax * ones(horizon, 1); ...
    deltaMax * ones(horizon, 1); ...
    rateUpper; ...
    -rateLower; ...
    zeros(horizon, 1) ...
];

if strcmp(solverAlgorithm, 'mpc-active-set')
    activeSetOptions = mpcActiveSetOptions('double');
    activeSetOptions.ConstraintTolerance = 1e-10;
    initialActiveSet = false(size(inequalityBound));
    [solution, exitFlag] = mpcActiveSetSolver( ...
        hessian, gradient, inequalityMatrix, inequalityBound, ...
        zeros(0, size(hessian, 2)), zeros(0, 1), ...
        initialActiveSet, activeSetOptions);
    iterations = NaN;
else
    options = optimoptions('quadprog', ...
        'Algorithm', solverAlgorithm, ...
        'ConstraintTolerance', 1e-10, ...
        'OptimalityTolerance', 1e-10, ...
        'Display', 'off');
    initialPoint = [];
end
if strcmp(solverAlgorithm, 'active-set')
    % A constant sequence at the measured wheel angle satisfies magnitude and
    % rate bounds. Add only the corridor slack needed to make it feasible.
    initialSteering = repmat( ...
        min(max(previousSteeringRad, -deltaMax), deltaMax), horizon, 1);
    initialLateralError = eFree + eRows * initialSteering;
    initialSlack = max([ ...
        initialLateralError - leftCorridorPreview, ...
        -initialLateralError - rightCorridorPreview, ...
        zeros(horizon, 1) ...
    ], [], 2);
    initialPoint = [initialSteering; initialSlack + 1e-10];
end
if ~strcmp(solverAlgorithm, 'mpc-active-set')
    [solution, ~, exitFlag, output] = quadprog( ...
        hessian, gradient, inequalityMatrix, inequalityBound, [], [], [], [], ...
        initialPoint, options);
    iterations = output.iterations;
end
if exitFlag <= 0 || any(~isfinite(solution))
    error('neurogrip:PythonEquivalentMpcFailure', ...
        'quadprog failed to solve the Python-equivalent MPC QP (exit flag %d).', exitFlag);
end

steering = solution(1:horizon);
slack = solution(horizon + 1:end);
predictedStates = reshape(fMatrix * x0 + gMatrix * steering + hMatrix * disturbance, 4, horizon).';
result = struct( ...
    'first_move_rad', steering(1), ...
    'steering_sequence_rad', steering, ...
    'slack_m', slack, ...
    'predicted_states', predictedStates, ...
    'exit_flag', exitFlag, ...
    'iterations', iterations ...
);
end

function [fMatrix, gMatrix, hMatrix] = prediction_matrices(aTracking, bTracking, horizon)
% Return X = F*x0 + G*U + H*d exactly as the Python reference constructs it.
stateDimension = 4;
fMatrix = zeros(horizon * stateDimension, stateDimension);
gMatrix = zeros(horizon * stateDimension, horizon);
hMatrix = zeros(horizon * stateDimension, horizon);
for step = 1:horizon
    row = (step - 1) * stateDimension + (1:stateDimension);
    fMatrix(row, :) = aTracking^step;
    for inner = 1:step
        aPower = aTracking^(step - inner);
        gMatrix(row, inner) = aPower * bTracking(:, 1);
        hMatrix(row, inner) = aPower * bTracking(:, 2);
    end
end
end
