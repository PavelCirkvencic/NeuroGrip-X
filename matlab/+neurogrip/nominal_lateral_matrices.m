function [aContinuous, bContinuous, aDiscrete, bDiscrete] = nominal_lateral_matrices(vxMps, fit, sampleTimeS)
%NOMINAL_LATERAL_MATRICES Shared physical bicycle model and exact ZOH matrices.
%
% State: x = [v_y; yaw_rate], input: steering angle delta. The equations and
% speed clamp exactly mirror `controllers/python/bicycle_model.py` and
% `physics_residual_node.py`. ZOH uses an augmented matrix exponential, the
% same method used internally by scipy.signal.cont2discrete for this system.

arguments
    vxMps (1, 1) double {mustBeFinite}
    fit (1, 1) struct
    sampleTimeS (1, 1) double {mustBePositive, mustBeFinite} = 0.02
end

vxMps = max(vxMps, 1.0);
mass = fit.mass_kg;
inertia = fit.yaw_inertia_kgm2;
wheelbase = fit.wheelbase_m;
frontFraction = fit.front_fraction;
front = fit.front_cornering_stiffness_n_rad;
rear = fit.rear_cornering_stiffness_n_rad;
frontLength = frontFraction * wheelbase;
rearLength = (1.0 - frontFraction) * wheelbase;

aContinuous = [ ...
    -(front + rear) / (mass * vxMps), ...
    -vxMps - (frontLength * front - rearLength * rear) / (mass * vxMps); ...
    -(frontLength * front - rearLength * rear) / (inertia * vxMps), ...
    -(frontLength^2 * front + rearLength^2 * rear) / (inertia * vxMps) ...
];
bContinuous = [front / mass; frontLength * front / inertia];

% Exact zero-order-hold discretisation for x_dot = A*x + B*u.
augmented = [aContinuous, bContinuous; zeros(1, 3)];
transition = expm(augmented * sampleTimeS);
aDiscrete = transition(1:2, 1:2);
bDiscrete = transition(1:2, 3);
end
