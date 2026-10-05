function controller = configure_shadow_mpc(controller)
%CONFIGURE_SHADOW_MPC Make an MPC object safe for measured-state shadowing.
%
% The live MATLAB path is deliberately read-only.  All four tracking states
% are measured by the shared ROS contract, so this configuration removes the
% default output-disturbance integrators and chooses MATLAB's custom-estimator
% mode.  The caller must set mpcstate.Plant and mpcstate.LastMove from fresh
% telemetry before every mpcmoveAdaptive call.  This prevents an un-actuated
% controller from integrating fictitious state or steering history.

arguments
    controller
end

if ~isa(controller, 'mpc') || numel(controller) ~= 1
    error('neurogrip:InvalidShadowController', ...
        'configure_shadow_mpc requires one scalar MPC controller object.');
end

numberOfOutputs = numel(controller.OutputVariables);
zeroOutputDisturbance = ss([], [], [], zeros(numberOfOutputs, 1), controller.Ts);
setoutdist(controller, 'model', zeroOutputDisturbance);
setEstimator(controller, 'custom');
end
