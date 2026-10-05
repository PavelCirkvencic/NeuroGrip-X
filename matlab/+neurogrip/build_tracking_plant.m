function [aTracking, bTracking, aLateral, bLateral] = build_tracking_plant(vxMps, fit, sampleTimeS)
%BUILD_TRACKING_PLANT Create the Python-compatible 4-state MPC prediction plant.
%
% State: [e_y; e_psi; v_y; yaw_rate]
% Input 1: steering angle (MV)
% Input 2: d_kappa = vx * curvature (measured disturbance)
%
% This is a direct MATLAB translation of
% `controllers/python/bicycle_model.py:build_tracking_plant_from_discrete`.

arguments
    vxMps (1, 1) double {mustBeFinite}
    fit (1, 1) struct
    sampleTimeS (1, 1) double {mustBePositive, mustBeFinite} = 0.02
end

vxMps = max(vxMps, 1.0);
[~, ~, aLateral, bLateral] = neurogrip.nominal_lateral_matrices( ...
    vxMps, fit, sampleTimeS);
[aTracking, bTracking] = neurogrip.build_tracking_plant_from_lateral( ...
    vxMps, aLateral, bLateral, sampleTimeS);
end
