function plant = online_mpc_plant_from_update(update)
%ONLINE_MPC_PLANT_FROM_UPDATE Convert a validated physical update to MPC `ss`.
%
% This is intentionally downstream of prepare_adaptive_model_update: callers
% cannot turn arbitrary ROS bytes into an MPC plant. Inputs are ordered
% [steering_angle_rad; d_kappa_rad_s], with steering as MV and curvature rate
% as measured disturbance. Outputs are the complete tracking state.

arguments
    update (1, 1) struct
end

if ~isfield(update, 'valid') || ~update.valid
    error('neurogrip:InvalidAdaptiveUpdate', ...
        'A validated Adaptive MPC update is required to create an online plant.');
end
plant = ss(update.a_tracking, update.b_tracking, eye(4), zeros(4, 2), ...
    update.dynamics.sample_time_s);
plant = setmpcsignals(plant, 'MV', 1, 'MD', 2, 'MO', 1:4);
plant.InputName = {'steering_angle_rad', 'd_kappa_rad_s'};
plant.OutputName = {'e_y_m', 'e_psi_rad', 'v_y_mps', 'yaw_rate_rps'};
end
