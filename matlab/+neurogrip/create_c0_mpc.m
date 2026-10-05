function [controller, plant] = create_c0_mpc(vxMps, fit, sampleTimeS)
%CREATE_C0_MPC Configure MATLAB C0 MPC from the shared nominal tracking plant.
%
% This is a functional parity scaffold, not a claim that the MATLAB and Python
% QP solvers are already numerically identical. The next golden-move gate will
% compare their constraints and first move explicitly.

arguments
    vxMps (1, 1) double {mustBeFinite}
    fit (1, 1) struct
    sampleTimeS (1, 1) double {mustBePositive, mustBeFinite} = 0.02
end

[aTracking, bTracking] = neurogrip.build_tracking_plant(vxMps, fit, sampleTimeS);
plant = ss(aTracking, bTracking, eye(4), zeros(4, 2), sampleTimeS);
plant = setmpcsignals(plant, 'MV', 1, 'MD', 2, 'MO', 1:4);
plant.InputName = {'steering_angle_rad', 'd_kappa_rad_s'};
plant.OutputName = {'e_y_m', 'e_psi_rad', 'v_y_mps', 'yaw_rate_rps'};

controller = mpc(plant, sampleTimeS, 30, 30);
controller.Weights.OutputVariables = [40.0, 60.0, 5.0, 10.0];
controller.Weights.ManipulatedVariables = 20.0;
controller.Weights.ManipulatedVariablesRate = 5.0;
controller.MV.Min = -0.37;
controller.MV.Max = 0.37;
controller.MV.RateMin = -2.5 * sampleTimeS;
controller.MV.RateMax = 2.5 * sampleTimeS;

% Python uses a soft |e_y| <= 1.5 m corridor. Assign the same output limits
% now; the exact slack penalty is checked separately in the golden-move gate.
controller.OV(1).Min = -1.5;
controller.OV(1).Max = 1.5;
controller.OV(1).MinECR = 1.0;
controller.OV(1).MaxECR = 1.0;
end
