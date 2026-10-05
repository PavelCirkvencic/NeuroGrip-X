function result = latency_compensated_mpc_move(frame, solverAlgorithm)
%LATENCY_COMPENSATED_MPC_MOVE Compensate one transport/control sample delay.
%
% The ROS-to-MATLAB-to-ROS path adds approximately one 20 ms control sample.
% Propagate the measured tracking state once while holding the measured wheel
% angle, then solve against the correspondingly shifted track preview.

arguments
    frame (1, 1) struct
    solverAlgorithm (1, :) char = 'mpc-active-set'
end

heldInput = [ ...
    frame.previous_steering_rad; ...
    frame.v_x_mps * frame.curvature_preview(1) ...
];
predictedState = frame.a_tracking * frame.state + ...
    frame.b_tracking * heldInput;
curvaturePreview = shift_preview(frame.curvature_preview);
leftCorridorPreview = shift_preview(frame.left_corridor_preview);
rightCorridorPreview = shift_preview(frame.right_corridor_preview);
result = neurogrip.python_equivalent_mpc_move( ...
    predictedState, curvaturePreview, frame.v_x_mps, ...
    frame.a_tracking, frame.b_tracking, frame.sample_time_s, ...
    frame.previous_steering_rad, leftCorridorPreview, ...
    rightCorridorPreview, solverAlgorithm);
result.delay_compensation_steps = 1;
result.delay_compensated_state = predictedState;
end

function shifted = shift_preview(values)
values = values(:);
shifted = [values(2:end); values(end)];
end
