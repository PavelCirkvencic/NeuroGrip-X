function result = test_ros2_adaptive_model_adapter()
%TEST_ROS2_ADAPTIVE_MODEL_ADAPTER Verify actual MATLAB ROS 2 subscriber callbacks.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
addpath(matlabRoot);
suffix = char(string(randi([100000, 999999])));
adapter = neurogrip.Ros2AdaptiveModelAdapter( ...
    ['/neurogrip_adapter_', suffix], 0.20, true);
publisherNode = ros2node(['/neurogrip_adapter_publisher_', suffix]);
dynamicsPublisher = ros2publisher(publisherNode, '/neurogrip/dynamics_flat', ...
    'std_msgs/Float64MultiArray');
trackingPublisher = ros2publisher(publisherNode, '/neurogrip/tracking_state', ...
    'std_msgs/Float64MultiArray');
appliedSteeringPublisher = ros2publisher(publisherNode, ...
    '/neurogrip/applied_steering_flat', 'std_msgs/Float64MultiArray');
cleanup = onCleanup(@() release_entities(adapter, dynamicsPublisher, ...
    trackingPublisher, appliedSteeringPublisher, publisherNode)); %#ok<NASGU>

pause(0.6); % allow DDS discovery before the bounded test exchange
dynamicsMessage = ros2message(dynamicsPublisher);
dynamicsMessage.data = [ ...
    4, 0.02, 1, 0.90, 0.01, 0.02, 0.95, 0.30, 1.10, ...
    0.01, 0.20, 1.20, 1.70, 0.78, 0.66, 0.02, 0.03, 0.06, 0.13];
trackingMessage = ros2message(trackingPublisher);
trackingMessage.data = [1, 42, 0.20, 0.05, 0.10, 0.40, 4.5, 0.08, 0.36, 0.60];
appliedSteeringMessage = ros2message(appliedSteeringPublisher);
appliedSteeringMessage.data = [1, 42, 0.012];
for index = 1:20
    send(dynamicsPublisher, dynamicsMessage);
    send(trackingPublisher, trackingMessage);
    send(appliedSteeringPublisher, appliedSteeringMessage);
    pause(0.02);
end
update = adapter.poll();
assert(update.valid, 'MATLAB ROS 2 callbacks did not produce a valid model update.');
assert(update.dynamics.schema_version == uint32(4) && ...
    abs(update.dynamics.estimated_front_grip - 0.78) < 1e-14 && ...
    abs(update.dynamics.rear_grip_error_q90 - 0.13) < 1e-14, ...
    'MATLAB ROS 2 adapter lost schema-4 grip/uncertainty fields.');
assert(isequal(size(update.model_bus.B), [4, 2, 31]), ...
    'Live MATLAB ROS 2 update produced an invalid B model-bus dimension.');
assert(update.applied_steering.valid && ...
    abs(update.applied_steering.steering_rad - 0.012) < 1e-14, ...
    'Live MATLAB ROS 2 adapter did not attach fresh measured steering.');
pause(0.25);
staleUpdate = adapter.poll();
assert(~staleUpdate.valid && strcmp(staleUpdate.reason, 'stale_dynamics'), ...
    'A stale live C2 model did not fail closed.');
result = struct('model_pages', size(update.model_bus.A, 3), ...
    'sample_time_s', update.dynamics.sample_time_s);
fprintf('ROS2_ADAPTIVE_MODEL_ADAPTER_PASS pages=%d Ts=%.3f\n', ...
    result.model_pages, result.sample_time_s);
end

function release_entities(adapter, dynamicsPublisher, trackingPublisher, appliedSteeringPublisher, publisherNode)
delete(adapter);
clear dynamicsPublisher trackingPublisher appliedSteeringPublisher publisherNode
end
