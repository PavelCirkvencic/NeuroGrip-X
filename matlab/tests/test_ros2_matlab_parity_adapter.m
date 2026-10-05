function result = test_ros2_matlab_parity_adapter()
%TEST_ROS2_MATLAB_PARITY_ADAPTER Verify live subscriber and stale watchdog.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
addpath(matlabRoot);
suffix = char(string(randi([100000, 999999])));
adapter = neurogrip.Ros2MatlabParityAdapter( ...
    ['/neurogrip_parity_adapter_', suffix], 0.20);
publisherNode = ros2node(['/neurogrip_parity_publisher_', suffix]);
publisher = ros2publisher(publisherNode, ...
    '/neurogrip/matlab_parity_frame', 'std_msgs/Float64MultiArray');
cleanup = onCleanup(@() release_entities(adapter, publisher, publisherNode));

values = zeros(1, 155);
values(1:3) = [3.0, 42.0, 0.02];
values(22) = 8.0;
values(30:33) = [0.1, 0.02, 0.0, 0.1];
values(40:55) = reshape(eye(4).', 1, []);
values(56:63) = reshape(zeros(4, 2).', 1, []);
values(94:153) = 1.5;
values(154:155) = [-0.04, 1.0];
message = ros2message(publisher);
message.data = values;
pause(0.6);
for index = 1:20
    send(publisher, message);
    pause(0.02);
end
frame = adapter.poll();
assert(frame.valid && frame.control_time_s == 42.0 && ...
    frame.first_move_rad == -0.04, ...
    'MATLAB ROS 2 parity callback did not return the transmitted frame.');
pause(0.25);
staleFrame = adapter.poll();
assert(~staleFrame.valid && strcmp(staleFrame.reason, 'stale_parity_frame'), ...
    'The MATLAB parity adapter did not fail closed on stale telemetry.');
result = struct('frame_length', numel(values), 'sample_time_s', frame.sample_time_s);
fprintf('ROS2_MATLAB_PARITY_ADAPTER_PASS fields=%d Ts=%.3f\n', ...
    result.frame_length, result.sample_time_s);
end

function release_entities(adapter, publisher, publisherNode) %#ok<INUSD>
delete(adapter);
clear publisher publisherNode
end
