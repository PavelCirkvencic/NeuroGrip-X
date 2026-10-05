function result = test_ros2_matlab_control_adapter()
%TEST_ROS2_MATLAB_CONTROL_ADAPTER Verify continuous QP frame callbacks.

thisFile = mfilename('fullpath');
matlabRoot = fileparts(fileparts(thisFile));
addpath(matlabRoot);
values = zeros(1, 125);
values(1:5) = [1.0, 42.0, 0.02, 8.0, 0.01];
values(6:9) = [0.1, 0.2, 0.3, 0.4];
values(10:25) = 1:16;
values(26:33) = 21:28;
values(34:63) = linspace(-0.1, 0.1, 30);
values(64:123) = 1.5;
values(124:125) = [-0.04, 1.0];
frame = neurogrip.parse_matlab_control_frame(values);
assert(frame.valid, 'A valid MATLAB control frame was rejected.');
assert(isequal(frame.a_tracking, reshape(1:16, 4, 4).'), ...
    'Control-frame A row-major order was decoded incorrectly.');
assert(isequal(frame.b_tracking, reshape(21:28, 2, 4).'), ...
    'Control-frame B row-major order was decoded incorrectly.');

suffix = char(string(randi([100000, 999999])));
adapter = neurogrip.Ros2MatlabControlAdapter( ...
    ['/neurogrip_control_adapter_', suffix], 0.20);
publisherNode = ros2node(['/neurogrip_control_publisher_', suffix]);
publisher = ros2publisher(publisherNode, ...
    '/neurogrip/matlab_control_frame', 'std_msgs/Float64MultiArray');
cleanup = onCleanup(@() release_entities(adapter, publisher, publisherNode));
message = ros2message(publisher);
message.data = values;
pause(0.6);
for index = 1:20
    send(publisher, message);
    pause(0.02);
end
liveFrame = adapter.poll();
assert(liveFrame.valid && liveFrame.control_time_s == 42.0 && ...
    liveFrame.first_move_rad == -0.04, ...
    'MATLAB ROS 2 control callback did not return the transmitted frame.');
pause(0.25);
staleFrame = adapter.poll();
assert(~staleFrame.valid && strcmp(staleFrame.reason, 'stale_control_frame'), ...
    'The MATLAB control adapter did not fail closed on stale telemetry.');
result = struct('frame_length', numel(values), ...
    'sample_time_s', liveFrame.sample_time_s);
fprintf('ROS2_MATLAB_CONTROL_ADAPTER_PASS fields=%d Ts=%.3f\n', ...
    result.frame_length, result.sample_time_s);
end

function release_entities(adapter, publisher, publisherNode) %#ok<INUSD>
delete(adapter);
clear publisher publisherNode
end
