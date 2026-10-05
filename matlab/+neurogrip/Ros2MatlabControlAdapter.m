classdef Ros2MatlabControlAdapter < handle
    %ROS2MATLABCONTROLADAPTER Read-only live QP-input adapter.

    properties (SetAccess = private)
        Node
        Subscriber
        LatestValues = []
        ReceiptS = -Inf
    end

    properties (SetAccess = private)
        MonotonicClock
        MaxAgeS (1, 1) double {mustBePositive, mustBeFinite} = 0.20
    end

    methods
        function adapter = Ros2MatlabControlAdapter(nodeName, maxAgeS)
            arguments
                nodeName (1, :) char {mustBeNonzeroLengthText} = ...
                    '/neurogrip_matlab_control_adapter'
                maxAgeS (1, 1) double {mustBePositive, mustBeFinite} = 0.20
            end
            adapter.MaxAgeS = maxAgeS;
            adapter.MonotonicClock = tic;
            adapter.Node = ros2node(nodeName);
            adapter.Subscriber = ros2subscriber(adapter.Node, ...
                '/neurogrip/matlab_control_frame', ...
                'std_msgs/Float64MultiArray');
            adapter.Subscriber.NewMessageFcn = @(message) ...
                adapter.capture(message);
        end

        function frame = poll(adapter)
            %POLL Return a fresh decoded control frame or fail closed.
            if isempty(adapter.LatestValues)
                frame = invalid_frame('missing_control_frame');
                return;
            end
            if toc(adapter.MonotonicClock) - adapter.ReceiptS > adapter.MaxAgeS
                frame = invalid_frame('stale_control_frame');
                return;
            end
            frame = neurogrip.parse_matlab_control_frame(adapter.LatestValues);
        end

        function delete(adapter)
            % Release only MATLAB-owned ROS entities.
            adapter.Subscriber = [];
            adapter.Node = [];
        end
    end

    methods (Access = private)
        function capture(adapter, message)
            adapter.LatestValues = double(message.data(:).');
            adapter.ReceiptS = toc(adapter.MonotonicClock);
        end
    end
end

function frame = invalid_frame(reason)
frame = struct('valid', false, 'reason', reason);
end
