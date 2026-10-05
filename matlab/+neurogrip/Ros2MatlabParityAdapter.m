classdef Ros2MatlabParityAdapter < handle
    %ROS2MATLABPARITYADAPTER Read-only live schema-3 decision adapter.
    %
    % The adapter owns one subscriber and no publisher. It fails closed when
    % the latest `/neurogrip/matlab_parity_frame` packet is missing, stale or
    % invalid, and exposes only the decoded telemetry needed for shadow MPC.

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
        function adapter = Ros2MatlabParityAdapter(nodeName, maxAgeS)
            arguments
                nodeName (1, :) char {mustBeNonzeroLengthText} = ...
                    '/neurogrip_matlab_parity_adapter'
                maxAgeS (1, 1) double {mustBePositive, mustBeFinite} = 0.20
            end
            adapter.MaxAgeS = maxAgeS;
            adapter.MonotonicClock = tic;
            adapter.Node = ros2node(nodeName);
            adapter.Subscriber = ros2subscriber(adapter.Node, ...
                '/neurogrip/matlab_parity_frame', ...
                'std_msgs/Float64MultiArray');
            adapter.Subscriber.NewMessageFcn = @(message) ...
                adapter.capture(message);
        end

        function frame = poll(adapter)
            %POLL Return a fresh decoded frame or a fail-closed reason.
            if isempty(adapter.LatestValues)
                frame = invalid_frame('missing_parity_frame');
                return;
            end
            if toc(adapter.MonotonicClock) - adapter.ReceiptS > adapter.MaxAgeS
                frame = invalid_frame('stale_parity_frame');
                return;
            end
            frame = neurogrip.parse_matlab_parity_frame(adapter.LatestValues);
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
