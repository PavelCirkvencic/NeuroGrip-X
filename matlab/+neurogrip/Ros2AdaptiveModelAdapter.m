classdef Ros2AdaptiveModelAdapter < handle
    %ROS2ADAPTIVEMODELADAPTER Read-only live C2-to-Adaptive-MPC adapter.
    %
    % This class owns no publisher and therefore cannot bypass ackermann_guard.
    % It consumes only standard ROS messages, validates both compatibility
    % contracts, applies a short receipt-time watchdog, and returns a physical
    % Adaptive MPC model bus.  The caller must choose safe nominal fallback on
    % any invalid result.

    properties (SetAccess = private)
        Node
        DynamicsSubscriber
        TrackingSubscriber
        AppliedSteeringSubscriber
        LatestDynamicsValues = []
        LatestTrackingValues = []
        LatestAppliedSteeringValues = []
        DynamicsReceiptS = -Inf
        TrackingReceiptS = -Inf
        AppliedSteeringReceiptS = -Inf
        RequireAppliedSteering (1, 1) logical = false
    end

    properties (SetAccess = private)
        MonotonicClock
        MaxAgeS (1, 1) double {mustBePositive, mustBeFinite} = 0.20
    end

    methods
        function adapter = Ros2AdaptiveModelAdapter(nodeName, maxAgeS, requireAppliedSteering)
            arguments
                nodeName (1, :) char {mustBeNonzeroLengthText} = '/neurogrip_matlab_model_adapter'
                maxAgeS (1, 1) double {mustBePositive, mustBeFinite} = 0.20
                requireAppliedSteering (1, 1) logical = false
            end
            adapter.MaxAgeS = maxAgeS;
            adapter.RequireAppliedSteering = requireAppliedSteering;
            adapter.MonotonicClock = tic;
            adapter.Node = ros2node(nodeName);
            adapter.DynamicsSubscriber = ros2subscriber(adapter.Node, ...
                '/neurogrip/dynamics_flat', 'std_msgs/Float64MultiArray');
            adapter.TrackingSubscriber = ros2subscriber(adapter.Node, ...
                '/neurogrip/tracking_state', 'std_msgs/Float64MultiArray');
            adapter.AppliedSteeringSubscriber = ros2subscriber(adapter.Node, ...
                '/neurogrip/applied_steering_flat', 'std_msgs/Float64MultiArray');
            % MATLAB R2026a ROS 2 delivers the message itself as the sole
            % NewMessageFcn argument (unlike several older ROS callback APIs).
            adapter.DynamicsSubscriber.NewMessageFcn = @(message) ...
                adapter.capture_dynamics(message);
            adapter.TrackingSubscriber.NewMessageFcn = @(message) ...
                adapter.capture_tracking(message);
            adapter.AppliedSteeringSubscriber.NewMessageFcn = @(message) ...
                adapter.capture_applied_steering(message);
        end

        function update = poll(adapter)
            %POLL Return only a fresh, fully validated physical model bus.
            nowS = toc(adapter.MonotonicClock);
            if isempty(adapter.LatestDynamicsValues)
                update = invalid_update('missing_dynamics');
                return;
            end
            if isempty(adapter.LatestTrackingValues)
                update = invalid_update('missing_tracking');
                return;
            end
            if nowS - adapter.DynamicsReceiptS > adapter.MaxAgeS
                update = invalid_update('stale_dynamics');
                return;
            end
            if nowS - adapter.TrackingReceiptS > adapter.MaxAgeS
                update = invalid_update('stale_tracking');
                return;
            end
            if adapter.RequireAppliedSteering && isempty(adapter.LatestAppliedSteeringValues)
                update = invalid_update('missing_applied_steering');
                return;
            end
            if adapter.RequireAppliedSteering && ...
                    nowS - adapter.AppliedSteeringReceiptS > adapter.MaxAgeS
                update = invalid_update('stale_applied_steering');
                return;
            end
            update = neurogrip.prepare_adaptive_model_update( ...
                adapter.LatestDynamicsValues, adapter.LatestTrackingValues);
            if ~update.valid || ~adapter.RequireAppliedSteering
                return;
            end
            appliedSteering = neurogrip.parse_applied_steering_flat( ...
                adapter.LatestAppliedSteeringValues);
            if ~appliedSteering.valid
                update.valid = false;
                update.reason = appliedSteering.reason;
                return;
            end
            if abs(appliedSteering.time_s - update.tracking.time_s) > 0.05
                update.valid = false;
                update.reason = 'applied_steering_time_mismatch';
                return;
            end
            update.applied_steering = appliedSteering;
        end

        function delete(adapter)
            % Release only MATLAB-owned ROS entities; never touch ROS processes.
            adapter.DynamicsSubscriber = [];
            adapter.TrackingSubscriber = [];
            adapter.AppliedSteeringSubscriber = [];
            adapter.Node = [];
        end
    end

    methods (Access = private)
        function capture_dynamics(adapter, message)
            adapter.LatestDynamicsValues = double(message.data(:).');
            adapter.DynamicsReceiptS = toc(adapter.MonotonicClock);
        end

        function capture_tracking(adapter, message)
            adapter.LatestTrackingValues = double(message.data(:).');
            adapter.TrackingReceiptS = toc(adapter.MonotonicClock);
        end

        function capture_applied_steering(adapter, message)
            adapter.LatestAppliedSteeringValues = double(message.data(:).');
            adapter.AppliedSteeringReceiptS = toc(adapter.MonotonicClock);
        end
    end
end

function update = invalid_update(reason)
update = struct( ...
    'valid', false, ...
    'reason', reason, ...
    'dynamics', [], ...
    'tracking', [], ...
    'applied_steering', [], ...
    'a_tracking', nan(4, 4), ...
    'b_tracking', nan(4, 2), ...
    'model_bus', [] ...
);
end
