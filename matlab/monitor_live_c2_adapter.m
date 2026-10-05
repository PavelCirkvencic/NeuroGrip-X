function result = monitor_live_c2_adapter(durationS, minimumValidSamples)
%MONITOR_LIVE_C2_ADAPTER Verify live ROS C2 packets without commanding a car.
%
% Start a C2_NEUROGRIP EUFS episode first, then run this function from the
% repository root.  It subscribes and validates only; it contains no ROS
% publisher and cannot bypass ackermann_guard.

arguments
    durationS (1, 1) double {mustBePositive, mustBeFinite} = 10.0
    minimumValidSamples (1, 1) double {mustBeInteger, mustBePositive} = 20
end

adapter = neurogrip.Ros2AdaptiveModelAdapter('/neurogrip_matlab_live_monitor', 0.20);
cleanup = onCleanup(@() delete(adapter)); %#ok<NASGU>
clock = tic;
validSamples = 0;
lastReason = 'no_packets';
maxSpectralRadius = NaN;
while toc(clock) < durationS
    update = adapter.poll();
    if update.valid
        validSamples = validSamples + 1;
        maxSpectralRadius = max_or_first(maxSpectralRadius, ...
            max(abs(eig(update.dynamics.a_lateral))));
    else
        lastReason = update.reason;
    end
    pause(0.02);
end
result = struct( ...
    'duration_s', durationS, ...
    'valid_samples', validSamples, ...
    'max_lateral_spectral_radius', maxSpectralRadius, ...
    'last_invalid_reason', lastReason ...
);
assert(validSamples >= minimumValidSamples, ...
    ['Live C2 MATLAB adapter received only %d valid samples (need %d); ' ...
    'last reason: %s.'], validSamples, minimumValidSamples, lastReason);
fprintf(['LIVE_C2_ADAPTER_PASS duration_s=%.2f valid_samples=%d ' ...
    'max_lateral_spectral_radius=%.6f\n'], result.duration_s, ...
    result.valid_samples, result.max_lateral_spectral_radius);
end

function value = max_or_first(current, candidate)
if isnan(current)
    value = candidate;
else
    value = max(current, candidate);
end
end
