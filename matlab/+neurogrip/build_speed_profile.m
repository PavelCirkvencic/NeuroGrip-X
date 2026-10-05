function profile = build_speed_profile(sM, lengthM, curvature1pm, ...
    frictionCoefficient, lateralUtilisation, maximumSpeedMps, minimumSpeedMps, ...
    accelerationLimitMps2, brakingLimitMps2, passes)
%BUILD_SPEED_PROFILE Port the periodic Python curvature/grip speed planner.

arguments
    sM (:, 1) double {mustBeFinite}
    lengthM (1, 1) double {mustBeFinite}
    curvature1pm (:, 1) double {mustBeFinite}
    frictionCoefficient (1, 1) double {mustBeFinite}
    lateralUtilisation (1, 1) double {mustBeFinite}
    maximumSpeedMps (1, 1) double {mustBeFinite}
    minimumSpeedMps (1, 1) double {mustBeFinite}
    accelerationLimitMps2 (1, 1) double {mustBeFinite}
    brakingLimitMps2 (1, 1) double {mustBeFinite}
    passes (1, 1) double {mustBeInteger, mustBePositive} = 4
end

s = sM(:);
curvature = curvature1pm(:);
if numel(s) < 3 || numel(s) ~= numel(curvature)
    error('neurogrip:InvalidSpeedProfileArrays', ...
        'Speed-profile arrays must have equal non-trivial length.');
end
if abs(s(end) - lengthM) <= 1e-6
    s = s(1:end - 1);
    curvature = curvature(1:end - 1);
end
if any(diff(s) <= 0.0) || ~(lengthM > s(end))
    error('neurogrip:InvalidSpeedProfileArcLength', ...
        'Arc length must increase and remain below periodic length.');
end
limits = [frictionCoefficient, lateralUtilisation, maximumSpeedMps, ...
    minimumSpeedMps, accelerationLimitMps2, brakingLimitMps2];
if any(~isfinite(limits)) || any(limits <= 0.0)
    error('neurogrip:InvalidSpeedProfileLimits', ...
        'Speed-profile limits must be finite and positive.');
end
if minimumSpeedMps > maximumSpeedMps
    error('neurogrip:InvalidSpeedProfileBounds', ...
        'Minimum speed cannot exceed maximum speed.');
end

gravityMps2 = 9.81;
lateralLimitMps2 = frictionCoefficient * lateralUtilisation * gravityMps2;
curveLimit = sqrt(lateralLimitMps2 ./ max(abs(curvature), 1e-5));
speed = min(max(curveLimit, minimumSpeedMps), maximumSpeedMps);
segment = diff([s; lengthM]);
numberOfSamples = numel(speed);

for passIndex = 1:passes
    for index = numberOfSamples:-1:1
        following = mod(index, numberOfSamples) + 1;
        allowed = sqrt(max(speed(following)^2 + ...
            2.0 * brakingLimitMps2 * segment(index), 0.0));
        speed(index) = min(speed(index), allowed);
    end
    for index = 1:numberOfSamples
        previous = mod(index - 2, numberOfSamples) + 1;
        allowed = sqrt(max(speed(previous)^2 + ...
            2.0 * accelerationLimitMps2 * segment(previous), 0.0));
        speed(index) = min(speed(index), allowed);
    end
end

profile = struct( ...
    'progress', s / lengthM, ...
    'speed_mps', speed, ...
    'lateral_limit_mps2', lateralLimitMps2, ...
    'maximum_speed_mps', maximumSpeedMps ...
);
end
