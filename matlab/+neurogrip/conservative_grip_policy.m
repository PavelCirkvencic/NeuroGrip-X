function [effectiveGrip, frontGrip, rearGrip] = conservative_grip_policy( ...
    filteredFront, filteredRear, frontStd, rearStd, ...
    frontErrorQ90, rearErrorQ90, marginScale)
%CONSERVATIVE_GRIP_POLICY Apply calibrated margins and weaker-axle rule.

arguments
    filteredFront (1, 1) double {mustBeFinite}
    filteredRear (1, 1) double {mustBeFinite}
    frontStd (1, 1) double {mustBeFinite}
    rearStd (1, 1) double {mustBeFinite}
    frontErrorQ90 (1, 1) double {mustBeFinite}
    rearErrorQ90 (1, 1) double {mustBeFinite}
    marginScale (1, 1) double {mustBeFinite} = 0.50
end
if any([frontStd, rearStd, frontErrorQ90, rearErrorQ90] < 0.0)
    error('neurogrip:InvalidGripUncertainty', ...
        'Grip uncertainty values must be non-negative.');
end
if marginScale < 0.0 || marginScale > 1.0
    error('neurogrip:InvalidGripMarginScale', ...
        'Grip margin scale must be in [0, 1].');
end
frontMargin = frontStd + marginScale * frontErrorQ90;
rearMargin = rearStd + marginScale * rearErrorQ90;
frontGrip = min(max(filteredFront - frontMargin, 0.35), 1.30);
rearGrip = min(max(filteredRear - rearMargin, 0.35), 1.30);
effectiveGrip = min(frontGrip, rearGrip);
end
