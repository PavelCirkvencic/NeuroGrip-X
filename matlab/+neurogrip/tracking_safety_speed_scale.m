function scale = tracking_safety_speed_scale(lateralErrorM, headingErrorRad)
%TRACKING_SAFETY_SPEED_SCALE Port the shared Python tracking recovery rule.

arguments
    lateralErrorM (1, 1) double {mustBeFinite}
    headingErrorRad (1, 1) double {mustBeFinite}
end
severity = max(abs(lateralErrorM) / 0.25, abs(headingErrorRad) / 0.10);
if severity <= 1.0
    scale = 1.0;
    return;
end
scale = min(max(1.0 - 0.45 * (severity - 1.0), 0.45), 1.0);
end
