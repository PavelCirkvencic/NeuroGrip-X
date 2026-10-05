function observable = grip_is_observable(vxMps, yawRateRps, appliedSteeringRad)
%GRIP_IS_OBSERVABLE Match the runtime excitation gate for schema-4 updates.

arguments
    vxMps (1, 1) double
    yawRateRps (1, 1) double
    appliedSteeringRad (1, 1) double
end

if any(~isfinite([vxMps, yawRateRps, appliedSteeringRad]))
    observable = false;
    return;
end
lateralExcitationMps2 = abs(vxMps * yawRateRps);
observable = vxMps >= 6.0 && lateralExcitationMps2 >= 1.0 && ...
    abs(appliedSteeringRad) >= 0.01;
end
