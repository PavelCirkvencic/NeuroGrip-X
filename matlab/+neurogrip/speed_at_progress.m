function speedMps = speed_at_progress(profile, lapProgress)
%SPEED_AT_PROGRESS Interpolate one periodic speed-profile target.

arguments
    profile (1, 1) struct
    lapProgress (1, 1) double {mustBeFinite}
end
query = mod(lapProgress, 1.0);
source = [profile.progress(:); 1.0];
values = [profile.speed_mps(:); profile.speed_mps(1)];
speedMps = interp1(source, values, query, 'linear');
end
