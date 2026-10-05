function fit = load_nominal_fit(path)
%LOAD_NOMINAL_FIT Load and validate the shared NeuroGrip nominal model JSON.
%
% The Python reference controller and the C2 runtime read the same JSON file.
% This function deliberately accepts no hidden MATLAB-only defaults except the
% documented yaw inertia fallback used by the Python implementation.

arguments
    path (1, :) char {mustBeNonzeroLengthText}
end

if ~isfile(path)
    error('neurogrip:MissingNominalFit', 'Nominal fit does not exist: %s', path);
end

fit = jsondecode(fileread(path));
required = [ ...
    "mass_kg", "wheelbase_m", "front_fraction", ...
    "front_cornering_stiffness_n_rad", "rear_cornering_stiffness_n_rad" ...
];
for name = required
    field = char(name);
    if ~isfield(fit, field) || ~isscalar(fit.(field)) || ...
            ~isfinite(fit.(field)) || fit.(field) <= 0
        error('neurogrip:InvalidNominalFit', ...
            'Nominal fit field %s must be a finite positive scalar.', field);
    end
end

if ~isfield(fit, 'yaw_inertia_kgm2')
    fit.yaw_inertia_kgm2 = 172.44;
end
if ~isscalar(fit.yaw_inertia_kgm2) || ~isfinite(fit.yaw_inertia_kgm2) || ...
        fit.yaw_inertia_kgm2 <= 0
    error('neurogrip:InvalidNominalFit', ...
        'Nominal fit yaw_inertia_kgm2 must be a finite positive scalar.');
end
if fit.front_fraction <= 0 || fit.front_fraction >= 1
    error('neurogrip:InvalidNominalFit', ...
        'Nominal fit front_fraction must be strictly between zero and one.');
end
end
