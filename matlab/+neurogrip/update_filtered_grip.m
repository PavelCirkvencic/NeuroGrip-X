function [filteredFront, filteredRear] = update_filtered_grip( ...
    previousFront, previousRear, estimatedFront, estimatedRear, alpha)
%UPDATE_FILTERED_GRIP Apply the exact scalar EMA used by the C2 controller.

arguments
    previousFront (1, 1) double {mustBeFinite}
    previousRear (1, 1) double {mustBeFinite}
    estimatedFront (1, 1) double {mustBeFinite}
    estimatedRear (1, 1) double {mustBeFinite}
    alpha (1, 1) double {mustBeFinite} = 0.05
end
if alpha <= 0.0 || alpha > 1.0
    error('neurogrip:InvalidGripFilterAlpha', ...
        'Grip filter alpha must be in (0, 1].');
end
filteredFront = previousFront + alpha * (estimatedFront - previousFront);
filteredRear = previousRear + alpha * (estimatedRear - previousRear);
end
