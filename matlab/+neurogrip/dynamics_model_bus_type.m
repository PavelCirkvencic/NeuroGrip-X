function busType = dynamics_model_bus_type()
%DYNAMICS_MODEL_BUS_TYPE Define the versioned schema-4 Simulink input bus.
%
% The bus contains only deterministic numeric fields needed downstream of
% `/neurogrip/dynamics_flat`. Parser validity/reason handling remains outside
% the bus so malformed packets can never masquerade as controller inputs.

names = { ...
    'schema_version', 'sample_time_s', 'a_lateral', 'b_lateral', ...
    'ensemble_std', 'conformal_radius', 'ood_score', 'latency_ms', ...
    'estimated_front_grip', 'estimated_rear_grip', ...
    'front_grip_std', 'rear_grip_std', ...
    'front_grip_error_q90', 'rear_grip_error_q90' ...
};
dimensions = {1, 1, [2 2], [2 1], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
dataTypes = [{'uint32'}, repmat({'double'}, 1, numel(names) - 1)];
elements(1, numel(names)) = Simulink.BusElement;
for index = 1:numel(names)
    elements(index) = Simulink.BusElement;
    elements(index).Name = names{index};
    elements(index).DataType = dataTypes{index};
    elements(index).Dimensions = dimensions{index};
    elements(index).DimensionsMode = 'Fixed';
    elements(index).Complexity = 'real';
    elements(index).SampleTime = -1;
end
busType = Simulink.Bus;
busType.Elements = elements;
busType.DataScope = 'Auto';
busType.Alignment = -1;
busType.PreserveElementDimensions = 0;
end
