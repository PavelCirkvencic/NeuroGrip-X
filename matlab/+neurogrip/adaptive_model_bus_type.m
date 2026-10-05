function busType = adaptive_model_bus_type()
%ADAPTIVE_MODEL_BUS_TYPE Define a strongly typed online-model Simulink bus.
%
% Keeping the field names and dimensions here prevents silent flattening or a
% latent-network vector from being wired into the Adaptive MPC block.

names = {'A', 'B', 'C', 'D', 'X', 'Y', 'U', 'DX'};
dimensions = {[4 4 31], [4 2 31], [4 4 31], [4 2 31], ...
    [4 1 31], [4 1 31], [2 1 31], [4 1 31]};
elements(1, numel(names)) = Simulink.BusElement;
for index = 1:numel(names)
    elements(index) = Simulink.BusElement;
    elements(index).Name = names{index};
    elements(index).DataType = 'double';
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
