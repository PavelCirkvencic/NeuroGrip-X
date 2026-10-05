function directory = test_model_directory()
%TEST_MODEL_DIRECTORY Return an OS-temporary directory for generated .slx tests.
%
% Programmatic Simulink saves change ZIP metadata even when the model equations
% are unchanged. Test builds must therefore stay out of versioned `models/` so
% a green test run never dirties the repository's reviewed model artifacts.

directory = fullfile(tempdir, 'neurogrip_matlab_test_models');
if ~isfolder(directory)
    mkdir(directory);
end
end
