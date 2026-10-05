function run_pre_matlab_tests()
%RUN_PRE_MATLAB_TESTS Stable entry point for non-interactive MATLAB validation.

test_nominal_zoh_parity();
test_python_equivalent_mpc_parity();
test_latency_compensated_mpc_move();
test_schema4_policy_parity();
test_matlab_parity_frame_contract();
test_adaptive_model_bus_contract();
test_ros_flat_contract();
test_online_mpc_plant_from_update();
test_adaptive_mpc_smoke();
test_shadow_mpc_configuration();
test_programmatic_simulink_plant();
test_programmatic_c0_closed_loop();
test_programmatic_adaptive_mpc_closed_loop();
test_ros2_adaptive_model_adapter();
test_ros2_matlab_parity_adapter();
test_ros2_matlab_control_adapter();
fprintf('PRE_MATLAB_TESTS_PASS\n');
end
