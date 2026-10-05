# NeuroGrip-X v2 ROS data contract

This is the active ROS/MATLAB contract. All timestamps use `/clock` with
`use_sim_time=true` during an EUFS episode.

| Topic | Type | Producer | Consumer |
|---|---|---|---|
| `/odom` | `nav_msgs/Odometry` | EUFS | tracking, recorder, AI |
| `/imu/data` | `sensor_msgs/Imu` | EUFS | recorder, AI |
| `/ros_can/wheel_speeds` | `eufs_msgs/WheelSpeedsStamped` | EUFS | controller, recorder, AI |
| `/neurogrip/applied_steering_flat` | `std_msgs/Float64MultiArray` | read-only wheel bridge | MATLAB shadow only |
| `/neurogrip/tracking_state` | `std_msgs/Float64MultiArray` | tracking adapter | controller, recorder, AI |
| `/neurogrip/dynamics_model` | `neurogrip_interfaces/DynamicsModel` | C2 AI | C2 controller, logger |
| `/neurogrip/matlab_parity_frame` | `std_msgs/Float64MultiArray` | C2 controller | read-only parity recorder/MATLAB |
| `/neurogrip/matlab_control_frame` | `std_msgs/Float64MultiArray` | C2 controller | MATLAB live exact-QP adapter |
| `/neurogrip/oracle_mu` | `std_msgs/Float64` | scenario manager | C1 only |
| `/neurogrip/command_candidate/python` | `AckermannDriveStamped` | Python controller | candidate mux |
| `/neurogrip/command_candidate/matlab_flat` | `std_msgs/Float64MultiArray` | MATLAB MPC | candidate mux |
| `/neurogrip/command_mux_status` | `std_msgs/Float64MultiArray` | candidate mux | evidence recorder |
| `/neurogrip/command_candidate` | `AckermannDriveStamped` | candidate mux | guard |
| `/neurogrip/command_safe` | `AckermannDriveStamped` | guard | recorder, AI |
| `/cmd` | `AckermannDriveStamped` | guard only | EUFS |

## Flat schemas

`tracking_state` schema 1:

```text
[1, stamp_s, e_y_m, e_psi_rad, v_y_mps, yaw_rate_rps,
 v_x_mps, curvature_1pm, d_kappa_rad_s, lap_progress_0_to_1]
```

`dynamics_flat` legacy schemas 2/3:

```text
[2, sample_time_s, valid_0_or_1, A11, A12, A21, A22, B1, B2,
 ensemble_prediction_std, conformal_radius, mahalanobis_ood, latency_ms]
```

`dynamics_flat` grip-aware schema 4:

```text
[4, sample_time_s, valid_0_or_1, A11, A12, A21, A22, B1, B2,
 ensemble_prediction_std, conformal_radius, mahalanobis_ood, latency_ms,
 estimated_front_grip, estimated_rear_grip,
 front_grip_std, rear_grip_std,
 front_grip_error_q90, rear_grip_error_q90]
```

Schema 4 always has exactly 19 finite values. A 13-value packet labelled as
schema 4 is invalid; MATLAB must not silently invent missing grip uncertainty.

The physical `DynamicsModel` is authoritative. `dynamics_flat` is the
versioned standard-message Simulink bridge and must be decoded with this exact
ordering. Python flattens matrices in row-major order; MATLAB explicitly
transposes the 2-by-2 reshape.

`applied_steering_flat` schema 1:

```text
[1, stamp_s, measured_steering_rad]
```

`matlab_parity_frame` schema 3 is a fixed 155-value, self-contained controller-decision
snapshot. It carries raw and filtered schema-4 grip, uncertainty, the exact
held uncertainty state, lap progress, speed-policy limits, raw lateral A/B, lifted tracking A/B,
all three 30-step previews and the Python first move.
`experiments_v2/record_matlab_parity.py` records it without
publishing commands. The detailed field order is executable in
`record_matlab_parity.FRAME_FIELDS` and decoded by
`neurogrip.parse_matlab_parity_frame`.
`Ros2MatlabParityAdapter` is a subscriber-only live decoder with a 200 ms
receipt watchdog. `monitor_live_python_equivalent_shadow` uses the complete
frame with `mpcActiveSetSolver`; it owns no publisher and cannot command EUFS.

`matlab_control_frame` schema 1 has exactly 125 finite values:

```text
[schema, control_time_s, sample_time_s, v_x_mps, previous_steering_rad,
 state_4, A_tracking_16, B_tracking_8,
 curvature_preview_30, left_corridor_preview_30,
 right_corridor_preview_30, python_first_move_rad, solution_valid]
```

`matlab_tcp_bridge` validates and transfers this frame to
`run_live_matlab_tcp_candidate`, which applies one-step latency compensation
and solves the same full preview QP with `mpcActiveSetSolver`. Its output
schema is:

```text
[1, control_time_s, steering_rad, solution_valid, solve_time_ms]
```

Only one TCP request is in flight and at most the newest pending frame is
retained, so obsolete frames cannot queue. The candidate mux rejects
malformed, stale, non-monotonic, out-of-range or over-20-ms packets. MATLAB
takeover requires a 3 s controller-stabilisation delay and 10 consecutive
valid candidates. Before takeover it uses the Python candidate; after takeover
any invalid MATLAB packet fails over to Python, or to a safe stop if Python is
also stale. The mux changes only steering and retains the shared Python
longitudinal command.

`applied_steering_bridge` receives only `/ros_can/wheel_speeds` and republishes
this standard-message mirror for MATLAB, which cannot consume the EUFS custom
message without generated support. It is a telemetry bridge, has no command
subscriber and must never be made an actuator path.

## Authority and leakage rules

- Exactly one publisher may exist on `/cmd`: `ackermann_guard`.
- MATLAB never publishes `/cmd` or the guard input directly.
- C0 and C2 must not subscribe to `/neurogrip/oracle_mu`.
- Injected scenario parameters are run metadata only and cannot be model
  features.
- `steering_applied` always comes from `WheelSpeedsStamped.speeds.steering`, not
  `/neurogrip/command_safe` or `/neurogrip/command_candidate`.
- MATLAB shadow uses the same measured steering as `mpcstate.LastMove`; a stale,
  malformed, out-of-bound or timestamp-mismatched bridge packet fails closed.
