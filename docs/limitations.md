# Limitations and claim boundary

The accepted result is limited to the three frozen FSUK-2023 trackdrive
scenarios in `final_holdout_matlab_actuated_v7.json`: balanced, front-limited
and rear-limited grip transitions with nominal zero added actuator delay.

Within that scope, all six laps completed without cone collisions or boundary
exits and MATLAB-actuated NeuroGrip-X reduced paired lap time by 5.17% on
average (4.82–5.73%). This is a simulation result, not a real-car claim or
proof for arbitrary tracks.

## Known limitations

- EUFS tyre, sensor and actuator parameters are not calibrated to an FSB
  Racing Team vehicle.
- C2's advantage is system-level: causal per-axle context adapts physical
  cornering stiffness and an uncertainty-aware speed profile. It is not an
  identical-speed, matrix-only ablation.
- C2 lateral RMS is slightly higher because it drives faster. Minimum body
  margin across all accepted laps remains 0.576 m.
- Direct learned-Jacobian blending is disabled (`c2_model_blend=0`). Neural
  Koopman context and the grip head remain active; the physical bicycle prior
  supplies the final local A/B after per-axle stiffness adaptation.
- The 0.5 s causal model needs sufficient speed, steering and lateral
  excitation. Before that, C2 uses the declared fallback estimate.
- Conformal error margins and Mahalanobis OOD detection are calibrated to the
  simulation development distribution, not a formal safety guarantee.
- Added actuator-delay robustness is not established. The accepted scope uses
  one-step compensation for measured compute/transport latency but no injected
  mechanical steering delay.
- The release video is a replay of recorded EUFS telemetry, not a Gazebo
  camera capture.
- Simulink contains executable nominal and adaptive MPC models and automated
  tests, but a MATLAB function using `mpcActiveSetSolver` actuated v7; a
  Simulink block did not command the accepted laps.

## Safety boundary

- `ackermann_guard` is the sole `/cmd` publisher.
- The neural node publishes estimates and dynamics only; it never commands the
  vehicle.
- The TCP node is transport-only and keeps one request in flight with at most
  one newest pending request, preventing stale control backlog.
- The candidate mux requires a valid streak before takeover and rejects stale,
  malformed, invalid, late or out-of-range MATLAB results.
- Both controllers share the same path, 12 m/s global cap, startup speed cap,
  steering/rate limits, acceleration limits and final command guard.
- C2 subtracts ensemble standard deviation and a calibrated q90 margin before
  computing the weaker-axle speed limit.
- The evaluator rejects boundary exits, collisions, invalid QPs, source
  fallbacks, stale diagnostics, provenance mismatch and insufficient
  post-transition grip-policy activity.

## Unsupported claims

Do not claim that NeuroGrip-X:

- is validated on a physical vehicle;
- always beats every MPC or an oracle with full tyre-state knowledge;
- has lower tracking error than C0;
- is robust to arbitrary actuator delay or sensor failures;
- receives ground-truth grip at runtime;
- was actuated by a Simulink block in the accepted holdout;
- produced a Gazebo-camera video.

A truthful summary is: ROS 2 and EUFS supplied the closed-loop simulation,
PyTorch estimated causal per-axle grip context, MATLAB solved C2 steering
online through a fail-closed transport and mux, and Simulink supplied
executable MPC integration/regression models.
