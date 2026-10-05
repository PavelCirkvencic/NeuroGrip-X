# MATLAB / Simulink parity layer

This folder is the MATLAB implementation boundary for NeuroGrip-X. It does
not replace the Python/EUFS final-holdout result. Its first task is stricter:
produce the same physical discrete lateral bicycle model as the frozen Python
reference before an MPC or Simulink model can be trusted.

## First automated gate: nominal ZOH parity

`data/nominal_zoh_golden_v1.json` stores independent Python reference vectors
created with `controllers.python.bicycle_model` and
`scipy.signal.cont2discrete(..., 'zoh')`. The MATLAB implementation uses an
augmented `expm` ZOH calculation, so it does not depend on an implicit
Simulink discretisation setting.

Run from the repository root:

```bash
/opt/MATLAB/R2026a/bin/matlab -batch "addpath('matlab/tests'); run_pre_matlab_tests"
```

The expected terminal line is `PRE_MATLAB_TESTS_PASS`.

The same batch test also calls `mpcmoveAdaptive` with an explicitly supplied
online state-space model. It only proves the MATLAB Adaptive MPC API and basic
constraints are live; it does not yet claim a Python-to-MATLAB first-move
match. That requires a separate golden-move test because the Python reference
uses a custom OSQP formulation.

`test_python_equivalent_mpc_parity.m` is the stricter controller-mathematics
gate. It translates the Python QP exactly and solves it with MATLAB
`quadprog`, then compares five first steering moves against Python OSQP golden
vectors. This keeps an exact, inspectable parity reference while the standard
Adaptive MPC object is prepared for online-model integration.

`build_neurogrip_controller.m` also programmatically creates
`matlab/models/neurogrip_c0_nominal_lateral_plant.slx` from the same JSON
parameters and runs it headlessly. The model is a fixed-speed two-state
lateral plant for the C0 MPC wrapper; it is deliberately not labelled as the
future adaptive closed loop.

`build_neurogrip_c0_closed_loop.m` creates a second model,
`neurogrip_c0_closed_loop.slx`, that uses the actual Simulink MPC Controller
block around the Python-compatible 4-state tracking plant. Its headless smoke
test checks finite signals, steering magnitude/rate limits and the expected
initial corrective steering direction. A short 0.4 s smoke must not be
misrepresented as a complete vehicle-level tracking benchmark.

## Adaptive MPC online-model bus gate

`build_neurogrip_adaptive_mpc.m` creates
`matlab/models/neurogrip_adaptive_mpc_closed_loop.slx`. It contains the actual
**Adaptive MPC Controller** block and a named online-model bus with precisely
the documented fields `A`, `B`, `C`, `D`, `X`, `Y`, `U`, and `DX`.

`DynamicsModel.msg` publishes the C2 physical 2-state discrete matrices
`A_dyn`, `B_dyn` for `[v_y; yaw_rate]`. The function
`neurogrip.build_tracking_plant_from_lateral` lifts those physical matrices to
the shared four-state tracking plant `[e_y; e_psi; v_y; yaw_rate]`; it never
uses normalised network features or a latent state. `make_adaptive_model_bus`
then repeats that local model over 31 pages: one present model plus 30 future
prediction steps. That 31-page shape is required by the installed Adaptive
MPC block for the release controller's prediction horizon of 30 and is covered by the
headless contract test.

The prototype feeds the nominal physical model through this same bus. Its
regression test then replaces only online `A/B` with a valid asymmetric
low-grip model and requires the steering trajectory to change. This proves the
Adaptive MPC model input is active rather than decorative. It remains an
**online-model-bus integration test**, not a vehicle-level performance claim.

## Schema-4 golden parity

`scripts/export_matlab_schema4_golden.py` deterministically exports the Python
release controller's observability gate, 0.05 EMA, calibrated uncertainty
subtraction, weaker-axle rule, periodic speed profile, tracking recovery rule,
grip-scaled matrices and 30-step OSQP first moves. The MATLAB test
`test_schema4_policy_parity.m` checks 40 sequential policy samples and 20 full
model/control vectors. Current maximum errors are approximately `6.7e-16` for
`A/B` and `1.8e-6 rad` for the first move; the latter reflects the deployed
OSQP real-time tolerance versus MATLAB's tighter `quadprog` solution.

The deployed Python controller and MATLAB now share a 30-step horizon and a
`2.5 rad/s` steering-rate limit. Older MATLAB artifacts using 20 steps and
`0.39 rad/s` are superseded.

## Live ROS 2 monitor (read-only)

`Ros2AdaptiveModelAdapter` subscribes to the standard
`/neurogrip/dynamics_flat` and `/neurogrip/tracking_state` topics. It decodes
Python's row-major matrix wire order, validates schemas, dimensions, stability
and receipt-time freshness, then prepares the physical 31-page Adaptive MPC
bus. It has **no ROS publisher**. An invalid or stale packet returns
`valid=false`; a future control supervisor must select the existing nominal C0
fallback in that case.

With a C2 EUFS episode already running, the following command verifies real
MATLAB ROS 2 DDS traffic without sending a vehicle command:

```bash
/opt/MATLAB/R2026a/bin/matlab -batch "addpath('matlab'); monitor_live_c2_adapter(10, 20)"
```

The batch regression suite also exercises a real local MATLAB ROS 2
publisher/subscriber callback exchange; it does not require Gazebo.

For an additional no-actuation end-to-end check, the shadow monitor also
requires `/neurogrip/applied_steering_flat`. The launch starts its source,
`applied_steering_bridge`, which is a one-way mirror of EUFS measured wheel
steering in a versioned `Float64MultiArray`; it has no command input. The
monitor sets all four measured tracking states in `mpcstate.Plant`, uses this
measured wheel angle as `mpcstate.LastMove`, removes default output-disturbance
integrators, and selects MATLAB's custom-estimator mode. It then converts the
live C2 packet to a physical online `ss` plant and calls `mpcmoveAdaptive`, but
records the resulting steering moves instead of publishing them:

```bash
/opt/MATLAB/R2026a/bin/matlab -batch "addpath('matlab'); monitor_live_adaptive_mpc_shadow(10, 20)"
```

Pass a third argument to write a CSV for a timestamp-aligned diagnostic:

```bash
/opt/MATLAB/R2026a/bin/matlab -batch "addpath('matlab'); monitor_live_adaptive_mpc_shadow(28, 40, '/tmp/neurogrip_shadow.csv')"
python experiments_v2/evaluate_shadow_comparison.py \
  --episode-csv /absolute/path/to/episode.csv \
  --shadow-csv /tmp/neurogrip_shadow.csv \
  --output-json /tmp/neurogrip_shadow_comparison.json --min-samples 40
```

The generic Adaptive MPC monitor is an integration diagnostic, not the exact
release QP. On the current 8 m/s schema-4 lap it completed safely but differed
from Python by `0.0466 rad` MAE because it consumed a single local model and
constant disturbance without the release controller's 30-step curvature and
corridor preview. It was therefore correctly rejected as an actuation
candidate. The older 2026-10-02 showcase agreement is retained as historical
development evidence, not as the current acceptance gate.

`monitor_live_python_equivalent_shadow.m` is the exact live gate. It subscribes
only to `/neurogrip/matlab_parity_frame`, reconstructs the complete release QP
and solves it with Model Predictive Control Toolbox's `mpcActiveSetSolver`.
On the accepted development run it independently recomputed 773 decisions
during a complete 8 m/s lap: median/max first-move errors were
`1.07e-6/6.41e-4 rad`, median/p95 solve times were `2.37/3.72 ms`, maximum solve
time was `7.15 ms`, and there were zero 20 ms deadline misses and zero sign
mismatches. The hash-bound result is in
`artifacts/reports/matlab_exact_live_shadow_v1.json`.

The accepted actuation path is deliberately guarded.
`/neurogrip/matlab_control_frame` carries the current state, full A/B sequence
and 30-step curvature/corridor previews. `matlab_tcp_bridge` moves one current
frame at a time to the localhost server in
`run_live_matlab_tcp_candidate.m`; MATLAB applies one-step delay compensation,
solves the full QP with `mpcActiveSetSolver` and returns a schema-1 candidate.
The bridge performs no control calculation and retains at most the newest
pending frame, preventing a stale backlog.

`candidate_mux` validates bounds, freshness, monotonicity, the 20 ms deadline,
a 3 s startup delay and a 10-sample valid streak before replacing Python
steering. MATLAB never publishes `/cmd`. The TCP boundary isolates MATLAB's
middleware runtime from ROS 2 Humble after direct DDS exposed an incompatible
Fast DDS runtime; ROS remains on both sides of the transport-only bridge.

In the accepted v7 performance holdout MATLAB actuated all three C2 laps. From
takeover through lap completion there were zero invalid solutions, deadline
misses, source changes, mux rejections or fallbacks. All 6 C0/C2 laps completed
without a collision or boundary exit, and C2 reduced mean paired lap time by
5.17%. See `artifacts/reports/final_holdout_matlab_actuated_v7/`.

`run_live_matlab_candidate.m` is retained as the direct-ROS development path;
it is not the transport used by the accepted v7 runs.

Programmatic Simulink regression tests build their disposable models under the
operating system temporary directory. This is intentional: MATLAB updates ZIP
metadata inside `.slx` files even if no equation changed, so test execution
must not make a clean Git worktree appear modified.

## Contracts that must remain fixed

- State is `[v_y; yaw_rate]` in SI units; steering angle is the sole input.
- `sample_time_s = 0.02` and zero-order hold are fixed.
- Longitudinal speed is clamped to at least `1.0 m/s`, matching C2 runtime.
- `runs/eufs_v1/models/nominal_fit.json` is the shared source of vehicle
  parameters. The golden JSON pins its current SHA-256.
- A later Adaptive MPC consumes physical `A_dyn`, `B_dyn`, never a normalised
  network latent state.
- The Adaptive MPC model bus carries 31 model pages for this 30-step
  controller horizon. Do not flatten or truncate these arrays at the ROS 2
  boundary.

The accepted seed-823 episode CSV predates raw schema-4 parity logging: it
contains the conservative profile grip and final command, but not the raw
neural `A/B`, axle standard deviations or q90 values. It therefore cannot
honestly prove sample-exact neural-message replay by itself. Preserve that
episode unchanged; use a new explicitly labelled MATLAB parity capture for
the remaining recorded-log gate.

That gate is now implemented by `/neurogrip/matlab_parity_frame` schema 3 and
`experiments_v2/record_matlab_parity.py`. The self-contained frame freezes 155
numeric values at the actual Python MPC decision: raw/filtered/held grip state,
policy parameters, state, lap progress, raw and lifted matrices, corridor and
curvature previews, applied steering and first move. It is telemetry-only.

`replay_matlab_parity_capture.m` processed all 1,398 valid decisions from the
safety-complete 8 m/s development lap. Results are versioned in
`artifacts/reports/matlab_schema4_recorded_parity_v1.json`: policy and target
speed errors were exactly zero, maximum A/B error was `8.88e-16`, median first
move error was `5.33e-7 rad`, maximum first-move error was `4.58e-4 rad`, and
there were zero sign or saturation mismatches. This is the recorded-log
parity gate; it is not a new performance benchmark.
