# NeuroGrip-X



https://github.com/user-attachments/assets/2de6381b-0af4-43b8-853c-42be63de9655





Grip-aware model predictive control for a Formula Student car in simulation.
NeuroGrip-X estimates front- and rear-axle grip from causal telemetry, adapts a
physical bicycle model, and sends the resulting control problem to MATLAB. ROS 2
owns state estimation, experiment orchestration and the fail-closed actuation
path; EUFS Sim 2 provides the vehicle and track dynamics.

The video is a synchronized replay of recorded EUFS telemetry. The final five
seconds contain a Simulink replay of the accepted front/rear grip estimates.

## Result

The release benchmark contains three frozen FSUK 2023 trackdrive scenarios:
balanced grip loss, a front-limited case and a rear-limited case. Each scenario
uses paired sensor noise, the same 12 m/s speed ceiling and identical safety
constraints.

| Scenario | Fixed-μ MPC | NeuroGrip-X | Reduction |
|---|---:|---:|---:|
| balanced | 30.70 s | 28.94 s | 5.73% |
| front-limited | 30.70 s | 29.22 s | 4.82% |
| rear-limited | 30.74 s | 29.22 s | 4.94% |
| **mean** | **30.71 s** | **29.13 s** | **5.17%** |

All six measured laps completed with zero cone collisions and no boundary
exit. The minimum vehicle-body boundary margin was 0.576 m. MATLAB supplied
every accepted NeuroGrip-X steering candidate after takeover; no active
deadline miss, invalid solution, mux rejection or fallback was recorded.

The comparison is deliberately system-level. The baseline uses a conservative
fixed assumption (`μ=0.75`, 70% utilization). NeuroGrip-X uses 98% of the
uncertainty-reduced weaker-axle estimate. It is not an isolated claim about one
MPC matrix update. The report, per-scenario metrics and artifact hashes are in
[docs/RESULTS.md](docs/RESULTS.md).

## Method

The estimator consumes a 0.5 s history at 50 Hz:

```text
[v_x, v_y, yaw rate, applied steering, acceleration command, a_x, a_y]
                                  |
                       GRU context encoder
                                  |
             low-rank neural Koopman dynamics + F/R grip head
                                  |
                three-model ensemble and calibrated margin
                                  |
        cornering-stiffness adaptation + grip-limited speed profile
```

Injected friction values, scenario identifiers and future samples are not
runtime inputs. The deployed controller uses the learned context to scale
front/rear cornering stiffness and the curvature-dependent speed envelope. A
physical bicycle model remains the control prior.

## Runtime path

```text
EUFS Sim 2
    |
ROS state and path projection ----> neural-Koopman ensemble
    |                                      |
    |                              grip + uncertainty
    |                                      |
    +------ full-horizon control frame -----+
                                           |
                                  localhost TCP bridge
                                           |
                              MATLAB mpcActiveSetSolver
                                           |
                              candidate validation and mux
                                           |
                                ackermann_guard -> /cmd
```

The TCP boundary avoids mixing MATLAB's ROS middleware runtime with the ROS 2
Humble process. Only one request may be in flight, and only the newest pending
frame is retained. The mux rejects stale, late, malformed or invalid solutions;
`ackermann_guard` is the only publisher with vehicle-command authority.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | ROS 2 interfaces, launch packages, logging, transport and safety nodes |
| `controllers/python/` | fixed-μ baseline, shared speed policy and MPC reference implementation |
| `learning/neurogrip/` | neural-Koopman model, data pipeline and uncertainty calibration |
| `experiments_v2/` | scenario runner, recorders, evaluators and video renderer |
| `matlab/` | online MPC implementation, parity tools and Simulink regression models |
| `config/` | seeded development scenarios and frozen holdout manifests |
| `runs/eufs_v1/models/` | versioned model weights and deployment manifests |
| `artifacts/reports/` | accepted benchmark and MATLAB integration evidence |
| `third_party/` | pinned EUFS repositories, notices and local patches |

## Environment

The verified development stack is Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress,
Python 3.10 and MATLAB R2026a with Simulink and Model Predictive Control Toolbox.
The Python and ROS tests do not require a MATLAB license.

From the repository root:

```bash
bash scripts/bootstrap_env.sh
source scripts/env.sh
bash scripts/bootstrap_eufs.sh
bash scripts/build_eufs.sh
colcon build --symlink-install
```

`bootstrap_eufs.sh` imports pinned revisions from `third_party/eufs.repos` and
applies the vehicle-dynamics patches used by the experiments. It does not run
`sudo`; missing system packages are reported before the bootstrap starts.

## Verification

Run the ROS, controller, learning and experiment test suites:

```bash
bash scripts/run_tests.sh
```

Run the MATLAB and Simulink regressions:

```bash
/opt/MATLAB/R2026a/bin/matlab -batch \
  "cd(pwd); addpath(fullfile(pwd,'matlab','tests')); run_pre_matlab_tests"
```

Validate a locally recorded holdout without starting a new simulation:

```bash
python experiments_v2/run_paired_holdout.py \
  --holdout-manifest config/final_holdout_matlab_actuated_v7.json \
  --run-root runs/eufs_v1/final_holdout_matlab_actuated_v7 \
  --dry-run --resume
```

Raw episode telemetry is intentionally excluded from Git. Once those episodes
exist locally, the release replay can be regenerated with:

```bash
bash scripts/render_matlab_holdout_v7.sh
```

## Scope

This is a simulation study, not validation of a real tyre, vehicle or circuit.
The reported advantage applies to the frozen scenarios and controller policies
described above. Current limitations and the permitted claim boundary are
documented in [docs/limitations.md](docs/limitations.md); model and dataset
provenance are in [docs/model_card.md](docs/model_card.md) and
[docs/dataset_card.md](docs/dataset_card.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
