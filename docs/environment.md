# Environment and reproducibility

## Verified stack

| Component | Version / location |
|---|---|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble (`/opt/ros/humble`) |
| Gazebo | Ignition Fortress 6.18 (`ign`, `ignition.msgs.*`) |
| Python (venv) | 3.10.12, created with `--system-site-packages` |
| PyTorch | 2.11.0+cu128 (CUDA available, RTX 3050) |
| NumPy / pandas / scikit-learn | 1.26.4 / 2.3.3 / 1.7.2 |
| MATLAB | R2026a (26.1.0) at `/opt/MATLAB/R2026a` |
| Simulink / ROS Toolbox / MPC Toolbox | installed under `/opt/MATLAB/R2026a/toolbox` |
| `mpcmoveAdaptive.m` | `/opt/MATLAB/R2026a/toolbox/mpc/mpc/@mpc/mpcmoveAdaptive.m` |

## Setup (from a clean checkout)

```bash
cd NeuroGrip-X
bash scripts/bootstrap_env.sh          # creates .venv and patches activate
source scripts/env.sh                  # venv + ROS + overlay + PYTHONPATH
colcon build --symlink-install
```

`scripts/bootstrap_env.sh` installs `requirements.txt` into the workspace venv
and appends a `PYTHONPATH` export to `.venv/bin/activate`. This is required
because colcon's pytest runner executes under system Python; the export lets it
import torch/numpy from the venv. It is an environment configuration, not a
global package install.

## Verification

```bash
python scripts/preflight_check.py          # 20 checks, non-zero on missing deps
python scripts/preflight_check.py --allow-missing-matlab   # CI without MATLAB
```

## One-command test entry point

```bash
bash scripts/run_tests.sh
```

This runs `colcon build`, `colcon test` (including lint), `colcon test-result`
and the standalone `learning/` + `experiments/` pytest suites. No Gazebo or
MATLAB process is started.

Local CI including preflight and worktree hygiene:

```bash
bash scripts/ci_check.sh
```

## Gazebo notes

- GUI rendering depends on EGL and is flaky on this host; automated recordings
  use `headless:=true` (server-only) which has proven reliable.
- If a run reports `Publisher count: 0` although the server log shows the world
  loading, clear stale FastRTPS shared memory before retrying:
  `rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*` (with no ROS running).
- The managed orchestrator `experiments/run_episode_batch.py` only signals its
  own process group and processes whose command line references the episode's
  unique world file; it never uses broad process kills.

## Reproducibility rules

- Scenario generation is seeded: `ros2 run neurogrip_sim scenario_builder
  <profile> --seed <n>`. Historical profiles keep byte-identical SDFs.
- Dataset splits are assigned to complete `scenario_id`s, never rows.
- Every model artifact records the catalog SHA-256, Git commit, seed,
  hyperparameters, normalization and feature/target schema.
- `experiments/benchmark_summary.py` refuses to combine artifacts whose catalog
  SHA, split schema or feature/target schema disagree (non-zero exit).
