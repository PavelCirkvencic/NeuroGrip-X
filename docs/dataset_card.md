# Dataset card — neural Koopman development data

## Collection

Training data comes from EUFS Sim 2 open-loop identification episodes at 50 Hz.
Each scenario defines a seeded combination of axle grip, grip transition,
vehicle parameters and excitation. `record_episode.py` stores physical
telemetry, events and immutable provenance; scenario parameters are never
model inputs.

## Model fields

The causal history has 25 rows and these seven SI-unit features:

```text
v_x_mps
v_y_mps
yaw_rate_rps
steering_applied_rad
acceleration_command_mps2
a_x_mps2
a_y_mps2
```

The physical state target is `[v_y, yaw_rate]`; steering angle is the model
input. Actual EUFS wheel steering is used rather than the requested command.

## Scenario-level split

`config/c2_development_provenance_v2.json` is authoritative:

| Role | Scenario count | Use |
|---|---:|---|
| Train | 6 | weight fitting and feature statistics |
| Validation | 1 | early stopping only |
| Calibration | 1 | residual/OOD calibration |
| Closed-loop development | 1 | controller tuning and threshold-only recalibration |
| Publication holdout | 3 | final paired evaluation only |

Splits are by complete scenario identity, never by shuffled telemetry rows.
Publication seeds 501–503 are absent from every development role.

## Quality and leakage controls

- feature order is stored in every manifest and checked at runtime;
- checkpoints record feature scaling and nominal-model identity;
- all three model members use independent random seeds;
- validation selects epochs; holdout data never selects weights or policy;
- closed-loop recalibration updates uncertainty/OOD thresholds only and records
  source episode hashes; neural weights remain unchanged;
- the publication runner verifies scenario YAML bytes and model manifest hash
  before starting any ROS process.

## Scope

The development set is intentionally small and simulation-only. It covers the
grip variations needed for this portfolio benchmark, not the full Formula
Student operating domain. The final claim is therefore restricted to the
frozen EUFS grip-transition holdout described in `docs/RESULTS.md`.
