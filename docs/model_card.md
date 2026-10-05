# Model card — grip-aware contextual neural Koopman ensemble

## Intended use

The model supplies causal front/rear grip context and local lateral-dynamics
diagnostics to the NeuroGrip-X MPC in EUFS simulation. It is read-only and is
not approved for a physical vehicle.

## Inputs and outputs

Each of three members consumes 25 causal samples at 50 Hz (0.5 s):

```text
[v_x, v_y, yaw_rate, applied_steering, acceleration_command, a_x, a_y]
```

It does not receive grip labels, scenario ID, seed, split identity or future
telemetry. `DynamicsModel` schema 4 publishes:

- physical 2×2/2×1 discrete local dynamics diagnostics;
- bounded front/rear grip estimates;
- per-axle ensemble standard deviation;
- calibrated per-axle q90 absolute-error margins;
- artifact identity and runtime health.

## Architecture

Every seed uses a GRU causal context encoder, learned Koopman lift and
context-conditioned low-rank operator around a physical bicycle prior. An
auxiliary two-output head maps the same causal context through SiLU layers to
bounded `[0.35, 1.30]` front/rear grip scales.

The release ensemble contains seeds 42, 43 and 44. Runtime checkpoints are
SHA-256 verified. The controller applies an EMA only when grip is observable
(`v_x ≥ 6 m/s`, lateral excitation and steering thresholds), then subtracts
ensemble/calibration uncertainty before using the weaker axle.

## Data and acceptance

Training mixes high-speed identification trajectories and closed-loop
trackdrive telemetry. Validation and calibration scenarios are disjoint from
training and from the release holdout.

| Split | Front MAE | Rear MAE |
|---|---:|---:|
| validation | 0.0410 | 0.0149 |
| calibration | 0.0228 | 0.0439 |

The predeclared per-axle MAE ceiling was 0.08; the maximum observed validation
or calibration MAE was 0.0439. Simulation grip labels are targets for offline
evaluation only and never runtime inputs.

## Release integration

The release controller uses schema-4 estimates to scale physical front/rear
cornering stiffness and build an uncertainty-reduced speed profile. Direct
blending of the raw learned A/B Jacobian is set to zero after development
tests showed poorer steering phase. This should be described as
"neural-Koopman context-conditioned physical MPC", not as a fully learned
dynamics controller.

The final holdout requires one immutable artifact identity, activation before
the transition and at least 90% post-transition NeuroGrip speed-policy
activity. All three release runs had 100% post-transition policy activity.

## Limitations

The per-axle outputs are simulation-calibrated estimates, not measurements of
physical tyre friction. OOD and conformal calibration are distribution
dependent. See [limitations.md](limitations.md) and [RESULTS.md](RESULTS.md).
