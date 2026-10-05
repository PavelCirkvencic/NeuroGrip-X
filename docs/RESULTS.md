# MATLAB-actuated release-holdout results

The accepted result comes from
`config/final_holdout_matlab_actuated_v7.json`. Scenario bytes, controller
order, model identity, speed-policy parameters and the acceptance rule were
frozen before the v7 episodes ran. No v7 episode was repeated by outcome.

| Scenario | Controller | Actuator | Lap [s] | RMS e_y [m] | Min margin [m] |
|---|---|---|---:|---:|---:|
| balanced seed 933 | C0 fixed-μ | Python | 30.70 | 0.1033 | 0.633 |
| balanced seed 933 | C2 NeuroGrip-X | MATLAB | 28.94 | 0.1251 | 0.576 |
| front-limited seed 931 | C0 fixed-μ | Python | 30.70 | 0.1019 | 0.654 |
| front-limited seed 931 | C2 NeuroGrip-X | MATLAB | 29.22 | 0.1198 | 0.590 |
| rear-limited seed 932 | C0 fixed-μ | Python | 30.74 | 0.1060 | 0.630 |
| rear-limited seed 932 | C2 NeuroGrip-X | MATLAB | 29.22 | 0.1224 | 0.595 |

Aggregate:

- mean C0 lap time: **30.71 s**;
- mean C2 lap time: **29.13 s**;
- mean paired reduction: **5.17%**;
- paired range: **4.82–5.73%**;
- safety-valid completed laps: **6/6**;
- cone collisions and boundary exits: **0**;
- minimum vehicle-body track margin: **0.576 m**;
- publication target: 5%, acceptance floor: 4% mean with every pair positive
  — **PASS**.

All three C2 episodes passed the strict MATLAB authority gate. After takeover,
MATLAB was the active source continuously through lap completion with zero
invalid solutions, deadline misses, mux rejections or fallbacks. The measured
p95 solve time was approximately 4 ms and the maximum remained below 13.5 ms,
under the declared 20 ms deadline.

The evaluator measures only `experiment_start` through a safety-valid
`lap_complete`. Any boundary exit, collision, invalid QP, stale diagnostics,
source change, provenance mismatch or insufficient post-transition
NeuroGrip-policy activity invalidates publication.

## Integrity identifiers

| Artifact | SHA-256 |
|---|---|
| v7 holdout manifest | `bbc1aef3503fb66f452a07cecd3b33c954ef9972f84eb75cae285bc6eec16630` |
| deployed ensemble manifest | `896dc9dd8589238d1b88cee68caf7d53be0c3978bfc00766bec301068e039702` |
| accepted benchmark JSON | `24c4b73dd42e5ca3c8897497a3bfc7d17c137d5c8443bb838b244d192a9d7801` |
| release MP4 | `03e5eac21a3f3b1c972891d32f50311c234610022a70f717ae98fbddef91a6d9` |
| release poster | `82365b5ecc596e9bbda5acc78ff519f19ee93463ef6198f1ba416fa6df2e258b` |

The benchmark JSON also binds each episode, event stream, metadata file,
MATLAB-candidate trace, mux trace and the source files on the MATLAB actuation
path by SHA-256.

The final five-second video panel is generated from the accepted balanced C2
episode. A temporary programmatic Simulink model replays the final five seconds
of `estimated_front_grip` and `estimated_rear_grip` through a Scope and
`To Workspace`; MATLAB animates that captured Simulink output. It is a signal
replay of recorded estimates, not a second vehicle simulation.

## Fair-comparison interpretation

C0 and C2 share the FSUK-2023 track, 12 m/s cap, reference, constrained QP,
steering and acceleration limits, command guard, and a 6.5 m/s cap over the
first 12% of the lap. C0 uses a fixed `μ=0.75` assumption and 70% scalar grip
utilisation. C2 uses causal front/rear estimates reduced by ensemble and
calibration uncertainty, then uses 98% of the conservative weaker-axle
capacity. MATLAB solves C2's lateral QP; the shared Python policy supplies the
longitudinal target.

C2 has slightly higher lateral RMS because it drives faster, while all six
runs remain comfortably inside the declared track boundary. This is a
system-level grip-aware planning/control result. It does not claim lower
tracking error, real-vehicle validity or robustness outside the frozen scope.

## Superseded results

The Python-actuated v3 benchmark remains valid development history but is
superseded by v7 for publication. Earlier small-track, globally speed-capped
and spinning-baseline previews—including the old 20.06% number—must not be
published. Failed v4–v6 experiments were used to correct startup transients,
actuation diagnostics and deterministic sensor seeding; they are not included
in the v7 aggregate.
