# MATLAB release demo

The approved portfolio clip is
`artifacts/videos/neurogrip_x_matlab_holdout_v7.mp4`.

| View | Controller | Balanced holdout lap |
|---|---|---:|
| Left | fixed-μ MPC (`μ=0.75`, Python) | 30.70 s |
| Right | NeuroGrip-X per-axle MPC (MATLAB) | 28.94 s |

The clip replays recorded EUFS Sim 2 pose and telemetry on the FSUK-2023
trackdrive reference. It is not a Gazebo-camera recording. Both sides begin at
the explicit `experiment_start`; the earlier finisher remains at the finish
while the other car completes its lap.

The timing tower is centred vertically and placed at 45% of the frame width.
It is recalculated once per video second. It unwraps and normalises each
recorded lap-progress trace, identifies the current leader and compares the
two recorded passage times at the trailing vehicle's current track progress.
It therefore reports a causal same-distance interval, handles the early
Standard lead and subsequent NeuroGripX pass, and never estimates time from
screen-space vehicle separation.

The five-second outro reports the complete frozen three-scenario aggregate,
not only the displayed balanced scenario: **5.17% lower mean lap time**, 6/6
completed laps, zero cone collisions and a 0.576 m minimum vehicle-body
boundary margin. Its right half animates the accepted balanced C2 episode's
final five seconds of front/rear grip estimates. The data is replayed through
a temporary programmatic Simulink model containing `From Workspace`, `Scope`
and `To Workspace` blocks before MATLAB renders the captured output.

The Red Bull Ring and Nordschleife sentences are scale analogies, not evidence
from either circuit. They apply the measured 5.17% aggregate to published lap
records:

- Red Bull Ring, 1:05.619: `65.619 s × 0.0517 = 3.3925 s`, displayed as about
  3.39 seconds. Source: [Red Bull Ring Formula 1 circuit facts](https://www.redbullring.com/en/events-tickets/formula-1/formula-1-circuit/).
- Nürburgring Nordschleife, 5:19.546: `319.546 s × 0.0517 = 16.5225 s`,
  displayed as about 16.52 seconds. Source: [official Nürburgring record list](https://nuerburgring.de/info/nuerburgring/records?locale=en).

Re-render:

```bash
source scripts/env.sh
bash scripts/render_matlab_holdout_v7.sh
```

Do not use the old small-track/spinning-C0 preview, imply this is camera
footage, or quote the superseded 20.06% result.
