# Closed-loop paired benchmark

Metrics include only the explicit `experiment_start` → safety-valid `lap_complete` interval. Any boundary exit, cone collision or controller failure invalidates the run.

| Scenario | Controller | Lap [s] | RMS e_y [m] | Min margin [m] | C2 active |
|---|---|---:|---:|---:|---:|
| matlabrelease_trackdrive_balanced_seed0933 | C0_FIXED | 30.70 | 0.1033 | 0.633 | - |
| matlabrelease_trackdrive_balanced_seed0933 | C2_NEUROGRIP | 28.94 | 0.1251 | 0.576 | 93.6% |
| matlabrelease_trackdrive_front_split_seed0931 | C0_FIXED | 30.70 | 0.1019 | 0.654 | - |
| matlabrelease_trackdrive_front_split_seed0931 | C2_NEUROGRIP | 29.22 | 0.1198 | 0.590 | 88.3% |
| matlabrelease_trackdrive_rear_split_seed0932 | C0_FIXED | 30.74 | 0.1060 | 0.630 | - |
| matlabrelease_trackdrive_rear_split_seed0932 | C2_NEUROGRIP | 29.22 | 0.1224 | 0.595 | 94.9% |

Paired mean lap-time reduction: **5.17%**; worst paired reduction: **4.82%**; publication gate: **PASS**.
