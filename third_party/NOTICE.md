# Third-party attribution

## EUFS Sim 2 ecosystem (MIT License)

The NeuroGrip-X v2 backend uses the following pinned repositories. All are
distributed under the MIT License; the upstream copyright notices are retained
in their checkouts and are not removed by the patches in `third_party/patches/`.

| Repository | URL | Commit |
|---|---|---|
| `eufs_sim2` | https://gitlab.com/eufs/public/eufs_sim2.git | `9f5df79a03725ea7d10542fc2ce8224d90836560` |
| `eufs_msgs` | https://gitlab.com/eufs/eufs_msgs.git | `9e918686c9e9292c613f321e6fd85e3a5d87cd87` |
| `eufs-gmock-matchers` | https://gitlab.com/eufs/public/eufs-gmock-matchers.git | `7ef83d030746c6a31bcf4f888d4121fcf4b7e8a9` |
| `eufs-logger` | https://gitlab.com/eufs/public/eufs-logger.git | `375ea1d8f8885af66809129e444624ba13353fa7` |
| `state_lib` | https://gitlab.com/eufs/public/state_lib.git | `ec83a141f188e8a4c39a381f4666485d8cc83e20` |
| `map_lib` | https://gitlab.com/eufs/public/map_lib.git | `1919b36062850c9ba4553d1833a9b517c61c2e86` |
| `vehicle_models` | https://gitlab.com/eufs/public/vehicle_models.git | `3508bec2c3d77e0ff16f08794675d4f7b52479b7` |

MIT License (upstream): Copyright (c) 2023 Edinburgh University Formula Student
(`eufs_sim2`, `map_lib`), Copyright (c) 2020 Edinburgh University Formula
Student (`eufs_msgs`), and the corresponding notices in the remaining
repositories. The full MIT text is included verbatim in each checkout.

### Assets used

- ADS-DV vehicle mesh and wheel meshes from the pinned `eufs_sim2` checkout,
  referenced at runtime through the installed `neurogrip_visualization`
  package. The meshes are used unmodified for the simulation frontend only.
- EUFS cone-track CSV maps from the pinned `map_lib` checkout, read by the
  offline track builder.

No upstream copyright notice is removed or altered by the project patches.
Patches only remove an unbuildable optional Python binding, mark IMU
orientation as unavailable, and add runtime grip/steering instrumentation.
