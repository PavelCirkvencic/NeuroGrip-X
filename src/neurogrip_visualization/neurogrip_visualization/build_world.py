"""Generate the Gazebo visual world from a track SDF plus the ADS-DV mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

CAR_MODEL_TEMPLATE = """
    <model name="{entity_name}">
      <static>true</static>
      <link name="body">
        <visual name="car_mesh">
          <geometry>
            <mesh><uri>file://{mesh_path}</uri><scale>1 1 1</scale></mesh>
          </geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient><diffuse>0.9 0.1 0.1 1</diffuse></material>
        </visual>
        <sensor name="chase_camera" type="camera">
          <pose>-6.0 0.0 3.0 0.0 0.20 0.0</pose>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <visualize>false</visualize>
          <topic>/camera/image</topic>
          <camera>
            <horizontal_fov>1.5</horizontal_fov>
            <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>300</far></clip>
          </camera>
        </sensor>
      </link>
      <pose>0 0 0 0 0 0</pose>
    </model>
"""


def parse_arguments() -> argparse.Namespace:
    """Parse track SDF, mesh, output and entity arguments."""
    parser = argparse.ArgumentParser(description="Build the NeuroGrip visual world.")
    parser.add_argument("--track-sdf", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-name", default="neurogrip_visual")
    parser.add_argument("--entity-name", default="formula_student_visual")
    return parser.parse_args()


def main() -> int:
    """Insert the car model and camera into the track world."""
    arguments = parse_arguments()
    track_sdf = arguments.track_sdf.expanduser().resolve()
    mesh = arguments.mesh.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not track_sdf.is_file():
        raise FileNotFoundError(track_sdf)
    if not mesh.is_file():
        raise FileNotFoundError(mesh)

    text = track_sdf.read_text(encoding="utf-8")
    if f'<world name="{arguments.world_name}">' not in text:
        raise ValueError("track SDF world name mismatch")
    model = CAR_MODEL_TEMPLATE.format(
        entity_name=arguments.entity_name, mesh_path=mesh
    )
    text = text.replace("</world>", model + "  </world>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Visual world: {output}")
    print(f"Entity: {arguments.entity_name} (mesh {mesh.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
