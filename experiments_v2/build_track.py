#!/usr/bin/env python3
"""
Deterministic EUFS cone-track to centerline reference builder.

Reads an EUFS CSV map (``tag,x,y,...`` with an optional ``car_start`` row),
transforms cones into the car-start frame exactly like ``map_lib`` (translate by
``-car_pos`` then rotate by ``-theta``), builds the blue/yellow centerline with
Delaunay cross-colour midpoints, fits a periodic cubic spline by arc length and
resamples it at a fixed spacing.

Outputs (in ``--output-dir``):
  <name>_reference.npz / .mat   reference arrays
  <name>_visual.sdf             Gazebo world with ground and cones
  <name>_manifest.json          hashes and provenance
  <name>_preview.png            centerline + cones preview
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.io import savemat
from scipy.spatial import Delaunay

EUFS_HEADER = [
    "tag",
    "x",
    "y",
    "direction",
    "x_variance",
    "y_variance",
    "xy_covariance",
]
SUPPORTED_COLORS = {"blue", "yellow", "orange", "big_orange", "unknown"}
CENTERLINE_COLORS = {"blue", "yellow"}
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the file is in this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass
class Cone:
    """One cone in the car-start frame."""

    color: str
    x: float
    y: float


def parse_arguments() -> argparse.Namespace:
    """Parse input map, name, spacing and output directory."""
    parser = argparse.ArgumentParser(description="Build an EUFS track reference.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--ds", type=float, default=0.20)
    parser.add_argument("--smoothing-per-midpoint-m2", type=float, default=0.03)
    return parser.parse_args()


def load_eufs_csv(csv_path: Path) -> tuple[list[Cone], tuple[float, float, float]]:
    """Parse an EUFS CSV and return (cones in car frame, car_start pose)."""
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"empty map file: {csv_path}")
    header = [name.strip() for name in lines[0].split(",")]
    if header[: len(EUFS_HEADER)] != EUFS_HEADER:
        raise ValueError(f"invalid EUFS header: {header}")
    if header != EUFS_HEADER:
        raise ValueError("unsupported extra CSV columns")

    raw_cones: list[tuple[str, float, float]] = []
    car_x, car_y, car_theta = 0.0, 0.0, 0.0
    car_start_found = False
    for line in lines[1:]:
        if not line.strip():
            continue
        row = line.split(",")
        if len(row) < 3:
            raise ValueError(f"malformed CSV row: {line}")
        tag = row[0].strip()
        if tag == "car_start":
            car_x, car_y, car_theta = float(row[1]), float(row[2]), float(row[3])
            car_start_found = True
            continue
        if tag not in SUPPORTED_COLORS:
            raise ValueError(f"unknown cone tag: {tag}")
        raw_cones.append((tag, float(row[1]), float(row[2])))
    if not raw_cones:
        raise ValueError("map contains no cones")

    cos_theta = np.cos(-car_theta)
    sin_theta = np.sin(-car_theta)
    cones = []
    for color, x, y in raw_cones:
        translated_x = x - car_x
        translated_y = y - car_y
        cones.append(
            Cone(
                color=color,
                x=float(translated_x * cos_theta - translated_y * sin_theta),
                y=float(translated_x * sin_theta + translated_y * cos_theta),
            )
        )
    if not car_start_found:
        car_theta = 0.0
    return cones, (car_x, car_y, car_theta)


def build_centerline(
    cones: list[Cone], start_point: np.ndarray | None = None
) -> np.ndarray:
    """Return ordered blue/yellow cross-colour midpoints as an (N, 2) cycle.

    Algorithm (validated in the v2 risk-spike):
      1. Delaunay triangulation over blue and yellow cones;
      2. keep triangle edges whose endpoints have opposite colours -> midpoint;
      3. inside every triangle with exactly two cross-colour edges, connect the
         two midpoints;
      4. the largest connected component is the centerline cycle.
    """
    points = np.array(
        [(cone.x, cone.y) for cone in cones if cone.color in CENTERLINE_COLORS]
    )
    colors = np.array(
        [cone.color for cone in cones if cone.color in CENTERLINE_COLORS]
    )
    if len(points) < 4:
        raise ValueError("not enough centerline cones")
    triangulation = Delaunay(points)

    edge_to_midpoint: dict[tuple[int, int], int] = {}
    midpoints: list[np.ndarray] = []

    def midpoint_id(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in edge_to_midpoint:
            edge_to_midpoint[key] = len(midpoints)
            midpoints.append(0.5 * (points[first] + points[second]))
        return edge_to_midpoint[key]

    adjacency: dict[int, set[int]] = {}
    for simplex in triangulation.simplices:
        cross_edges = []
        for local in range(3):
            first = int(simplex[local])
            second = int(simplex[(local + 1) % 3])
            if colors[first] != colors[second]:
                cross_edges.append(midpoint_id(first, second))
        if len(cross_edges) == 2:
            first_mid, second_mid = cross_edges
            adjacency.setdefault(first_mid, set()).add(second_mid)
            adjacency.setdefault(second_mid, set()).add(first_mid)

    if not adjacency:
        raise ValueError("no cross-colour centerline edges found")
    components = _connected_components(adjacency)
    component = max(components, key=len)
    if start_point is None:
        start_point = np.zeros(2)
    cycle_ids = _order_cycle(component, adjacency, midpoints, np.asarray(start_point, dtype=float))
    return np.array([midpoints[index] for index in cycle_ids])


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Return connected components of an undirected graph."""
    seen: set[int] = set()
    components = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _order_cycle(
    component: list[int],
    adjacency: dict[int, set[int]],
    midpoints: list[np.ndarray],
    start_point: np.ndarray,
) -> list[int]:
    """Order a single-degree-2 component into a cycle starting at car start."""
    degrees = {node: len(adjacency[node]) for node in component}
    if any(degree != 2 for degree in degrees.values()):
        raise ValueError("centerline component is not a single closed cycle")

    start = min(
        component, key=lambda node: np.linalg.norm(midpoints[node] - start_point)
    )
    next_node = min(adjacency[start])
    ordered = [start]
    previous = start
    current = next_node
    while current != start:
        ordered.append(current)
        candidates = [node for node in adjacency[current] if node != previous]
        if not candidates:
            raise ValueError("cycle traversal reached a dead end")
        previous, current = current, candidates[0]
    return ordered


def _orient_cycle(cycle: np.ndarray, heading_rad: float) -> np.ndarray:
    """Orient the cycle so its initial tangent matches the car-start heading."""
    tangent = cycle[1] - cycle[0]
    heading = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    if float(np.dot(tangent, heading)) < 0.0:
        # Reverse direction but keep cycle[0] as the start node.
        return np.concatenate([cycle[:1], cycle[:0:-1]])
    return cycle


def resample_reference(
    cycle: np.ndarray,
    ds: float,
    smoothing_per_midpoint_m2: float = 0.03,
) -> dict[str, np.ndarray]:
    """Fit a trackable periodic smoothing spline and resample by arc length."""
    if ds <= 0.0 or smoothing_per_midpoint_m2 < 0.0:
        raise ValueError("spacing must be positive and smoothing non-negative")
    closed = np.vstack([cycle, cycle[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    node_arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    node_parameter = node_arc / node_arc[-1]
    spline, _ = splprep(
        [closed[:, 0], closed[:, 1]],
        u=node_parameter,
        s=smoothing_per_midpoint_m2 * len(cycle),
        per=True,
        k=3,
    )
    dense_parameter = np.linspace(0.0, 1.0, max(10000, len(cycle) * 200))
    dense_x, dense_y = splev(dense_parameter, spline)
    dense_arc = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(dense_x), np.diff(dense_y)))]
    )
    total_length = float(dense_arc[-1])
    if total_length <= ds:
        raise ValueError("track shorter than one sample spacing")
    sample_count = int(np.floor(total_length / ds))
    s = np.arange(sample_count + 1) * ds
    s[-1] = total_length
    sample_parameter = np.interp(s, dense_arc, dense_parameter)
    x, y = splev(sample_parameter, spline)
    dx, dy = splev(sample_parameter, spline, der=1)
    ddx, ddy = splev(sample_parameter, spline, der=2)
    x, y = np.asarray(x), np.asarray(y)
    dx, dy = np.asarray(dx), np.asarray(dy)
    ddx, ddy = np.asarray(ddx), np.asarray(ddy)
    yaw = np.arctan2(dy, dx)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = (dx * ddy - dy * ddx) / denominator
    return {
        "s": s,
        "x": x,
        "y": y,
        "yaw": yaw,
        "curvature": curvature,
        "length": np.array([total_length]),
    }


def align_reference_start(
    reference: dict[str, np.ndarray], cycle: np.ndarray, ds: float, blend_length_m: float = 7.0
) -> dict[str, np.ndarray]:
    """Blend the smoothed path into the measured start position and tangent."""
    s_m = reference["s"]
    length_m = float(reference["length"][0])
    points = np.column_stack([reference["x"], reference["y"]])
    current_tangent = np.asarray(
        [np.cos(reference["yaw"][0]), np.sin(reference["yaw"][0])]
    )
    desired_tangent = cycle[1] - cycle[-1]
    desired_tangent /= np.linalg.norm(desired_tangent)
    signed_distance = np.where(s_m <= length_m / 2.0, s_m, s_m - length_m)
    blend = np.exp(-np.square(signed_distance / blend_length_m))
    points += (cycle[0] - points[0]) * blend[:, None]
    points += (
        signed_distance[:, None]
        * (desired_tangent - current_tangent)
        * blend[:, None]
    )
    points[-1] = points[0]

    corrected_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    corrected_length = float(corrected_arc[-1])
    sample_count = int(np.floor(corrected_length / ds))
    new_s = np.arange(sample_count + 1) * ds
    new_s[-1] = corrected_length
    spline_x = CubicSpline(corrected_arc, points[:, 0], bc_type="periodic")
    spline_y = CubicSpline(corrected_arc, points[:, 1], bc_type="periodic")
    x, y = spline_x(new_s), spline_y(new_s)
    dx, dy = spline_x(new_s, 1), spline_y(new_s, 1)
    ddx, ddy = spline_x(new_s, 2), spline_y(new_s, 2)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    return {
        "s": new_s,
        "x": x,
        "y": y,
        "yaw": np.arctan2(dy, dx),
        "curvature": (dx * ddy - dy * ddx) / denominator,
        "length": np.asarray([corrected_length]),
    }


def compute_widths(
    reference: dict[str, np.ndarray], cones: list[Cone]
) -> dict[str, np.ndarray]:
    """Return left/right track width from the nearest cone on each side."""
    points = np.column_stack([reference["x"], reference["y"]])
    tangents = np.column_stack([np.cos(reference["yaw"]), np.sin(reference["yaw"])])
    centerline_cones = [cone for cone in cones if cone.color in CENTERLINE_COLORS]
    cone_points = np.array([(cone.x, cone.y) for cone in centerline_cones])
    left_width = np.full(len(points), np.inf)
    right_width = np.full(len(points), np.inf)
    for index, (point, tangent) in enumerate(zip(points, tangents)):
        offsets = cone_points - point
        lateral = tangent[0] * offsets[:, 1] - tangent[1] * offsets[:, 0]
        distances = np.linalg.norm(offsets, axis=1)
        positive = distances[lateral > 0.0]
        negative = distances[lateral < 0.0]
        if positive.size:
            left_width[index] = float(np.min(positive))
        if negative.size:
            right_width[index] = float(np.min(negative))
    return {"left_width": left_width, "right_width": right_width}


def render_visual_sdf(cones: list[Cone], name: str) -> str:
    """Return a Gazebo world SDF with the ground and every cone."""
    cone_materials = {
        "blue": "0.1 0.2 0.9 1",
        "yellow": "0.95 0.85 0.1 1",
        "orange": "0.95 0.45 0.05 1",
        "big_orange": "1.0 0.35 0.0 1",
        "unknown": "0.5 0.5 0.5 1",
    }
    models = []
    for index, cone in enumerate(cones):
        material = cone_materials.get(cone.color, "0.5 0.5 0.5 1")
        models.append(
            f"""
    <model name="cone_{index:04d}_{cone.color}">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>0.11</radius><length>0.325</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.11</radius><length>0.325</length></cylinder></geometry>
          <material><ambient>{material}</ambient><diffuse>{material}</diffuse></material>
        </visual>
      </link>
      <pose>{cone.x:.4f} {cone.y:.4f} 0.1625 0 0 0</pose>
    </model>"""
        )
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="neurogrip_visual">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="libignition-gazebo-scene-broadcaster-system.so"
            name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="libignition-gazebo-physics-system.so"
            name="ignition::gazebo::systems::Physics"/>
    <plugin filename="libignition-gazebo-user-commands-system.so"
            name="ignition::gazebo::systems::UserCommands"/>
    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.8 0.85 0.9 1</background>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.25 0.25 0.27 1</ambient><diffuse>0.25 0.25 0.27 1</diffuse></material>
        </visual>
      </link>
    </model>{''.join(models)}
  </world>
</sdf>
"""


def render_preview(
    cones: list[Cone], reference: dict[str, np.ndarray], output_path: Path
) -> None:
    """Save a centerline and cone preview PNG."""
    figure, axis = plt.subplots(figsize=(8, 8))
    colors = {"blue": "#1f4fd8", "yellow": "#d8c81f", "orange": "#d87c1f", "big_orange": "#e05200", "unknown": "#888888"}
    for cone in cones:
        axis.scatter(cone.x, cone.y, s=6, c=colors.get(cone.color, "#888888"))
    axis.plot(reference["x"], reference["y"], "-", color="#00d0a0", linewidth=1.2)
    axis.set_aspect("equal")
    axis.set_title("EUFS track reference")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def build_track(
    input_csv: Path,
    name: str,
    output_dir: Path,
    ds: float,
    smoothing_per_midpoint_m2: float = 0.03,
) -> dict:
    """Build every artifact for one track and return the manifest."""
    input_csv = input_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cones, car_start = load_eufs_csv(input_csv)
    # In the transformed frame the car start is exactly the origin.
    cycle = build_centerline(cones, start_point=np.zeros(2))
    cycle = _orient_cycle(cycle, car_start[2])
    reference = resample_reference(cycle, ds, smoothing_per_midpoint_m2)
    reference = align_reference_start(reference, cycle, ds)
    widths = compute_widths(reference, cones)
    reference.update(widths)

    for key in ("s", "x", "y", "yaw", "curvature", "left_width", "right_width"):
        if not np.all(np.isfinite(reference[key])):
            raise ValueError(f"non-finite value in reference array '{key}'")
    if not np.all(np.diff(reference["s"]) > 0.0):
        raise ValueError("arc-length samples are not strictly increasing")
    if np.min(reference["left_width"]) <= 0.0 or np.min(reference["right_width"]) <= 0.0:
        raise ValueError("non-positive track width")

    npz_path = output_dir / f"{name}_reference.npz"
    mat_path = output_dir / f"{name}_reference.mat"
    np.savez(npz_path, **reference)
    savemat(mat_path, {"reference": reference}, do_compression=True)

    sdf_path = output_dir / f"{name}_visual.sdf"
    sdf_path.write_text(render_visual_sdf(cones, name), encoding="utf-8")
    render_preview(cones, reference, output_dir / f"{name}_preview.png")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "input_csv": portable_path(input_csv),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "car_start": {"x": car_start[0], "y": car_start[1], "theta": car_start[2]},
        "cone_count": len(cones),
        "centerline_midpoints": int(len(cycle)),
        "centerline_length_m": float(reference["length"][0]),
        "sample_spacing_m": ds,
        "smoothing_per_midpoint_m2": smoothing_per_midpoint_m2,
        "samples": int(len(reference["s"])),
        "min_left_width_m": float(np.min(reference["left_width"])),
        "min_right_width_m": float(np.min(reference["right_width"])),
        "max_abs_curvature_1pm": float(np.max(np.abs(reference["curvature"]))),
        "outputs": {
            "npz": portable_path(npz_path),
            "mat": portable_path(mat_path),
            "sdf": portable_path(sdf_path),
            "preview": portable_path(output_dir / f"{name}_preview.png"),
        },
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "mat_sha256": hashlib.sha256(mat_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / f"{name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """Build the track and print a short report."""
    arguments = parse_arguments()
    manifest = build_track(
        arguments.input,
        arguments.name,
        arguments.output_dir,
        arguments.ds,
        arguments.smoothing_per_midpoint_m2,
    )
    print(f"Track: {manifest['name']}")
    print(f"Cones: {manifest['cone_count']}")
    print(f"Centerline midpoints: {manifest['centerline_midpoints']}")
    print(f"Length: {manifest['centerline_length_m']:.2f} m")
    print(f"Samples: {manifest['samples']} @ {manifest['sample_spacing_m']} m")
    print(f"Min widths L/R: {manifest['min_left_width_m']:.2f}/{manifest['min_right_width_m']:.2f} m")
    print(f"Manifest: {manifest['outputs']['npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
