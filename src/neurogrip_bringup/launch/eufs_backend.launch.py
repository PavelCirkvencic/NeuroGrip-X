"""
Launch the pinned EUFS Sim 2 dynamic backend with the NeuroGrip v2 contract.

The authoritative simulation time and vehicle state come from EUFS.  Gazebo is
only a visual frontend (Phase 2) and is not started here.

Arguments:
---------
  core_config    immutable per-episode vehicle YAML (mass/inertia/Pacejka)
  plugin_config  NeuroGrip 50 Hz plugin YAML
  track          EUFS CSV track to load
  headless       accepted for API compatibility (EUFS is always headless)
  seed           recorded for provenance only

"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_core_config() -> str:
    """Return the pinned nominal EUFS vehicle parameter file."""
    return os.path.join(
        get_package_share_directory("vehicle_models"),
        "config",
        "DynamicBicycle",
        "ads-dv-calculated.yaml",
    )


def default_plugin_config() -> str:
    """Return the versioned NeuroGrip plugin configuration."""
    return os.path.join(
        get_package_share_directory("neurogrip_bringup"),
        "config",
        "eufs_plugin_params.yaml",
    )


def default_track() -> str:
    """Return the pinned small track CSV."""
    return os.path.join(
        get_package_share_directory("map_lib"), "maps", "tracks", "small_track.csv"
    )


def launch_eufs(context):
    """Start the EUFS node, URDF publisher and track loader."""
    core_config = LaunchConfiguration("core_config").perform(context)
    plugin_config = LaunchConfiguration("plugin_config").perform(context)
    track = LaunchConfiguration("track").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    seed = LaunchConfiguration("seed").perform(context)

    eufs_sim2_share = get_package_share_directory("eufs_sim2")
    sim_node = Node(
        package="eufs_sim2",
        executable="eufs_sim2_node",
        name="eufs_sim2",
        parameters=[
            {"core_params": core_config},
            plugin_config,
        ],
        remappings=[
            ("/plugin/vehicle_state_plugin/ground_truth/state", "/odom"),
            ("/plugin/wheel_speed_plugin/wheel_speed", "/ros_can/wheel_speeds"),
            ("/plugin/imu_plugin/imu/data", "/imu/data"),
            ("/plugin/twist_publisher/twist", "/ros_can/twist"),
            ("/plugin/force_publisher/car_forces", "/plugin/force_publisher/car_forces"),
            ("/plugin/cone_fusion/gt_cones", "/cones/lenient"),
            ("/plugin/cone_fusion/cones", "/cones"),
            ("/plugin/cone_fusion/map", "/map"),
            ("/plugin/state_publisher/ros_can/state_str", "sim/ros_can/state_str"),
            ("/plugin/state_publisher/ros_can/state", "sim/ros_can/state"),
            ("/complete_mission_flag", "/ros_can/mission_completed"),
            ("/fix", "/ros_can/fix"),
            ("/plugin/cone_fusion/camera/cones", "/camera/cones"),
            ("/plugin/cone_fusion/lidar_grid/cones", "/lidar_grid/cones"),
        ],
        output="screen",
    )
    urdf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(eufs_sim2_share, "launch", "publish_urdf.launch.py")
        ),
        launch_arguments={"save_urdf": "False"}.items(),
    )
    set_track = Node(
        package="neurogrip_bringup",
        executable="set_track",
        name="neurogrip_set_track",
        parameters=[{"track_path": track, "timeout_s": 60.0}],
        output="screen",
    )
    print(f"[eufs_backend] core_config={core_config}")
    print(f"[eufs_backend] plugin_config={plugin_config}")
    print(f"[eufs_backend] track={track} headless={headless} seed={seed}")
    return [urdf, sim_node, set_track]


def generate_launch_description() -> LaunchDescription:
    """Declare arguments and start the EUFS backend."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("core_config", default_value=default_core_config()),
            DeclareLaunchArgument("plugin_config", default_value=default_plugin_config()),
            DeclareLaunchArgument("track", default_value=default_track()),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("seed", default_value="0"),
            OpaqueFunction(function=launch_eufs),
        ]
    )
