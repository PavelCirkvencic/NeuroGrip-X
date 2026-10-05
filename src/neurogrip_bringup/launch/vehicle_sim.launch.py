"""Launch the NeuroGrip-X Ackermann vehicle simulation."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_gazebo(context):
    """Start Gazebo, optionally server-only for reproducible batch recordings."""
    headless = LaunchConfiguration("headless").perform(context).strip().lower()
    world_path = LaunchConfiguration("world_path").perform(context)
    prefix = "-s -r" if headless in ("true", "1", "yes") else "-r"
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("ros_gz_sim"),
                        "launch",
                        "gz_sim.launch.py",
                    ]
                )
            ),
            # Navodnici su važni jer putanja projekta sadrži razmake.
            launch_arguments={"gz_args": f'{prefix} "{world_path}"'}.items(),
        )
    ]


def generate_launch_description():
    """Launch the vehicle world and bridge core Gazebo topics to ROS 2."""
    default_world_path = PathJoinSubstitution(
        [
            FindPackageShare("neurogrip_sim"),
            "worlds",
            "neurogrip_ackermann.sdf",
        ]
    )
    declare_world_path = DeclareLaunchArgument(
        "world_path",
        default_value=default_world_path,
        description=(
            "Absolute path to an SDF world. Defaults to the packaged nominal "
            "NeuroGrip-X world; scenario_builder generates reproducible variants."
        ),
    )
    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description=(
            "Run the Gazebo server without the GUI. Recommended for automated "
            "batch recordings; interactive use can leave it false."
        ),
    )
    scenario_manifest_path = LaunchConfiguration("scenario_manifest_path")
    declare_scenario_manifest = DeclareLaunchArgument(
        "scenario_manifest_path",
        default_value="",
        description=(
            "Optional generated scenario manifest. When set, the command guard "
            "applies that scenario's hidden steering actuator delay/gain."
        ),
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        output="screen",
    )

    command_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
        ],
        output="screen",
    )

    odometry_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        ],
        output="screen",
    )

    command_guard = Node(
        package="neurogrip_control",
        executable="command_guard",
        parameters=[
            {"use_sim_time": True},
            {"scenario_manifest_path": scenario_manifest_path},
        ],
        output="screen",
    )

    imu_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/vehicle_blue/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
        ],
        output="screen",
    )

    # The bridge publishes the raw Gazebo IMU on /model/vehicle_blue/imu; the
    # relay reads that topic and publishes the filtered stream on
    # /model/vehicle_blue/imu_filtered (the logger's default).  It applies the
    # scenario's hidden noise/dropout, or passes messages through unchanged.
    imu_relay = Node(
        package="neurogrip_control",
        executable="imu_relay",
        name="imu_relay",
        parameters=[
            {"use_sim_time": True},
            {"scenario_manifest_path": scenario_manifest_path},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_world_path,
            declare_headless,
            declare_scenario_manifest,
            OpaqueFunction(function=launch_gazebo),
            clock_bridge,
            command_bridge,
            odometry_bridge,
            command_guard,
            imu_bridge,
            imu_relay,
        ]
    )
