"""Run the v2 EUFS system-identification excitation through the normal guard."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start independent tracking, the single command guard and excitation driver."""
    track_npz = LaunchConfiguration("track_npz")
    excitation_profile = LaunchConfiguration("excitation_profile")
    return LaunchDescription(
        [
            DeclareLaunchArgument("track_npz"),
            DeclareLaunchArgument("excitation_profile", default_value="system_id_v2"),
            DeclareLaunchArgument("steering_delay_s", default_value="0.0"),
            DeclareLaunchArgument("steering_gain", default_value="1.0"),
            Node(
                package="neurogrip_control",
                executable="tracking_state_node",
                name="neurogrip_tracking_state",
                parameters=[{"track_npz": track_npz, "use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="neurogrip_control",
                executable="ackermann_guard",
                name="ackermann_guard",
                parameters=[
                    {
                        "input_topic": "/neurogrip/command_candidate",
                        "output_topic": "/cmd",
                        "use_sim_time": True,
                        "steering_delay_s": LaunchConfiguration("steering_delay_s"),
                        "steering_gain": LaunchConfiguration("steering_gain"),
                    }
                ],
                output="screen",
            ),
            Node(
                package="neurogrip_control",
                executable="eufs_excitation_driver",
                name="neurogrip_eufs_excitation",
                parameters=[{"profile": excitation_profile, "use_sim_time": True}],
                output="screen",
            ),
        ]
    )
