"""Closed-loop C0/C1 Python MPC + Ackermann guard against the EUFS backend."""

from __future__ import annotations

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def maybe_ai_node(context):
    """Start the read-only physical C2 model only for the C2 controller."""
    if LaunchConfiguration("controller").perform(context) != "C2_NEUROGRIP":
        return []
    manifest = LaunchConfiguration("c2_ensemble_manifest").perform(context)
    if not manifest:
        raise RuntimeError("C2_NEUROGRIP requires c2_ensemble_manifest")
    return [
        Node(
            package="neurogrip_ai",
            executable="koopman_inference_node",
            name="neurogrip_learned_dynamics",
            parameters=[{"ensemble_manifest": manifest, "use_sim_time": True}],
            output="screen",
        )
    ]


def maybe_matlab_tcp_bridge(context):
    """Join the localhost MATLAB server only for MATLAB actuation runs."""
    if LaunchConfiguration("actuation_source").perform(context) != "matlab":
        return []
    return [
        Node(
            package="neurogrip_control",
            executable="matlab_tcp_bridge",
            name="neurogrip_matlab_tcp_bridge",
            parameters=[
                {
                    "port": LaunchConfiguration("matlab_tcp_port"),
                    "use_sim_time": False,
                }
            ],
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Start the command guard and the Python controller."""
    track_npz = LaunchConfiguration("track_npz")
    controller = LaunchConfiguration("controller")
    target_speed = LaunchConfiguration("target_speed")
    target_speed_min = LaunchConfiguration("target_speed_min")
    fixed_mu = LaunchConfiguration("fixed_mu")
    nominal_fit = LaunchConfiguration("nominal_fit")
    c2_model_blend = LaunchConfiguration("c2_model_blend")
    c2_adaptive_speed_max = LaunchConfiguration("c2_adaptive_speed_max")
    scalar_mu_utilisation = LaunchConfiguration("scalar_mu_utilisation")
    neurogrip_utilisation = LaunchConfiguration("neurogrip_utilisation")
    grip_error_margin_scale = LaunchConfiguration("grip_error_margin_scale")
    startup_speed_cap = LaunchConfiguration("startup_speed_cap_mps")
    startup_speed_cap_end = LaunchConfiguration(
        "startup_speed_cap_end_progress"
    )
    steering_delay = LaunchConfiguration("steering_delay_s")
    steering_gain = LaunchConfiguration("steering_gain")
    actuation_source = LaunchConfiguration("actuation_source")
    matlab_activation_delay = LaunchConfiguration("matlab_activation_delay_s")
    guard = Node(
        package="neurogrip_control",
        executable="ackermann_guard",
        name="ackermann_guard",
        parameters=[
            {
                "input_topic": "/neurogrip/command_candidate",
                "output_topic": "/cmd",
                "use_sim_time": True,
                "steering_delay_s": steering_delay,
                "steering_gain": steering_gain,
            }
        ],
        output="screen",
    )
    candidate_mux = Node(
        package="neurogrip_control",
        executable="candidate_mux",
        name="neurogrip_candidate_mux",
        parameters=[
            {
                "selected_source": actuation_source,
                "matlab_activation_delay_s": matlab_activation_delay,
                "use_sim_time": True,
            }
        ],
        output="screen",
    )
    tracking = Node(
        package="neurogrip_control",
        executable="tracking_state_node",
        name="neurogrip_tracking_state",
        parameters=[{"track_npz": track_npz, "use_sim_time": True}],
        output="screen",
    )
    applied_steering = Node(
        package="neurogrip_control",
        executable="applied_steering_bridge",
        name="neurogrip_applied_steering_bridge",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    controller_process = ExecuteProcess(
        cmd=[
            sys.executable,
            "-m",
            "controllers.python.controller_node",
            "--track-npz",
            track_npz,
            "--controller",
            controller,
            "--target-speed",
            target_speed,
            "--target-speed-min",
            target_speed_min,
            "--fixed-mu",
            fixed_mu,
            "--c2-adaptive-speed-max",
            c2_adaptive_speed_max,
            "--nominal-fit",
            nominal_fit,
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            ["c2_model_blend:=", c2_model_blend],
            "-p",
            ["scalar_mu_utilisation:=", scalar_mu_utilisation],
            "-p",
            ["neurogrip_utilisation:=", neurogrip_utilisation],
            "-p",
            ["grip_error_margin_scale:=", grip_error_margin_scale],
            "-p",
            ["startup_speed_cap_mps:=", startup_speed_cap],
            "-p",
            ["startup_speed_cap_end_progress:=", startup_speed_cap_end],
        ],
        cwd=os.getcwd(),
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("track_npz"),
            DeclareLaunchArgument("controller", default_value="C0_FIXED"),
            DeclareLaunchArgument("target_speed", default_value="3.0"),
            DeclareLaunchArgument("target_speed_min", default_value="1.2"),
            DeclareLaunchArgument("fixed_mu", default_value="1.0"),
            DeclareLaunchArgument(
                "nominal_fit",
                default_value="runs/eufs_v1/models/nominal_fit.json",
            ),
            DeclareLaunchArgument("c2_ensemble_manifest", default_value=""),
            DeclareLaunchArgument("c2_model_blend", default_value="0.02"),
            DeclareLaunchArgument("c2_adaptive_speed_max", default_value="-1.0"),
            DeclareLaunchArgument("scalar_mu_utilisation", default_value="0.78"),
            DeclareLaunchArgument("neurogrip_utilisation", default_value="0.95"),
            DeclareLaunchArgument("grip_error_margin_scale", default_value="0.50"),
            DeclareLaunchArgument("startup_speed_cap_mps", default_value="8.0"),
            DeclareLaunchArgument(
                "startup_speed_cap_end_progress", default_value="0.12"
            ),
            DeclareLaunchArgument("steering_delay_s", default_value="0.0"),
            DeclareLaunchArgument("steering_gain", default_value="1.0"),
            DeclareLaunchArgument("actuation_source", default_value="python"),
            DeclareLaunchArgument("matlab_activation_delay_s", default_value="3.0"),
            DeclareLaunchArgument("matlab_tcp_port", default_value="55980"),
            tracking,
            applied_steering,
            OpaqueFunction(function=maybe_matlab_tcp_bridge),
            candidate_mux,
            guard,
            OpaqueFunction(function=maybe_ai_node),
            controller_process,
        ]
    )
