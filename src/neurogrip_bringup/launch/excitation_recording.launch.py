"""
Run one traceable NeuroGrip-X excitation recording.

Start ``vehicle_sim.launch.py`` with the matching generated SDF first.  This
launch deliberately requires the matching scenario manifest, then starts the
logger before the excitation driver and shuts down after the logger completes.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def require_manifest(context):
    """Fail before recording if the scenario provenance was not provided."""
    manifest_path = LaunchConfiguration("scenario_manifest_path").perform(context)
    if not manifest_path:
        raise RuntimeError(
            "scenario_manifest_path is required. Generate it with "
            "'ros2 run neurogrip_sim scenario_builder <profile> --seed <seed>'."
        )
    return []


def maybe_contextual_node(context):
    """Start the read-only contextual inference node when a checkpoint is set."""
    checkpoint_path = LaunchConfiguration("ai_checkpoint_path").perform(context)
    if not checkpoint_path:
        return []
    model_root = LaunchConfiguration("ai_model_root").perform(context)
    return [
        Node(
            package="neurogrip_ai",
            executable="contextual_inference_node",
            name="neurogrip_contextual_inference",
            parameters=[
                {"use_sim_time": True},
                {"checkpoint_path": checkpoint_path},
                {"model_root": model_root},
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    """Launch state logging and deterministic excitation as one experiment."""
    scenario_manifest_path = LaunchConfiguration("scenario_manifest_path")
    output_dir = LaunchConfiguration("output_dir")
    sample_rate_hz = LaunchConfiguration("sample_rate_hz")
    excitation_profile = LaunchConfiguration("excitation_profile")

    declare_scenario_manifest = DeclareLaunchArgument(
        "scenario_manifest_path",
        default_value="",
        description="Required generated scenario .manifest.yaml file.",
    )
    declare_output_dir = DeclareLaunchArgument(
        "output_dir",
        default_value="data/raw",
        description="Directory for raw CSV recordings and metadata.",
    )
    declare_sample_rate = DeclareLaunchArgument(
        "sample_rate_hz",
        default_value="50.0",
        description="State logger sampling frequency in Hz.",
    )
    declare_excitation_profile = DeclareLaunchArgument(
        "excitation_profile",
        default_value="baseline_v1",
        description="Excitation profile name, e.g. baseline_v1 or dynamic_v2.",
    )
    declare_ai_checkpoint = DeclareLaunchArgument(
        "ai_checkpoint_path",
        default_value="",
        description=(
            "Optional N1 checkpoint. When set, the read-only contextual "
            "inference node runs alongside the recording."
        ),
    )
    declare_ai_model_root = DeclareLaunchArgument(
        "ai_model_root",
        default_value="learning",
        description="Repository learning/ root that provides the neurogrip package.",
    )

    logger = Node(
        package="neurogrip_control",
        executable="state_logger",
        name="state_logger",
        parameters=[
            {"use_sim_time": True},
            {"output_dir": output_dir},
            {
                "sample_rate_hz": ParameterValue(
                    sample_rate_hz, value_type=float
                )
            },
            {"stop_on_completion": True},
            {"scenario_manifest_path": scenario_manifest_path},
            {"excitation_profile": excitation_profile},
        ],
        output="screen",
    )
    driver = Node(
        package="neurogrip_control",
        executable="excitation_driver",
        name="excitation_driver",
        parameters=[
            {"use_sim_time": True},
            {"profile_name": excitation_profile},
        ],
        output="screen",
    )
    grip_scheduler = Node(
        package="neurogrip_control",
        executable="grip_scheduler",
        name="grip_scheduler",
        parameters=[
            {"use_sim_time": True},
            {"scenario_manifest_path": scenario_manifest_path},
        ],
        output="screen",
    )

    shutdown_when_complete = RegisterEventHandler(
        OnProcessExit(
            target_action=logger,
            on_exit=[
                LogInfo(msg="Recording complete; shutting down experiment launch."),
                EmitEvent(event=Shutdown(reason="state logger completed")),
            ],
        )
    )

    return LaunchDescription(
        [
            declare_scenario_manifest,
            declare_output_dir,
            declare_sample_rate,
            declare_excitation_profile,
            declare_ai_checkpoint,
            declare_ai_model_root,
            OpaqueFunction(function=require_manifest),
            OpaqueFunction(function=maybe_contextual_node),
            logger,
            grip_scheduler,
            # Give subscriptions time to connect before the first phase label.
            TimerAction(period=1.0, actions=[driver]),
            shutdown_when_complete,
        ]
    )
