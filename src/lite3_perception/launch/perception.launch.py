import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap

# camera topics read here; with sensor_prefix:=/local they come from lite3_explorer's sensor_relay instead of the robot
CAMERA_TOPICS = ["/camera/color/image_raw", "/camera/color/image_raw/compressed", "/camera/color/camera_info",
                 "/camera/depth/image_raw", "/camera/depth/camera_info"]


def _camera_remaps(context):
    prefix = LaunchConfiguration("sensor_prefix").perform(context).strip("/")
    return [SetRemap(src=t, dst=f"/{prefix}{t}") for t in CAMERA_TOPICS] if prefix else []


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("lite3_perception"), "config", "detector.yaml")
    qr_launch = os.path.join(get_package_share_directory("qr_scanner"), "launch", "qr_scanner.launch.py")
    return LaunchDescription([
        DeclareLaunchArgument("with_bridge", default_value="false",
                              description="also start the reference LLM bridge"),
        DeclareLaunchArgument("with_qr", default_value="true",
                              description="also start qr_scanner (needed when detector.yaml has qr_source: scanner)"),
        DeclareLaunchArgument("qr_image_topic", default_value="/camera/color/image_raw/compressed",
                              description="front camera stream for qr_scanner (same one perception uses)"),
        DeclareLaunchArgument("qr_compressed", default_value="true",
                              description="true if qr_image_topic is a CompressedImage topic"),
        DeclareLaunchArgument("qr_camera_info", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("sensor_prefix", default_value="",
                              description="e.g. /local: read the camera topics relayed by lite3_explorer sensor_relay "
                                          "(explorer.sh), so this launch adds no subscriber on the robot"),
        GroupAction([
            OpaqueFunction(function=_camera_remaps),
            Node(package="lite3_perception", executable="object_detector", name="object_detector",
                 parameters=[cfg], output="screen"),
            Node(package="lite3_perception", executable="llm_event_bridge", name="llm_event_bridge",
                 parameters=[cfg], output="screen", condition=IfCondition(LaunchConfiguration("with_bridge"))),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(qr_launch),
                launch_arguments={
                    "front_image_topic": LaunchConfiguration("qr_image_topic"),
                    "front_compressed": LaunchConfiguration("qr_compressed"),
                    "front_camera_info": LaunchConfiguration("qr_camera_info"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("with_qr"))),
        ]),
    ])
