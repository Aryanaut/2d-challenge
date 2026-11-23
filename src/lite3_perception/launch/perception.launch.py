import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("lite3_perception"), "config", "detector.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("with_bridge", default_value="false",
                              description="also start the reference LLM bridge"),
        Node(package="lite3_perception", executable="object_detector", name="object_detector",
             parameters=[cfg], output="screen"),
        Node(package="lite3_perception", executable="llm_event_bridge", name="llm_event_bridge",
             parameters=[cfg], output="screen", condition=IfCondition(LaunchConfiguration("with_bridge"))),
    ])
