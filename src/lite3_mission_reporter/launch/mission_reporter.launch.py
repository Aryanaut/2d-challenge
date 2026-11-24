import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("lite3_mission_reporter"), "config", "mission_reporter.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("use_llm_result", default_value="false"),
        Node(package="lite3_mission_reporter", executable="mission_reporter", name="mission_reporter",
             parameters=[cfg, {
                 # launch args actually reach the node now (before, use_llm_result was ignored)
                 "use_llm_result": ParameterValue(LaunchConfiguration("use_llm_result"), value_type=bool),
             }], output="screen"),
    ])
