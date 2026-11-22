import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('lite3_terrain_severity'),
                                  'config', 'terrain_severity.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML with terrain_severity parameters'),
        DeclareLaunchArgument('use_governor', default_value='false',
                              description='false = only report severity; true = also scale /cmd_vel'),
        Node(package='lite3_terrain_severity', executable='terrain_severity', name='terrain_severity',
             output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'use_governor': ParameterValue(LaunchConfiguration('use_governor'), value_type=bool)}]),
    ])
