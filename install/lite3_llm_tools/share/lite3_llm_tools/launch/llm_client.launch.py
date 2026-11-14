"""Start the AI client node. Any argument left empty keeps the value from config/llm_client.yaml."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_llm_tools'
OVERRIDES = ('server_url', 'group_id', 'image_topic', 'scan_topic', 'map_topic', 'prompt_file', 'output_dir')


def _node(context):
    overrides = {}
    for key in OVERRIDES:
        value = LaunchConfiguration(key).perform(context)
        if value:
            overrides[key] = value
    for key in ('image_compressed', 'lidar_context'):
        value = LaunchConfiguration(key).perform(context)
        if value:
            overrides[key] = value.lower() == 'true'
    params = [LaunchConfiguration('llm_params_file').perform(context)]
    if overrides:
        params.append(overrides)
    return [Node(package=PKG, executable='llm_client', name='llm_client', output='screen', parameters=params)]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    args = [DeclareLaunchArgument('llm_params_file', default_value=os.path.join(share, 'config', 'llm_client.yaml'))]
    args += [DeclareLaunchArgument(k, default_value='', description='Overrides llm_client.yaml') for k in
             OVERRIDES + ('image_compressed', 'lidar_context')]
    return LaunchDescription(args + [OpaqueFunction(function=_node)])
