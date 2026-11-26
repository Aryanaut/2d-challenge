"""Launch the stair climber (and optionally the VLM checker).

    ros2 launch stair_climber stair_climber.launch.py
    ros2 launch stair_climber stair_climber.launch.py dry_run:=true use_sim_time:=true
    ros2 launch stair_climber stair_climber.launch.py with_vlm:=true
    ros2 launch stair_climber stair_climber.launch.py state_bridge:=false state_port:=43897

state_bridge (default true) also runs robot_state_bridge: the robot's 0x0901 UDP state -> /robot_state, used to check
every gait / AI switch (stair_climber and the mission manager resend a command until the robot reports it).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('stair_climber'),
                                  'config', 'stair_climber.yaml')
    params = LaunchConfiguration('params')
    sim = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('dry_run', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('with_vlm', default_value='false'),
        DeclareLaunchArgument('state_bridge', default_value='true'),
        DeclareLaunchArgument('state_port', default_value='43897'),

        Node(
            package='stair_climber', executable='stair_climber_node',
            name='stair_climber', output='screen',
            parameters=[params, {
                'dry_run': LaunchConfiguration('dry_run'),
                'autostart': LaunchConfiguration('autostart'),
                'use_sim_time': sim,
            }],
        ),
        Node(
            package='stair_climber', executable='vlm_stair_checker',
            name='vlm_stair_checker', output='screen',
            parameters=[params, {'use_sim_time': sim}],
            condition=IfCondition(LaunchConfiguration('with_vlm')),
        ),
        Node(
            package='stair_climber', executable='robot_state_bridge',
            name='robot_state_bridge', output='screen',
            parameters=[{'port': LaunchConfiguration('state_port')}],
            condition=IfCondition(LaunchConfiguration('state_bridge')),
        ),
    ])
