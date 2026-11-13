"""Terrain classifier on its own - for manual driving with the remote, or next to any autonomy stack.

Manual:
  ros2 launch lite3_terrain_classifier terrain_classifier.launch.py
  ros2 topic echo /terrain_type

Auto: kitchen_exploration.launch.py (lite3_kitchen_explorer) includes this file (use_terrain:=true).

If the preflight said there is no TF base_link -> livox_frame, add
  publish_lidar_tf:=true lidar_x:=.. lidar_z:=.. lidar_pitch:=..
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_terrain_classifier'


def _classifier(context):
    lc = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    overrides = {'use_sim_time': lc('use_sim_time').lower() == 'true'}
    for arg, param in (('cloud_topic', 'lidar.topic'), ('depth_topic', 'depth.image_topic'),
                       ('depth_info_topic', 'depth.info_topic'), ('fusion_mode', 'fusion_mode'),
                       ('base_frame', 'base_frame')):
        if lc(arg):
            overrides[param] = lc(arg)
    if lc('use_depth'):
        overrides['use_depth'] = lc('use_depth').lower() == 'true'
    return [Node(package=PKG, executable='terrain_classifier', name='terrain_classifier',
                 output='screen', parameters=[lc('terrain_params'), overrides])]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('terrain_params', default_value=os.path.join(share, 'config', 'terrain.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('cloud_topic', default_value='', description='Overrides lidar.topic'),
        DeclareLaunchArgument('depth_topic', default_value='', description='Overrides depth.image_topic'),
        DeclareLaunchArgument('depth_info_topic', default_value='', description='Overrides depth.info_topic'),
        DeclareLaunchArgument('use_depth', default_value='', description='true/false, overrides use_depth'),
        DeclareLaunchArgument('fusion_mode', default_value='',
                              description='lidar_primary | require_agreement | any_rugged'),
        DeclareLaunchArgument('base_frame', default_value=''),
        # base_link -> lidar (only if nothing else publishes it)
        DeclareLaunchArgument('publish_lidar_tf', default_value='false'),
        DeclareLaunchArgument('lidar_frame', default_value='livox_frame'),
        DeclareLaunchArgument('lidar_x', default_value='0.20'),
        DeclareLaunchArgument('lidar_y', default_value='0.0'),
        DeclareLaunchArgument('lidar_z', default_value='0.10'),
        DeclareLaunchArgument('lidar_roll', default_value='0.0'),
        DeclareLaunchArgument('lidar_pitch', default_value='0.0'),
        DeclareLaunchArgument('lidar_yaw', default_value='0.0'),
        # base_link -> camera (optional: the depth check also works in the camera frame)
        DeclareLaunchArgument('publish_camera_tf', default_value='false'),
        DeclareLaunchArgument('camera_frame', default_value='camera_link',
                              description='Root frame of the camera driver TF tree'),
        DeclareLaunchArgument('camera_x', default_value='0.30'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.05'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.3', description='rad, positive = looking down'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
    ]

    def static_tf(name, child, prefix, flag):
        return Node(package='tf2_ros', executable='static_transform_publisher', name=name,
                    arguments=['--frame-id', 'base_link', '--child-frame-id', lc(child),
                               '--x', lc(prefix + '_x'), '--y', lc(prefix + '_y'), '--z', lc(prefix + '_z'),
                               '--roll', lc(prefix + '_roll'), '--pitch', lc(prefix + '_pitch'),
                               '--yaw', lc(prefix + '_yaw')],
                    condition=IfCondition(lc(flag)))

    return LaunchDescription(args + [
        static_tf('terrain_lidar_static_tf', 'lidar_frame', 'lidar', 'publish_lidar_tf'),
        static_tf('terrain_camera_static_tf', 'camera_frame', 'camera', 'publish_camera_tf'),
        OpaqueFunction(function=_classifier),
    ])
