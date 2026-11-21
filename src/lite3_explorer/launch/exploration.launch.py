"""Full stack: leg-odom TF, cloud->scan, terrain classifier, slam_toolbox, Nav2, gait switcher, the mission manager
and (lite3_llm_tools) the AI client.

Example:
  ros2 launch lite3_explorer exploration.launch.py server_url:=http://192.168.10.141:8000 group_id:=team7
  ros2 launch lite3_explorer exploration.launch.py use_ai:=false mission_params:=/path/to/my_mission.yaml
  ros2 launch lite3_explorer exploration.launch.py controller:=mppi      # holonomic MPPI controller (vx, vy, wz)
  ros2 launch lite3_explorer exploration.launch.py use_curtains:=true    # walk through curtains on tile edges
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_explorer'
LLM_ARGS = ('server_url', 'group_id', 'image_topic', 'prompt_file')


def _nav2(context):
    """Nav2 bringup with the parameter file of the chosen local controller (controller:=rpp | mppi)."""
    if LaunchConfiguration('use_nav2').perform(context).lower() != 'true':
        return []
    controller = LaunchConfiguration('controller').perform(context).lower()
    if controller not in ('rpp', 'mppi'):
        raise RuntimeError(f"controller must be 'rpp' or 'mppi', not {controller!r}")
    params = LaunchConfiguration('nav2_params').perform(context) or os.path.join(
        get_package_share_directory(PKG), 'config', 'nav2_params_mppi.yaml' if controller == 'mppi' else 'nav2_params.yaml')
    return [LogInfo(msg=f'Nav2 local controller: {controller.upper()} ({params})'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
                launch_arguments={'params_file': params, 'use_sim_time': LaunchConfiguration('use_sim_time').perform(context),
                                  'autostart': 'true'}.items())]


def _llm_client(context):
    """The AI client node (lite3_llm_tools). Skipped when use_ai is false or the package is not installed."""
    if LaunchConfiguration('use_mission').perform(context).lower() != 'true' \
            or LaunchConfiguration('use_ai').perform(context).lower() == 'false':
        return []
    try:
        share = get_package_share_directory('lite3_llm_tools')
    except PackageNotFoundError:
        return [LogInfo(msg='lite3_llm_tools is not installed: running without the AI.')]
    args = {k: LaunchConfiguration(k).perform(context) for k in LLM_ARGS + ('image_compressed',)}
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'llm_client.launch.py')),
        launch_arguments={k: v for k, v in args.items() if v}.items())]


def _mission_manager(context):
    if LaunchConfiguration('use_mission').perform(context).lower() != 'true':
        return []
    params = [LaunchConfiguration('mission_params').perform(context),
              {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'}]
    use_ai = LaunchConfiguration('use_ai').perform(context)
    if use_ai:
        params.append({'use_ai': use_ai.lower() == 'true'})
    edge = LaunchConfiguration('quickscan_on_edge').perform(context)
    if edge:
        params.append({'modes.exploring.quickscan_on_edge': edge.lower() == 'true'})
    if LaunchConfiguration('use_curtains').perform(context).lower() == 'true':
        params.append({'modes.exploring.curtains': True})
    for arg in ('scene_max_free', 'scene_min_known'):
        value = LaunchConfiguration(arg).perform(context)
        if value:
            params.append({f'tiles.{arg}': float(value)})
    auto = LaunchConfiguration('auto_start').perform(context)
    if auto:
        params.append({'modes.idle.auto_start': auto.lower() == 'true'})
    return [Node(package=PKG, executable='mission_manager', name='mission_manager', output='screen',
                 parameters=params)]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    cfg = lambda name: os.path.join(share, 'config', name)  # noqa: E731
    lc = LaunchConfiguration

    args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('publish_odom_tf', default_value='true',
                              description='Broadcast odom->base_link from /leg_odom2'),
        DeclareLaunchArgument('odom_topic', default_value='/leg_odom2'),
        DeclareLaunchArgument('use_cloud_to_scan', default_value='true',
                              description='Make /scan from a 3D lidar point cloud'),
        DeclareLaunchArgument('cloud_topic', default_value='/livox/points'),
        DeclareLaunchArgument('scan_min_height', default_value='-0.12',
                              description='Lowest point kept, metres in base_link (filters floor tiles)'),
        DeclareLaunchArgument('scan_max_height', default_value='0.60'),
        DeclareLaunchArgument('publish_lidar_tf', default_value='false'),
        DeclareLaunchArgument('lidar_frame', default_value='livox_frame'),
        DeclareLaunchArgument('lidar_x', default_value='0.20'),
        DeclareLaunchArgument('lidar_y', default_value='0.0'),
        DeclareLaunchArgument('lidar_z', default_value='0.10'),
        DeclareLaunchArgument('lidar_yaw', default_value='0.0'),
        DeclareLaunchArgument('lidar_pitch', default_value='0.0'),
        DeclareLaunchArgument('lidar_roll', default_value='0.0'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_terrain', default_value='true'),
        DeclareLaunchArgument('terrain_on_tile_entry', default_value='true',
                              description='true: classify the terrain only when entering a new tile'),
        DeclareLaunchArgument('use_gait', default_value='true'),
        DeclareLaunchArgument('use_mission', default_value='true'),
        DeclareLaunchArgument('auto_start', default_value='',
                              description='true/false: leave idle by itself (empty = mission.yaml)'),
        DeclareLaunchArgument('quickscan_on_edge', default_value='',
                              description='true: stop on each tile edge and run the quickscan animation (empty = mission.yaml)'),
        DeclareLaunchArgument('scene_max_free', default_value='',
                              description='Scene tile: scanned, but less than this share of it traversable, e.g. 0.6 '
                                          '(empty = mission.yaml tiles.scene_max_free, 0.80)'),
        DeclareLaunchArgument('scene_min_known', default_value='',
                              description='Scene tile: scanned at least this much (empty = mission.yaml, 0.95)'),
        DeclareLaunchArgument('use_curtains', default_value='false',
                              description='Optional: detect hanging curtains in the depth image (low rate); one along the '
                                          "current tile's edge is squared up to and walked through with vx commands"),
        DeclareLaunchArgument('curtain_params', default_value=cfg('curtain.yaml')),
        DeclareLaunchArgument('slam_params', default_value=cfg('slam_toolbox.yaml')),
        DeclareLaunchArgument('controller', default_value='rpp',
                              description='Nav2 local controller: rpp (regulated pure pursuit: vx/wz only) | '
                                          'mppi (MPPI, holonomic: vx, vy and wz)'),
        DeclareLaunchArgument('nav2_params', default_value='',
                              description="'' = nav2_params.yaml (controller:=rpp) or nav2_params_mppi.yaml (mppi)"),
        DeclareLaunchArgument('mission_params', default_value=cfg('mission.yaml')),
        DeclareLaunchArgument('use_ai', default_value='', description='true/false (empty = mission.yaml)'),
        DeclareLaunchArgument('server_url', default_value=''),
        DeclareLaunchArgument('group_id', default_value=''),
        DeclareLaunchArgument('image_topic', default_value=''),
        DeclareLaunchArgument('image_compressed', default_value=''),
        DeclareLaunchArgument('prompt_file', default_value=''),
    ]

    nodes = [
        Node(package=PKG, executable='odom_tf_bridge', name='odom_tf_bridge', output='screen',
             parameters=[{'odom_topic': lc('odom_topic'), 'republish_topic': '/odom'}],
             condition=IfCondition(lc('publish_odom_tf'))),
        Node(package='tf2_ros', executable='static_transform_publisher', name='lidar_static_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', lc('lidar_frame'),
                        '--x', lc('lidar_x'), '--y', lc('lidar_y'), '--z', lc('lidar_z'),
                        '--yaw', lc('lidar_yaw'), '--pitch', lc('lidar_pitch'), '--roll', lc('lidar_roll')],
             condition=IfCondition(lc('publish_lidar_tf'))),
        Node(package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
             name='cloud_to_scan', output='screen',
             remappings=[('cloud_in', lc('cloud_topic')), ('scan', '/scan')],
             parameters=[{'target_frame': 'base_link', 'transform_tolerance': 0.05,
                          'min_height': lc('scan_min_height'), 'max_height': lc('scan_max_height'),
                          'angle_min': -3.14159, 'angle_max': 3.14159, 'angle_increment': 0.0087,
                          'scan_time': 0.1, 'range_min': 0.35, 'range_max': 12.0, 'use_inf': True}],
             condition=IfCondition(lc('use_cloud_to_scan'))),
        Node(package=PKG, executable='curtain_detector', name='curtain_detector', output='screen',
             parameters=[lc('curtain_params'), {'use_sim_time': lc('use_sim_time')}],
             condition=IfCondition(lc('use_curtains'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('lite3_terrain_classifier'), 'launch', 'terrain_classifier.launch.py')),
            launch_arguments={'use_sim_time': lc('use_sim_time'), 'cloud_topic': lc('cloud_topic'),
                              'trigger_mode': lc('terrain_on_tile_entry')}.items(),
            condition=IfCondition(lc('use_terrain'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
            launch_arguments={'slam_params_file': lc('slam_params'), 'use_sim_time': lc('use_sim_time')}.items(),
            condition=IfCondition(lc('use_slam'))),
        OpaqueFunction(function=_nav2),
        Node(package=PKG, executable='gait_switcher', name='gait_switcher', output='screen',
             parameters=[{'flat_gait': 'flat_medium', 'rugged_gait': 'rug_general'}],
             condition=IfCondition(lc('use_gait'))),
        Node(package=PKG, executable='animation_player', name='animation_player', output='screen'),
        OpaqueFunction(function=_llm_client),
        OpaqueFunction(function=_mission_manager),
    ]
    return LaunchDescription(args + nodes)
