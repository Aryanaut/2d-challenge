"""Full stack: leg-odom TF, cloud->scan, slam_toolbox, Nav2, frontier explorer, tile reporter and (lite3_llm_tools) the AI client.

Example:
  ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py \
      server_url:=http://192.168.10.141:8000 group_id:=team7
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_kitchen_explorer'


LLM_ARGS = ('server_url', 'group_id', 'image_topic', 'prompt_file')


def _llm_client(context):
    """The AI client node (lite3_llm_tools). Skipped when use_ai is false or the package is not installed."""
    if LaunchConfiguration('use_reporter').perform(context).lower() != 'true' \
            or LaunchConfiguration('use_ai').perform(context).lower() == 'false':
        return []
    try:
        share = get_package_share_directory('lite3_llm_tools')
    except PackageNotFoundError:
        return [LogInfo(msg='lite3_llm_tools is not installed: running without the AI (use_ai is ignored).')]
    args = {k: LaunchConfiguration(k).perform(context) for k in LLM_ARGS + ('image_compressed',)}
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'llm_client.launch.py')),
        launch_arguments={k: v for k, v in args.items() if v}.items())]


def _explorer_and_reporter(context):
    use_reporter = LaunchConfiguration('use_reporter').perform(context).lower() == 'true'
    overrides = {}
    use_ai = LaunchConfiguration('use_ai').perform(context)
    if use_ai:
        overrides['use_ai'] = use_ai.lower() == 'true'
    reporter_params = [LaunchConfiguration('reporter_params').perform(context)]
    if overrides:
        reporter_params.append(overrides)
    sim = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    nodes = []
    if LaunchConfiguration('use_explorer').perform(context).lower() == 'true':
        # With the reporter running, the explorer waits for it: first report at the start pose.
        nodes.append(Node(package=PKG, executable='frontier_explorer', name='frontier_explorer',
                          output='screen',
                          parameters=[LaunchConfiguration('explorer_params').perform(context),
                                      {'use_sim_time': sim, 'start_active': not use_reporter}]))
    if use_reporter:
        nodes.append(Node(package=PKG, executable='environment_reporter', name='environment_reporter',
                          output='screen', parameters=reporter_params + [{'use_sim_time': sim}]))
    return nodes


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
        DeclareLaunchArgument('cloud_topic', default_value='/livox/points',
                              description='Livox MID-360 PointCloud2 topic'),
        DeclareLaunchArgument('scan_min_height', default_value='-0.12',
                              description='Lowest point kept, metres in base_link (filters floor tiles)'),
        DeclareLaunchArgument('scan_max_height', default_value='0.60'),
        DeclareLaunchArgument('publish_lidar_tf', default_value='false'),
        DeclareLaunchArgument('lidar_frame', default_value='livox_frame',
                              description='frame_id of /livox/points (see preflight output)'),
        DeclareLaunchArgument('lidar_x', default_value='0.20', description='base_link->lidar (m)'),
        DeclareLaunchArgument('lidar_y', default_value='0.0'),
        DeclareLaunchArgument('lidar_z', default_value='0.10'),
        DeclareLaunchArgument('lidar_yaw', default_value='0.0', description='rad'),
        DeclareLaunchArgument('lidar_pitch', default_value='0.0', description='rad (MID-360 is often tilted)'),
        DeclareLaunchArgument('lidar_roll', default_value='0.0', description='rad'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_terrain', default_value='true',
                              description='Run the rugged / non-rugged classifier (publishes /terrain_type)'),
        DeclareLaunchArgument('use_gait', default_value='true',
                              description='Switch gait from /terrain_type: rugged -> rug_general, non_rugged -> flat_medium'),
        DeclareLaunchArgument('use_explorer', default_value='true'),
        DeclareLaunchArgument('use_reporter', default_value='true',
                              description='Ask the AI to describe the environment at every stop'),
        DeclareLaunchArgument('slam_params', default_value=cfg('slam_toolbox.yaml')),
        DeclareLaunchArgument('nav2_params', default_value=cfg('nav2_params.yaml')),
        DeclareLaunchArgument('explorer_params', default_value=cfg('explorer.yaml')),
        DeclareLaunchArgument('reporter_params', default_value=cfg('reporter.yaml')),
        DeclareLaunchArgument('use_ai', default_value='', description='true/false: use the AI (also starts / skips the llm_client node); empty = reporter.yaml'),
        DeclareLaunchArgument('server_url', default_value='', description='Overrides llm_client.yaml (lite3_llm_tools)'),
        DeclareLaunchArgument('group_id', default_value='', description='Overrides llm_client.yaml (lite3_llm_tools)'),
        DeclareLaunchArgument('image_topic', default_value='', description='Overrides llm_client.yaml (lite3_llm_tools)'),
        DeclareLaunchArgument('image_compressed', default_value='', description='true/false, overrides llm_client.yaml'),
        DeclareLaunchArgument('prompt_file', default_value='', description='Overrides llm_client.yaml (lite3_llm_tools)'),
    ]

    nodes = [
        Node(package=PKG, executable='odom_tf_bridge', name='odom_tf_bridge', output='screen',
             parameters=[{'odom_topic': lc('odom_topic'), 'republish_topic': '/odom'}],
             condition=IfCondition(lc('publish_odom_tf'))),
        Node(package='tf2_ros', executable='static_transform_publisher', name='lidar_static_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', lc('lidar_frame'),
                        '--x', lc('lidar_x'), '--y', lc('lidar_y'), '--z', lc('lidar_z'),
                        '--yaw', lc('lidar_yaw'), '--pitch', lc('lidar_pitch'),
                        '--roll', lc('lidar_roll')],
             condition=IfCondition(lc('publish_lidar_tf'))),
        Node(package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
             name='cloud_to_scan', output='screen',
             remappings=[('cloud_in', lc('cloud_topic')), ('scan', '/scan')],
             parameters=[{
                 'target_frame': 'base_link',
                 'transform_tolerance': 0.05,
                 'min_height': lc('scan_min_height'),
                 'max_height': lc('scan_max_height'),
                 'angle_min': -3.14159,
                 'angle_max': 3.14159,
                 'angle_increment': 0.0087,
                 'scan_time': 0.1,
                 'range_min': 0.35,   # ignore the robot's own legs
                 'range_max': 12.0,
                 'use_inf': True,
             }],
             condition=IfCondition(lc('use_cloud_to_scan'))),
        # Terrain classifier -> /terrain_type. The lidar TF is already published above
        # (lidar_static_tf), so the classifier does not publish it again.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('lite3_terrain_classifier'), 'launch',
                'terrain_classifier.launch.py')),
            launch_arguments={'use_sim_time': lc('use_sim_time'),
                              'cloud_topic': lc('cloud_topic')}.items(),
            condition=IfCondition(lc('use_terrain'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')),
            launch_arguments={'slam_params_file': lc('slam_params'),
                              'use_sim_time': lc('use_sim_time')}.items(),
            condition=IfCondition(lc('use_slam'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
            launch_arguments={'params_file': lc('nav2_params'),
                              'use_sim_time': lc('use_sim_time'),
                              'autostart': 'true'}.items(),
            condition=IfCondition(lc('use_nav2'))),
        Node(package=PKG, executable='gait_switcher', name='gait_switcher', output='screen',
             parameters=[{'flat_gait': 'flat_medium', 'rugged_gait': 'rug_general'}],
             condition=IfCondition(lc('use_gait'))),
        OpaqueFunction(function=_llm_client),
        OpaqueFunction(function=_explorer_and_reporter),
    ]
    return LaunchDescription(args + nodes)
