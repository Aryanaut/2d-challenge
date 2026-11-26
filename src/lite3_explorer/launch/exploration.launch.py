"""Full stack: leg-odom TF, cloud->scan, terrain classifier, slam_toolbox, Nav2, gait switcher, the mission manager
and (lite3_llm_tools) the AI client.

Example:
  ros2 launch lite3_explorer exploration.launch.py server_url:=http://192.168.10.141:8000 group_id:=team7
  ros2 launch lite3_explorer exploration.launch.py use_ai:=false mission_params:=/path/to/my_mission.yaml
  ros2 launch lite3_explorer exploration.launch.py controller:=mppi      # holonomic MPPI controller (vx, vy, wz)
  ros2 launch lite3_explorer exploration.launch.py use_curtains:=true    # walk through curtains on tile edges
  ros2 launch lite3_explorer exploration.launch.py use_stairs:=true      # climb over staircases in AI mode
  ros2 launch lite3_explorer exploration.launch.py use_rollers:=true     # cross roller tiles in the h-step gait
  ros2 launch lite3_explorer exploration.launch.py hstep_speed:=0.3      # Nav2 speed cap in the h-step gait (m/s)
  ros2 launch lite3_explorer exploration.launch.py step_pause:=false     # no balance pauses stepping up onto a tile
"""
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue

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
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
        launch_arguments={'params_file': params, 'use_sim_time': LaunchConfiguration('use_sim_time').perform(context),
                          'autostart': 'true'}.items())
    out = [LogInfo(msg=f'Nav2 local controller: {controller.upper()} ({params})')]
    if LaunchConfiguration('step_pause').perform(context).lower() != 'true':
        return out + [nav2]
    # step_pause: the velocity smoother publishes to /cmd_vel_nav_smoothed instead of /cmd_vel (a global remap comes
    # before the node's own cmd_vel_smoothed -> cmd_vel), and step_pause_gate forwards it, stopping the robot briefly
    # while it steps up onto a higher tile so it can balance.
    return out + [
        GroupAction([SetRemap(src='cmd_vel_smoothed', dst='/cmd_vel_nav_smoothed'), nav2]),
        Node(package=PKG, executable='step_pause_gate', name='step_pause_gate', output='screen',
             parameters=[LaunchConfiguration('step_pause_params').perform(context),
                         {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower() == 'true',
                          'in_topic': '/cmd_vel_nav_smoothed', 'out_topic': '/cmd_vel'}])]


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
    if LaunchConfiguration('use_stairs').perform(context).lower() == 'true':
        params.append({'modes.exploring.stairs': True})
    params.append({'modes.scanning.opening_stairs':
                   LaunchConfiguration('stairs_on_opening').perform(context).lower() == 'true'})
    if LaunchConfiguration('use_rollers').perform(context).lower() == 'true':
        params.append({'modes.exploring.rollers': True})
    angle = LaunchConfiguration('roller_angle_deg').perform(context)
    if angle:                                          # '' = modes.roller.approach_angle_deg from mission.yaml
        params.append({'modes.roller.approach_angle_deg': float(angle)})
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
        DeclareLaunchArgument('use_tilt_gate', default_value='true',
                              description='Drop scans while the robot / lidar IMU is tilted away from its start reference '
                                          '(stairs): /scan_raw -> tilt_gate -> /scan'),
        DeclareLaunchArgument('tilt_params', default_value=cfg('tilt_gate.yaml')),
        DeclareLaunchArgument('use_scan_fill', default_value='false',   # not yet checked on a replay
                              description='slam_toolbox maps /scan_slam: /scan with no-return rays ahead filled just below '
                                          'range_max, so open floor (outside the maze) is cleared free (scan_fill.py)'),
        DeclareLaunchArgument('use_slam', default_value='true'),
        DeclareLaunchArgument('use_nav2', default_value='true'),
        DeclareLaunchArgument('use_terrain', default_value='true'),
        DeclareLaunchArgument('terrain_on_tile_entry', default_value='true',
                              description='true: classify the terrain only when entering a new tile'),
        DeclareLaunchArgument('use_gait', default_value='true'),
        DeclareLaunchArgument('hstep_speed', default_value='0.38',
                              description='Nav2 linear speed cap (m/s) while the h-step gait (rug_hstep) is in use; 0 = no cap'),
        DeclareLaunchArgument('step_pause', default_value='true',
                              description='Stop briefly (step_pause.yaml: 2 x 0.2 s) while Nav2 drives up onto a higher tile, '
                                          'so the robot can balance; false = Nav2 publishes /cmd_vel directly'),
        DeclareLaunchArgument('step_pause_params', default_value=cfg('step_pause.yaml')),
        DeclareLaunchArgument('use_mission', default_value='true'),
        DeclareLaunchArgument('auto_start', default_value='',
                              description='true/false: leave idle by itself (empty = mission.yaml)'),
        DeclareLaunchArgument('quickscan_on_edge', default_value='',
                              description='true: stop on each tile edge and run the quickscan animation (empty = mission.yaml)'),
        DeclareLaunchArgument('scene_max_free', default_value='',
                              description='Scene tile: scanned, but less than this share of the scanned part traversable, e.g. 0.6 '
                                          '(empty = mission.yaml tiles.scene_max_free, 0.80)'),
        DeclareLaunchArgument('scene_min_known', default_value='',
                              description='Scene tile: scanned at least this much (empty = mission.yaml, 0.85)'),
        DeclareLaunchArgument('use_curtains', default_value='false',
                              description='Optional: detect hanging curtains in the depth image (low rate); one along the '
                                          "current tile's edge is squared up to and walked through with vx commands"),
        DeclareLaunchArgument('curtain_params', default_value=cfg('curtain.yaml')),
        DeclareLaunchArgument('use_elevation', default_value='',
                              description='true/false overrides terrain.yaml elevation.enabled (empty: the yaml decides). On: '
                                          'the terrain classifier fits a ground-plane reference at start, the mission prefers '
                                          'the tile with the least elevation change (prefer_flat_tiles) and holds the h-step '
                                          'gait (/step_up) when driving to a higher tile'),
        DeclareLaunchArgument('use_stairs', default_value='false',
                              description='Optional: detect staircases in the depth image (low rate); one on a tile '
                                          'beside the current tile is climbed over in the robot AI mode, then walled off'),
        DeclareLaunchArgument('stairs_on_opening', default_value='true',
                              description='Run the stair detector during the opening scan; a staircase beside the start '
                                          'tile is climbed (stairs mode) right after it, even without use_stairs'),
        DeclareLaunchArgument('stairs_params', default_value=cfg('stairs.yaml')),
        DeclareLaunchArgument('use_rollers', default_value='false',
                              description='Optional: detect roller tiles (depth camera, confirmed by the 3D lidar; low rate); '
                                          'one beside the current tile is squared up to and crossed in the h-step gait'),
        DeclareLaunchArgument('roller_params', default_value=cfg('roller.yaml')),
        DeclareLaunchArgument('roller_angle_deg', default_value='',
                              description="Roller mode: angle (deg, + = turned left) the body keeps to the roller plane while "
                                          "approaching it; 0 = square on; '' = approach_angle_deg in mission.yaml"),
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
             remappings=[('cloud_in', lc('cloud_topic')), ('scan', PythonExpression(["'/scan_raw' if '", lc('use_tilt_gate'), "'.lower() == 'true' else '/scan'"]))],
             parameters=[{'target_frame': 'base_link', 'transform_tolerance': 0.05,
                          'min_height': lc('scan_min_height'), 'max_height': lc('scan_max_height'),
                          'angle_min': -3.14159, 'angle_max': 3.14159, 'angle_increment': 0.0087,
                          'scan_time': 0.1, 'range_min': 0.35, 'range_max': 12.0, 'use_inf': True}],
             condition=IfCondition(lc('use_cloud_to_scan'))),
        Node(package=PKG, executable='tilt_gate', name='tilt_gate', output='screen',
             parameters=[lc('tilt_params'), {'use_sim_time': lc('use_sim_time')}],
             condition=IfCondition(lc('use_tilt_gate'))),
        Node(package=PKG, executable='curtain_detector', name='curtain_detector', output='screen',
             parameters=[lc('curtain_params'), {'use_sim_time': lc('use_sim_time')}],
             condition=IfCondition(lc('use_curtains'))),
        # use_stairs: detecting the whole mission; else only while the mission enables it (/stairs/enable: opening scan)
        Node(package=PKG, executable='stair_detector', name='stair_detector', output='screen',
             parameters=[lc('stairs_params'), {'use_sim_time': lc('use_sim_time'),
                                               'start_enabled': PythonExpression(["'", lc('use_stairs'), "'.lower() == 'true'"])}],
             condition=IfCondition(PythonExpression(["'", lc('use_stairs'), "'.lower() == 'true' or '",
                                                     lc('stairs_on_opening'), "'.lower() == 'true'"]))),
        Node(package=PKG, executable='roller_detector', name='roller_detector', output='screen',
             parameters=[lc('roller_params'), {'use_sim_time': lc('use_sim_time'), 'cloud_topic': lc('cloud_topic')}],
             condition=IfCondition(lc('use_rollers'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('lite3_terrain_classifier'), 'launch', 'terrain_classifier.launch.py')),
            launch_arguments={'use_sim_time': lc('use_sim_time'), 'cloud_topic': lc('cloud_topic'),
                              'trigger_mode': lc('terrain_on_tile_entry'),
                              'elevation': lc('use_elevation')}.items(),
            condition=IfCondition(lc('use_terrain'))),
        # respawn: modes.stairs.new_map clears the map by stopping slam_toolbox (Humble has no reset service)
        Node(package=PKG, executable='scan_fill', name='scan_fill', output='screen',
             parameters=[{'use_sim_time': lc('use_sim_time')}],
             condition=IfCondition(PythonExpression(["'", lc('use_scan_fill'), "'.lower() == 'true' and '",
                                                     lc('use_slam'), "'.lower() == 'true'"]))),
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
             parameters=[lc('slam_params'), {'use_sim_time': lc('use_sim_time'),
                                             'scan_topic': PythonExpression(["'/scan_slam' if '", lc('use_scan_fill'),
                                                                             "'.lower() == 'true' else '/scan'"])}],
             respawn=True, respawn_delay=0.5,
             condition=IfCondition(lc('use_slam'))),
        OpaqueFunction(function=_nav2),
        Node(package=PKG, executable='gait_switcher', name='gait_switcher', output='screen',
             parameters=[{'flat_gait': 'flat_slow', 'rugged_gait': 'rug_hstep',
                          'hstep_speed_limit': ParameterValue(lc('hstep_speed'), value_type=float)}],
             condition=IfCondition(lc('use_gait'))),
        Node(package=PKG, executable='animation_player', name='animation_player', output='screen'),
        OpaqueFunction(function=_llm_client),
        OpaqueFunction(function=_mission_manager),
    ]
    return LaunchDescription(args + nodes)
