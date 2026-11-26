"""lite3_sim: the simulated maze + lite3_explorer's unmodified mission_manager + RViz.

  ros2 launch lite3_sim sim.launch.py                                   # mazes/arena_example.yaml
  ros2 launch lite3_sim sim.launch.py maze:=dead_ends                   # a bundled maze by name
  ros2 launch lite3_sim sim.launch.py maze:=/home/me/my_maze.yaml       # your own maze
  ros2 launch lite3_sim sim.launch.py use_curtains:=false controller:=omni speed_scale:=2.0
  ros2 launch lite3_sim sim.launch.py maze:=stairs_loop                 # staircase ('M') climbed in AI mode
  ros2 launch lite3_sim sim.launch.py use_mission:=false                # sim + RViz only; run mission_manager yourself
  ros2 launch lite3_sim sim.launch.py maze:=mazeA reporter:=URLA        # + mission reporter POSTing to task_report_A
  ros2 launch lite3_sim sim.launch.py maze:=mazeA reporter:=URLB        # ... to task_report_B (reporter:=local: no POST)

Scene tiles '1'..'8' in the maze name their scene (1 clothes_shop, 2 fitness_centre, 3 kitchen, 4 market, 5 workshop,
6 clinic, 7 cleaning_room, 8 furniture_shop; '0' = unnamed). With such tiles (scene_ai:=auto) the sim stands in for
the AI's scene answers (use_ai on in the mission manager), so /scene/result carries the scene type, as on the robot.

The mission manager gets lite3_explorer's config/mission.yaml (mission_params:=...), then config/sim_mission.yaml
(no AI, no terrain classifier, no animations, curtains and stairs on), then tiles.size_m from the maze file.
"""
import os

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_sim'
REPORTER_URLS = {'A': '/task_report_A', 'B': '/task_report_B'}    # reporter:=URLA / URLB, appended to reporter_host


def _reporter_url(value, host):
    """reporter:= URLA / URLB (or A / B) -> the server URL; local -> '' (topic + results file only)."""
    key = value.strip().upper()
    if key in ('LOCAL', 'NONE', 'OFF'):
        return ''
    key = key[3:] if key.startswith('URL') else key
    if key not in REPORTER_URLS:
        raise RuntimeError(f'reporter:={value!r}: use URLA, URLB or local')
    return host.rstrip('/') + REPORTER_URLS[key]


def _resolve_maze(value, share):
    if os.path.isfile(os.path.expanduser(value)):
        return os.path.abspath(os.path.expanduser(value))
    for cand in (os.path.join(share, 'mazes', value), os.path.join(share, 'mazes', value + '.yaml')):
        if os.path.isfile(cand):
            return cand
    bundled = sorted(f[:-5] for f in os.listdir(os.path.join(share, 'mazes')) if f.endswith('.yaml'))
    raise RuntimeError(f'maze {value!r} not found: give a path to a maze YAML or one of {bundled}')


def _setup(context):
    share = get_package_share_directory(PKG)
    lc = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    maze = _resolve_maze(lc('map') or lc('maze'), share)          # map:= is accepted as another name for maze:=
    with open(maze, encoding='utf-8') as fh:
        tile_size = float((yaml.safe_load(fh) or {}).get('tile_size', 1.02))
    from lite3_sim.maze import load_maze
    scene_types = load_maze(maze).scene_types
    scene_ai = lc('scene_ai').lower()
    scene_ai = bool(scene_types) if scene_ai == 'auto' else scene_ai == 'true'

    sim_over = {'maze': maze, 'controller.type': lc('controller'), 'map_origin': lc('map_origin'),
                'speed_scale': float(lc('speed_scale')), 'scene_ai': scene_ai}
    if lc('lidar_range'):
        sim_over['lidar.range_max'] = float(lc('lidar_range'))
    actions = [
        LogInfo(msg=f'lite3_sim: maze {maze} (tiles {tile_size:g} m); scene AI stand-in '
                    + ('on: ' + (', '.join(f'{i},{j} {n}' for (i, j), n in sorted(scene_types.items())) or
                                 'no named scene tiles') if scene_ai else 'off')),
        Node(package=PKG, executable='sim_node', name='lite3_sim', output='screen',
             parameters=[lc('sim_params'), sim_over]),
    ]
    if lc('use_mission').lower() == 'true':
        mission = [lc('mission_params'), lc('sim_mission_params'),
                   {'tiles.size_m': tile_size,
                    'modes.exploring.curtains': lc('use_curtains').lower() == 'true',
                    'modes.exploring.stairs': lc('use_stairs').lower() == 'true',
                    'modes.idle.auto_start': lc('auto_start').lower() == 'true'}]
        if scene_ai:
            # the sim answers the scene tiles (one picture: no animation_player for the scenescan in the sim)
            mission.append({'use_ai': True, 'modes.scene.quickscan': False})
        actions.append(Node(package='lite3_explorer', executable='mission_manager', name='mission_manager',
                            output='screen', parameters=mission))
    if lc('reporter'):
        url = _reporter_url(lc('reporter'), lc('reporter_host'))
        try:
            rep_share = get_package_share_directory('lite3_mission_reporter')
        except PackageNotFoundError:
            raise RuntimeError('reporter:= needs lite3_mission_reporter: colcon build --packages-select '
                               'lite3_mission_reporter') from None
        actions += [
            LogInfo(msg=f'lite3_sim: mission reporter -> {url or "no server (topic + results file only)"}'),
            Node(package='lite3_mission_reporter', executable='mission_reporter', name='mission_reporter',
                 output='screen',
                 parameters=[os.path.join(rep_share, 'config', 'mission_reporter.yaml'),
                             {'server_url': url, 'output_dir': lc('reporter_output_dir')}]),
        ]
    if lc('rviz').lower() == 'true':
        actions.append(Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
                            arguments=['-d', lc('rviz_config')]))
    return actions


def generate_launch_description():
    share = get_package_share_directory(PKG)
    explorer = get_package_share_directory('lite3_explorer')
    args = [
        DeclareLaunchArgument('maze', default_value='arena_example',
                              description='path to a maze YAML, or the name of a bundled one (mazes/)'),
        DeclareLaunchArgument('map', default_value='', description='same as maze (wins when given)'),
        DeclareLaunchArgument('use_curtains', default_value='true', description='mission curtain mode'),
        DeclareLaunchArgument('use_stairs', default_value='true',
                              description="mission stairs mode: climb over 'M' staircase tiles in AI mode"),
        DeclareLaunchArgument('controller', default_value='rpp', description='rpp (vx, wz) | omni (vx, vy, wz)'),
        DeclareLaunchArgument('map_origin', default_value='start',
                              description='start: map frame at the start pose (like SLAM) | maze: at the maze corner'),
        DeclareLaunchArgument('speed_scale', default_value='1.0'),
        DeclareLaunchArgument('lidar_range', default_value='', description="'' = config/sim.yaml"),
        DeclareLaunchArgument('auto_start', default_value='true',
                              description='false: start with  ros2 service call /mission_manager/set_active '
                                          'std_srvs/srv/SetBool "{data: true}"'),
        DeclareLaunchArgument('use_mission', default_value='true',
                              description='false: only the sim (+ RViz); run mission_manager yourself'),
        DeclareLaunchArgument('scene_ai', default_value='auto',
                              description="answer the scene mode's AI with the maze's scene types: auto = when the maze "
                                          "has '1'..'8' scene tiles | true | false"),
        DeclareLaunchArgument('reporter', default_value='',
                              description="'' = no mission reporter | URLA | URLB (POST to <reporter_host>/task_report_A "
                                          "or _B) | local (reporter without POSTing)"),
        DeclareLaunchArgument('reporter_host', default_value='http://192.168.10.141:8001'),
        DeclareLaunchArgument('reporter_output_dir', default_value='~/lite3_sim_runs/mission_results',
                              description="the reporter's results.jsonl / results.json"),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share, 'rviz', 'lite3_sim.rviz')),
        DeclareLaunchArgument('sim_params', default_value=os.path.join(share, 'config', 'sim.yaml')),
        DeclareLaunchArgument('mission_params', default_value=os.path.join(explorer, 'config', 'mission.yaml')),
        DeclareLaunchArgument('sim_mission_params', default_value=os.path.join(share, 'config', 'sim_mission.yaml')),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
