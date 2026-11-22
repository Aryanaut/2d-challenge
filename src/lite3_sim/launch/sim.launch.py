"""lite3_sim: the simulated maze + lite3_explorer's unmodified mission_manager + RViz.

  ros2 launch lite3_sim sim.launch.py                                   # mazes/arena_example.yaml
  ros2 launch lite3_sim sim.launch.py maze:=dead_ends                   # a bundled maze by name
  ros2 launch lite3_sim sim.launch.py maze:=/home/me/my_maze.yaml       # your own maze
  ros2 launch lite3_sim sim.launch.py use_curtains:=false controller:=omni speed_scale:=2.0
  ros2 launch lite3_sim sim.launch.py use_mission:=false                # sim + RViz only; run mission_manager yourself

The mission manager gets lite3_explorer's config/mission.yaml (mission_params:=...), then config/sim_mission.yaml
(no AI, no terrain classifier, no animations, curtains on), then tiles.size_m from the maze file.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'lite3_sim'


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

    sim_over = {'maze': maze, 'controller.type': lc('controller'), 'map_origin': lc('map_origin'),
                'speed_scale': float(lc('speed_scale'))}
    if lc('lidar_range'):
        sim_over['lidar.range_max'] = float(lc('lidar_range'))
    actions = [
        LogInfo(msg=f'lite3_sim: maze {maze} (tiles {tile_size:g} m)'),
        Node(package=PKG, executable='sim_node', name='lite3_sim', output='screen',
             parameters=[lc('sim_params'), sim_over]),
    ]
    if lc('use_mission').lower() == 'true':
        mission = [lc('mission_params'), lc('sim_mission_params'),
                   {'tiles.size_m': tile_size,
                    'modes.exploring.curtains': lc('use_curtains').lower() == 'true',
                    'modes.idle.auto_start': lc('auto_start').lower() == 'true'}]
        actions.append(Node(package='lite3_explorer', executable='mission_manager', name='mission_manager',
                            output='screen', parameters=mission))
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
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share, 'rviz', 'lite3_sim.rviz')),
        DeclareLaunchArgument('sim_params', default_value=os.path.join(share, 'config', 'sim.yaml')),
        DeclareLaunchArgument('mission_params', default_value=os.path.join(explorer, 'config', 'mission.yaml')),
        DeclareLaunchArgument('sim_mission_params', default_value=os.path.join(share, 'config', 'sim_mission.yaml')),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
