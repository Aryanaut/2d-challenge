"""Test one robot animation on the robot: starts animation_player and plays the animation once.

  ros2 launch lite3_explorer test_animation.launch.py                          # quickscan
  ros2 launch lite3_explorer test_animation.launch.py animation:=quickscan pause_s:=1.0 lean:=15000 yaw:=4000

The launch exits when the animation has finished (or failed). Animations: see lite3_explorer/animations.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lite3_explorer.animations import ANIMATIONS


def _nodes(context):
    name = LaunchConfiguration('animation').perform(context).strip().lower()
    if name not in ANIMATIONS:
        raise RuntimeError(f'unknown animation {name!r}; use one of {sorted(ANIMATIONS)}')
    # Optional overrides, applied to the chosen animation's parameters (player parameter <animation>.<key>).
    params = {}
    for key, cast in (('pause_s', float), ('move_s', float), ('lean', int), ('yaw', int)):
        value = LaunchConfiguration(key).perform(context)
        if value:
            params[f'{name}.{key}'] = cast(value)
    player = Node(package='lite3_explorer', executable='animation_player', name='animation_player',
                  output='screen', parameters=[params] if params else [])
    test = Node(package='lite3_explorer', executable='animation_test', name='animation_test',
                output='screen', parameters=[{'animation': name}])
    return [player, test,
            RegisterEventHandler(OnProcessExit(target_action=test, on_exit=[EmitEvent(event=Shutdown())]))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('animation', default_value='quickscan',
                              description='animation to test: ' + ', '.join(sorted(ANIMATIONS))),
        DeclareLaunchArgument('pause_s', default_value='', description='pause duration (empty = default)'),
        DeclareLaunchArgument('move_s', default_value='', description='seconds per move (empty = default)'),
        DeclareLaunchArgument('lean', default_value='', description='forward lean, max 20000 (empty = default)'),
        DeclareLaunchArgument('yaw', default_value='', description='yaw to each side, must exceed 9553 (empty = default)'),
        OpaqueFunction(function=_nodes),
    ])
