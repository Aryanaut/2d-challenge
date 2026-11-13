"""Only the frontier explorer + environment reporter, for robots where SLAM and Nav2 already run."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('lite3_kitchen_explorer')
    full = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'kitchen_exploration.launch.py')),
        launch_arguments={'publish_odom_tf': 'false', 'use_cloud_to_scan': 'false',
                          'use_slam': 'false', 'use_nav2': 'false'}.items())
    return LaunchDescription([full])
