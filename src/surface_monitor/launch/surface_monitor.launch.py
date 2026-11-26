"""Launch the surface monitor, optionally with the Logitech webcam driver.

  ros2 launch surface_monitor surface_monitor.launch.py group:=group07 \
      client_script:=/home/user/scene_understanding/client.py

  # also start the webcam (needs: sudo apt install ros-$ROS_DISTRO-v4l2-camera)
  ros2 launch surface_monitor surface_monitor.launch.py start_camera:=true \
      video_device:=/dev/v4l/by-id/usb-046d_<your-webcam>-video-index0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("surface_monitor")
    params = os.path.join(share, "config", "surface_monitor.yaml")
    prompt = os.path.join(share, "prompts", "surface_prompt.txt")

    args = [
        DeclareLaunchArgument("group", default_value="groupXX"),
        DeclareLaunchArgument("client_script",
                              default_value="/home/user/scene_understanding/client.py"),
        DeclareLaunchArgument("client_python", default_value="python3"),
        DeclareLaunchArgument("mode", default_value="hybrid"),
        DeclareLaunchArgument("vlm_backend", default_value="direct"),   # broker = share the gateway via vlm_broker
        DeclareLaunchArgument("image_topic", default_value="/down_camera/image_raw"),
        DeclareLaunchArgument("start_camera", default_value="false"),
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]

    camera = Node(
        package="v4l2_camera", executable="v4l2_camera_node", name="down_camera",
        namespace="down_camera", condition=IfCondition(LaunchConfiguration("start_camera")),
        parameters=[{"video_device": LaunchConfiguration("video_device"),
                     "image_size": [640, 480],
                     "use_sim_time": LaunchConfiguration("use_sim_time")}],
    )  # publishes /down_camera/image_raw

    monitor = Node(
        package="surface_monitor", executable="surface_monitor", name="surface_monitor",
        output="screen",
        parameters=[params, {
            "group": LaunchConfiguration("group"),
            "client_script": LaunchConfiguration("client_script"),
            "client_python": LaunchConfiguration("client_python"),
            "mode": LaunchConfiguration("mode"),
            "vlm_backend": LaunchConfiguration("vlm_backend"),
            "image_topic": LaunchConfiguration("image_topic"),
            "prompt_file": prompt,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )
    return LaunchDescription(args + [camera, monitor])
