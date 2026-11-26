"""Publish the left and right USB webcams.

  ros2 launch usb_cameras usb_cameras.launch.py left_device:=/dev/video0 right_device:=/dev/video2
  # stable names (recommended):
  ros2 launch usb_cameras usb_cameras.launch.py \
      left_device:=/dev/v4l/by-path/<left-port>-video-index0 \
      right_device:=/dev/v4l/by-path/<right-port>-video-index0

Find the devices first:  ros2 run usb_cameras probe_cameras
Set a device to "none" to run only one camera.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    params = os.path.join(get_package_share_directory("usb_cameras"), "config", "usb_cameras.yaml")
    left = LaunchConfiguration("left_device").perform(context)
    right = LaunchConfiguration("right_device").perform(context)
    names, devices = [], []
    for name, dev in (("left", left), ("right", right)):
        if dev and dev.lower() != "none":
            names.append(name)
            devices.append(dev)
    if not names:
        raise RuntimeError("give left_device:=... and/or right_device:=...")
    return [Node(package="usb_cameras", executable="usb_cameras", name="usb_cameras", output="screen",
                 parameters=[params, {"names": names, "devices": devices}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("left_device", default_value="/dev/video0"),
        DeclareLaunchArgument("right_device", default_value="/dev/video2"),
        OpaqueFunction(function=_nodes),
    ])
