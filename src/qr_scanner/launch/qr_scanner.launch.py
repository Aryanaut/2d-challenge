"""Launch the QR scanner on up to three cameras.

  ros2 launch qr_scanner qr_scanner.launch.py \
      front_image_topic:=/camera/color/image_raw front_camera_info:=/camera/color/camera_info \
      left_image_topic:=/left_camera/image_raw right_image_topic:=/right_camera/image_raw

  # USB cameras visible on this computer as devices (no camera driver needed):
  ros2 launch qr_scanner qr_scanner.launch.py left_device:=/dev/video2 right_device:=/dev/video4

  # side cameras on the laptop/Pi that already READ the codes (cam_sender.py + net_receiver):
  ros2 launch qr_scanner qr_scanner.launch.py left_detections_topic:=/left/qr_codes \
      right_detections_topic:=/right/qr_codes

Leave a camera's topic, device and detections topic as "none" to switch it off.
Find topics with:  ros2 topic list | grep -i -E "image|camera"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CAMERA_ARGS = {
    # name: (image_topic, camera_info, depth_topic)
    "front": ("/camera/color/image_raw", "/camera/color/camera_info", "none"),
    "left": ("none", "none", "none"),
    "right": ("none", "none", "none"),
}


def camera_launch_args():
    args = []
    for name, (img, info, depth) in CAMERA_ARGS.items():
        args += [
            DeclareLaunchArgument(f"{name}_image_topic", default_value=img),
            DeclareLaunchArgument(f"{name}_compressed", default_value="false"),
            DeclareLaunchArgument(f"{name}_camera_info", default_value=info),
            DeclareLaunchArgument(f"{name}_depth_topic", default_value=depth),
            DeclareLaunchArgument(f"{name}_device", default_value="none"),
            DeclareLaunchArgument(f"{name}_detections_topic", default_value="none"),
        ]
    return args


def camera_params():
    s = lambda n: ParameterValue(LaunchConfiguration(n), value_type=str)
    params = {}
    for name in CAMERA_ARGS:
        params.update({
            f"{name}_image_topic": s(f"{name}_image_topic"),
            f"{name}_compressed": ParameterValue(LaunchConfiguration(f"{name}_compressed"), value_type=bool),
            f"{name}_camera_info_topic": s(f"{name}_camera_info"),
            f"{name}_depth_topic": s(f"{name}_depth_topic"),
            f"{name}_device": s(f"{name}_device"),
            f"{name}_detections_topic": s(f"{name}_detections_topic"),
        })
    return params


def generate_launch_description():
    params = os.path.join(get_package_share_directory("qr_scanner"), "config", "qr_scanner.yaml")
    node = Node(
        package="qr_scanner", executable="qr_scanner", name="qr_scanner", output="screen",
        parameters=[params, camera_params(),
                    {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}],
    )
    return LaunchDescription(camera_launch_args() +
                             [DeclareLaunchArgument("use_sim_time", default_value="false"), node])
