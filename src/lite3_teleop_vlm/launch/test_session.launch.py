"""Everything except the keyboard console (that needs its own terminal).

  # terminal 1 (robot or workstation): broker + surface monitor + QR scanner
  ros2 launch lite3_teleop_vlm test_session.launch.py group:=group07 \
      client_script:=/home/user/scene_understanding/client.py \
      front_image_topic:=/camera/color/image_raw down_image_topic:=/down_camera/image_raw

  # terminal 2 (SSH on the robot's group computer, transfer_ws sourced): drive + test
  ros2 run lite3_teleop_vlm teleop_console --ros-args \
      --params-file $(ros2 pkg prefix lite3_teleop_vlm)/share/lite3_teleop_vlm/config/test_session.yaml

Switch parts off with start_surface:=false / start_qr:=false / start_down_camera:=true.
Side QR cameras: left_image_topic:=/... right_image_topic:=/...  (or left_device:=/dev/videoN ...).

Side cameras on the laptop/Pi (cam_sender.py reads the QR codes there at full resolution):
  ros2 launch lite3_teleop_vlm test_session.launch.py net_cameras:=true ...
starts the network receiver too and wires /left/qr_codes + /right/qr_codes into qr_scanner
(the preview images /left|right/image_raw/compressed are only used for the debug pictures).
Don't also run "ros2 run usb_cameras net_receiver" then - both would want port 5600.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _share(pkg):
    try:
        return get_package_share_directory(pkg)
    except Exception:
        return None


def generate_launch_description():
    here = _share("lite3_teleop_vlm")
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("group", default_value="groupXX"),
        DeclareLaunchArgument("client_script", default_value="/home/user/scene_understanding/client.py"),
        DeclareLaunchArgument("client_python", default_value="python3"),
        DeclareLaunchArgument("front_image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("front_compressed", default_value="false"),
        DeclareLaunchArgument("front_camera_info", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("front_depth_topic", default_value="none"),   # only a depth image ALIGNED to colour
        # side USB cameras for QR scanning: a ROS topic, or a /dev/video* device; "none" = off
        DeclareLaunchArgument("left_image_topic", default_value="none"),
        DeclareLaunchArgument("left_compressed", default_value="false"),
        DeclareLaunchArgument("left_camera_info", default_value="none"),
        DeclareLaunchArgument("left_device", default_value="none"),
        DeclareLaunchArgument("left_detections_topic", default_value="none"),
        DeclareLaunchArgument("right_image_topic", default_value="none"),
        DeclareLaunchArgument("right_compressed", default_value="false"),
        DeclareLaunchArgument("right_camera_info", default_value="none"),
        DeclareLaunchArgument("right_device", default_value="none"),
        DeclareLaunchArgument("right_detections_topic", default_value="none"),
        DeclareLaunchArgument("down_image_topic", default_value="/down_camera/image_raw"),
        DeclareLaunchArgument("down_compressed", default_value="false"),
        DeclareLaunchArgument("start_surface", default_value="true"),
        DeclareLaunchArgument("start_qr", default_value="true"),
        DeclareLaunchArgument("start_down_camera", default_value="false"),
        DeclareLaunchArgument("down_video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("surface_mode", default_value="hybrid"),
        DeclareLaunchArgument("net_cameras", default_value="false"),
        DeclareLaunchArgument("net_port", default_value="5600"),
    ]
    b = lambda name: ParameterValue(L(name), value_type=bool)
    s = lambda name: ParameterValue(L(name), value_type=str)

    nodes = [Node(
        package="lite3_teleop_vlm", executable="vlm_broker", name="vlm_broker", output="screen",
        parameters=[os.path.join(here, "config", "test_session.yaml"), {
            "group": L("group"), "client_script": L("client_script"), "client_python": L("client_python"),
            "front_image_topic": L("front_image_topic"), "front_compressed": b("front_compressed"),
            "down_image_topic": L("down_image_topic"), "down_compressed": b("down_compressed"),
            "prompts_dir": os.path.join(here, "prompts")}])]

    nodes.append(Node(
        package="v4l2_camera", executable="v4l2_camera_node", name="down_camera", namespace="down_camera",
        condition=IfCondition(L("start_down_camera")),
        parameters=[{"video_device": L("down_video_device"), "image_size": [640, 480]}]))

    surface = _share("surface_monitor")
    if surface:
        nodes.append(Node(
            package="surface_monitor", executable="surface_monitor", name="surface_monitor", output="screen",
            condition=IfCondition(L("start_surface")),
            parameters=[os.path.join(surface, "config", "surface_monitor.yaml"), {
                "vlm_backend": "broker", "mode": L("surface_mode"),
                "image_topic": L("down_image_topic"), "compressed": b("down_compressed"),
                "prompt_file": os.path.join(surface, "prompts", "surface_prompt.txt")}]))

    def net_or(value_if_net, arg):
        """value_if_net when net_cameras:=true, else the launch argument `arg`."""
        return PythonExpression(["'", value_if_net, "' if '", L("net_cameras"), "'.lower() == 'true' else '",
                                 L(arg), "'"])

    if _share("usb_cameras"):
        nodes.append(Node(
            package="usb_cameras", executable="net_receiver", name="net_camera_receiver", output="screen",
            condition=IfCondition(L("net_cameras")),
            parameters=[{"port": ParameterValue(L("net_port"), value_type=int)}]))

    qr = _share("qr_scanner")
    if qr:
        nodes.append(Node(
            package="qr_scanner", executable="qr_scanner", name="qr_scanner", output="screen",
            condition=IfCondition(L("start_qr")),
            parameters=[os.path.join(qr, "config", "qr_scanner.yaml"), {
                "front_image_topic": s("front_image_topic"), "front_compressed": b("front_compressed"),
                "front_camera_info_topic": s("front_camera_info"), "front_depth_topic": s("front_depth_topic"),
                "left_image_topic": ParameterValue(net_or("/left/image_raw/compressed", "left_image_topic"),
                                                   value_type=str),
                "left_compressed": ParameterValue(net_or("true", "left_compressed"), value_type=bool),
                "left_camera_info_topic": s("left_camera_info"), "left_device": s("left_device"),
                "left_detections_topic": ParameterValue(net_or("/left/qr_codes", "left_detections_topic"),
                                                        value_type=str),
                "right_image_topic": ParameterValue(net_or("/right/image_raw/compressed", "right_image_topic"),
                                                    value_type=str),
                "right_compressed": ParameterValue(net_or("true", "right_compressed"), value_type=bool),
                "right_camera_info_topic": s("right_camera_info"), "right_device": s("right_device"),
                "right_detections_topic": ParameterValue(net_or("/right/qr_codes", "right_detections_topic"),
                                                         value_type=str)}]))

    return LaunchDescription(args + nodes)
