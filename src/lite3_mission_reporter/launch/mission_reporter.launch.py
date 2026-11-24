import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SERVER = "http://192.168.10.141:8001"
URLS = {"A": f"{SERVER}/task_report_A", "B": f"{SERVER}/task_report_B"}


def resolve_url(value):
    """'A' / 'URLA' -> task_report_A, 'B' / 'URLB' -> task_report_B, a full http(s) URL as given, '' = the yaml's server_url."""
    v = value.strip()
    key = v.upper().removeprefix("URL")
    if key in URLS:
        return URLS[key]
    if v and not v.lower().startswith(("http://", "https://")):
        raise RuntimeError(f"url:={value!r} is not A, B (URLA / URLB) or an http(s) URL")
    return v


def launch_setup(context):
    cfg = os.path.join(get_package_share_directory("lite3_mission_reporter"), "config", "mission_reporter.yaml")
    params = [cfg, {
        # launch args actually reach the node now (before, use_llm_result was ignored)
        "use_llm_result": ParameterValue(LaunchConfiguration("use_llm_result"), value_type=bool),
    }]
    url = resolve_url(LaunchConfiguration("url").perform(context))
    if url:                                                   # after the yaml, so it wins over server_url there
        params.append({"server_url": url})
    return [Node(package="lite3_mission_reporter", executable="mission_reporter", name="mission_reporter",
                 parameters=params, output="screen")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_llm_result", default_value="false"),
        DeclareLaunchArgument("url", default_value="",
                              description="A / URLA -> " + URLS["A"] + ", B / URLB -> " + URLS["B"]
                                          + ", or a full URL; empty = server_url from the yaml"),
        OpaqueFunction(function=launch_setup),
    ])
