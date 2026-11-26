from glob import glob

from setuptools import setup

package_name = "lite3_teleop_vlm"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/prompts", glob("prompts/*.txt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Legged Robot Challenge team",
    maintainer_email="todo@todo.todo",
    description="Keyboard teleop console for the Lite3 plus a VLM broker that shares the scene-understanding gateway between nodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleop_console = lite3_teleop_vlm.console_node:main",
            "vlm_broker = lite3_teleop_vlm.broker_node:main",
        ],
    },
)
