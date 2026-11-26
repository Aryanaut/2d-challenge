from glob import glob

from setuptools import setup

package_name = "usb_cameras"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/scripts", glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Legged Robot Challenge team",
    maintainer_email="todo@todo.todo",
    description="Publishes several USB webcams (left/right) as ROS 2 image topics, with a probe tool to find them.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "usb_cameras = usb_cameras.camera_node:main",
            "probe_cameras = usb_cameras.probe:main",
            "net_receiver = usb_cameras.net_receiver:main",
        ],
    },
)
