from glob import glob

from setuptools import setup

package_name = "surface_monitor"

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
    description="Reports the surface under the robot (wood / metal stairs / off course) from a downward camera, using a local classifier plus the Lab 6 VLM gateway.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "surface_monitor = surface_monitor.surface_node:main",
            "surface_tune = surface_monitor.tune:main",
            "surface_collect = surface_monitor.collect:main",
            "surface_extract = surface_monitor.extract_bag:main",
            "surface_train = surface_monitor.train:main",
        ],
    },
)
