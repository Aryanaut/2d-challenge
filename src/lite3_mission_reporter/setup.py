from glob import glob
from setuptools import setup

package_name = "lite3_mission_reporter"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Mission result reporter (task_id / task_type / scene_type / label / pos / message JSON)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mission_reporter = lite3_mission_reporter.mission_reporter_node:main",
        ],
    },
)
