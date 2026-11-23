from glob import glob
from setuptools import setup

package_name = "lite3_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/hazard_templates", glob("hazard_templates/*.png") + glob("hazard_templates/*.jpg")),
        ("share/" + package_name + "/models", glob("models/*.onnx") + glob("models/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Object detection / trigger node that publishes events for the LLM scene-understanding package",
    license="MIT",
    entry_points={
        "console_scripts": [
            "object_detector = lite3_perception.object_detector_node:main",
            "llm_event_bridge = lite3_perception.llm_event_bridge:main",
        ],
    },
)
