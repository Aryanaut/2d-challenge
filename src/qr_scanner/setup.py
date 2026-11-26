from glob import glob

from setuptools import setup

package_name = "qr_scanner"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Legged Robot Challenge team",
    maintainer_email="todo@todo.todo",
    description="Reads QR codes on walls from the front camera at varying distances, decodes them and localises them in the map.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "qr_scanner = qr_scanner.qr_node:main",
            "qr_test = qr_scanner.qr_test:main",
        ],
    },
)
