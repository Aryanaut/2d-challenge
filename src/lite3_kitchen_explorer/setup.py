from glob import glob
import os

from setuptools import setup

package_name = 'lite3_kitchen_explorer'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'prompts'), glob('prompts/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@example.com',
    description='Lite3 frontier exploration + AI environment reports at every stop',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_explorer = lite3_kitchen_explorer.frontier_explorer:main',
            'environment_reporter = lite3_kitchen_explorer.environment_reporter:main',
            'odom_tf_bridge = lite3_kitchen_explorer.odom_tf_bridge:main',
            'gait_switcher = lite3_kitchen_explorer.gait_switcher:main',
            'preflight = lite3_kitchen_explorer.preflight:main',
            'survey_monitor = lite3_kitchen_explorer.survey_monitor:main',
        ],
    },
)
