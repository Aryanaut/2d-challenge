from glob import glob
import os

from setuptools import setup

package_name = 'lite3_explorer'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.mission'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@example.com',
    description='Lite3 tile exploration with a mission manager (the AI comes from lite3_llm_tools)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_manager = lite3_explorer.mission_manager:main',
            'odom_tf_bridge = lite3_explorer.odom_tf_bridge:main',
            'gait_switcher = lite3_explorer.gait_switcher:main',
            'animation_player = lite3_explorer.animation_player:main',
            'animation_test = lite3_explorer.animation_test:main',
            'preflight =lite3_explorer.preflight:main',
            'survey_monitor = lite3_explorer.survey_monitor:main',
            'curtain_detector = lite3_explorer.curtain_detector:main',
            'curtain_costmap_filter = lite3_explorer.curtain_costmap_filter:main',
            'stair_detector = lite3_explorer.stair_detector:main',
            'roller_detector = lite3_explorer.roller_detector:main',
            'tilt_gate = lite3_explorer.tilt_gate:main',
            'scan_fill = lite3_explorer.scan_fill:main',
            'controller_monitor = lite3_explorer.controller_monitor:main',
            'robot_state_bridge = lite3_explorer.robot_state_bridge:main',
            'step_pause_gate = lite3_explorer.step_pause_gate:main',
            'step_ahead_detector = lite3_explorer.step_ahead_detector:main',
            'sensor_relay = lite3_explorer.sensor_relay:main',
        ],
    },
)
