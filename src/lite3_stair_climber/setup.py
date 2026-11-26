from glob import glob

from setuptools import setup

package_name = 'stair_climber'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hadi Priyatna',
    maintainer_email='hadipriyatna1806@gmail.com',
    description='Autonomous stair detection and crossing for the DEEP Robotics Lite3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stair_climber_node = stair_climber.stair_climber_node:main',
            'vlm_stair_checker = stair_climber.vlm_stair_checker:main',
            'robot_state_bridge = stair_climber.robot_state_bridge:main',
        ],
    },
)
