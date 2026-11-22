from glob import glob

from setuptools import setup

package_name = 'lite3_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/mazes', glob('mazes/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryanaut',
    maintainer_email='you@example.com',
    description='Maze simulator for lite3_explorer (exploration, tile grid, curtain mode) with RViz.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = lite3_sim.sim_node:main',
            'headless = lite3_sim.headless:main',
            'maze_preview = lite3_sim.maze_preview:main',
        ],
    },
)
