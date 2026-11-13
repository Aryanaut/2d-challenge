from glob import glob
from setuptools import setup

pkg = 'lite3_terrain_severity'
setup(
    name=pkg,
    version='1.0.0',
    packages=[pkg],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + pkg]),
        ('share/' + pkg, ['package.xml']),
        ('share/' + pkg + '/config', glob('config/*.yaml')),
        ('share/' + pkg + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'terrain_severity = lite3_terrain_severity.terrain_severity:main',
    ]},
)
