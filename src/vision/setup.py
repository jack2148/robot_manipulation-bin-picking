import os
from glob import glob
from setuptools import setup

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch 파일 설치
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # YOLO weight 설치
        (os.path.join('share', package_name, 'weights'), ['weights/best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'pose_publisher = vision.pose_publisher:main',
        ],
    },
)