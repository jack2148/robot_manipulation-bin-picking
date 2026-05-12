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
        (os.path.join('share', package_name, 'weights'), ['weights/best.pt', 'weights/insert_best.pt', 'weights/ob_in_best.pt']),

        # template png 설치
        (os.path.join('share', package_name, 'templates'), glob('templates/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'pose_publisher = vision.pose_publisher:main',
            'pose_publisher_yaw = vision.pose_publisher_yaw:main',
            'pose_publisher_newenw = vision.pose_publisher_newenw:main',
            'pose_publisher_ob_in = vision.pose_publisher_ob_in:main',
            'new_pose_publisher_ob_in = vision.new_pose_publisher_ob_in:main',

        ],
    },
)