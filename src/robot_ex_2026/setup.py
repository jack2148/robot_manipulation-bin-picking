from setuptools import find_packages, setup
# RoCogMan 연구실은 항상 열려있답니다~
package_name = 'robot_ex_2026'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taekyong',
    maintainer_email='taekyong@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "motor_id_check=robot_ex_2026.motor_id_check:main",
            "grip_current=robot_ex_2026.grip_current:main",
        ],
    },
)
