from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision',
            executable='pose_publisher',
            name='pose_publisher',
            output='screen',
        ),
    ])