from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision',
            executable='pose_publisher_ob_in',
            name='pose_publisher_ob_in',
            output='screen',
        ),
    ])
