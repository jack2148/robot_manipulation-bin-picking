from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    object_pose_transform_params = {
        # 비워두면 기본 경로 사용:
        # calib/config/handeye_capture_rs/handeye_result.json
        'handeye_result_path': '',

        # YOLO confidence threshold
        'min_confidence': 0.3,

        # 입력 object pose topic
        'object_topic': '/object_poses',

        # controller가 TCP pose를 보내는 trigger topic
        'peg_trigger_topic': '/manipulation/trigger_peg',
        'hole_trigger_topic': '/manipulation/trigger_hole',

        # 변환 결과 publish topic
        'peg_output_topic': '/vision/peg_targets',
        'hole_output_topic': '/vision/hole_targets',

        # peg / hole class 이름
        'peg_classes': ['cylinder', 'square', 'cross'],
        'hole_classes': ['cylinder_hole', 'square_hole', 'cross_hole'],
    }

    return LaunchDescription([
        Node(
            package='vision',
            executable='object_pose_transform_node',
            name='object_pose_transform_node',
            output='screen',
            parameters=[object_pose_transform_params],
        ),
    ])