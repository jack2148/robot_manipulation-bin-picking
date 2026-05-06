"""
cd ~/course/robot_manipulation-bin-picking
colcon build --symlink-install --packages-select vision calib control bringup
source install/setup.bash

"""


from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # Path settings
    # ============================================================
    handeye_result_path = (
        '/home/choisuhyun/course/robot_manipulation-bin-picking/'
        'src/calib/config/handeye_capture_rs/handeye_result.json'
    )

    # 참고:
    # pose_publisher.py 내부에서 아래 경로를 자동으로 사용함.
    # get_package_share_directory('vision') / 'weights' / 'best.pt'
    #
    # 원본 src 경로:
    # /home/choisuhyun/course/robot_manipulation-bin-picking/src/vision/weights/best.pt

    # ============================================================
    # 1. Pose Publisher
    # RealSense + YOLO -> /object_poses
    # package: vision
    # executable: pose_publisher
    # ============================================================
    pose_publisher_node = Node(
        package='vision',
        executable='pose_publisher',
        name='pose_publisher',
        output='screen',
    )

    # ============================================================
    # 2. Object Pose Transform
    # /object_poses + /manipulation/trigger_* -> /vision/*_targets
    # package: calib
    # executable: object_pose_transform_node
    # ============================================================
    object_pose_transform_params = {
        'handeye_result_path': handeye_result_path,
        'min_confidence': 0.3,

        'object_topic': '/object_poses',

        'peg_trigger_topic': '/manipulation/trigger_peg',
        'hole_trigger_topic': '/manipulation/trigger_hole',

        'peg_output_topic': '/vision/peg_targets',
        'hole_output_topic': '/vision/hole_targets',

        'peg_classes': ['cylinder', 'square', 'cross'],
        'hole_classes': ['cylinder_hole', 'square_hole', 'cross_hole'],
    }

    object_pose_transform_node = Node(
        package='calib',
        executable='object_pose_transform_node',
        name='object_pose_transform_node',
        output='screen',
        parameters=[object_pose_transform_params],
    )

    # ============================================================
    # 3. Peg-in-hole Controller
    # Robot control + trigger publish + target subscribe
    # package: control
    # executable: peg_in_hole_controller
    # ============================================================
    peg_in_hole_params = {
        # ===== 로봇/ROS 설정 =====
        'robot_ip': '192.168.1.10',
        'use_simulation_mode': False,
        'gripper_topic': '/grip_state',

        'grip_open': 1,
        'grip_close': 0,
        'grip_stop': 2,

        # ===== 고정 joint 자세 =====
        'home_joint': [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0],
        'peg_camera_joint': [10.87, 2.78, 79.15, 8.07, 90.0, 34.16],
        'hole_camera_joint': [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16],
        'peg_return_mid_joint': [-47.0, 2.78, 79.15, 8.07, 90.0, 34.16],

        # ===== 작업 z 좌표 =====
        'pick_down_target_z_mm': 69.83,
        'pick_approach_offset_z_mm': 30.0,
        'pick_up_target_z_mm': 110.0,

        'place_approach_target_z_mm': 108.0,
        'place_down_target_z_mm': 98.0,
        'place_up_target_z_mm': 110.0,

        # ===== 속도/가속도 =====
        'move_j_speed': 30.0,
        'move_j_acc': 40.0,
        'move_l_speed': 40.0,
        'move_l_acc': 60.0,

        # ===== 비전 토픽 =====
        'peg_targets_topic': '/vision/peg_targets',
        'hole_targets_topic': '/vision/hole_targets',
        'trigger_peg_topic': '/manipulation/trigger_peg',
        'trigger_hole_topic': '/manipulation/trigger_hole',
        'camera_settle_sec': 0.5,
        'vision_wait_timeout_sec': 2.0,

        # ===== 대기/timeout =====
        'grasp_wait_sec': 1.0,
        'release_wait_sec': 1.0,
        'move_start_timeout_sec': 3.0,
    }

    peg_in_hole_controller_node = Node(
        package='control',
        executable='peg_in_hole_controller',
        name='peg_in_hole_controller',
        output='screen',
        parameters=[peg_in_hole_params],
    )

    return LaunchDescription([
        pose_publisher_node,
        object_pose_transform_node,

        TimerAction(
            period=5.0,
            actions=[
                peg_in_hole_controller_node,
            ],
        ),
    ])