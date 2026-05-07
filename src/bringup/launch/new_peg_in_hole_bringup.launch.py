"""
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select vision calib control bringup
source install/setup.bash

ros2 launch bringup peg_in_hole_bringup_yaw.launch.py
"""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # Path settings
    # ============================================================
    handeye_result_path = (
        '/home/chu/robot_manipulation-bin-picking/'
        'src/calib/config/handeye_capture_rs/handeye_result.json'
    )

    # ============================================================
    # 1. Pose Publisher (Template Yaw Version)
    #
    # RealSense + YOLO + Template Matching Yaw
    #
    # /detect_mode = object
    #   -> best.pt 실행
    #   -> /object_poses publish
    #
    # /detect_mode = insert
    #   -> insert_best.pt 실행
    #   -> /insert_poses publish
    #
    # 추가:
    #   -> yaw_deg
    #   -> yaw_score
    #   -> yaw_source
    # ============================================================
    pose_publisher_yaw_node = Node(
        package='vision',
        executable='pose_publisher_yaw',
        name='pose_publisher_yaw',
        output='screen',
    )

    # ============================================================
    # 2. Object Pose Transform
    #
    # yaw 우선순위:
    #   1. yaw_deg 존재 시 template yaw 사용
    #   2. 없으면 PCA orientation 기반 yaw 사용
    #
    # insert끼리 동일 위치면
    # confidence 높은 것만 유지
    # ============================================================
    object_pose_transform_params = {
        'handeye_result_path': handeye_result_path,
        'min_confidence': 0.3,

        'object_topic': '/object_poses',
        'insert_topic': '/insert_poses',
        'detect_mode_topic': '/detect_mode',

        'peg_trigger_topic': '/manipulation/trigger_peg',
        'hole_trigger_topic': '/manipulation/trigger_hole',

        'peg_output_topic': '/vision/peg_targets',
        'hole_output_topic': '/vision/hole_targets',

        # object와 insert가 가까우면
        # 같은 물체로 판단하여 insert 제거
        'exclude_dist_mm': 30.0,

        # insert끼리 너무 가까우면
        # confidence 높은 것만 유지
        'insert_duplicate_dist_mm': 10.0,
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
        'peg_camera_joint': [9.68, 8.52, 55.63, 25.85, 90.0, 35.39],
        'hole_camera_joint': [-173.76, 8.07, 56.16, 25.78, 90.0, 38.83],
        'peg_return_mid_joint': [-47.0, 2.78, 79.15, 8.07, 90.0, 34.16],

        # ===== 작업 z 좌표 =====
        'pick_down_target_z_mm': 69.83,
        'pick_approach_offset_z_mm': 30.0,
        'pick_up_target_z_mm': 200.0,

        'place_approach_target_z_mm': 108.0,
        'place_down_target_z_mm': 98.0,
        'place_up_target_z_mm': 200.0,

        # ===== 속도/가속도 =====
        'move_j_speed': 60.0,
        'move_j_acc': 60.0,

        'move_l_speed': 180.0,
        'move_l_acc': 360.0,

        'approach_move_l_speed': 180.0,
        'approach_move_l_acc': 360.0,

        'descend_move_l_speed': 60.0,
        'descend_move_l_acc': 120.0,

        # ===== 비전 토픽 =====
        'peg_targets_topic': '/vision/peg_targets',
        'hole_targets_topic': '/vision/hole_targets',

        'trigger_peg_topic': '/manipulation/trigger_peg',
        'trigger_hole_topic': '/manipulation/trigger_hole',

        # ===== 비전 timing =====
        'camera_settle_sec': 0.5,
        'vision_wait_timeout_sec': 5.0,

        # ===== 대기 =====
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
        pose_publisher_yaw_node,

        object_pose_transform_node,

        TimerAction(
            period=5.0,
            actions=[
                peg_in_hole_controller_node,
            ],
        ),
    ])