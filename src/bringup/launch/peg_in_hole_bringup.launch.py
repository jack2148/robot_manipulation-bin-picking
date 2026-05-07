"""
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select vision calib control bringup
source install/setup.bash

ros2 launch bringup peg_in_hole_bringup.launch.py
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
    # 1. Pose Publisher
    # RealSense + YOLO
    #
    # /detect_mode = object
    #   -> best.pt 실행
    #   -> /object_poses publish
    #
    # /detect_mode = insert
    #   -> insert_best.pt 실행
    #   -> /insert_poses publish
    # ============================================================
    pose_publisher_node = Node(
        package='vision',
        executable='pose_publisher',
        name='pose_publisher',
        output='screen',
    )

    # ============================================================
    # 2. Object Pose Transform
    #
    # trigger_peg:
    #   /detect_mode object 요청
    #   fresh /object_poses 수신
    #   -> /vision/peg_targets publish
    #
    # trigger_hole:
    #   /detect_mode object 요청
    #   fresh /object_poses 저장
    #   /detect_mode insert 요청
    #   fresh /insert_poses 수신
    #   object와 insert 비교 후 가까운 insert 제외
    #   -> /vision/hole_targets publish
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

        # robot base 좌표계 XY 기준 mm 거리
        # 같은 id이고 이 거리보다 가까우면 같은 물체로 보고 insert 후보에서 제외
        'exclude_dist_mm': 30.0,
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
        'move_j_speed': 60.0,
        'move_j_acc': 60.0,

        'move_l_speed': 60.0,
        'move_l_acc': 120.0,

        'approach_move_l_speed': 60.0,
        'approach_move_l_acc': 120.0,

        'descend_move_l_speed': 20.0,
        'descend_move_l_acc': 40.0,

        # ===== 비전 토픽 =====
        'peg_targets_topic': '/vision/peg_targets',
        'hole_targets_topic': '/vision/hole_targets',
        'trigger_peg_topic': '/manipulation/trigger_peg',
        'trigger_hole_topic': '/manipulation/trigger_hole',

        # trigger 후 fresh pose 받을 시간 고려
        'camera_settle_sec': 0.5,
        'vision_wait_timeout_sec': 3.0,

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