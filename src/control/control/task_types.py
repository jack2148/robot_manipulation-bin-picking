from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np


class TaskState(Enum):
    IDLE_HOME = auto()                         # 초기자세
    MOVE_TO_PEG_CAMERA_POSE = auto()          # peg를 보기 위한 카메라 자세로 이동
    MOVE_TO_PEG_CAMERA_POSE_VIA_MID = auto()  # hole 작업 후 peg 자세로 복귀
    INSPECT_PEGS = auto()                     # peg를 인식
    MOVE_TO_TARGET_PEG = auto()               # 목표 peg 위로 이동
    DESCEND_TO_PEG = auto()                   # peg 잡는 높이로 내려감
    GRASP_PEG = auto()                        # peg 잡기
    LIFT_WITH_PEG = auto()                    # peg 잡고 상승
    MOVE_TO_HOLE_CAMERA_POSE = auto()         # hole 보기 위한 자세로 이동
    INSPECT_HOLES = auto()                    # hole 인식
    MOVE_TO_TARGET_HOLE = auto()              # 목표 hole 위로 이동
    DESCEND_TO_HOLE = auto()                  # hole 삽입 높이로 내려감
    RELEASE_PEG = auto()                      # peg 놓기
    LIFT_FROM_HOLE = auto()                   # 놓고 상승
    CHECK_REMAINING_TASK = auto()             # 남은 작업 확인
    RETURN_HOME = auto()                      # 홈 자세 복귀
    DONE = auto()                             # 종료
    ERROR = auto()                            # 예외


@dataclass
class VisionTarget:
    """
    비전 노드에서 수신한 하나의 객체 정보.

    object_id:
        0 = 원통
        1 = 직사각형
        2 = 십자가

    pose:
        [x, y, z, rx, ry, rz]
        x, y는 vision에서 받은 base 좌표계 물체 위치.
        z는 이후 상태에서 작업 높이로 덮어쓴다.
    """
    pose: np.ndarray
    object_id: int


@dataclass
class TaskContext:
    # ===== 고정 자세 =====
    home_joint: np.ndarray
    peg_camera_joint: np.ndarray
    hole_camera_joint: np.ndarray

    # ===== 그리퍼 수평 TCP 자세 =====
    # MoveL에서는 어떤 target pose가 들어와도 이 RPY로 강제한다.
    flat_tcp_rx_deg: float = 90.0
    flat_tcp_ry_deg: float = 0.0
    flat_tcp_rz_deg: float = 0.0

    # ===== 작업 z 파라미터 =====
    pick_down_target_z_mm: float = 69.83
    pick_approach_offset_z_mm: float = 30.0
    pick_up_target_z_mm: float = 110.0

    place_approach_target_z_mm: float = 108.0
    place_down_target_z_mm: float = 98.0
    place_up_target_z_mm: float = 110.0

    # ===== motion 파라미터 =====
    move_j_speed: float = 60.0
    move_j_acc: float = 80.0
    move_l_speed: float = 80.0
    move_l_acc: float = 120.0

    # 기존 rbpodo wait는 더 이상 완료 판단에 사용하지 않지만 파라미터는 남겨둠
    move_start_timeout_sec: float = 0.5

    # ===== MoveJ 완료 판단 파라미터 =====
    joint_tol_deg: float = 0.5
    joint_stable_count_required: int = 5
    joint_wait_timeout_sec: float = 40.0
    joint_polling_dt_sec: float = 0.05

    # ===== MoveL 완료 판단 파라미터 =====
    tcp_pos_tol_mm: float = 1.0
    tcp_rot_tol_deg: float = 2.0
    tcp_stable_count_required: int = 5
    tcp_wait_timeout_sec: float = 40.0
    tcp_polling_dt_sec: float = 0.05

    # ===== gripper 파라미터 =====
    grasp_wait_sec: float = 1.0
    release_wait_sec: float = 1.0

    # ===== vision 파라미터 =====
    vision_wait_timeout_sec: float = 2.0

    # 기존 파라미터 호환용. 실제 MoveL 자세에는 flat_tcp_*를 사용한다.
    vision_fixed_rx_deg: float = 90.0
    vision_fixed_rz_deg: float = 0.0

    # ===== 런타임 데이터 =====
    peg_targets: list[VisionTarget] = field(default_factory=list)
    hole_targets: list[VisionTarget] = field(default_factory=list)

    current_peg_index: int = -1
    current_hole_index: int = -1

    current_peg_pick_pose: np.ndarray | None = None
    current_hole_place_pose: np.ndarray | None = None
    current_target_id: int | None = None
