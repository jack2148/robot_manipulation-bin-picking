import time
from enum import Enum, auto
from dataclasses import dataclass, field

import numpy as np
import rbpodo as rb

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64MultiArray


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
    peg_targets: list[np.ndarray] = field(default_factory=list)
    hole_targets: list[np.ndarray] = field(default_factory=list)

    current_peg_index: int = -1
    current_hole_index: int = -1

    current_peg_pick_pose: np.ndarray | None = None
    current_hole_place_pose: np.ndarray | None = None


class PegInHoleController(Node):
    """
    - rbpodo로 로봇 제어
    - /grip_state publish로 그리퍼 제어
    - /vision/peg_targets, /vision/hole_targets 구독
    """

    def __init__(self):
        super().__init__("peg_in_hole_controller")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("robot_ip", "192.168.1.10"),
                ("use_simulation_mode", False),
                ("gripper_topic", "grip_state"),

                ("grip_open", 1),
                ("grip_close", 0),
                ("grip_stop", 2),

                ("home_joint", [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0]),
                ("peg_camera_joint", [10.87, 2.78, 79.15, 8.07, 90.0, 34.16]),
                ("hole_camera_joint", [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16]),

                # MoveL에서 항상 강제할 수평 TCP 자세
                ("flat_tcp_rx_deg", 90.0),
                ("flat_tcp_ry_deg", 0.0),
                ("flat_tcp_rz_deg", 0.0),

                ("pick_down_target_z_mm", 69.83),
                ("pick_approach_offset_z_mm", 30.0),
                ("pick_up_target_z_mm", 110.0),

                ("place_approach_target_z_mm", 108.0),
                ("place_down_target_z_mm", 98.0),
                ("place_up_target_z_mm", 110.0),

                ("move_j_speed", 60.0),
                ("move_j_acc", 80.0),
                ("move_l_speed", 80.0),
                ("move_l_acc", 120.0),
                ("move_start_timeout_sec", 0.5),

                ("joint_tol_deg", 0.5),
                ("joint_stable_count_required", 5),
                ("joint_wait_timeout_sec", 40.0),
                ("joint_polling_dt_sec", 0.05),

                ("tcp_pos_tol_mm", 1.0),
                ("tcp_rot_tol_deg", 2.0),
                ("tcp_stable_count_required", 5),
                ("tcp_wait_timeout_sec", 40.0),
                ("tcp_polling_dt_sec", 0.05),

                ("grasp_wait_sec", 1.0),
                ("release_wait_sec", 1.0),

                ("peg_targets_topic", "/vision/peg_targets"),
                ("hole_targets_topic", "/vision/hole_targets"),
                ("vision_wait_timeout_sec", 2.0),
                ("vision_fixed_rx_deg", 90.0),
                ("vision_fixed_rz_deg", 0.0),
            ],
        )

        self.use_simulation_mode = self._get_bool_param("use_simulation_mode")

        self.GRIP_OPEN = self._get_int_param("grip_open")
        self.GRIP_CLOSE = self._get_int_param("grip_close")
        self.GRIP_STOP = self._get_int_param("grip_stop")

        robot_ip = self._get_str_param("robot_ip")
        gripper_topic = self._get_str_param("gripper_topic")

        # 명령 전송용
        self.robot = rb.Cobot(robot_ip)

        # 상태 읽기용
        self.robot_data = rb.CobotData(robot_ip)

        self.grip_pub = self.create_publisher(Int32, gripper_topic, 10)

        self.state = TaskState.IDLE_HOME

        self.ctx = TaskContext(
            home_joint=self._get_array_param("home_joint", 6),
            peg_camera_joint=self._get_array_param("peg_camera_joint", 6),
            hole_camera_joint=self._get_array_param("hole_camera_joint", 6),

            flat_tcp_rx_deg=self._get_float_param("flat_tcp_rx_deg"),
            flat_tcp_ry_deg=self._get_float_param("flat_tcp_ry_deg"),
            flat_tcp_rz_deg=self._get_float_param("flat_tcp_rz_deg"),

            pick_down_target_z_mm=self._get_float_param("pick_down_target_z_mm"),
            pick_approach_offset_z_mm=self._get_float_param("pick_approach_offset_z_mm"),
            pick_up_target_z_mm=self._get_float_param("pick_up_target_z_mm"),

            place_approach_target_z_mm=self._get_float_param("place_approach_target_z_mm"),
            place_down_target_z_mm=self._get_float_param("place_down_target_z_mm"),
            place_up_target_z_mm=self._get_float_param("place_up_target_z_mm"),

            move_j_speed=self._get_float_param("move_j_speed"),
            move_j_acc=self._get_float_param("move_j_acc"),
            move_l_speed=self._get_float_param("move_l_speed"),
            move_l_acc=self._get_float_param("move_l_acc"),
            move_start_timeout_sec=self._get_float_param("move_start_timeout_sec"),

            joint_tol_deg=self._get_float_param("joint_tol_deg"),
            joint_stable_count_required=self._get_int_param("joint_stable_count_required"),
            joint_wait_timeout_sec=self._get_float_param("joint_wait_timeout_sec"),
            joint_polling_dt_sec=self._get_float_param("joint_polling_dt_sec"),

            tcp_pos_tol_mm=self._get_float_param("tcp_pos_tol_mm"),
            tcp_rot_tol_deg=self._get_float_param("tcp_rot_tol_deg"),
            tcp_stable_count_required=self._get_int_param("tcp_stable_count_required"),
            tcp_wait_timeout_sec=self._get_float_param("tcp_wait_timeout_sec"),
            tcp_polling_dt_sec=self._get_float_param("tcp_polling_dt_sec"),

            grasp_wait_sec=self._get_float_param("grasp_wait_sec"),
            release_wait_sec=self._get_float_param("release_wait_sec"),

            vision_wait_timeout_sec=self._get_float_param("vision_wait_timeout_sec"),
            vision_fixed_rx_deg=self._get_float_param("vision_fixed_rx_deg"),
            vision_fixed_rz_deg=self._get_float_param("vision_fixed_rz_deg"),
        )

        self.peg_targets_topic = self._get_str_param("peg_targets_topic")
        self.hole_targets_topic = self._get_str_param("hole_targets_topic")

        self.latest_peg_xyyaw: list[tuple[float, float, float]] = []
        self.latest_hole_xyyaw: list[tuple[float, float, float]] = []

        self.peg_msg_received = False
        self.hole_msg_received = False

        self.peg_sub = self.create_subscription(
            Float64MultiArray,
            self.peg_targets_topic,
            self.peg_targets_callback,
            10,
        )

        self.hole_sub = self.create_subscription(
            Float64MultiArray,
            self.hole_targets_topic,
            self.hole_targets_callback,
            10,
        )

        self.get_logger().info(f"Robot IP: {robot_ip}")
        self.get_logger().info(f"Gripper topic: {gripper_topic}")
        self.get_logger().info(f"Peg target topic: {self.peg_targets_topic}")
        self.get_logger().info(f"Hole target topic: {self.hole_targets_topic}")
        self.get_logger().info(f"Use simulation mode: {self.use_simulation_mode}")
        self.get_logger().info(
            f"Flat TCP RPY: "
            f"[{self.ctx.flat_tcp_rx_deg}, "
            f"{self.ctx.flat_tcp_ry_deg}, "
            f"{self.ctx.flat_tcp_rz_deg}]"
        )

    # ------------------------------------------------------------------
    # parameter helper
    # ------------------------------------------------------------------
    def _get_str_param(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _get_bool_param(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _get_int_param(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _get_float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _get_array_param(self, name: str, expected_len: int) -> np.ndarray:
        value = list(self.get_parameter(name).value)

        if len(value) != expected_len:
            raise ValueError(
                f"Parameter '{name}' must have length {expected_len}, but got {len(value)}"
            )

        return np.array(value, dtype=float)

    # ------------------------------------------------------------------
    # rbpodo helper
    # ------------------------------------------------------------------
    def set_operation_mode(self):
        rc = rb.ResponseCollector()

        if self.use_simulation_mode:
            self.robot.set_operation_mode(rc, rb.OperationMode.Simulation)
            mode_name = "Simulation"
        else:
            self.robot.set_operation_mode(rc, rb.OperationMode.Real)
            mode_name = "Real"

        rc.error().throw_if_not_empty()
        self.get_logger().info(f"Operation mode set to {mode_name}")

    def request_valid_state(
        self,
        retry: int = 30,
        wait_sec: float = 0.05,
    ):
        """
        robot_data.request_data()가 None을 반환할 수 있으므로 여러 번 재시도.
        """
        for _ in range(retry):
            state = self.robot_data.request_data()

            if state is not None:
                return state

            time.sleep(wait_sec)

        raise RuntimeError("robot_data.request_data() returned None repeatedly")

    def _angle_abs_error_deg(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        각도 오차를 -180~180 기준으로 정규화한 뒤 절댓값으로 반환.
        """
        current = np.array(current, dtype=float)
        target = np.array(target, dtype=float)

        return np.abs((current - target + 180.0) % 360.0 - 180.0)

    def get_current_joint(self) -> np.ndarray:
        """
        현재 로봇의 j1~j6 관절각을 읽어옴.

        반환:
            np.ndarray([j1, j2, j3, j4, j5, j6])

        단위:
            degree
        """
        state = self.request_valid_state()
        sdata = state.sdata

        if not hasattr(sdata, "jnt_ang"):
            raise RuntimeError(
                f"sdata has no attribute 'jnt_ang'. "
                f"Available fields: {dir(sdata)}"
            )

        current_joint = np.array(sdata.jnt_ang, dtype=float)

        if current_joint.shape[0] < 6:
            raise RuntimeError(f"Invalid jnt_ang length: {current_joint.shape[0]}")

        return current_joint[:6]

    def get_current_tcp_pose(self) -> np.ndarray:
        """
        현재 TCP pose를 CobotData 상태값에서 읽어옴.

        반환:
            np.ndarray([x, y, z, rx, ry, rz])

        단위:
            x, y, z = mm
            rx, ry, rz = deg

        주의:
            robot.get_tcp_info(rc)는 사용하지 않음.
            태블릿/컨트롤러에 print(get_tcp_info(), ...)가 뜨는 문제를 피하기 위해
            robot_data.request_data() 기반으로 읽음.
        """
        state = self.request_valid_state()
        sdata = state.sdata

        if hasattr(sdata, "tcp"):
            tcp_info = np.array(sdata.tcp, dtype=float)

        elif hasattr(sdata, "tcp_pos"):
            tcp_info = np.array(sdata.tcp_pos, dtype=float)

        elif hasattr(sdata, "cur_pos"):
            tcp_info = np.array(sdata.cur_pos, dtype=float)

        elif hasattr(sdata, "tcp_info"):
            tcp_info = np.array(sdata.tcp_info, dtype=float)

        else:
            raise RuntimeError(
                "TCP pose field를 찾지 못했습니다. "
                f"Available fields: {dir(sdata)}"
            )

        if tcp_info.shape[0] < 6:
            raise RuntimeError(f"Invalid TCP pose length: {tcp_info.shape[0]}")

        return tcp_info[:6]

    def force_flat_gripper_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        어떤 TCP pose가 들어와도 MoveL 실행 전 그리퍼 자세를 항상 수평 RPY로 강제한다.

        x, y, z는 유지하고,
        rx, ry, rz만 flat_tcp_* 값으로 덮어쓴다.
        """
        pose = np.array(pose, dtype=float).copy()

        pose[3] = self.ctx.flat_tcp_rx_deg
        pose[4] = self.ctx.flat_tcp_ry_deg
        pose[5] = self.ctx.flat_tcp_rz_deg

        return pose

    def wait_until_joint_reached(self, target_joint: np.ndarray) -> bool:
        """
        move_j 이후 현재 joint 값을 직접 읽어서 목표 joint에 도달했는지 판단.

        완료 조건:
            모든 관절 오차가 joint_tol_deg 이하인 상태가
            joint_stable_count_required번 연속 유지되면 완료로 판단.
        """
        target_joint = np.array(target_joint, dtype=float)

        start_time = time.monotonic()
        stable_count = 0

        while rclpy.ok():
            current_joint = self.get_current_joint()
            joint_error = self._angle_abs_error_deg(current_joint, target_joint)

            self.get_logger().info(
                f"[WAIT_JOINT] current = {current_joint}, "
                f"target = {target_joint}, "
                f"error = {joint_error}, "
                f"stable = {stable_count}/{self.ctx.joint_stable_count_required}"
            )

            if np.all(joint_error <= self.ctx.joint_tol_deg):
                stable_count += 1

                if stable_count >= self.ctx.joint_stable_count_required:
                    self.get_logger().info("[WAIT_JOINT] target reached")
                    return True
            else:
                stable_count = 0

            elapsed = time.monotonic() - start_time

            if elapsed > self.ctx.joint_wait_timeout_sec:
                self.get_logger().error(
                    f"[WAIT_JOINT] timeout after {elapsed:.2f} sec. "
                    f"current = {current_joint}, "
                    f"target = {target_joint}, "
                    f"error = {joint_error}"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(self.ctx.joint_polling_dt_sec)

        return False

    def wait_until_tcp_reached(self, target_pose: np.ndarray) -> bool:
        """
        move_l 이후 현재 TCP pose를 직접 읽어서 목표 TCP pose에 도달했는지 판단.

        완료 조건:
            position error <= tcp_pos_tol_mm
            rotation error <= tcp_rot_tol_deg
            위 조건이 tcp_stable_count_required번 연속 유지되면 완료.
        """
        target_pose = np.array(target_pose, dtype=float)

        start_time = time.monotonic()
        stable_count = 0

        while rclpy.ok():
            current_pose = self.get_current_tcp_pose()

            pos_error = np.abs(current_pose[:3] - target_pose[:3])
            rot_error = self._angle_abs_error_deg(current_pose[3:6], target_pose[3:6])

            self.get_logger().info(
                f"[WAIT_TCP] current = {current_pose}, "
                f"target = {target_pose}, "
                f"pos_error = {pos_error}, "
                f"rot_error = {rot_error}, "
                f"stable = {stable_count}/{self.ctx.tcp_stable_count_required}"
            )

            pos_ok = np.all(pos_error <= self.ctx.tcp_pos_tol_mm)
            rot_ok = np.all(rot_error <= self.ctx.tcp_rot_tol_deg)

            if pos_ok and rot_ok:
                stable_count += 1

                if stable_count >= self.ctx.tcp_stable_count_required:
                    self.get_logger().info("[WAIT_TCP] target reached")
                    return True
            else:
                stable_count = 0

            elapsed = time.monotonic() - start_time

            if elapsed > self.ctx.tcp_wait_timeout_sec:
                self.get_logger().error(
                    f"[WAIT_TCP] timeout after {elapsed:.2f} sec. "
                    f"current = {current_pose}, "
                    f"target = {target_pose}, "
                    f"pos_error = {pos_error}, "
                    f"rot_error = {rot_error}"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(self.ctx.tcp_polling_dt_sec)

        return False

    def move_j_and_wait(
        self,
        joint: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
    ):
        """
        rbpodo 기본 move_j 명령은 그대로 사용하고,
        완료 판단은 현재 joint polling 방식으로 수행.
        """
        if speed is None:
            speed = self.ctx.move_j_speed
        if acc is None:
            acc = self.ctx.move_j_acc

        rc = rb.ResponseCollector()

        joint = np.array(joint, dtype=float)

        self.get_logger().info(f"[MOVE_J] target joint = {joint}")

        self.robot.move_j(rc, joint, speed, acc)
        rc.error().throw_if_not_empty()

        reached = self.wait_until_joint_reached(joint)

        if not reached:
            raise RuntimeError(f"MoveJ target not reached: {joint}")

        rc.error().throw_if_not_empty()

        self.get_logger().info("[MOVE_J] finished by joint polling")

    def move_l_and_wait(
        self,
        pose: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
    ):
        """
        rbpodo Python API 기준 move_l 사용.

        move_l signature:
            move_l(rc, point, speed, acceleration, timeout=-1.0, return_on_err=False)

        point:
            [x, y, z, rx, ry, rz]

        단위:
            x, y, z = mm
            rx, ry, rz = deg

        중요:
            어떤 pose가 들어와도 rx, ry, rz는 flat_tcp_* 값으로 강제한다.
            즉, MoveL 구간에서는 그리퍼를 항상 땅과 평행한 자세로 유지한다.
        """
        if speed is None:
            speed = self.ctx.move_l_speed
        if acc is None:
            acc = self.ctx.move_l_acc

        rc = rb.ResponseCollector()

        raw_pose = np.array(pose, dtype=float)
        pose = self.force_flat_gripper_pose(raw_pose)

        self.get_logger().info(f"[MOVE_L] raw target pose = {raw_pose}")
        self.get_logger().info(f"[MOVE_L] flat target pose = {pose}")
        self.get_logger().info(f"[MOVE_L] speed = {speed}, acc = {acc}")

        try:
            current_pose = self.get_current_tcp_pose()
            self.get_logger().info(f"[MOVE_L DEBUG] current tcp = {current_pose}")
            self.get_logger().info(f"[MOVE_L DEBUG] target tcp  = {pose}")
            self.get_logger().info(f"[MOVE_L DEBUG] delta tcp   = {pose - current_pose}")
        except Exception as e:
            self.get_logger().warn(f"[MOVE_L DEBUG] current tcp read failed: {e}")

        # rb.ReferenceFrame.Base 넣지 않음.
        # rbpodo Python API의 5번째 인자는 ReferenceFrame이 아니라 timeout임.
        self.robot.move_l(rc, pose, speed, acc)
        rc.error().throw_if_not_empty()

        reached = self.wait_until_tcp_reached(pose)

        if not reached:
            raise RuntimeError(f"MoveL target not reached: {pose}")

        rc.error().throw_if_not_empty()

        self.get_logger().info("[MOVE_L] finished by TCP polling")

    def move_j1_only_and_wait(
        self,
        target_j1_deg: float,
        speed: float | None = None,
        acc: float | None = None,
    ):
        """
        현재 관절각을 저장한 뒤 j2~j6는 그대로 두고 j1만 target_j1_deg로 이동.
        """
        saved_joint = self.get_current_joint()

        target_joint = saved_joint.copy()
        target_joint[0] = float(target_j1_deg)

        self.get_logger().info(f"[MOVE_J1_ONLY] saved joint = {saved_joint}")
        self.get_logger().info(f"[MOVE_J1_ONLY] target joint = {target_joint}")

        self.move_j_and_wait(target_joint, speed=speed, acc=acc)

    # ------------------------------------------------------------------
    # gripper helper
    # ------------------------------------------------------------------
    def publish_grip(self, state_value: int):
        msg = Int32()
        msg.data = int(state_value)
        self.grip_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)

    def gripper_open(self):
        self.get_logger().info("[GRIP] OPEN")
        self.publish_grip(self.GRIP_OPEN)

    def gripper_close(self):
        self.get_logger().info("[GRIP] CLOSE")
        self.publish_grip(self.GRIP_CLOSE)

    def gripper_stop(self):
        self.get_logger().info("[GRIP] STOP")
        self.publish_grip(self.GRIP_STOP)

    # ------------------------------------------------------------------
    # vision callback
    # ------------------------------------------------------------------
    def _parse_xyyaw_msg(
        self,
        msg: Float64MultiArray,
        label: str,
    ) -> list[tuple[float, float, float]]:
        data = list(msg.data)

        if len(data) % 3 != 0:
            self.get_logger().warn(
                f"[VISION SUB] Invalid {label} data length: {len(data)}. "
                f"Expected [x1, y1, yaw1, x2, y2, yaw2, ...]"
            )
            return []

        targets: list[tuple[float, float, float]] = []

        for i in range(0, len(data), 3):
            targets.append(
                (
                    float(data[i]),
                    float(data[i + 1]),
                    float(data[i + 2]),
                )
            )

        return targets

    def peg_targets_callback(self, msg: Float64MultiArray):
        self.latest_peg_xyyaw = self._parse_xyyaw_msg(msg, "peg")
        self.peg_msg_received = True

        self.get_logger().info(
            f"[VISION SUB] peg targets received: {len(self.latest_peg_xyyaw)}"
        )

    def hole_targets_callback(self, msg: Float64MultiArray):
        self.latest_hole_xyyaw = self._parse_xyyaw_msg(msg, "hole")
        self.hole_msg_received = True

        self.get_logger().info(
            f"[VISION SUB] hole targets received: {len(self.latest_hole_xyyaw)}"
        )

    def _wait_for_peg_msg(self) -> bool:
        self.peg_msg_received = False
        self.latest_peg_xyyaw = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self, timeout_sec=0.05)

            if self.peg_msg_received:
                return True

        return False

    def _wait_for_hole_msg(self) -> bool:
        self.hole_msg_received = False
        self.latest_hole_xyyaw = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self, timeout_sec=0.05)

            if self.hole_msg_received:
                return True

        return False

    def _xyyaw_to_tcp_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        """
        [x, y, yaw] -> [x, y, z, rx, ry, rz]

        그리퍼를 항상 땅과 평행하게 유지하기 위해
        vision yaw는 TCP 자세에 사용하지 않는다.

        x, y:
            vision에서 받은 목표 위치

        yaw:
            현재는 사용하지 않음.
            필요하면 나중에 gripper yaw 정렬용으로 별도 로직에서 사용.

        z:
            이후 state에서 덮어씀.
        """
        return np.array(
            [
                x,
                y,
                0.0,
                self.ctx.flat_tcp_rx_deg,
                self.ctx.flat_tcp_ry_deg,
                self.ctx.flat_tcp_rz_deg,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # vision inspect
    # ------------------------------------------------------------------
    def inspect_pegs(self) -> list[np.ndarray]:
        self.get_logger().info("[VISION] Waiting for peg targets...")

        if not self._wait_for_peg_msg():
            self.get_logger().warn("[VISION] Peg target wait timeout")
            return []

        peg_candidates = [
            self._xyyaw_to_tcp_pose(x, y, yaw)
            for x, y, yaw in self.latest_peg_xyyaw
        ]

        self.get_logger().info(f"[VISION] detected peg count = {len(peg_candidates)}")
        return peg_candidates

    def inspect_holes(self) -> list[np.ndarray]:
        self.get_logger().info("[VISION] Waiting for hole targets...")

        if not self._wait_for_hole_msg():
            self.get_logger().warn("[VISION] Hole target wait timeout")
            return []

        hole_candidates = [
            self._xyyaw_to_tcp_pose(x, y, yaw)
            for x, y, yaw in self.latest_hole_xyyaw
        ]

        self.get_logger().info(f"[VISION] detected hole count = {len(hole_candidates)}")
        return hole_candidates

    # ------------------------------------------------------------------
    # selection helper
    # ------------------------------------------------------------------
    def select_next_peg(self):
        if len(self.ctx.peg_targets) == 0:
            self.ctx.current_peg_index = -1
            self.ctx.current_peg_pick_pose = None
            return

        self.ctx.current_peg_index = 0
        self.ctx.current_peg_pick_pose = self.ctx.peg_targets[0]
        self.get_logger().info(f"[SELECT] current peg index = {self.ctx.current_peg_index}")

    def select_next_hole(self):
        if len(self.ctx.hole_targets) == 0:
            self.ctx.current_hole_index = -1
            self.ctx.current_hole_place_pose = None
            return

        self.ctx.current_hole_index = 0
        self.ctx.current_hole_place_pose = self.ctx.hole_targets[0]
        self.get_logger().info(f"[SELECT] current hole index = {self.ctx.current_hole_index}")

    def consume_current_task(self):
        if self.ctx.current_peg_index >= 0:
            if len(self.ctx.peg_targets) > self.ctx.current_peg_index:
                del self.ctx.peg_targets[self.ctx.current_peg_index]

        if self.ctx.current_hole_index >= 0:
            if len(self.ctx.hole_targets) > self.ctx.current_hole_index:
                del self.ctx.hole_targets[self.ctx.current_hole_index]

        self.ctx.current_peg_index = -1
        self.ctx.current_hole_index = -1
        self.ctx.current_peg_pick_pose = None
        self.ctx.current_hole_place_pose = None

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def step(self):
        self.get_logger().info(f"[STATE] {self.state.name}")

        if self.state == TaskState.IDLE_HOME:
            self.move_j_and_wait(self.ctx.home_joint)
            self.gripper_open()
            time.sleep(0.5)
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE:
            self.move_j_and_wait(self.ctx.peg_camera_joint)
            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID:
            # 현재 j1~j6 저장 후 j1만 home_joint[0]으로 변경
            self.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.move_j_and_wait(self.ctx.peg_camera_joint)
            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.INSPECT_PEGS:
            self.ctx.peg_targets = self.inspect_pegs()

            if len(self.ctx.peg_targets) == 0:
                self.get_logger().info("[INFO] No peg remaining")
                self.state = TaskState.RETURN_HOME
            else:
                self.select_next_peg()
                self.state = TaskState.MOVE_TO_TARGET_PEG

        elif self.state == TaskState.MOVE_TO_TARGET_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.ctx.current_peg_pick_pose.copy()
            target_pose[2] = (
                self.ctx.pick_down_target_z_mm
                + self.ctx.pick_approach_offset_z_mm
            )

            self.move_l_and_wait(target_pose)
            self.state = TaskState.DESCEND_TO_PEG

        elif self.state == TaskState.DESCEND_TO_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.ctx.current_peg_pick_pose.copy()
            target_pose[2] = self.ctx.pick_down_target_z_mm

            self.move_l_and_wait(target_pose)
            self.state = TaskState.GRASP_PEG

        elif self.state == TaskState.GRASP_PEG:
            self.gripper_close()
            time.sleep(self.ctx.grasp_wait_sec)
            self.state = TaskState.LIFT_WITH_PEG

        elif self.state == TaskState.LIFT_WITH_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.ctx.current_peg_pick_pose.copy()
            target_pose[2] = self.ctx.pick_up_target_z_mm

            self.move_l_and_wait(target_pose)
            self.state = TaskState.MOVE_TO_HOLE_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_HOLE_CAMERA_POSE:
            # 현재 j1~j6 저장 후 j1만 home_joint[0]으로 변경
            self.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.move_j_and_wait(self.ctx.hole_camera_joint)
            self.state = TaskState.INSPECT_HOLES

        elif self.state == TaskState.INSPECT_HOLES:
            self.ctx.hole_targets = self.inspect_holes()

            if len(self.ctx.hole_targets) == 0:
                raise RuntimeError("No available hole detected")

            self.select_next_hole()
            self.state = TaskState.MOVE_TO_TARGET_HOLE

        elif self.state == TaskState.MOVE_TO_TARGET_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_approach_target_z_mm

            self.move_l_and_wait(target_pose)
            self.state = TaskState.DESCEND_TO_HOLE

        elif self.state == TaskState.DESCEND_TO_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_down_target_z_mm

            self.move_l_and_wait(target_pose)
            self.state = TaskState.RELEASE_PEG

        elif self.state == TaskState.RELEASE_PEG:
            self.gripper_open()
            time.sleep(self.ctx.release_wait_sec)
            self.state = TaskState.LIFT_FROM_HOLE

        elif self.state == TaskState.LIFT_FROM_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_up_target_z_mm

            self.move_l_and_wait(target_pose)
            self.state = TaskState.CHECK_REMAINING_TASK

        elif self.state == TaskState.CHECK_REMAINING_TASK:
            self.consume_current_task()
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID

        elif self.state == TaskState.RETURN_HOME:
            self.move_j_and_wait(self.ctx.home_joint)
            self.state = TaskState.DONE

        elif self.state == TaskState.DONE:
            self.get_logger().info("[DONE] Task completed")

        elif self.state == TaskState.ERROR:
            self.get_logger().error("[ERROR] Task stopped")

        else:
            raise RuntimeError(f"Unhandled state: {self.state}")

    def run(self):
        try:
            self.set_operation_mode()

            while rclpy.ok() and self.state not in (TaskState.DONE, TaskState.ERROR):
                self.step()
                rclpy.spin_once(self, timeout_sec=0.01)

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            self.state = TaskState.ERROR

        finally:
            try:
                self.gripper_stop()
            except Exception:
                pass


def main():
    rclpy.init()
    node = PegInHoleController()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Exit")


if __name__ == "__main__":
    main()