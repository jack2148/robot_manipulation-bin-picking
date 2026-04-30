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

    move_start_timeout_sec: float = 0.5

    # ===== gripper 파라미터 =====
    grasp_wait_sec: float = 1.0
    release_wait_sec: float = 1.0

    # ===== vision 파라미터 =====
    vision_wait_timeout_sec: float = 2.0
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
                ("use_simulation_mode", True),
                ("gripper_topic", "grip_state"),

                ("grip_open", 1),
                ("grip_close", 0),
                ("grip_stop", 2),

                ("home_joint", [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0]),
                ("peg_camera_joint", [10.87, 2.78, 79.15, 8.07, 90.0, 34.16]),
                ("hole_camera_joint", [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16]),

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

        self.robot = rb.Cobot(robot_ip)
        self.grip_pub = self.create_publisher(Int32, gripper_topic, 10)

        self.state = TaskState.IDLE_HOME

        self.ctx = TaskContext(
            home_joint=self._get_array_param("home_joint", 6),
            peg_camera_joint=self._get_array_param("peg_camera_joint", 6),
            hole_camera_joint=self._get_array_param("hole_camera_joint", 6),

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
    def set_simulation_mode(self):
        rc = rb.ResponseCollector()
        self.robot.set_operation_mode(rc, rb.OperationMode.Real)
        rc.error().throw_if_not_empty()
        self.get_logger().info("Operation mode set to Simulation")

    def move_j_and_wait(
        self,
        joint: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
    ):
        if speed is None:
            speed = self.ctx.move_j_speed
        if acc is None:
            acc = self.ctx.move_j_acc

        rc = rb.ResponseCollector()

        joint = np.array(joint, dtype=float)

        self.get_logger().info(f"[MOVE_J] target joint = {joint}")

        self.robot.move_j(rc, joint, speed, acc)
        rc.error().throw_if_not_empty()

        if self.robot.wait_for_move_started(rc, self.ctx.move_start_timeout_sec).is_success():
            self.robot.wait_for_move_finished(rc)

        rc.error().throw_if_not_empty()

    def move_l_and_wait(
        self,
        pose: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
    ):
        if speed is None:
            speed = self.ctx.move_l_speed
        if acc is None:
            acc = self.ctx.move_l_acc

        rc = rb.ResponseCollector()

        pose = np.array(pose, dtype=float)

        self.get_logger().info(f"[MOVE_L] target pose = {pose}")

        self.robot.move_l(rc, pose, speed, acc, rb.ReferenceFrame.Base)
        rc.error().throw_if_not_empty()

        if self.robot.wait_for_move_started(rc, self.ctx.move_start_timeout_sec).is_success():
            self.robot.wait_for_move_finished(rc)

        rc.error().throw_if_not_empty()

    def get_current_joint(self) -> np.ndarray:
        """
        현재 로봇의 j1~j6 관절각을 읽어옴.

        rbpodo 사용 방식:
            sdata = robot.get_sdata()
            current_joints = sdata.jnt_ang

        반환:
            np.ndarray([j1, j2, j3, j4, j5, j6])

        단위:
            degree
        """
        sdata = self.robot.get_sdata()

        if sdata is None:
            raise RuntimeError("robot.get_sdata() returned None")

        if not hasattr(sdata, "jnt_ang"):
            raise RuntimeError(
                f"robot.get_sdata() result has no attribute 'jnt_ang'. "
                f"Available fields: {dir(sdata)}"
            )

        current_joint = np.array(sdata.jnt_ang, dtype=float)

        if current_joint.shape[0] < 6:
            raise RuntimeError(f"Invalid jnt_ang length: {current_joint.shape[0]}")

        current_joint = current_joint[:6]
        self.get_logger().info(f"[GET_SDATA] current joint = {current_joint}")

        return current_joint

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

        현재 자세 기준:
            rx = 90 고정
            ry = yaw
            rz = 0 고정

        z는 이후 state에서 덮어씀.
        """
        return np.array(
            [
                x,
                y,
                0.0,
                self.ctx.vision_fixed_rx_deg,
                yaw,
                self.ctx.vision_fixed_rz_deg,
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
            target_pose[2] = self.ctx.pick_down_target_z_mm + self.ctx.pick_approach_offset_z_mm

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
            if self.use_simulation_mode:
                self.set_simulation_mode()
            else:
                self.get_logger().info("Simulation mode setup skipped")

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