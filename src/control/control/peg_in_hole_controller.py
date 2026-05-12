import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node

try:
    from .task_types import TaskContext, TaskState
    from .robot_motion import RobotMotion
    from .gripper_interface import GripperInterface
    from .vision_interface import VisionInterface
except ImportError:  # direct script/debug execution support
    from task_types import TaskContext, TaskState
    from robot_motion import RobotMotion
    from gripper_interface import GripperInterface
    from vision_interface import VisionInterface


class PegInHoleController(Node):
    """
    상태머신만 담당하는 peg-in-hole 메인 컨트롤러.

    - 로봇 이동: RobotMotion
    - 그리퍼 publish: GripperInterface
    - 비전 trigger/subscribe/parsing: VisionInterface

    trigger 토픽 데이터:
        Float64MultiArray.data = [x, y, z, rx, ry, rz]

    vision 결과 토픽 데이터:
        Float64MultiArray.data = [x, y, yaw, id, x, y, yaw, id, ...]
        id: 0=원통, 1=직사각형, 2=십자가
    """

    def __init__(self):
        super().__init__("peg_in_hole_controller")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("robot_ip", "192.168.1.10"),
                ("use_simulation_mode", False),
                ("gripper_topic", "/grip_state"),

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

                # peg/hole 공통 접근 MoveL 속도
                ("approach_move_l_speed", 40.0),
                ("approach_move_l_acc", 80.0),

                # peg/hole 공통 하강 MoveL 속도
                ("descend_move_l_speed", 8.0),
                ("descend_move_l_acc", 40.0),

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
                ("trigger_peg_topic", "/manipulation/trigger_peg"),
                ("trigger_hole_topic", "/manipulation/trigger_hole"),
                ("camera_settle_sec", 0.5),

                ("vision_wait_timeout_sec", 2.0),
                ("vision_fixed_rx_deg", 90.0),
                ("vision_fixed_rz_deg", 0.0),

                # 방금 원위치에 되돌려놓은 peg를 한 번 제외할 때 쓰는 xy 거리 기준
                ("skip_once_xy_tol_mm", 20.0),
            ],
        )

        self.state = TaskState.IDLE_HOME
        self.use_simulation_mode = self._get_bool_param("use_simulation_mode")

        robot_ip = self._get_str_param("robot_ip")
        gripper_topic = self._get_str_param("gripper_topic")
        peg_targets_topic = self._get_str_param("peg_targets_topic")
        hole_targets_topic = self._get_str_param("hole_targets_topic")
        trigger_peg_topic = self._get_str_param("trigger_peg_topic")
        trigger_hole_topic = self._get_str_param("trigger_hole_topic")
        camera_settle_sec = self._get_float_param("camera_settle_sec")

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

            approach_move_l_speed=self._get_float_param("approach_move_l_speed"),
            approach_move_l_acc=self._get_float_param("approach_move_l_acc"),
            descend_move_l_speed=self._get_float_param("descend_move_l_speed"),
            descend_move_l_acc=self._get_float_param("descend_move_l_acc"),

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

            skip_once_xy_tol_mm=self._get_float_param("skip_once_xy_tol_mm"),
        )

        self.motion = RobotMotion(
            node=self,
            ctx=self.ctx,
            robot_ip=robot_ip,
            use_simulation_mode=self.use_simulation_mode,
        )

        self.gripper = GripperInterface(
            node=self,
            topic=gripper_topic,
            grip_open=self._get_int_param("grip_open"),
            grip_close=self._get_int_param("grip_close"),
            grip_stop=self._get_int_param("grip_stop"),
        )

        self.vision = VisionInterface(
            node=self,
            ctx=self.ctx,
            robot_motion=self.motion,
            peg_targets_topic=peg_targets_topic,
            hole_targets_topic=hole_targets_topic,
            trigger_peg_topic=trigger_peg_topic,
            trigger_hole_topic=trigger_hole_topic,
            camera_settle_sec=camera_settle_sec,
        )

        self.get_logger().info(f"Robot IP: {robot_ip}")
        self.get_logger().info(f"Gripper topic: {gripper_topic}")
        self.get_logger().info(f"Peg target topic: {peg_targets_topic}")
        self.get_logger().info(f"Hole target topic: {hole_targets_topic}")
        self.get_logger().info(f"Trigger peg topic: {trigger_peg_topic}")
        self.get_logger().info(f"Trigger hole topic: {trigger_hole_topic}")
        self.get_logger().info(f"Camera settle sec: {camera_settle_sec}")
        self.get_logger().info(f"Use simulation mode: {self.use_simulation_mode}")
        self.get_logger().info(
            f"Flat TCP RPY: "
            f"[{self.ctx.flat_tcp_rx_deg}, "
            f"{self.ctx.flat_tcp_ry_deg}, "
            f"{self.ctx.flat_tcp_rz_deg}]"
        )
        self.get_logger().info(
            f"Approach MoveL speed/acc: "
            f"{self.ctx.approach_move_l_speed}, "
            f"{self.ctx.approach_move_l_acc}"
        )
        self.get_logger().info(
            f"Descend MoveL speed/acc: "
            f"{self.ctx.descend_move_l_speed}, "
            f"{self.ctx.descend_move_l_acc}"
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
    # selection helper
    # ------------------------------------------------------------------
    def is_skip_once_target(self, peg) -> bool:
        """
        matching jig가 없어서 원래 위치에 다시 내려놓은 peg와
        같은 id의 peg는 다음 선택에서 딱 1번만 제외한다.

        위치 기준이 아니라 id 기준이다.
        예:
            방금 원 peg를 되돌려놓음
            → 다음 선택에서 object_id == 0인 peg는 모두 한 번 제외
        """
        if self.ctx.skip_once_pick_id is None:
            return False

        if peg.object_id == self.ctx.skip_once_pick_id:
            self.get_logger().info(
                f"[SKIP_ONCE] skip same id once. "
                f"id = {peg.object_id} "
                f"({self.vision.shape_name(peg.object_id)})"
            )
            return True

        return False

    def _set_current_peg(self, peg_index: int):
        selected_peg = self.ctx.peg_targets[peg_index]

        self.ctx.current_peg_index = peg_index
        self.ctx.current_peg_pick_pose = selected_peg.pose.copy()
        self.ctx.current_target_id = selected_peg.object_id

        self.get_logger().info(
            f"[SELECT] current peg index = {self.ctx.current_peg_index}, "
            f"id = {self.ctx.current_target_id} "
            f"({self.vision.shape_name(self.ctx.current_target_id)}), "
            f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
        )

    def select_next_peg(self):
        if len(self.ctx.peg_targets) == 0:
            self.ctx.current_peg_index = -1
            self.ctx.current_peg_pick_pose = None
            self.ctx.current_target_id = None
            return

        selected_index = None

        # remaining_jig_counts가 비어 있으면 새 사이클로 판단한다.
        # 이때는 기존처럼 첫 번째 peg를 아무거나 잡되,
        # 방금 원위치에 되돌려놓은 peg는 한 번만 제외한다.
        if len(self.ctx.remaining_jig_counts) == 0:
            self.get_logger().info(
                "[SELECT] No remaining jig information. Select first non-skipped peg."
            )

            for i, peg in enumerate(self.ctx.peg_targets):
                if self.is_skip_once_target(peg):
                    continue

                selected_index = i
                break

            if selected_index is None:
                self.get_logger().warn(
                    "[SELECT] Only skipped peg is available. "
                    "Clear skip_once and select first peg."
                )
                self.clear_skip_once()
                selected_index = 0

            self._set_current_peg(selected_index)
            self.clear_skip_once()
            return

        # 남은 jig 타입을 알고 있으면,
        # 그 jig 타입에 대응되는 peg만 선택한다.
        # 이때도 방금 되돌려놓은 peg는 한 번만 제외한다.
        for i, peg in enumerate(self.ctx.peg_targets):
            if self.ctx.remaining_jig_counts.get(peg.object_id, 0) <= 0:
                continue

            if self.is_skip_once_target(peg):
                continue

            selected_index = i
            break

        if selected_index is not None:
            peg = self.ctx.peg_targets[selected_index]
            self.get_logger().info(
                f"[SELECT] Select peg matched with remaining jig id = {peg.object_id} "
                f"({self.vision.shape_name(peg.object_id)})"
            )
            self._set_current_peg(selected_index)
            self.clear_skip_once()
            return

        # skip_once 때문에 선택할 peg가 없을 수도 있다.
        # 이 경우 한 번 제외 조건을 해제하고 다시 남은 jig 타입 기준으로 선택한다.
        if self.ctx.skip_once_pick_pose is not None:
            self.get_logger().warn(
                "[SELECT] No candidate after skip_once. "
                "Clear skip_once and retry selection."
            )
            self.clear_skip_once()

            for i, peg in enumerate(self.ctx.peg_targets):
                if self.ctx.remaining_jig_counts.get(peg.object_id, 0) > 0:
                    selected_index = i
                    break

            if selected_index is not None:
                self._set_current_peg(selected_index)
                return

        # 남은 jig 타입에 해당하는 peg가 없으면,
        # 현재 jig 세트로는 더 할 수 없다고 보고 새 사이클로 넘어간다.
        # 이후 첫 번째 peg를 잡고 hole 촬영 결과에 따라 다시 판단한다.
        self.get_logger().warn(
            "[SELECT] No peg matched with remaining jig ids. "
            "Clear remaining_jig_counts and select first peg."
        )
        self.ctx.remaining_jig_counts.clear()
        self._set_current_peg(0)

    def _update_remaining_jig_counts_if_needed(self):
        """
        remaining_jig_counts가 비어 있을 때만 현재 촬영된 hole 목록으로 초기화한다.

        의미:
            - 비어 있음: 새 jig 세트 시작
            - 비어 있지 않음: 이전에 본 jig 세트에서 아직 안 쓴 jig가 남아 있음
        """
        if len(self.ctx.remaining_jig_counts) != 0:
            return

        self.ctx.remaining_jig_counts = Counter(
            target.object_id for target in self.ctx.hole_targets
        )

        self.get_logger().info(
            f"[JIG] Initialize remaining_jig_counts = "
            f"{dict(self.ctx.remaining_jig_counts)}"
        )

    def select_next_hole(self) -> bool:
        if self.ctx.current_target_id is None:
            raise RuntimeError("No selected peg id. Cannot select matching hole.")

        if len(self.ctx.hole_targets) == 0:
            self.ctx.current_hole_index = -1
            self.ctx.current_hole_place_pose = None
            self.get_logger().warn("[SELECT] No available hole detected")
            return False

        self.ctx.current_hole_index = -1
        self.ctx.current_hole_place_pose = None

        for i, hole in enumerate(self.ctx.hole_targets):
            if hole.object_id == self.ctx.current_target_id:
                self.ctx.current_hole_index = i
                self.ctx.current_hole_place_pose = hole.pose.copy()

                self.get_logger().info(
                    f"[SELECT] matched hole index = {i}, "
                    f"id = {self.ctx.current_target_id} "
                    f"({self.vision.shape_name(self.ctx.current_target_id)})"
                )
                return True

        available_ids = [target.object_id for target in self.ctx.hole_targets]

        self.get_logger().warn(
            f"[SELECT] No matching hole found for selected peg id = "
            f"{self.ctx.current_target_id} "
            f"({self.vision.shape_name(self.ctx.current_target_id)}). "
            f"Available hole ids = {available_ids}"
        )

        if self.ctx.current_target_id in self.ctx.remaining_jig_counts:
            del self.ctx.remaining_jig_counts[self.ctx.current_target_id]

            self.get_logger().warn(
                f"[JIG] Remove unavailable jig id = {self.ctx.current_target_id} "
                f"from remaining_jig_counts. "
                f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
            )

        return False

    def mark_current_jig_used(self):
        if self.ctx.current_target_id is None:
            return

        object_id = self.ctx.current_target_id

        if self.ctx.remaining_jig_counts.get(object_id, 0) > 0:
            self.ctx.remaining_jig_counts[object_id] -= 1

            if self.ctx.remaining_jig_counts[object_id] <= 0:
                del self.ctx.remaining_jig_counts[object_id]

        self.get_logger().info(
            f"[JIG] Used jig id = {object_id} "
            f"({self.vision.shape_name(object_id)}), "
            f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
        )

    def save_last_pick_pose(self):
        if self.ctx.current_peg_pick_pose is None:
            raise RuntimeError("No selected peg target")

        up_pose = self.ctx.current_peg_pick_pose.copy()
        up_pose[2] = self.ctx.pick_up_target_z_mm

        down_pose = self.ctx.current_peg_pick_pose.copy()
        down_pose[2] = self.ctx.pick_down_target_z_mm

        self.ctx.last_pick_up_pose = up_pose.copy()
        self.ctx.last_pick_down_pose = down_pose.copy()
        self.ctx.last_pick_id = self.ctx.current_target_id

        self.get_logger().info(
            f"[RECOVERY SAVE] last_pick_id = {self.ctx.last_pick_id}, "
            f"up_pose = {self.ctx.last_pick_up_pose}, "
            f"down_pose = {self.ctx.last_pick_down_pose}"
        )

    def set_skip_once_from_last_pick(self):
        """
        방금 matching jig가 없어서 되돌려놓은 peg의 id를
        다음 peg 선택에서 딱 1번 제외하기 위해 저장한다.
        """
        self.ctx.skip_once_pick_id = self.ctx.last_pick_id

        self.get_logger().info(
            f"[SKIP_ONCE] set skip id once = {self.ctx.skip_once_pick_id} "
            f"({self.vision.shape_name(self.ctx.skip_once_pick_id)})"
        )

    def clear_skip_once(self):
        self.ctx.skip_once_pick_pose = None
        self.ctx.skip_once_pick_id = None

    def clear_current_task(self):
        self.ctx.current_peg_index = -1
        self.ctx.current_hole_index = -1
        self.ctx.current_peg_pick_pose = None
        self.ctx.current_hole_place_pose = None
        self.ctx.current_target_id = None

    def clear_recovery_pose(self):
        self.ctx.last_pick_up_pose = None
        self.ctx.last_pick_down_pose = None
        self.ctx.last_pick_id = None

    def consume_current_task(self):
        if self.ctx.current_peg_index >= 0:
            if len(self.ctx.peg_targets) > self.ctx.current_peg_index:
                del self.ctx.peg_targets[self.ctx.current_peg_index]

        if self.ctx.current_hole_index >= 0:
            if len(self.ctx.hole_targets) > self.ctx.current_hole_index:
                del self.ctx.hole_targets[self.ctx.current_hole_index]

        self.clear_current_task()
        self.clear_recovery_pose()

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def step(self):
        self.get_logger().info(f"[STATE] {self.state.name}")

        if self.state == TaskState.IDLE_HOME:
            self.motion.move_j_and_wait(self.ctx.home_joint)
            self.gripper.open()
            time.sleep(0.5)
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE:
            self.motion.move_j_and_wait(self.ctx.peg_camera_joint)
            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID:
            # 현재 j1~j6 저장 후 j1만 home_joint[0]으로 변경
            self.motion.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.motion.move_j_and_wait(self.ctx.peg_camera_joint)
            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.INSPECT_PEGS:
            self.ctx.peg_targets = self.vision.inspect_pegs()

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

            # matching hole이 없을 때 원래 위치로 되돌리기 위해 저장한다.
            # 복구 시에는 pick_up_target_z_mm 높이로 먼저 돌아온 뒤,
            # move_l로 pick_down_target_z_mm까지 내려간다.
            self.save_last_pick_pose()

            # peg 잡기 전, 조금 높은 접근 위치로 이동
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
            )
            self.state = TaskState.DESCEND_TO_PEG

        elif self.state == TaskState.DESCEND_TO_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.ctx.current_peg_pick_pose.copy()
            target_pose[2] = self.ctx.pick_down_target_z_mm

            # 그리퍼 닫기 직전, peg 잡는 높이로 하강
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
            )
            self.state = TaskState.GRASP_PEG

        elif self.state == TaskState.GRASP_PEG:
            self.gripper.close()
            time.sleep(self.ctx.grasp_wait_sec)
            self.state = TaskState.LIFT_WITH_PEG

        elif self.state == TaskState.LIFT_WITH_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.ctx.current_peg_pick_pose.copy()
            target_pose[2] = self.ctx.pick_up_target_z_mm

            self.motion.move_l_and_wait(target_pose)
            self.state = TaskState.MOVE_TO_HOLE_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_HOLE_CAMERA_POSE:
            # 현재 j1~j6 저장 후 j1만 home_joint[0]으로 변경
            self.motion.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.motion.move_j_and_wait(self.ctx.hole_camera_joint)
            self.state = TaskState.INSPECT_HOLES

        elif self.state == TaskState.INSPECT_HOLES:
            self.ctx.hole_targets = self.vision.inspect_holes()

            self._update_remaining_jig_counts_if_needed()

            matched = self.select_next_hole()

            if matched:
                self.state = TaskState.MOVE_TO_TARGET_HOLE
            else:
                self.get_logger().warn(
                    "[RECOVERY] Matching hole is not found. "
                    "Return peg to original pick place through J1 midpoint."
                )
                self.state = TaskState.RETURN_TO_PICK_VIA_MID

        elif self.state == TaskState.MOVE_TO_TARGET_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_approach_target_z_mm

            # hole 위 접근 위치로 이동
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
            )
            self.state = TaskState.DESCEND_TO_HOLE

        elif self.state == TaskState.DESCEND_TO_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_down_target_z_mm

            # peg를 hole에 넣기 직전, 삽입 높이로 하강
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
            )
            self.state = TaskState.RELEASE_PEG

        elif self.state == TaskState.RELEASE_PEG:
            self.gripper.open()
            time.sleep(self.ctx.release_wait_sec)

            # 정상 삽입 성공으로 보고 현재 타입의 jig 사용 count를 감소시킨다.
            self.mark_current_jig_used()

            self.state = TaskState.LIFT_FROM_HOLE

        elif self.state == TaskState.LIFT_FROM_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()
            target_pose[2] = self.ctx.place_up_target_z_mm

            self.motion.move_l_and_wait(target_pose)
            self.state = TaskState.CHECK_REMAINING_TASK

        elif self.state == TaskState.CHECK_REMAINING_TASK:
            if len(self.ctx.remaining_jig_counts) == 0:
                self.get_logger().info(
                    "[INFO] All remembered jigs are used. "
                    "Start new cycle from first peg."
                )

            self.consume_current_task()
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID

        elif self.state == TaskState.RETURN_TO_PICK_VIA_MID:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            # hole/camera 쪽에서 바로 cartesian으로 돌아가지 않고,
            # 현재 joint를 기준으로 J1만 home 쪽으로 먼저 경유한다.
            self.motion.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.state = TaskState.RETURN_TO_PICK_UP_POSE

        elif self.state == TaskState.RETURN_TO_PICK_UP_POSE:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] return to original pick up pose = "
                f"{self.ctx.last_pick_up_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_up_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
            )
            self.state = TaskState.DESCEND_TO_PICK_PLACE

        elif self.state == TaskState.DESCEND_TO_PICK_PLACE:
            if self.ctx.last_pick_down_pose is None:
                raise RuntimeError("No saved pick down pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] descend to original pick pose = "
                f"{self.ctx.last_pick_down_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_down_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
            )
            self.state = TaskState.RELEASE_BACK_TO_PICK_PLACE

        elif self.state == TaskState.RELEASE_BACK_TO_PICK_PLACE:
            self.get_logger().info("[RECOVERY] open gripper and return peg")
            self.gripper.open()
            time.sleep(self.ctx.release_wait_sec)

            # 방금 원위치에 되돌려놓은 peg는 다음 선택에서 한 번만 제외한다.
            self.set_skip_once_from_last_pick()

            self.state = TaskState.LIFT_FROM_PICK_PLACE

        elif self.state == TaskState.LIFT_FROM_PICK_PLACE:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] lift after returning peg = "
                f"{self.ctx.last_pick_up_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_up_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
            )

            self.clear_current_task()
            self.clear_recovery_pose()

            # 방금 되돌려놓은 뒤에는 home/j1 경유 없이 바로 peg 사진 자세로 이동한다.
            # remaining_jig_counts는 유지한다.
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE

        elif self.state == TaskState.RETURN_HOME:
            self.motion.move_j_and_wait(self.ctx.home_joint)
            self.state = TaskState.DONE

        elif self.state == TaskState.DONE:
            self.get_logger().info("[DONE] Task completed")

        elif self.state == TaskState.ERROR:
            self.get_logger().error("[ERROR] Task stopped")

        else:
            raise RuntimeError(f"Unhandled state: {self.state}")

    def run(self):
        try:
            self.motion.set_operation_mode()

            while rclpy.ok() and self.state not in (TaskState.DONE, TaskState.ERROR):
                self.step()
                rclpy.spin_once(self, timeout_sec=0.01)

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            self.state = TaskState.ERROR

        finally:
            try:
                self.gripper.stop()
            except Exception:
                pass