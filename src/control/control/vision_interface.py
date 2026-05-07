import time

import numpy as np
import rclpy
from std_msgs.msg import Float64MultiArray

try:
    from .task_types import TaskContext, VisionTarget
except ImportError:  # direct script/debug execution support
    from task_types import TaskContext, VisionTarget


class VisionInterface:
    """
    비전 촬영 trigger publish, peg/hole target subscribe, [x,y,yaw,id] 파싱 담당.
    """

    VALID_OBJECT_IDS = (0, 1, 2)
    OBJECT_ID_NAME = {
        0: "cylinder",
        1: "rectangle",
        2: "cross",
    }

    # ------------------------------------------------------------------
    # hole/place 전용 yaw 추가 보정값
    # ------------------------------------------------------------------
    # peg를 잡을 때 yaw는 건드리지 않고,
    # hole에 놓을 때 rz만 추가로 보정하고 싶으면 여기 값을 수정한다.
    #
    # 예:
    #   id=0 원통     놓을 때 +5도 필요하면 0: 5.0
    #   id=1 사각형   놓을 때 -3도 필요하면 1: -3.0
    #   id=2 십자가   놓을 때 +10도 필요하면 2: 10.0
    #
    # 현재는 추가 보정 없음.
    HOLE_YAW_OFFSET_BY_ID_DEG = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
    }

    def __init__(
        self,
        node,
        ctx: TaskContext,
        robot_motion,
        peg_targets_topic: str,
        hole_targets_topic: str,
        trigger_peg_topic: str,
        trigger_hole_topic: str,
        camera_settle_sec: float,
    ):
        self.node = node
        self.ctx = ctx
        self.robot_motion = robot_motion

        self.peg_targets_topic = peg_targets_topic
        self.hole_targets_topic = hole_targets_topic
        self.trigger_peg_topic = trigger_peg_topic
        self.trigger_hole_topic = trigger_hole_topic
        self.camera_settle_sec = float(camera_settle_sec)

        self.latest_peg_xyyawid: list[tuple[float, float, float, int]] = []
        self.latest_hole_xyyawid: list[tuple[float, float, float, int]] = []

        self.peg_msg_received = False
        self.hole_msg_received = False

        self.trigger_peg_pub = self.node.create_publisher(
            Float64MultiArray,
            self.trigger_peg_topic,
            10,
        )

        self.trigger_hole_pub = self.node.create_publisher(
            Float64MultiArray,
            self.trigger_hole_topic,
            10,
        )

        self.peg_sub = self.node.create_subscription(
            Float64MultiArray,
            self.peg_targets_topic,
            self.peg_targets_callback,
            10,
        )

        self.hole_sub = self.node.create_subscription(
            Float64MultiArray,
            self.hole_targets_topic,
            self.hole_targets_callback,
            10,
        )

    def shape_name(self, object_id: int | None) -> str:
        if object_id is None:
            return "none"
        return self.OBJECT_ID_NAME.get(object_id, "unknown")

    # ------------------------------------------------------------------
    # vision trigger helper
    # ------------------------------------------------------------------
    def publish_ee_pose_trigger(self, pub, label: str):
        """
        사진 촬영 trigger용으로 현재 TCP pose를 publish한다.

        publish data:
            [x, y, z, rx, ry, rz]

        단위:
            x, y, z = mm
            rx, ry, rz = deg
        """
        if self.camera_settle_sec > 0.0:
            time.sleep(self.camera_settle_sec)

        ee_pose = self.robot_motion.get_current_tcp_pose()

        msg = Float64MultiArray()
        msg.data = [float(v) for v in ee_pose[:6]]

        pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.05)

        self.node.get_logger().info(
            f"[VISION TRIGGER] {label} trigger published. "
            f"ee_pose = {ee_pose}"
        )

    def trigger_peg_capture(self):
        self.publish_ee_pose_trigger(self.trigger_peg_pub, "PEG")

    def trigger_hole_capture(self):
        self.publish_ee_pose_trigger(self.trigger_hole_pub, "HOLE")

    # ------------------------------------------------------------------
    # vision callback / parsing
    # ------------------------------------------------------------------
    def _parse_xyyawid_msg(
        self,
        msg: Float64MultiArray,
        label: str,
    ) -> list[tuple[float, float, float, int]]:
        data = list(msg.data)

        if len(data) % 4 != 0:
            self.node.get_logger().warn(
                f"[VISION SUB] Invalid {label} data length: {len(data)}. "
                f"Expected [x1, y1, yaw1, id1, x2, y2, yaw2, id2, ...]"
            )
            return []

        targets: list[tuple[float, float, float, int]] = []

        for i in range(0, len(data), 4):
            x = float(data[i])
            y = float(data[i + 1])
            yaw = float(data[i + 2])
            object_id = int(round(float(data[i + 3])))

            if object_id not in self.VALID_OBJECT_IDS:
                self.node.get_logger().warn(
                    f"[VISION SUB] Unknown {label} id: {object_id}. "
                    f"Expected 0=cylinder, 1=rectangle, 2=cross. "
                    f"This target will be ignored."
                )
                continue

            targets.append((x, y, yaw, object_id))

        return targets

    def peg_targets_callback(self, msg: Float64MultiArray):
        self.latest_peg_xyyawid = self._parse_xyyawid_msg(msg, "peg")
        self.peg_msg_received = True

        self.node.get_logger().info(
            f"[VISION SUB] peg targets received: {len(self.latest_peg_xyyawid)}"
        )

    def hole_targets_callback(self, msg: Float64MultiArray):
        self.latest_hole_xyyawid = self._parse_xyyawid_msg(msg, "hole")
        self.hole_msg_received = True

        self.node.get_logger().info(
            f"[VISION SUB] hole targets received: {len(self.latest_hole_xyyawid)}"
        )

    def _wait_for_peg_msg(self, reset: bool = True) -> bool:
        if reset:
            self.peg_msg_received = False
            self.latest_peg_xyyawid = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.peg_msg_received:
                return True

        return False

    def _wait_for_hole_msg(self, reset: bool = True) -> bool:
        if reset:
            self.hole_msg_received = False
            self.latest_hole_xyyawid = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.hole_msg_received:
                return True

        return False

    def _xyyaw_to_tcp_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        object_id: int,
        target_kind: str = "peg",
    ) -> np.ndarray:
        """
        [x, y, yaw, id] -> [x, y, z, rx, ry, rz]

        yaw는 object_id에 따라 보정한 뒤 rz에 넣는다.
        z는 이후 상태머신에서 작업 높이에 맞게 덮어쓴다.

        target_kind:
            "peg"  : peg를 잡을 때 사용하는 yaw 보정
            "hole" : hole에 놓을 때 사용하는 yaw 보정
        """
        if target_kind == "hole":
            corrected_yaw = self._correct_hole_yaw_by_object_id(yaw, object_id)
        else:
            corrected_yaw = self._correct_yaw_by_object_id(yaw, object_id)

        return np.array(
            [
                x,
                y,
                0.0,
                self.ctx.flat_tcp_rx_deg,
                self.ctx.flat_tcp_ry_deg,
                corrected_yaw,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # vision inspect
    # ------------------------------------------------------------------
    def inspect_pegs(self) -> list[VisionTarget]:
        self.node.get_logger().info("[VISION] Trigger peg capture and wait for peg targets...")

        # 중요:
        # trigger 직후 빠르게 결과가 들어올 수 있으므로,
        # wait 함수 내부에서 다시 reset하지 않도록 여기서 먼저 초기화한다.
        self.peg_msg_received = False
        self.latest_peg_xyyawid = []

        self.trigger_peg_capture()

        if not self._wait_for_peg_msg(reset=False):
            self.node.get_logger().warn("[VISION] Peg target wait timeout")
            return []

        peg_candidates = [
            VisionTarget(
                pose=self._xyyaw_to_tcp_pose(
                    x,
                    y,
                    yaw,
                    object_id,
                    target_kind="peg",
                ),
                object_id=object_id,
            )
            for x, y, yaw, object_id in self.latest_peg_xyyawid
        ]

        for i, target in enumerate(peg_candidates):
            self.node.get_logger().info(
                f"[VISION] peg[{i}] id = {target.object_id} "
                f"({self.shape_name(target.object_id)}), "
                f"pose = {target.pose}"
            )

        self.node.get_logger().info(f"[VISION] detected peg count = {len(peg_candidates)}")
        return peg_candidates

    def inspect_holes(self) -> list[VisionTarget]:
        self.node.get_logger().info("[VISION] Trigger hole capture and wait for hole targets...")

        # 중요:
        # trigger 직후 빠르게 결과가 들어올 수 있으므로,
        # wait 함수 내부에서 다시 reset하지 않도록 여기서 먼저 초기화한다.
        self.hole_msg_received = False
        self.latest_hole_xyyawid = []

        self.trigger_hole_capture()

        if not self._wait_for_hole_msg(reset=False):
            self.node.get_logger().warn("[VISION] Hole target wait timeout")
            return []

        hole_candidates = [
            VisionTarget(
                pose=self._xyyaw_to_tcp_pose(
                    x,
                    y,
                    yaw,
                    object_id,
                    target_kind="hole",
                ),
                object_id=object_id,
            )
            for x, y, yaw, object_id in self.latest_hole_xyyawid
        ]

        for i, target in enumerate(hole_candidates):
            self.node.get_logger().info(
                f"[VISION] hole[{i}] id = {target.object_id} "
                f"({self.shape_name(target.object_id)}), "
                f"pose = {target.pose}"
            )

        self.node.get_logger().info(f"[VISION] detected hole count = {len(hole_candidates)}")
        return hole_candidates

    def _normalize_yaw_deg(self, yaw: float) -> float:
        """
        yaw를 -180 ~ 180 deg 범위로 정규화한다.
        """
        return (float(yaw) + 180.0) % 360.0 - 180.0

    def _correct_yaw_by_object_id(self, yaw: float, object_id: int) -> float:
        """
        vision에서 받은 yaw를 object id에 따라 보정한다.

        id 의미:
            0: 원
            1: 사각형
            2: 십자가

        현재 peg 잡을 때 이 보정값이 잘 맞는 상태이므로,
        이 함수는 peg 기준 보정식으로 유지한다.
        """
        yaw = float(yaw)

        if object_id == 0:
            corrected_yaw = 135.0

        elif object_id == 1:
            corrected_yaw = (yaw % 90.0)+ 45.0

        elif object_id == 2:
            corrected_yaw = (yaw % 90.0) + 45.0

        else:
            self.node.get_logger().warn(
                f"[VISION] Unknown object id for yaw correction: {object_id}. "
                f"Use raw yaw = {yaw}"
            )
            corrected_yaw = yaw

        self.node.get_logger().info(
            f"[VISION] peg yaw correction: "
            f"id={object_id}, raw_yaw={yaw:.3f}, corrected_yaw={corrected_yaw:.3f}"
        )

        return corrected_yaw

    def _correct_hole_yaw_by_object_id(self, yaw: float, object_id: int) -> float:
        """
        hole/place 전용 yaw 보정.

        기본 보정은 peg와 동일하게 적용한 뒤,
        hole에 놓을 때만 추가 offset을 더한다.

        조정은 HOLE_YAW_OFFSET_BY_ID_DEG 값만 바꾸면 된다.
        """
        base_yaw = self._correct_yaw_by_object_id(yaw, object_id)
        hole_offset = float(self.HOLE_YAW_OFFSET_BY_ID_DEG.get(object_id, 0.0))

        corrected_yaw = base_yaw + hole_offset
        corrected_yaw = self._normalize_yaw_deg(corrected_yaw)

        self.node.get_logger().info(
            f"[VISION] hole yaw correction: "
            f"id={object_id}, raw_yaw={yaw:.3f}, "
            f"base_yaw={base_yaw:.3f}, "
            f"hole_offset={hole_offset:.3f}, "
            f"hole_corrected_yaw={corrected_yaw:.3f}"
        )

        return corrected_yaw