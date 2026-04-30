import json
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    # RB5 rx ry rz 를 회전행렬로 변환
    # R = Rz @ Ry @ Rx
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [0.0,         1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0,         0.0,        1.0],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def rb5_pose_array_to_T_mm(data):
    # RB5 현재 TCP pose를 base_T_ee로 변환
    # data = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    if len(data) < 6:
        raise ValueError(
            f"trigger data must have 6 values: "
            f"[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg], got {len(data)}"
        )

    x, y, z, rx, ry, rz = map(float, data[:6])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]

    return T


def object_json_to_cam_T_obj_mm(obj):
    """
    /object_poses JSON object -> camera_T_object

    비전부 position은 m단위 -> mm로 변환.
    orientation은 PCA axis 벡터 3개를 column으로 쌓아 R_cam_obj 구성.
    """
    pos = obj["position"]
    ori = obj["orientation"]

    p_cam_obj_mm = np.array([
        pos["x"],
        pos["y"],
        pos["z"],
    ], dtype=np.float64) * 1000.0

    R_cam_obj = np.column_stack([
        np.array(ori["axis_x"], dtype=np.float64),
        np.array(ori["axis_y"], dtype=np.float64),
        np.array(ori["axis_z"], dtype=np.float64),
    ])

    # PCA axis 수치 보정
    u, _, vt = np.linalg.svd(R_cam_obj)
    R_cam_obj = u @ vt

    # 오른손 좌표계 보정
    if np.linalg.det(R_cam_obj) < 0:
        R_cam_obj[:, 2] *= -1.0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam_obj
    T[:3, 3] = p_cam_obj_mm

    return T


def yaw_deg_from_R_base_obj(R):
    """
    base/world 좌표계 기준 yaw 추출.

    현재 yaw는 point cloud PCA 1번 축(axis_x)이
    base XY 평면에서 향하는 방향을 의미

    yaw = atan2(R[1, 0], R[0, 0])
    """
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)

    # -180 ~ 180 정규화
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

    return float(yaw_deg)


class ObjectPoseTransformNode(Node):
    def __init__(self):
        super().__init__("object_pose_transform_node")

        self.declare_parameter("handeye_result_path", "")
        self.declare_parameter("min_confidence", 0.3)


        # sub 토픽
        self.declare_parameter("object_topic", "/object_poses")
        self.declare_parameter("trigger_topic", "/manipulation/trigger")

        # pub 토픽
        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")

        # 물체 class에 따른 hole/peg 분류
        self.declare_parameter("peg_classes", ["cross", "cylinder", "hole"])
        self.declare_parameter("hole_classes", ["cross_hole", "cylinder_hole", "hole_hole"])

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        
        object_topic = self.get_parameter("object_topic").value
        trigger_topic = self.get_parameter("trigger_topic").value
        peg_output_topic = self.get_parameter("peg_output_topic").value
        hole_output_topic = self.get_parameter("hole_output_topic").value

        self.peg_classes = set(self.get_parameter("peg_classes").value)
        self.hole_classes = set(self.get_parameter("hole_classes").value)

        self.ee_T_cam = self.load_handeye_result_as_mm()

        self.latest_objects = []

        # sub
        self.object_sub = self.create_subscription(
            String,
            object_topic,
            self.object_callback,
            10,
        )
        self.trigger_sub = self.create_subscription(
            Float64MultiArray,
            trigger_topic,
            self.trigger_callback,
            10,
        )

        # pub
        self.peg_pub = self.create_publisher(
            Float64MultiArray,
            peg_output_topic,
            10,
        )

        self.hole_pub = self.create_publisher(
            Float64MultiArray,
            hole_output_topic,
            10,
        )

        self.get_logger().info(f"subscribe object topic: {object_topic} [std_msgs/String]")
        self.get_logger().info(f"subscribe trigger topic: {trigger_topic} "f"[Float64MultiArray: x,y,z,rx,ry,rz]")
        self.get_logger().info(f"publish peg topic: {peg_output_topic} [Float64MultiArray]")
        self.get_logger().info(f"publish hole topic: {hole_output_topic} [Float64MultiArray]")
        self.get_logger().info(f"peg_classes: {sorted(list(self.peg_classes))}")
        self.get_logger().info(f"hole_classes: {sorted(list(self.hole_classes))}")

    def load_handeye_result_as_mm(self):
        param_path = self.get_parameter("handeye_result_path").value

        if param_path:
            result_path = Path(param_path)
        else:
            result_path = (
                Path(get_package_share_directory("calib"))
                / "config"
                / "handeye_capture_rs"
                / "handeye_result.json"
            )

        if not result_path.exists():
            raise FileNotFoundError(f"handeye_result.json not found: {result_path}")

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "ee_T_cam" not in data:
            raise KeyError("handeye_result.json must contain key: ee_T_cam")

        ee_T_cam = np.array(data["ee_T_cam"], dtype=np.float64)

        if ee_T_cam.shape != (4, 4):
            raise ValueError("ee_T_cam must be 4x4 matrix")

        # hand-eye calibration 결과 translation은 meter 기준이므로 mm로 변환
        ee_T_cam_mm = ee_T_cam.copy()
        ee_T_cam_mm[:3, 3] *= 1000.0

        self.get_logger().info(f"loaded handeye result: {result_path}")
        self.get_logger().info(f"ee_T_cam [translation=mm]:\n{ee_T_cam_mm}")

        return ee_T_cam_mm

    def object_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_objects = data.get("objects", [])
        except Exception as e:
            self.latest_objects = []
            self.get_logger().warn(f"failed to parse /object_poses JSON: {e}")

    def transform_one_object_to_xyyaw(self, obj, base_T_ee):
        """
        object 1개를 base 기준 [x_mm, y_mm, yaw_deg]로 변환.
        """
        cam_T_obj = object_json_to_cam_T_obj_mm(obj)
        base_T_obj = base_T_ee @ self.ee_T_cam @ cam_T_obj

        p = base_T_obj[:3, 3]
        R = base_T_obj[:3, :3]

        x_mm = float(p[0])
        y_mm = float(p[1])
        yaw_deg = yaw_deg_from_R_base_obj(R)

        return x_mm, y_mm, yaw_deg

    def trigger_callback(self, trigger_msg):
        if not self.latest_objects:
            self.get_logger().warn("trigger received, but latest_objects is empty")
            return

        try:
            base_T_ee = rb5_pose_array_to_T_mm(trigger_msg.data)

            peg_data = []
            hole_data = []

            for obj in self.latest_objects:
                cls = obj.get("class", "")
                conf = float(obj.get("confidence", 0.0))

                # conf 일정 이하는 무시
                if conf < self.min_confidence:
                    continue

                # peg/hole 둘 다 해당 안 되는 class는 무시
                if cls not in self.peg_classes and cls not in self.hole_classes:
                    continue

                x_mm, y_mm, yaw_deg = self.transform_one_object_to_xyyaw(
                    obj=obj,
                    base_T_ee=base_T_ee,
                )

                item = [x_mm, y_mm, yaw_deg]

                if cls in self.peg_classes:
                    peg_data.extend(item)

                if cls in self.hole_classes:
                    hole_data.extend(item)

            if peg_data:
                peg_msg = Float64MultiArray()
                peg_msg.data = peg_data
                self.peg_pub.publish(peg_msg)

                self.get_logger().info(
                    f"published /vision/peg_targets | "
                    f"num_objects={len(peg_data) // 3}, data={peg_data}"
                )
            else:
                self.get_logger().warn("trigger received, but no valid peg target after filtering")

            if hole_data:
                hole_msg = Float64MultiArray()
                hole_msg.data = hole_data
                self.hole_pub.publish(hole_msg)

                self.get_logger().info(
                    f"published /vision/hole_targets | "
                    f"num_objects={len(hole_data) // 3}, data={hole_data}"
                )
            else:
                self.get_logger().warn("trigger received, but no valid hole target after filtering")

        except Exception as e:
            self.get_logger().error(f"failed to transform object poses: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = ObjectPoseTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()