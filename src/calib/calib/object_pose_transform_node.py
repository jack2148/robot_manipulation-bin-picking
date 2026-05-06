import json
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


# 오일러 tcp를 회전행렬로 변환
def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
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

# RB5 tcp를 T로 변환
def rb5_pose_array_to_T_mm(data):
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

# vision부에서 받은 정보(m단위)를 mm단위 정보로 변환
def object_json_to_cam_T_obj_mm(obj):
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

    u, _, vt = np.linalg.svd(R_cam_obj)
    R_cam_obj = u @ vt

    if np.linalg.det(R_cam_obj) < 0:
        R_cam_obj[:, 2] *= -1.0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam_obj
    T[:3, 3] = p_cam_obj_mm

    return T

# pca를 바탕으로 yaw값을 추정
def yaw_deg_from_R_base_obj(R):
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
    return float(yaw_deg)


class ObjectPoseTransformNode(Node):
    def __init__(self):
        super().__init__("object_pose_transform_node")

        self.declare_parameter("handeye_result_path", "")
        self.declare_parameter("min_confidence", 0.3)

        self.declare_parameter("object_topic", "/object_poses")

        self.declare_parameter("peg_trigger_topic", "/manipulation/trigger_peg")
        self.declare_parameter("hole_trigger_topic", "/manipulation/trigger_hole")

        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")

        self.declare_parameter("peg_classes", ["cylinder", "square", "cross"])
        self.declare_parameter("hole_classes", ["cylinder_hole", "square_hole", "cross_hole"])

        self.class_to_id = {
            # peg zone
            "cylinder": 0,
            "square": 1,
            "cross": 2,

            # hole zone
            "cylinder_hole": 0,
            "square_hole": 1,
            "cross_hole": 2,
        }

        self.min_confidence = float(self.get_parameter("min_confidence").value)

        object_topic = self.get_parameter("object_topic").value
        peg_trigger_topic = self.get_parameter("peg_trigger_topic").value
        hole_trigger_topic = self.get_parameter("hole_trigger_topic").value
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
        self.peg_trigger_sub = self.create_subscription(
            Float64MultiArray,
            peg_trigger_topic,
            self.peg_trigger_callback,
            10,
        )
        self.hole_trigger_sub = self.create_subscription(
            Float64MultiArray,
            hole_trigger_topic,
            self.hole_trigger_callback,
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
        self.get_logger().info(
            f"subscribe peg trigger topic: {peg_trigger_topic} "
            f"[Float64MultiArray: x,y,z,rx,ry,rz]"
        )
        self.get_logger().info(
            f"subscribe hole trigger topic: {hole_trigger_topic} "
            f"[Float64MultiArray: x,y,z,rx,ry,rz]"
        )
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
        cam_T_obj = object_json_to_cam_T_obj_mm(obj)
        base_T_obj = base_T_ee @ self.ee_T_cam @ cam_T_obj

        p = base_T_obj[:3, 3]
        R = base_T_obj[:3, :3]

        x_mm = float(p[0])
        y_mm = float(p[1])
        yaw_deg = yaw_deg_from_R_base_obj(R)

        return x_mm, y_mm, yaw_deg

    def build_target_data(self, trigger_msg, target_classes):
        if not self.latest_objects:
            self.get_logger().warn("trigger received, but latest_objects is empty")
            return []

        base_T_ee = rb5_pose_array_to_T_mm(trigger_msg.data)
        target_data = []

        for obj in self.latest_objects:
            cls = obj.get("class", "")
            conf = float(obj.get("confidence", 0.0))

            if conf < self.min_confidence:
                continue
            if cls not in target_classes:
                continue

            x_mm, y_mm, yaw_deg = self.transform_one_object_to_xyyaw(
                obj=obj,
                base_T_ee=base_T_ee,
            )

            obj_id = self.class_to_id.get(cls, -1)
            target_data.extend([
                float(x_mm),
                float(y_mm),
                float(yaw_deg),
                float(obj_id),
            ])

        return target_data

    # 1번 보내면 씹힐수도 있으니깬 귀찮으니 여러번 보내기
    def publish_repeated(self, publisher, msg, count=5):
        for _ in range(count):
            publisher.publish(msg)

    # peg zone 트리거를 받은 경우
    def peg_trigger_callback(self, trigger_msg):
        try:
            peg_data = self.build_target_data(
                trigger_msg=trigger_msg,
                target_classes=self.peg_classes,
            )

            if not peg_data:
                self.get_logger().warn("peg trigger received, but no valid peg target")
                return

            msg = Float64MultiArray()
            msg.data = peg_data
            self.publish_repeated(self.peg_pub, msg, count=10)

            self.get_logger().info(
                f"published /vision/peg_targets | "
                f"num_objects={len(peg_data) // 4}, data={peg_data}"
            )

        except Exception as e:
            self.get_logger().error(f"failed to publish peg targets: {e}")

    # hole zone 트리거를 받은 경우
    def hole_trigger_callback(self, trigger_msg):
        try:
            hole_data = self.build_target_data(
                trigger_msg=trigger_msg,
                target_classes=self.hole_classes,
            )

            if not hole_data:
                self.get_logger().warn("hole trigger received, but no valid hole target")
                return

            msg = Float64MultiArray()
            msg.data = hole_data
            self.publish_repeated(self.hole_pub, msg, count=10)

            self.get_logger().info(
                f"published /vision/hole_targets | "
                f"num_objects={len(hole_data) // 4}, data={hole_data}"
            )

        except Exception as e:
            self.get_logger().error(f"failed to publish hole targets: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPoseTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()