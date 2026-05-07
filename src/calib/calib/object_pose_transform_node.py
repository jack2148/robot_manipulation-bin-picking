import json
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


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
        self.declare_parameter("insert_topic", "/insert_poses")
        self.declare_parameter("detect_mode_topic", "/detect_mode")

        self.declare_parameter("peg_trigger_topic", "/manipulation/trigger_peg")
        self.declare_parameter("hole_trigger_topic", "/manipulation/trigger_hole")

        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")

        self.declare_parameter("exclude_dist_mm", 30.0)

        self.class_to_id = {
            "cylinder": 0,
            "cylinder_insert": 0,

            "hole": 1,
            "hole_insert": 1,

            "cross": 2,
            "cross_insert": 2,
        }

        self.min_confidence = float(self.get_parameter("min_confidence").value)

        object_topic = self.get_parameter("object_topic").value
        insert_topic = self.get_parameter("insert_topic").value
        detect_mode_topic = self.get_parameter("detect_mode_topic").value

        peg_trigger_topic = self.get_parameter("peg_trigger_topic").value
        hole_trigger_topic = self.get_parameter("hole_trigger_topic").value

        peg_output_topic = self.get_parameter("peg_output_topic").value
        hole_output_topic = self.get_parameter("hole_output_topic").value

        self.ee_T_cam = self.load_handeye_result_as_mm()

        self.latest_objects = []
        self.latest_inserts = []

        self.pending_task = None
        self.pending_trigger_msg = None
        self.pending_object_targets = None

        self.detect_mode_pub = self.create_publisher(
            String,
            detect_mode_topic,
            10,
        )

        self.object_sub = self.create_subscription(
            String,
            object_topic,
            self.object_callback,
            10,
        )

        self.insert_sub = self.create_subscription(
            String,
            insert_topic,
            self.insert_callback,
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

        self.get_logger().info(f"subscribe object topic: {object_topic}")
        self.get_logger().info(f"subscribe insert topic: {insert_topic}")
        self.get_logger().info(f"publish detect mode topic: {detect_mode_topic}")
        self.get_logger().info(f"subscribe peg trigger topic: {peg_trigger_topic}")
        self.get_logger().info(f"subscribe hole trigger topic: {hole_trigger_topic}")
        self.get_logger().info(f"publish peg topic: {peg_output_topic}")
        self.get_logger().info(f"publish hole topic: {hole_output_topic}")
        self.get_logger().info(
            f"exclude_dist_mm: {float(self.get_parameter('exclude_dist_mm').value):.1f}"
        )

    def publish_detect_mode(self, mode):
        msg = String()
        msg.data = mode
        self.detect_mode_pub.publish(msg)
        self.get_logger().info(f"request detect_mode: {mode}")

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
            return

        if self.pending_task == "peg_wait_object":
            self.process_peg_after_object_update()

        elif self.pending_task == "hole_wait_object":
            self.process_hole_after_object_update()

    def insert_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_inserts = data.get("objects", [])
        except Exception as e:
            self.latest_inserts = []
            self.get_logger().warn(f"failed to parse /insert_poses JSON: {e}")
            return

        if self.pending_task == "hole_wait_insert":
            self.process_hole_after_insert_update()

    def transform_one_object_to_xyyaw(self, obj, base_T_ee):
        cam_T_obj = object_json_to_cam_T_obj_mm(obj)
        base_T_obj = base_T_ee @ self.ee_T_cam @ cam_T_obj

        p = base_T_obj[:3, 3]
        R = base_T_obj[:3, :3]

        x_mm = float(p[0])
        y_mm = float(p[1])
        yaw_deg = yaw_deg_from_R_base_obj(R)

        return x_mm, y_mm, yaw_deg

    def object_to_target_dict(self, obj, base_T_ee):
        cls = obj.get("class", "")
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            return None

        if cls not in self.class_to_id:
            return None

        x_mm, y_mm, yaw_deg = self.transform_one_object_to_xyyaw(
            obj=obj,
            base_T_ee=base_T_ee,
        )

        obj_id = self.class_to_id[cls]

        return {
            "class": cls,
            "id": int(obj_id),
            "x": float(x_mm),
            "y": float(y_mm),
            "yaw": float(yaw_deg),
            "confidence": float(conf),
        }

    def make_targets_from_objects(self, objects, base_T_ee):
        targets = []

        for obj in objects:
            target = self.object_to_target_dict(obj, base_T_ee)
            if target is not None:
                targets.append(target)

        return targets

    def targets_to_msg_data(self, targets):
        data = []

        for t in targets:
            data.extend([
                float(t["x"]),
                float(t["y"]),
                float(t["yaw"]),
                float(t["id"]),
            ])

        return data

    def publish_repeated(self, publisher, msg, count=10):
        for _ in range(count):
            publisher.publish(msg)

    def reset_pending(self):
        self.pending_task = None
        self.pending_trigger_msg = None
        self.pending_object_targets = None

    def peg_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(
                f"ignore peg trigger because pending_task is running: {self.pending_task}"
            )
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_task = "peg_wait_object"

        self.publish_detect_mode("object")

        self.get_logger().info(
            "peg trigger received → switch detect_mode=object → wait fresh /object_poses"
        )

    def process_peg_after_object_update(self):
        try:
            if self.pending_trigger_msg is None:
                self.reset_pending()
                return

            if not self.latest_objects:
                self.get_logger().warn(
                    "fresh /object_poses received, but objects is empty"
                )
                self.reset_pending()
                return

            base_T_ee = rb5_pose_array_to_T_mm(self.pending_trigger_msg.data)

            peg_targets = self.make_targets_from_objects(
                self.latest_objects,
                base_T_ee,
            )

            if not peg_targets:
                self.get_logger().warn(
                    "peg trigger received, but no valid target from fresh /object_poses"
                )
                self.reset_pending()
                return

            msg = Float64MultiArray()
            msg.data = self.targets_to_msg_data(peg_targets)

            self.publish_repeated(self.peg_pub, msg, count=10)

            self.get_logger().info(
                f"published /vision/peg_targets | "
                f"num_objects={len(peg_targets)}, data={msg.data}"
            )

            self.reset_pending()

        except Exception as e:
            self.get_logger().error(f"failed to publish peg targets: {e}")
            self.reset_pending()

    def hole_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(
                f"ignore hole trigger because pending_task is running: {self.pending_task}"
            )
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_object_targets = None
        self.pending_task = "hole_wait_object"

        self.publish_detect_mode("object")

        self.get_logger().info(
            "hole trigger received → switch detect_mode=object → wait fresh /object_poses"
        )

    def process_hole_after_object_update(self):
        try:
            if self.pending_trigger_msg is None:
                self.reset_pending()
                return

            base_T_ee = rb5_pose_array_to_T_mm(self.pending_trigger_msg.data)

            self.pending_object_targets = self.make_targets_from_objects(
                self.latest_objects,
                base_T_ee,
            )

            self.pending_task = "hole_wait_insert"
            self.publish_detect_mode("insert")

            self.get_logger().info(
                f"fresh /object_poses captured for hole | "
                f"num_objects={len(self.pending_object_targets)} "
                f"→ switch detect_mode=insert → wait fresh /insert_poses"
            )

        except Exception as e:
            self.get_logger().error(f"failed during hole object capture step: {e}")
            self.reset_pending()

    def process_hole_after_insert_update(self):
        try:
            if self.pending_trigger_msg is None:
                self.reset_pending()
                return

            if not self.latest_inserts:
                self.get_logger().warn(
                    "fresh /insert_poses received, but inserts is empty"
                )
                self.reset_pending()
                return

            base_T_ee = rb5_pose_array_to_T_mm(self.pending_trigger_msg.data)
            exclude_dist_mm = float(self.get_parameter("exclude_dist_mm").value)

            object_targets = self.pending_object_targets
            if object_targets is None:
                object_targets = []

            insert_targets = self.make_targets_from_objects(
                self.latest_inserts,
                base_T_ee,
            )

            valid_insert_targets = []

            for ins in insert_targets:
                should_exclude = False

                for obj in object_targets:
                    if ins["id"] != obj["id"]:
                        continue

                    dist_mm = float(
                        np.hypot(
                            ins["x"] - obj["x"],
                            ins["y"] - obj["y"],
                        )
                    )

                    if dist_mm < exclude_dist_mm:
                        should_exclude = True
                        self.get_logger().info(
                            f"exclude insert target | "
                            f"insert_class={ins['class']}, "
                            f"object_class={obj['class']}, "
                            f"id={ins['id']}, dist={dist_mm:.1f} mm"
                        )
                        break

                if not should_exclude:
                    valid_insert_targets.append(ins)

            if not valid_insert_targets:
                self.get_logger().warn(
                    "hole trigger received, but no valid insert target after filtering"
                )
                self.reset_pending()
                return

            msg = Float64MultiArray()
            msg.data = self.targets_to_msg_data(valid_insert_targets)

            self.publish_repeated(self.hole_pub, msg, count=10)

            self.get_logger().info(
                f"published /vision/hole_targets | "
                f"num_objects={len(valid_insert_targets)}, data={msg.data}"
            )

            self.reset_pending()

        except Exception as e:
            self.get_logger().error(f"failed to publish hole targets: {e}")
            self.reset_pending()


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