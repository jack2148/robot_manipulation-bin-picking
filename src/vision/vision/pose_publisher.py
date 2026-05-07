import json
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(get_package_share_directory('vision')) / 'weights'

MODELS = {
    "object": {
        "path": str(WEIGHTS_DIR / "best.pt"),
        "names": ["cross", "cylinder", "hole"],
        "colors": {"cross": (0, 220, 0), "cylinder": (0, 140, 255), "hole": (220, 0, 220)},
        "topic": "/object_poses",
    },
    "insert": {
        "path": str(WEIGHTS_DIR / "insert_best.pt"),
        "names": ["cross_insert", "cylinder_insert", "hole_insert"],
        "colors": {"cross_insert": (0, 220, 0), "cylinder_insert": (0, 140, 255), "hole_insert": (220, 0, 220)},
        "topic": "/insert_poses",
    },
}

CONF_THRESH = 0.4


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def get_point_cloud(depth_image, mask_img, intrinsics, depth_scale):
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy

    rows, cols = np.where(mask_img > 0)
    z_vals = depth_image[rows, cols].astype(float) * depth_scale
    valid = z_vals > 0
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]

    if len(z_vals) < 10:
        return None

    z_med = np.median(z_vals)
    valid = np.abs(z_vals - z_med) < 0.05
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]

    if len(z_vals) < 10:
        return None

    x = (cols - ppx) * z_vals / fx
    y = (rows - ppy) * z_vals / fy
    return np.stack([x, y, z_vals], axis=1)


def estimate_pose(points):
    centroid = np.median(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    return centroid, axes


# ---------------------------------------------------------------------------
# ROS2 노드
# ---------------------------------------------------------------------------

class PosePublisher(Node):
    def __init__(self):
        super().__init__("pose_publisher")

        # 두 모델 미리 로드
        self.models = {}
        for mode, cfg in MODELS.items():
            if not Path(cfg["path"]).exists():
                self.get_logger().warn(f"모델 없음: {cfg['path']}")
                continue
            self.models[mode] = YOLO(cfg["path"])
            self.get_logger().info(f"[{mode}] 모델 로드 완료: {cfg['path']}")

        self.mode = "object"

        # 퍼블리셔 (두 토픽 모두 준비)
        self.publishers_ = {
            mode: self.create_publisher(String, cfg["topic"], 10)
            for mode, cfg in MODELS.items()
        }

        # 모드 전환 구독 (/detect_mode 토픽으로 "object" 또는 "insert" 발행)
        self.create_subscription(String, "/detect_mode", self._mode_callback, 10)

        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        self.intrinsics = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        ).get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

        self.get_logger().info(
            f"color intrinsics: fx={self.intrinsics.fx:.2f}, "
            f"fy={self.intrinsics.fy:.2f}, "
            f"ppx={self.intrinsics.ppx:.2f}, "
            f"ppy={self.intrinsics.ppy:.2f}, "
            f"depth_scale={self.depth_scale:.6f}"
        )

        self.timer = self.create_timer(0.1, self._timer_callback)  # 10Hz
        self.get_logger().info("PosePublisher ready. 현재 모드: object")

    def _mode_callback(self, msg):
        mode = msg.data.strip().lower()
        if mode not in MODELS:
            self.get_logger().warn(f"알 수 없는 모드: '{mode}' (object / insert 중 선택)")
            return
        if mode not in self.models:
            self.get_logger().warn(f"모델이 로드되지 않아 전환 불가: {mode}")
            return
        self.mode = mode
        self.get_logger().info(f"모드 전환 → {mode}")

    def _timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        cfg = MODELS[self.mode]
        model = self.models[self.mode]
        results = model(color_image, conf=CONF_THRESH, verbose=False)[0]

        objects = []
        display = color_image.copy()

        cv2.putText(display, f"MODE: {self.mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        n_det = len(results.boxes) if results.boxes is not None else 0
        self.get_logger().info(f"detections: {n_det}", throttle_duration_sec=1.0)

        if results.masks is not None:
            for i, mask_xy in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i])
                class_name = results.names[cls_id]
                confidence = float(results.boxes.conf[i])
                color = cfg["colors"].get(class_name, (255, 255, 255))

                mask_img = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_img, [mask_xy.astype(np.int32)], 255)

                points = get_point_cloud(depth_image, mask_img, self.intrinsics, self.depth_scale)
                if points is None:
                    continue

                centroid, axes = estimate_pose(points)

                objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "position": {
                        "x": round(float(centroid[0]), 4),
                        "y": round(float(centroid[1]), 4),
                        "z": round(float(centroid[2]), 4),
                    },
                    "orientation": {
                        "axis_x": [round(float(v), 4) for v in axes[:, 0]],
                        "axis_y": [round(float(v), 4) for v in axes[:, 1]],
                        "axis_z": [round(float(v), 4) for v in axes[:, 2]],
                    },
                })

                overlay = display.copy()
                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                cx = int(centroid[0] * self.intrinsics.fx / centroid[2] + self.intrinsics.ppx)
                cy = int(centroid[1] * self.intrinsics.fy / centroid[2] + self.intrinsics.ppy)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                label = (f"{class_name} {confidence:.2f} | "
                         f"X:{centroid[0]:+.2f} Y:{centroid[1]:+.2f} Z:{centroid[2]:.2f}m")
                cv2.putText(display, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("PosePublisher - ESC to quit", display)
        cv2.waitKey(1)

        msg = String()
        msg.data = json.dumps({"mode": self.mode, "objects": objects})
        self.publishers_[self.mode].publish(msg)

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = PosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
