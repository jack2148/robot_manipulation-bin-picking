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
# 설정 - object + insert 통합 모델 (6 클래스)
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(get_package_share_directory('vision')) / 'weights'
MODEL_PATH = str(WEIGHTS_DIR / 'ob_in_best.pt')

# 클래스 ID → 이름 (학습 시 순서와 동일)
CLASS_NAMES = {
    0: 'cylinder',
    1: 'hole',
    2: 'cross',
    3: 'cross_insert',
    4: 'cylinder_insert',
    5: 'hole_insert',
}

CLASS_COLORS = {
    'cylinder':        (0, 140, 255),
    'hole':            (220, 0, 220),
    'cross':           (0, 220, 0),
    'cross_insert':    (0, 180, 0),
    'cylinder_insert': (0, 100, 200),
    'hole_insert':     (180, 0, 180),
}

# object / insert 분류 (토픽 분리 발행용)
OBJECT_CLASSES = {'cylinder', 'hole', 'cross'}
INSERT_CLASSES = {'cross_insert', 'cylinder_insert', 'hole_insert'}

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

class PosePublisherObIn(Node):
    def __init__(self):
        super().__init__('pose_publisher_ob_in')

        # 통합 모델 로드
        if not Path(MODEL_PATH).exists():
            self.get_logger().error(f'모델 없음: {MODEL_PATH}')
            raise FileNotFoundError(MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'모델 로드 완료: {MODEL_PATH}')

        # 퍼블리셔
        # - /ob_in_poses  : object + insert 전체 (통합)
        # - /object_poses : object 클래스만
        # - /insert_poses : insert 클래스만
        self.pub_all = self.create_publisher(String, '/ob_in_poses', 10)
        self.pub_object = self.create_publisher(String, '/object_poses', 10)
        self.pub_insert = self.create_publisher(String, '/insert_poses', 10)

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
            f'color intrinsics: fx={self.intrinsics.fx:.2f}, '
            f'fy={self.intrinsics.fy:.2f}, '
            f'ppx={self.intrinsics.ppx:.2f}, '
            f'ppy={self.intrinsics.ppy:.2f}, '
            f'depth_scale={self.depth_scale:.6f}'
        )

        self.timer = self.create_timer(0.1, self._timer_callback)  # 10Hz
        self.get_logger().info('PosePublisherObIn ready. object + insert 통합 검출 중')

    def _timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = self.model(color_image, conf=CONF_THRESH, verbose=False)[0]

        all_objects, object_list, insert_list = [], [], []
        display = color_image.copy()

        n_det = len(results.boxes) if results.boxes is not None else 0
        self.get_logger().info(f'detections: {n_det}', throttle_duration_sec=1.0)

        if results.masks is not None:
            for i, mask_xy in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i])
                class_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
                confidence = float(results.boxes.conf[i])
                color = CLASS_COLORS.get(class_name, (255, 255, 255))

                mask_img = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_img, [mask_xy.astype(np.int32)], 255)

                points = get_point_cloud(
                    depth_image, mask_img, self.intrinsics, self.depth_scale
                )
                if points is None:
                    continue

                centroid, axes = estimate_pose(points)

                det = {
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'position': {
                        'x': round(float(centroid[0]), 4),
                        'y': round(float(centroid[1]), 4),
                        'z': round(float(centroid[2]), 4),
                    },
                    'orientation': {
                        'axis_x': [round(float(v), 4) for v in axes[:, 0]],
                        'axis_y': [round(float(v), 4) for v in axes[:, 1]],
                        'axis_z': [round(float(v), 4) for v in axes[:, 2]],
                    },
                }
                all_objects.append(det)

                if class_name in OBJECT_CLASSES:
                    object_list.append(det)
                elif class_name in INSERT_CLASSES:
                    insert_list.append(det)

                # 시각화
                overlay = display.copy()
                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                cx = int(centroid[0] * self.intrinsics.fx / centroid[2] + self.intrinsics.ppx)
                cy = int(centroid[1] * self.intrinsics.fy / centroid[2] + self.intrinsics.ppy)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                label = (f'{class_name} {confidence:.2f} | '
                         f'X:{centroid[0]:+.2f} Y:{centroid[1]:+.2f} Z:{centroid[2]:.2f}m')
                cv2.putText(display, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('PosePublisher ob_in - ESC to quit', display)
        cv2.waitKey(1)

        # 토픽 발행
        self.pub_all.publish(String(data=json.dumps({'objects': all_objects})))
        self.pub_object.publish(String(data=json.dumps({'objects': object_list})))
        self.pub_insert.publish(String(data=json.dumps({'objects': insert_list})))

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = PosePublisherObIn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
