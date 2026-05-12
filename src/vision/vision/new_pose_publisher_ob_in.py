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
# 설정 - object + insert 통합 모델 (6 클래스) + Template Yaw
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(get_package_share_directory('vision')) / 'weights'
TEMPLATE_DIR = Path(get_package_share_directory('vision')) / 'templates'

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

# object / insert 분류
OBJECT_CLASSES = {'cylinder', 'hole', 'cross'}
INSERT_CLASSES = {'cross_insert', 'cylinder_insert', 'hole_insert'}

CONF_THRESH = 0.4

# Template yaw 설정
ANGLE_STEP_DEG = 1
MATCH_SIZE = 160

TEMPLATE_FILES = {
    'cross': 'cross_top.png',
    'cylinder': 'circle_top.png',
    'hole': 'square_top.png',

    'cross_insert': 'cross_insert_top.png',
    'cylinder_insert': 'circle_insert_top.png',
    'hole_insert': 'square_insert_top.png',
}

NO_YAW_CLASSES = {
    'cylinder',
    'cylinder_insert',
}

ROTATED_TEMPLATES = {}


# ---------------------------------------------------------------------------
# Template yaw utils
# ---------------------------------------------------------------------------
def normalize_binary(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(
        img,
        (MATCH_SIZE, MATCH_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img.astype(np.uint8)


def rotate_keep_size(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def crop_mask_to_square(mask_img, pad_ratio=0.2):
    ys, xs = np.where(mask_img > 0)

    if len(xs) < 10:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    w = x1 - x0 + 1
    h = y1 - y0 + 1
    side = max(w, h)

    pad = int(round(side * pad_ratio))
    side = side + 2 * pad

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    x0 = cx - side // 2
    x1 = x0 + side
    y0 = cy - side // 2
    y1 = y0 + side

    out = np.zeros((side, side), dtype=np.uint8)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(mask_img.shape[1], x1)
    src_y1 = min(mask_img.shape[0], y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask_img[src_y0:src_y1, src_x0:src_x1]
    return out


def iou_score(a, b):
    a_bin = a > 0
    b_bin = b > 0

    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


def load_rotated_templates(logger=None):
    global ROTATED_TEMPLATES
    ROTATED_TEMPLATES = {}

    for cls, fname in TEMPLATE_FILES.items():
        path = TEMPLATE_DIR / fname
        tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if tmpl is None:
            if logger is not None:
                logger.warn(f'template not found: {path}')
            continue

        tmpl = normalize_binary(tmpl)

        ROTATED_TEMPLATES[cls] = []

        for angle in range(0, 180, ANGLE_STEP_DEG):
            rotated = rotate_keep_size(tmpl, angle)
            ROTATED_TEMPLATES[cls].append((float(angle), rotated))

        if logger is not None:
            logger.info(f'loaded template: class={cls}, path={path}')


def estimate_yaw_from_template(mask_img, class_name):
    if class_name in NO_YAW_CLASSES:
        return 0.0, 1.0, 'circle_no_yaw'

    if class_name not in ROTATED_TEMPLATES:
        return 0.0, 0.0, 'template_missing'

    crop = crop_mask_to_square(mask_img)

    if crop is None:
        return 0.0, 0.0, 'mask_invalid'

    query = normalize_binary(crop)

    if query is None:
        return 0.0, 0.0, 'query_invalid'

    best_angle = 0.0
    best_score = -1.0

    for angle, tmpl in ROTATED_TEMPLATES[class_name]:
        score = iou_score(query, tmpl)

        if score > best_score:
            best_score = score
            best_angle = angle

    yaw_deg = float(best_angle)

    return yaw_deg, float(best_score), 'template_iou'


# ---------------------------------------------------------------------------
# Point cloud / PCA utils
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

        load_rotated_templates(self.get_logger())

        if not Path(MODEL_PATH).exists():
            self.get_logger().error(f'모델 없음: {MODEL_PATH}')
            raise FileNotFoundError(MODEL_PATH)

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'모델 로드 완료: {MODEL_PATH}')

        # 퍼블리셔
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

        self.timer = self.create_timer(0.1, self._timer_callback)
        self.get_logger().info('PosePublisherObIn ready. object + insert 통합 검출 + template yaw')

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

        all_objects = []
        object_list = []
        insert_list = []

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
                    depth_image,
                    mask_img,
                    self.intrinsics,
                    self.depth_scale,
                )

                if points is None:
                    continue

                # 위치/axis는 기존 방식 그대로
                centroid, axes = estimate_pose(points)

                # yaw만 template matching 방식으로 추가
                yaw_deg, yaw_score, yaw_source = estimate_yaw_from_template(
                    mask_img=mask_img,
                    class_name=class_name,
                )

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
                    'yaw_deg': round(float(yaw_deg), 2),
                    'yaw_score': round(float(yaw_score), 3),
                    'yaw_source': yaw_source,
                }

                all_objects.append(det)

                if class_name in OBJECT_CLASSES:
                    object_list.append(det)
                elif class_name in INSERT_CLASSES:
                    insert_list.append(det)

                # ------------------------------------------------------------
                # 시각화
                # ------------------------------------------------------------
                overlay = display.copy()

                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)

                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                cx = int(centroid[0] * self.intrinsics.fx / centroid[2] + self.intrinsics.ppx)
                cy = int(centroid[1] * self.intrinsics.fy / centroid[2] + self.intrinsics.ppy)

                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                # yaw 방향 화살표
                if class_name not in NO_YAW_CLASSES:
                    length = 45
                    yaw_rad = np.radians(yaw_deg)

                    ex = int(cx + length * np.cos(yaw_rad))
                    ey = int(cy - length * np.sin(yaw_rad))

                    cv2.arrowedLine(
                        display,
                        (cx, cy),
                        (ex, ey),
                        (0, 0, 255),
                        2,
                        tipLength=0.25,
                    )

                label = (
                    f'{class_name} {confidence:.2f} | '
                    f'X:{centroid[0]:+.2f} Y:{centroid[1]:+.2f} Z:{centroid[2]:.2f}m | '
                    f'yaw:{yaw_deg:.1f} score:{yaw_score:.2f}'
                )

                cv2.putText(
                    display,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )

        cv2.imshow('PosePublisher ob_in + template yaw - ESC to quit', display)
        cv2.waitKey(1)

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