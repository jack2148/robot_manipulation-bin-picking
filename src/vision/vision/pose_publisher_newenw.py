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
TEMPLATE_DIR = Path(get_package_share_directory('vision')) / 'templates'

CONF_THRESH = 0.35

ANGLE_STEP_DEG = 2

# 기존 yaw-only 정규화용이었지만, 여기서는 sliding template의 기본 기준 크기로도 사용 가능
MATCH_SIZE = 160

# segmentation 중심 주변에서만 template sliding
LOCAL_SEARCH_RADIUS_PX = 55

# depth median window: 9x9
DEPTH_WINDOW_RADIUS_PX = 4

# template sliding score threshold
MIN_TEMPLATE_SCORE_FOR_CENTER = 0.25

# template 중심이 segmentation 중심에서 너무 멀리 튀면 fallback
MAX_CENTER_SHIFT_PX = 35

# template 크기 보정
TEMPLATE_SCALE = 1.10

MODELS = {
    "object": {
        "path": str(WEIGHTS_DIR / "best.pt"),
        "names": ["cross", "cylinder", "hole"],
        "colors": {
            "cross": (0, 220, 0),
            "cylinder": (0, 140, 255),
            "hole": (220, 0, 220),
        },
        "topic": "/object_poses",
    },
    "insert": {
        "path": str(WEIGHTS_DIR / "insert_best.pt"),
        "names": ["cross_insert", "cylinder_insert", "hole_insert"],
        "colors": {
            "cross_insert": (0, 220, 0),
            "cylinder_insert": (0, 140, 255),
            "hole_insert": (220, 0, 220),
        },
        "topic": "/insert_poses",
    },
}

TEMPLATE_FILES = {
    "cross": "cross_top.png",
    "cylinder": "circle_top.png",
    "hole": "square_top.png",

    "cross_insert": "cross_insert_top.png",
    "cylinder_insert": "circle_insert_top.png",
    "hole_insert": "square_insert_top.png",
}

NO_YAW_CLASSES = {
    "cylinder",
    "cylinder_insert",
}

RAW_TEMPLATES = {}


# ---------------------------------------------------------------------------
# Template utils
# ---------------------------------------------------------------------------
def binarize_template(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img.astype(np.uint8)


def crop_mask_to_square(mask_img, pad_ratio=0.15):
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


def rotate_keep_size(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated


def load_templates(logger=None):
    global RAW_TEMPLATES
    RAW_TEMPLATES = {}

    for cls, fname in TEMPLATE_FILES.items():
        path = TEMPLATE_DIR / fname
        tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if tmpl is None:
            if logger is not None:
                logger.warn(f"template not found: {path}")
            continue

        tmpl = binarize_template(tmpl)
        tmpl = crop_mask_to_square(tmpl, pad_ratio=0.15)

        if tmpl is None:
            if logger is not None:
                logger.warn(f"template invalid: {path}")
            continue

        RAW_TEMPLATES[cls] = tmpl

        if logger is not None:
            logger.info(f"loaded template: class={cls}, path={path}, shape={tmpl.shape}")


# ---------------------------------------------------------------------------
# Center / depth / 3D utils
# ---------------------------------------------------------------------------
def get_mask_centroid_2d(mask_img):
    m = cv2.moments(mask_img, binaryImage=True)

    if abs(m["m00"]) < 1e-6:
        ys, xs = np.where(mask_img > 0)
        if len(xs) < 10:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return float(cx), float(cy)


def get_mask_bbox(mask_img):
    ys, xs = np.where(mask_img > 0)

    if len(xs) < 10:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return int(x0), int(y0), int(x1), int(y1)


def get_depth_median_around(depth_image, cx, cy, depth_scale, radius_px=4):
    h, w = depth_image.shape[:2]

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    x0 = max(0, cx_i - radius_px)
    x1 = min(w, cx_i + radius_px + 1)
    y0 = max(0, cy_i - radius_px)
    y1 = min(h, cy_i + radius_px + 1)

    patch = depth_image[y0:y1, x0:x1].astype(float) * depth_scale
    vals = patch[patch > 0]

    if len(vals) < 5:
        return None

    z_med = np.median(vals)

    # 한 번 더 depth outlier 제거
    vals2 = vals[np.abs(vals - z_med) < 0.03]

    if len(vals2) >= 5:
        return float(np.median(vals2))

    return float(z_med)


def pixel_depth_to_3d(cx, cy, z, intrinsics):
    x = (cx - intrinsics.ppx) * z / intrinsics.fx
    y = (cy - intrinsics.ppy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=float)


def crop_local_roi(mask_img, cx, cy, radius_px):
    h, w = mask_img.shape[:2]

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    x0 = max(0, cx_i - radius_px)
    x1 = min(w, cx_i + radius_px + 1)
    y0 = max(0, cy_i - radius_px)
    y1 = min(h, cy_i + radius_px + 1)

    roi = mask_img[y0:y1, x0:x1].copy()

    if roi.size == 0:
        return None, None

    return roi, (x0, y0)


# ---------------------------------------------------------------------------
# Template sliding
# ---------------------------------------------------------------------------
def make_scaled_rotated_template(class_name, angle_deg, target_side):
    if class_name not in RAW_TEMPLATES:
        return None

    base = RAW_TEMPLATES[class_name]

    target_side = int(round(target_side))
    target_side = max(16, target_side)

    resized = cv2.resize(
        base,
        (target_side, target_side),
        interpolation=cv2.INTER_NEAREST,
    )

    rotated = rotate_keep_size(resized, angle_deg)

    # 회전 후 binary 재정리
    _, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)

    # 거의 빈 template 방지
    if np.count_nonzero(rotated) < 10:
        return None

    return rotated.astype(np.uint8)


def binary_template_sliding(mask_img, class_name):
    """
    segmentation 중심을 초기값으로 하고,
    중심 주변 ROI 안에서 rotated binary template을 sliding하여
    best center와 best yaw를 찾는다.
    """
    seg_center = get_mask_centroid_2d(mask_img)

    if seg_center is None:
        return None

    seg_cx, seg_cy = seg_center

    bbox = get_mask_bbox(mask_img)
    if bbox is None:
        return None

    bx0, by0, bx1, by1 = bbox
    bw = bx1 - bx0 + 1
    bh = by1 - by0 + 1

    # 실제 detection mask 크기에 맞춰 template 크기 결정
    target_side = int(round(max(bw, bh) * TEMPLATE_SCALE))
    target_side = max(16, target_side)

    # ROI는 template보다 충분히 커야 함
    search_radius = max(LOCAL_SEARCH_RADIUS_PX, target_side // 2 + 10)

    roi, origin = crop_local_roi(mask_img, seg_cx, seg_cy, search_radius)

    if roi is None:
        return {
            "cx": float(seg_cx),
            "cy": float(seg_cy),
            "seg_cx": float(seg_cx),
            "seg_cy": float(seg_cy),
            "yaw_deg": 0.0,
            "yaw_score": 0.0,
            "yaw_source": "roi_invalid",
            "center_source": "seg_center_fallback",
        }

    roi_h, roi_w = roi.shape[:2]

    if target_side >= roi_w or target_side >= roi_h:
        # ROI보다 template이 크면 template 크기를 줄임
        target_side = int(min(roi_w, roi_h) * 0.75)
        target_side = max(16, target_side)

    x0, y0 = origin

    roi_bin = (roi > 0).astype(np.uint8) * 255

    best_score = -1.0
    best_angle = 0.0
    best_loc = None
    best_template_shape = None

    if class_name not in RAW_TEMPLATES:
        return {
            "cx": float(seg_cx),
            "cy": float(seg_cy),
            "seg_cx": float(seg_cx),
            "seg_cy": float(seg_cy),
            "yaw_deg": 0.0,
            "yaw_score": 0.0,
            "yaw_source": "template_missing",
            "center_source": "seg_center_fallback",
        }

    for angle in range(0, 180, ANGLE_STEP_DEG):
        tmpl = make_scaled_rotated_template(class_name, angle, target_side)

        if tmpl is None:
            continue

        th, tw = tmpl.shape[:2]

        if th > roi_h or tw > roi_w:
            continue

        # binary overlap sliding
        # TM_CCORR_NORMED: binary mask끼리 겹치는 정도를 빠르게 계산
        result = cv2.matchTemplate(
            roi_bin,
            tmpl,
            cv2.TM_CCORR_NORMED,
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = float(max_val)
            best_angle = float(angle)
            best_loc = max_loc
            best_template_shape = (th, tw)

    if best_loc is None or best_template_shape is None:
        return {
            "cx": float(seg_cx),
            "cy": float(seg_cy),
            "seg_cx": float(seg_cx),
            "seg_cy": float(seg_cy),
            "yaw_deg": 0.0,
            "yaw_score": 0.0,
            "yaw_source": "sliding_failed",
            "center_source": "seg_center_fallback",
        }

    th, tw = best_template_shape
    loc_x, loc_y = best_loc

    refined_cx = x0 + loc_x + tw / 2.0
    refined_cy = y0 + loc_y + th / 2.0

    shift = float(np.hypot(refined_cx - seg_cx, refined_cy - seg_cy))

    use_refined = (
        best_score >= MIN_TEMPLATE_SCORE_FOR_CENTER
        and shift <= MAX_CENTER_SHIFT_PX
    )

    if use_refined:
        cx = refined_cx
        cy = refined_cy
        center_source = "template_sliding"
    else:
        cx = seg_cx
        cy = seg_cy

        if best_score < MIN_TEMPLATE_SCORE_FOR_CENTER:
            center_source = "seg_center_low_template_score"
        else:
            center_source = "seg_center_large_shift_fallback"

    return {
        "cx": float(cx),
        "cy": float(cy),
        "seg_cx": float(seg_cx),
        "seg_cy": float(seg_cy),
        "refined_cx": float(refined_cx),
        "refined_cy": float(refined_cy),
        "center_shift_px": float(shift),
        "yaw_deg": float(best_angle),
        "yaw_score": float(best_score),
        "yaw_source": "binary_template_sliding",
        "center_source": center_source,
        "template_side": int(target_side),
    }


def estimate_pose_from_best_method(mask_img, class_name, depth_image, intrinsics, depth_scale):
    """
    최선 방식:
    - cylinder류: seg center + depth median
    - non-cylinder류: binary template sliding center/yaw + depth median
    """
    seg_center = get_mask_centroid_2d(mask_img)

    if seg_center is None:
        return None

    seg_cx, seg_cy = seg_center

    if class_name in NO_YAW_CLASSES:
        center_info = {
            "cx": float(seg_cx),
            "cy": float(seg_cy),
            "seg_cx": float(seg_cx),
            "seg_cy": float(seg_cy),
            "refined_cx": float(seg_cx),
            "refined_cy": float(seg_cy),
            "center_shift_px": 0.0,
            "yaw_deg": 0.0,
            "yaw_score": 1.0,
            "yaw_source": "circle_no_yaw",
            "center_source": "seg_center_circle",
            "template_side": 0,
        }
    else:
        center_info = binary_template_sliding(mask_img, class_name)

        if center_info is None:
            return None

    cx = center_info["cx"]
    cy = center_info["cy"]

    z = get_depth_median_around(
        depth_image,
        cx,
        cy,
        depth_scale,
        radius_px=DEPTH_WINDOW_RADIUS_PX,
    )

    if z is None:
        # template 보정 중심에서 depth가 안 잡히면 seg center로 fallback
        z = get_depth_median_around(
            depth_image,
            center_info["seg_cx"],
            center_info["seg_cy"],
            depth_scale,
            radius_px=DEPTH_WINDOW_RADIUS_PX,
        )

        if z is None:
            return None

        cx = center_info["seg_cx"]
        cy = center_info["seg_cy"]
        center_info["cx"] = float(cx)
        center_info["cy"] = float(cy)
        center_info["center_source"] = "seg_center_depth_fallback"

    centroid = pixel_depth_to_3d(cx, cy, z, intrinsics)

    return centroid, center_info


# ---------------------------------------------------------------------------
# Point cloud / PCA utils
# 기존 orientation axes 계산용으로 유지
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


def estimate_axes_from_points(points, centroid):
    centered = points - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    return axes


# ---------------------------------------------------------------------------
# ROS2 노드
# ---------------------------------------------------------------------------
class PosePublisher(Node):
    def __init__(self):
        super().__init__("pose_publisher")

        load_templates(self.get_logger())

        self.models = {}

        for mode, cfg in MODELS.items():
            if not Path(cfg["path"]).exists():
                self.get_logger().warn(f"모델 없음: {cfg['path']}")
                continue

            self.models[mode] = YOLO(cfg["path"])
            self.get_logger().info(f"[{mode}] 모델 로드 완료: {cfg['path']}")

        self.mode = "object"

        self.publishers_ = {
            mode: self.create_publisher(String, cfg["topic"], 10)
            for mode, cfg in MODELS.items()
        }

        self.create_subscription(String, "/detect_mode", self._mode_callback, 10)

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

        self.timer = self.create_timer(0.1, self._timer_callback)
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

        cv2.putText(
            display,
            f"MODE: {self.mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

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

                pose_result = estimate_pose_from_best_method(
                    mask_img=mask_img,
                    class_name=class_name,
                    depth_image=depth_image,
                    intrinsics=self.intrinsics,
                    depth_scale=self.depth_scale,
                )

                if pose_result is None:
                    continue

                centroid, center_info = pose_result

                points = get_point_cloud(
                    depth_image,
                    mask_img,
                    self.intrinsics,
                    self.depth_scale,
                )

                if points is not None:
                    axes = estimate_axes_from_points(points, centroid)
                    pc_median = np.median(points, axis=0)
                else:
                    axes = np.eye(3)
                    pc_median = centroid.copy()

                yaw_deg = center_info["yaw_deg"]
                yaw_score = center_info["yaw_score"]
                yaw_source = center_info["yaw_source"]

                cx = int(round(center_info["cx"]))
                cy = int(round(center_info["cy"]))

                objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),

                    # 최종 사용 위치
                    "position": {
                        "x": round(float(centroid[0]), 4),
                        "y": round(float(centroid[1]), 4),
                        "z": round(float(centroid[2]), 4),
                    },

                    # 비교용: 기존 방식 point cloud median
                    "position_pc_median": {
                        "x": round(float(pc_median[0]), 4),
                        "y": round(float(pc_median[1]), 4),
                        "z": round(float(pc_median[2]), 4),
                    },

                    "orientation": {
                        "axis_x": [round(float(v), 4) for v in axes[:, 0]],
                        "axis_y": [round(float(v), 4) for v in axes[:, 1]],
                        "axis_z": [round(float(v), 4) for v in axes[:, 2]],
                    },

                    "yaw_deg": round(float(yaw_deg), 2),
                    "yaw_score": round(float(yaw_score), 3),
                    "yaw_source": yaw_source,
                    "center_source": center_info["center_source"],

                    "pixel_center": {
                        "cx": round(float(center_info["cx"]), 2),
                        "cy": round(float(center_info["cy"]), 2),
                    },
                    "seg_pixel_center": {
                        "cx": round(float(center_info["seg_cx"]), 2),
                        "cy": round(float(center_info["seg_cy"]), 2),
                    },
                    "refined_pixel_center": {
                        "cx": round(float(center_info.get("refined_cx", center_info["cx"])), 2),
                        "cy": round(float(center_info.get("refined_cy", center_info["cy"])), 2),
                    },
                    "center_shift_px": round(float(center_info.get("center_shift_px", 0.0)), 2),
                    "template_side": int(center_info.get("template_side", 0)),
                })

                overlay = display.copy()

                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color)

                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                seg_cx = int(round(center_info["seg_cx"]))
                seg_cy = int(round(center_info["seg_cy"]))

                refined_cx = int(round(center_info.get("refined_cx", center_info["cx"])))
                refined_cy = int(round(center_info.get("refined_cy", center_info["cy"])))

                # segmentation 중심: 파란 점
                cv2.circle(display, (seg_cx, seg_cy), 4, (255, 0, 0), -1)

                # template refined 중심 후보: 초록 점
                if class_name not in NO_YAW_CLASSES:
                    cv2.circle(display, (refined_cx, refined_cy), 4, (0, 255, 0), -1)

                # 최종 채택 중심: 빨간 점
                cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

                # local search ROI
                search_radius = LOCAL_SEARCH_RADIUS_PX
                cv2.rectangle(
                    display,
                    (max(0, seg_cx - search_radius), max(0, seg_cy - search_radius)),
                    (
                        min(display.shape[1] - 1, seg_cx + search_radius),
                        min(display.shape[0] - 1, seg_cy + search_radius),
                    ),
                    (255, 255, 0),
                    1,
                )

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
                    f"{class_name} {confidence:.2f} | "
                    f"X:{centroid[0]:+.2f} Y:{centroid[1]:+.2f} Z:{centroid[2]:.2f}m | "
                    f"yaw:{yaw_deg:.1f} score:{yaw_score:.2f} | "
                    f"{center_info['center_source']}"
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

        cv2.imshow("PosePublisher - ESC to quit", display)
        cv2.waitKey(1)

        msg = String()
        msg.data = json.dumps({
            "mode": self.mode,
            "objects": objects,
        })

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