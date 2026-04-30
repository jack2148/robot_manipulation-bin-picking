import os
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Optional
from dataclasses import dataclass


# =========================
# Pose / Detector
# =========================

@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray


class TargetDetector:
    def __init__(
        self,
        cam_mat: np.ndarray = None,
        dist_coeffs: np.ndarray = None,
    ):
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (8, 8),
            22.5 * 1e-3,
            16.875 * 1e-3,
            self.dict,
        )
        self.board.setLegacyPattern(True)
        self.update_intrinsic(cam_mat, dist_coeffs)

        self.charuco_corner_color = (0, 50, 200)

    def generate_img(self, img_res: int = 500) -> np.ndarray:
        return self.board.generateImage((img_res, img_res))

    def update_intrinsic(self, cam_mat: np.ndarray, dist_coeffs: np.ndarray):
        self.params = cv2.aruco.CharucoParameters()
        self.params.cameraMatrix = cam_mat
        self.params.distCoeffs = dist_coeffs

        self.marker_detector = cv2.aruco.ArucoDetector(self.dict)
        self.detector = cv2.aruco.CharucoDetector(self.board, self.params)

    def detect_aruco(self, img: np.ndarray):
        if "4.10" in cv2.__version__:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                img, self.board.getDictionary()
            )
        else:
            marker_corners, marker_ids, _ = self.marker_detector.detectMarkers(img)
        return marker_corners, marker_ids

    def detect_charuco(self, img, marker_corners, marker_ids):
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(
            img, markerCorners=marker_corners, markerIds=marker_ids
        )
        return charuco_corners, charuco_ids

    def draw_aruco(self, img, marker_corners, marker_ids):
        out = img.copy()
        if self._is_grayscale(out):
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
        return out

    def draw_charuco(self, img, charuco_corners, charuco_ids):
        out = img.copy()
        if self._is_grayscale(out):
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                out, charuco_corners, charuco_ids, self.charuco_corner_color
            )
        return out

    def _is_grayscale(self, img: np.ndarray) -> bool:
        return len(img.shape) != 3 or img.shape[-1] != 3

    def estimate_pose(
        self, img: np.ndarray, draw: bool = True
    ) -> Optional[tuple[Pose, Optional[np.ndarray]]]:
        cam_mat = self.params.cameraMatrix
        dist_coeffs = self.params.distCoeffs

        marker_corners, marker_ids = self.detect_aruco(img)
        if marker_ids is None or len(marker_ids) == 0:
            return None

        charuco_corners, charuco_ids = self.detect_charuco(
            img, marker_corners, marker_ids
        )

        if charuco_ids is None or len(charuco_ids) < 6:
            return None

        if "4.10" in cv2.__version__:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                cam_mat,
                dist_coeffs,
                None,
                None,
            )
        else:
            obj_points, img_points = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            valid, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, cam_mat, dist_coeffs
            )

        if not valid:
            return None

        pose = Pose(rvec=rvec, tvec=tvec)

        if not draw:
            return pose, None

        debug_img = self.draw_aruco(img, marker_corners, marker_ids)
        debug_img = self.draw_charuco(debug_img, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(
            debug_img,
            cam_mat,
            dist_coeffs,
            rvec,
            tvec,
            self.square_length,
        )
        return pose, debug_img

    @property
    def cam_mat(self):
        return self.params.cameraMatrix

    @property
    def dist_coeffs(self):
        return self.params.distCoeffs

    @property
    def square_length(self):
        return self.board.getSquareLength()


# =========================
# utils
# =========================

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, np.asarray(tvec).reshape(3))


def T_to_R_t(T):
    R = T[:3, :3].copy()
    t = T[:3, 3].reshape(3, 1).copy()
    return R, t


def rot_err_deg(R1, R2):
    R = R1.T @ R2
    val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


# =========================
# Collector
# =========================

class CharucoHandEyeCollector:
    def __init__(
        self,
        K,
        dist,
        min_charuco_corners=6,
        save_dir="handeye_capture",
    ):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        self.min_charuco_corners = min_charuco_corners

        self.save_dir = save_dir
        self.img_dir = os.path.join(save_dir, "images")
        self.dbg_dir = os.path.join(save_dir, "debug")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.dbg_dir, exist_ok=True)

        self.detector = TargetDetector(
            cam_mat=self.K,
            dist_coeffs=self.dist,
        )

        self.samples = []

    def detect_board_pose(self, image_bgr, draw=True):
        output = self.detector.estimate_pose(image_bgr, draw=draw)

        debug_img = image_bgr.copy() if draw else None

        if output is None:
            return False, None, None, debug_img, "charuco detection failed"

        pose, det_img = output
        if draw and det_img is not None:
            debug_img = det_img

        return True, pose.rvec, pose.tvec, debug_img, "ok"

    def capture_sample(self, image_bgr, base_T_ee):
        base_T_ee = np.asarray(base_T_ee, dtype=np.float64)
        if base_T_ee.shape != (4, 4):
            raise ValueError("base_T_ee must be 4x4")

        ok, rvec, tvec, debug_img, msg = self.detect_board_pose(image_bgr, draw=True)

        idx = len(self.samples)
        stem = f"{idx:03d}"

        img_path = os.path.join(self.img_dir, f"{stem}.png")
        dbg_path = os.path.join(self.dbg_dir, f"{stem}_debug.png")
        meta_path = os.path.join(self.save_dir, f"{stem}.json")

        cv2.imwrite(img_path, image_bgr)
        if debug_img is not None:
            cv2.imwrite(dbg_path, debug_img)

        if not ok:
            result = {
                "ok": False,
                "index": idx,
                "reason": msg,
                "saved_image": img_path,
                "saved_debug": dbg_path,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ok": False,
                        "reason": msg,
                        "base_T_ee": base_T_ee.tolist(),
                        "image_path": img_path,
                        "debug_path": dbg_path,
                    },
                    f,
                    indent=2,
                )
            return result

        cam_T_board = rvec_tvec_to_T(rvec, tvec)

        sample = {
            "index": idx,
            "base_T_ee": base_T_ee,
            "cam_T_board": cam_T_board,
            "image_path": img_path,
            "debug_path": dbg_path,
        }
        self.samples.append(sample)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": True,
                    "base_T_ee": base_T_ee.tolist(),
                    "cam_T_board": cam_T_board.tolist(),
                    "image_path": img_path,
                    "debug_path": dbg_path,
                },
                f,
                indent=2,
            )

        return {
            "ok": True,
            "index": idx,
            "num_valid_samples": len(self.samples),
            "saved_image": img_path,
            "saved_debug": dbg_path,
        }

    def run_calibration(self, method_name="TSAI"):
        if len(self.samples) < 5:
            return {
                "ok": False,
                "reason": f"valid samples too few: {len(self.samples)} (recommend >= 15)"
            }

        method_map = {
            "TSAI": cv2.CALIB_HAND_EYE_TSAI,
            "PARK": cv2.CALIB_HAND_EYE_PARK,
            "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        method = method_map[method_name.upper()]

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for s in self.samples:
            base_T_ee = s["base_T_ee"]
            cam_T_board = s["cam_T_board"]

            Rg, tg = T_to_R_t(base_T_ee)
            Rt, tt = T_to_R_t(cam_T_board)

            R_gripper2base.append(Rg)
            t_gripper2base.append(tg)
            R_target2cam.append(Rt)
            t_target2cam.append(tt)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=method,
        )

        ee_T_cam = make_T(R_cam2gripper, t_cam2gripper)

        base_T_board_list = []

        for s in self.samples:
            base_T_ee = s["base_T_ee"]
            cam_T_board = s["cam_T_board"]

            base_T_board = base_T_ee @ ee_T_cam @ cam_T_board
            base_T_board_list.append(base_T_board)

        positions = np.array([T[:3, 3] for T in base_T_board_list], dtype=np.float64)
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0)

        rotations = [T[:3, :3] for T in base_T_board_list]
        R_ref = rotations[0]
        rot_errs = [rot_err_deg(R_ref, R) for R in rotations]

        base_T_board_mean = np.eye(4, dtype=np.float64)
        base_T_board_mean[:3, :3] = rotations[0]
        base_T_board_mean[:3, 3] = pos_mean

        result = {
            "ok": True,
            "num_valid_samples": len(self.samples),
            "method": method_name.upper(),
            "ee_T_cam": ee_T_cam.tolist(),
            "base_T_board_mean": base_T_board_mean.tolist(),
            "position_std_mm": (pos_std * 1000.0).tolist(),
            "rotation_error_deg_mean": float(np.mean(rot_errs)),
            "rotation_error_deg_std": float(np.std(rot_errs)),
        }

        out_path = os.path.join(self.save_dir, "handeye_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["saved_result"] = out_path
        return result


# =========================
# RealSense wrapper
# =========================

class RealSenseCharucoHandEye:
    def __init__(
        self,
        width=1280,
        height=720,
        fps=30,
        min_charuco_corners=6,
        save_dir="handeye_capture",
        serial_number=None,
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_number is not None:
            self.config.enable_device(serial_number)

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        color_stream = self.profile.get_stream(rs.stream.color)
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.K = np.array([
            [color_intr.fx, 0.0, color_intr.ppx],
            [0.0, color_intr.fy, color_intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.dist = np.array(color_intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)

        self.collector = CharucoHandEyeCollector(
            K=self.K,
            dist=self.dist,
            min_charuco_corners=min_charuco_corners,
            save_dir=save_dir,
        )

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def get_latest_bgr(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def preview(self, window_name="realsense_preview"):
        while True:
            img = self.get_latest_bgr()
            if img is None:
                continue

            ok, _, _, debug_img, msg = self.collector.detect_board_pose(img, draw=True)

            show = debug_img if debug_img is not None else img.copy()
            text = f"charuco: {'OK' if ok else 'FAIL'} ({msg})"
            cv2.putText(
                show,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if ok else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        cv2.destroyWindow(window_name)

    def capture(self, base_T_ee):
        img = self.get_latest_bgr()
        if img is None:
            return {"ok": False, "reason": "failed to get color frame"}
        return self.collector.capture_sample(img, base_T_ee)

    def run_calibration(self, method_name="TSAI"):
        return self.collector.run_calibration(method_name=method_name)


# =========================
# test
# =========================

if __name__ == "__main__":
    calib = RealSenseCharucoHandEye(
        width=1280,
        height=720,
        fps=30,
        save_dir="handeye_capture_rs",
    )

    try:
        base_T_ee = np.array([
            [1.0, 0.0, 0.0, 0.45],
            [0.0, 1.0, 0.0, 0.10],
            [0.0, 0.0, 1.0, 0.30],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        print(calib.capture(base_T_ee))
        print(calib.run_calibration("TSAI"))

    finally:
        calib.stop()