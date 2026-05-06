import time
import json
import traceback
from pathlib import Path
from datetime import datetime

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import numpy as np
import rbpodo as rb

from calibration import RealSenseCharucoHandEye


# ============================================================
# USER CONFIG
# ============================================================

ROBOT_IP = "192.168.1.10"

USE_ROBOT = True
USE_CAMERA = True
RUN_MODE = "real"      # "real" or "sim"
SIM_MODE = (RUN_MODE == "sim")

MOVE_SPEED = 60.0
MOVE_ACCEL = 100.0

TCP_POS_TOL_MM = 2.0
TCP_RPY_TOL_DEG = 2.0
TCP_REACH_TIMEOUT_S = 30.0
TCP_STABLE_COUNT = 5
TCP_POLL_DT = 0.05

CAPTURE_DELAY_S = 1.5

MIN_SAFE_Z_MM = 150.0

# 기존 샘플러와 동일한 가정:
# camera forward = EE local -Y
CAMERA_FWD_AXIS = 1
CAMERA_FWD_SIGN = -1

# 검증용 이동 grid
RADIUS_MM_DEFAULT = 200.0
GRID_DIST_OFFSETS_MM = [0.0, 60.0, 120.0]
GRID_U_OFFSETS_MM = [-80.0, -40.0, 0.0, 40.0, 80.0]
GRID_V_OFFSETS_MM = [-50.0, 0.0, 50.0]

# 결과 저장 위치
SAVE_DIR = Path(__file__).resolve().parents[1] / "config" / "handeye_verify_rs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIXED HAND-EYE RESULT: ee_T_cam
# 단위: meter
# ============================================================

EE_T_CAM = np.array([
    [-0.7060207385535571,  0.7081758673326828, -0.004653779097927135, -0.021958140801196825],
    [ 0.007422930891674395, 0.0008290226932985423, -0.9999721060201385, 0.04480681220491634],
    [-0.7081522554008193, -0.7060355895059232, -0.0058420477365626056, 0.06951031763543669],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


# ============================================================
# Basic math utils
# ============================================================

def debug(msg):
    print(f"[DEBUG] {msg}", flush=True)


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n >= eps else v.copy()


def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    """
    pose: x y z rx ry rz
    rotation convention: R = Rz @ Ry @ Rx
    """
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def R_to_euler_zyx_deg(R):
    R = np.asarray(R, dtype=np.float64)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy >= 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    return np.degrees([rx, ry, rz])


def pose_vec_to_T_mm(pose6):
    """
    input pose:
        x y z in mm
        rx ry rz in deg
    output:
        T in mm translation
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    x, y, z, rx, ry, rz = pose6

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def T_to_pose_vec_mm(T):
    T = np.asarray(T, dtype=np.float64)
    rx, ry, rz = R_to_euler_zyx_deg(T[:3, :3])
    return np.array([T[0, 3], T[1, 3], T[2, 3], rx, ry, rz], dtype=np.float64)


def base_T_ee_from_tcp_pose_mm(pose6):
    """
    input:
        robot TCP pose x y z mm, rx ry rz deg
    output:
        base_T_ee, translation in meter
    """
    T_mm = pose_vec_to_T_mm(pose6)
    T_m = T_mm.copy()
    T_m[:3, 3] /= 1000.0
    return T_m


def rvec_tvec_to_T(rvec, tvec):
    """
    Charuco / solvePnP result:
        X_cam = R * X_board + t
    therefore output is cam_T_board.
    tvec unit is assumed meter.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_angle_deg(R):
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def mean_rotation_svd(R_list):
    """
    simple chordal mean:
        average matrix -> project to SO(3)
    """
    M = np.mean(np.stack(R_list, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt

    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt

    return Rm


def inv_T(T):
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# ============================================================
# Trajectory generation
# ============================================================

def get_camera_fwd_from_ee_R(R_ee):
    axis_vec = R_ee[:, CAMERA_FWD_AXIS]
    return normalize(CAMERA_FWD_SIGN * axis_vec)


def make_fwd_perp_axes(fwd):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if abs(np.dot(fwd, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    px = normalize(np.cross(up, fwd))
    py = normalize(np.cross(fwd, px))
    return px, py


def make_look_at_R(cam_pos_mm, target_mm, ref_up, ref_up_fallback):
    """
    EE rotation matrix.
    Assumption:
        camera forward direction = EE -local_y
    """
    view_dir = normalize(target_mm - cam_pos_mm)
    local_y = -view_dir

    up = ref_up if abs(np.dot(local_y, ref_up)) < 0.98 else ref_up_fallback

    local_z = normalize(np.cross(up, local_y))
    local_x = normalize(np.cross(local_y, local_z))
    local_z = normalize(np.cross(local_x, local_y))

    R = np.column_stack([local_x, local_y, local_z])

    # 기존 샘플러에서 썼던 roll offset
    ROLL_OFFSET_DEG = 25.0
    a = np.radians(ROLL_OFFSET_DEG)

    Ry_local = np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0        ],
        [-np.sin(a), 0, np.cos(a)]
    ], dtype=np.float64)

    return R @ Ry_local


def compute_board_center_from_initial_pose(initial_pose_vec, radius_mm):
    T0 = pose_vec_to_T_mm(initial_pose_vec)
    p0 = T0[:3, 3]
    fwd = get_camera_fwd_from_ee_R(T0[:3, :3])
    return p0 + radius_mm * fwd


def build_verify_pose_list(initial_pose_vec, base_dist_mm,
                           dist_offsets_mm,
                           u_offsets_mm,
                           v_offsets_mm):
    """
    initial pose에서 보드 중심을 추정하고,
    여러 거리/좌우/상하 포즈를 생성한다.
    """
    T0 = pose_vec_to_T_mm(initial_pose_vec)
    p0 = T0[:3, 3]
    fwd = get_camera_fwd_from_ee_R(T0[:3, :3])

    board_center_mm = p0 + base_dist_mm * fwd

    perp_u, perp_v = make_fwd_perp_axes(fwd)
    ref_up = perp_v
    ref_up_fallback = perp_u

    pose_list = []

    for d_off in dist_offsets_mm:
        dist_mm = base_dist_mm + d_off

        if dist_mm <= 0:
            continue

        for v in v_offsets_mm:
            for u in u_offsets_mm:
                tcp_pos_mm = (
                    board_center_mm
                    - dist_mm * fwd
                    + u * perp_u
                    + v * perp_v
                )

                R = make_look_at_R(
                    cam_pos_mm=tcp_pos_mm,
                    target_mm=board_center_mm,
                    ref_up=ref_up,
                    ref_up_fallback=ref_up_fallback,
                )

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = tcp_pos_mm

                pose_list.append(T_to_pose_vec_mm(T))

    return pose_list


def is_pose_safe_basic(pose_vec):
    x, y, z, rx, ry, rz = np.asarray(pose_vec, dtype=np.float64).reshape(6)

    if z < MIN_SAFE_Z_MM:
        return False, f"z too low: {z:.1f} < {MIN_SAFE_Z_MM:.1f}"

    return True, "ok"


def filter_unsafe_poses(pose_list):
    safe = []
    removed = []

    for i, p in enumerate(pose_list):
        ok, reason = is_pose_safe_basic(p)
        if ok:
            safe.append(p)
        else:
            removed.append((i, p, reason))

    print(f"[SAFE FILTER] kept={len(safe)}, removed={len(removed)}")

    for i, p, reason in removed:
        print(f"  removed idx={i}: {reason}, pose={p}")

    return safe


# ============================================================
# Robot helpers
# ============================================================

def angle_diff_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a - b + 180.0) % 360.0 - 180.0


def get_current_tcp_pose_vec_mm(robot_data):
    state = robot_data.request_data()

    if state is None:
        return None

    if hasattr(state.sdata, "tcp"):
        tcp_info = np.array(state.sdata.tcp, dtype=np.float64)
    elif hasattr(state.sdata, "tcp_pos"):
        tcp_info = np.array(state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(state.sdata, "cur_pos"):
        tcp_info = np.array(state.sdata.cur_pos, dtype=np.float64)
    else:
        raise RuntimeError("TCP pose field를 찾지 못했습니다.")

    return tcp_info[:6].copy()


def wait_until_tcp_reached(
    robot_data,
    target_pose_vec,
    pos_tol_mm=TCP_POS_TOL_MM,
    rpy_tol_deg=TCP_RPY_TOL_DEG,
    timeout_s=TCP_REACH_TIMEOUT_S,
    stable_count_required=TCP_STABLE_COUNT,
    poll_dt=TCP_POLL_DT,
):
    target_pose_vec = np.asarray(target_pose_vec, dtype=np.float64).reshape(6)

    start = time.time()
    stable_count = 0

    last_cur = None
    last_pos_err = float("inf")
    last_rpy_err = float("inf")

    while time.time() - start < timeout_s:
        cur = get_current_tcp_pose_vec_mm(robot_data)

        if cur is None:
            time.sleep(poll_dt)
            continue

        last_cur = cur

        pos_err = float(np.linalg.norm(cur[:3] - target_pose_vec[:3]))
        rpy_err_vec = angle_diff_deg(cur[3:6], target_pose_vec[3:6])
        rpy_err = float(np.linalg.norm(rpy_err_vec))

        last_pos_err = pos_err
        last_rpy_err = rpy_err

        if pos_err <= pos_tol_mm and rpy_err <= rpy_tol_deg:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_count_required:
            print(
                f"[TCP ARRIVED] pos_err={pos_err:.3f} mm, "
                f"rpy_err={rpy_err:.3f} deg"
            )
            return True, cur, pos_err, rpy_err

        time.sleep(poll_dt)

    print(
        f"[TCP TIMEOUT] pos_err={last_pos_err:.3f} mm, "
        f"rpy_err={last_rpy_err:.3f} deg"
    )
    print(f"  target = {target_pose_vec}")
    print(f"  current = {last_cur}")

    return False, last_cur, last_pos_err, last_rpy_err


def move_l_blocking(robot, rc, robot_data, pose_vec, speed, accel):
    pose_vec = np.asarray(pose_vec, dtype=np.float64).reshape(6)

    print("[MOVE_L] target:", pose_vec)
    robot.move_l(rc, pose_vec, speed, accel)
    rc.error().throw_if_not_empty()

    ok, cur_pose, pos_err, rpy_err = wait_until_tcp_reached(
        robot_data=robot_data,
        target_pose_vec=pose_vec,
    )

    rc.error().throw_if_not_empty()

    if not ok:
        raise RuntimeError(
            "TCP did not reach target pose. "
            f"pos_err={pos_err:.3f} mm, rpy_err={rpy_err:.3f} deg"
        )

    return cur_pose


# ============================================================
# Viewer helpers
# ============================================================

def make_overlay(img, text_lines):
    out = img.copy()
    y = 30

    for line in text_lines:
        cv2.putText(
            out, str(line), (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 28

    return out


def show_detection_viewer(calib, window_name, lines=None):
    lines = lines or []

    frame = None if calib is None else calib.get_latest_bgr()

    if frame is None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img = make_overlay(img, ["NO FRAME"] + lines)
        cv2.imshow(window_name, img)
        return None, None, None, None

    ok, rvec, tvec, debug_img, msg = calib.collector.detect_board_pose(frame, draw=True)

    if debug_img is None:
        debug_img = frame.copy()

    dist_text = "board_dist_mm: N/A"
    if ok and tvec is not None:
        dist_mm = float(np.linalg.norm(np.asarray(tvec).reshape(-1)) * 1000.0)
        dist_text = f"board_dist_mm: {dist_mm:.1f}"

    img = make_overlay(
        debug_img,
        [
            f"charuco: {'OK' if ok else 'FAIL'} ({msg})",
            dist_text,
        ] + lines
    )

    cv2.imshow(window_name, img)
    return ok, rvec, tvec, msg


# ============================================================
# Verification logic
# ============================================================

def capture_one_verification_sample(calib, base_T_ee, sample_idx):
    """
    Detect board pose and compute:
        base_T_board = base_T_ee @ EE_T_CAM @ cam_T_board
    """
    frame = calib.get_latest_bgr()

    if frame is None:
        return None, "no frame"

    ok, rvec, tvec, debug_img, msg = calib.collector.detect_board_pose(frame, draw=True)

    if not ok or rvec is None or tvec is None:
        return None, f"charuco fail: {msg}"

    cam_T_board = rvec_tvec_to_T(rvec, tvec)
    base_T_board = base_T_ee @ EE_T_CAM @ cam_T_board

    sample = {
        "sample_idx": int(sample_idx),
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "base_T_ee": base_T_ee.tolist(),
        "cam_T_board": cam_T_board.tolist(),
        "base_T_board": base_T_board.tolist(),
        "board_pos_base_m": base_T_board[:3, 3].tolist(),
        "board_pos_base_mm": (base_T_board[:3, 3] * 1000.0).tolist(),
        "msg": str(msg),
    }

    return sample, "ok"


def summarize_verification(samples):
    if len(samples) == 0:
        return None

    Ts = [np.asarray(s["base_T_board"], dtype=np.float64) for s in samples]

    positions_m = np.array([T[:3, 3] for T in Ts], dtype=np.float64)
    positions_mm = positions_m * 1000.0

    R_list = [T[:3, :3] for T in Ts]
    R_mean = mean_rotation_svd(R_list)

    pos_mean_mm = np.mean(positions_mm, axis=0)
    pos_std_mm = np.std(positions_mm, axis=0)
    pos_err_mm = np.linalg.norm(positions_mm - pos_mean_mm.reshape(1, 3), axis=1)

    rot_err_deg = []
    for R in R_list:
        dR = R_mean.T @ R
        rot_err_deg.append(rotation_angle_deg(dR))
    rot_err_deg = np.array(rot_err_deg, dtype=np.float64)

    # outlier 기준은 임시로 넉넉하게:
    # position error > mean + 2*std 또는 rotation error > mean + 2*std
    pos_thr = float(np.mean(pos_err_mm) + 2.0 * np.std(pos_err_mm))
    rot_thr = float(np.mean(rot_err_deg) + 2.0 * np.std(rot_err_deg))

    outliers = []
    for i, s in enumerate(samples):
        if pos_err_mm[i] > pos_thr or rot_err_deg[i] > rot_thr:
            outliers.append({
                "sample_idx": int(s["sample_idx"]),
                "pos_err_mm": float(pos_err_mm[i]),
                "rot_err_deg": float(rot_err_deg[i]),
            })

    summary = {
        "ok": True,
        "num_samples": len(samples),
        "ee_T_cam": EE_T_CAM.tolist(),

        "base_T_board_mean": {
            "translation_mm": pos_mean_mm.tolist(),
            "rotation_matrix": R_mean.tolist(),
        },

        "position_std_mm": pos_std_mm.tolist(),
        "position_std_norm_mm": float(np.linalg.norm(pos_std_mm)),

        "position_error_mm": {
            "mean": float(np.mean(pos_err_mm)),
            "std": float(np.std(pos_err_mm)),
            "max": float(np.max(pos_err_mm)),
            "per_sample": pos_err_mm.tolist(),
        },

        "rotation_error_deg": {
            "mean": float(np.mean(rot_err_deg)),
            "std": float(np.std(rot_err_deg)),
            "max": float(np.max(rot_err_deg)),
            "per_sample": rot_err_deg.tolist(),
        },

        "outlier_rule": {
            "position_threshold_mm": pos_thr,
            "rotation_threshold_deg": rot_thr,
        },
        "outliers": outliers,
    }

    return summary


def print_summary(summary):
    if summary is None:
        print("[SUMMARY] no valid samples")
        return

    print("\n" + "=" * 70)
    print("HAND-EYE FIXED RESULT VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"num_samples = {summary['num_samples']}")

    t = summary["base_T_board_mean"]["translation_mm"]
    print("\n[base_T_board mean translation]")
    print(f"x={t[0]:.3f} mm, y={t[1]:.3f} mm, z={t[2]:.3f} mm")

    std = summary["position_std_mm"]
    print("\n[position std]")
    print(f"x={std[0]:.3f} mm, y={std[1]:.3f} mm, z={std[2]:.3f} mm")
    print(f"std_norm={summary['position_std_norm_mm']:.3f} mm")

    pe = summary["position_error_mm"]
    print("\n[position error to mean]")
    print(f"mean={pe['mean']:.3f} mm, std={pe['std']:.3f} mm, max={pe['max']:.3f} mm")

    re = summary["rotation_error_deg"]
    print("\n[rotation error to mean]")
    print(f"mean={re['mean']:.3f} deg, std={re['std']:.3f} deg, max={re['max']:.3f} deg")

    print("\n[outliers]")
    if len(summary["outliers"]) == 0:
        print("none")
    else:
        for o in summary["outliers"]:
            print(
                f"sample={o['sample_idx']} "
                f"pos_err={o['pos_err_mm']:.3f} mm "
                f"rot_err={o['rot_err_deg']:.3f} deg"
            )

    print("=" * 70 + "\n")


# ============================================================
# Input helpers
# ============================================================

def input_initial_pose_vec():
    print("\n[INPUT INITIAL TCP POSE]")
    print("format: x(mm) y(mm) z(mm) rx(deg) ry(deg) rz(deg)")
    print("example: -117.99 -461.92 380.00 90.04 -0.06 44.45")

    while True:
        vals = input("initial pose >> ").strip().split()

        if len(vals) != 6:
            print("6개 입력해야 함")
            continue

        try:
            return np.array(list(map(float, vals)), dtype=np.float64)
        except Exception:
            print("숫자로 다시 입력")


def input_radius_mm(default_value):
    print("\n[INPUT BOARD DISTANCE / RADIUS]")
    print("초기 pose에서 보드까지의 대략 거리(mm)를 입력")
    print("엔터를 치면 기본값 사용")

    while True:
        s = input(f"radius mm [{default_value:.1f}] >> ").strip()

        if s == "":
            return float(default_value)

        try:
            return float(s)
        except ValueError:
            print("숫자로 입력")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    robot = None
    rc = None
    robot_data = None
    calib = None

    samples = []

    try:
        debug("program start")

        print("\n[FIXED ee_T_cam]")
        print(EE_T_CAM)

        print("\n[ee_T_cam translation]")
        trans_mm = EE_T_CAM[:3, 3] * 1000.0
        print(f"x={trans_mm[0]:.3f} mm, y={trans_mm[1]:.3f} mm, z={trans_mm[2]:.3f} mm")
        print(f"distance from TCP={np.linalg.norm(trans_mm):.3f} mm")

        cam_z_in_ee = EE_T_CAM[:3, 2]
        print("\n[camera optical z-axis in EE frame]")
        print(cam_z_in_ee)

        # Robot connect
        if USE_ROBOT:
            robot = rb.Cobot(ROBOT_IP)
            rc = rb.ResponseCollector()
            robot_data = rb.CobotData(ROBOT_IP)

            robot.set_operation_mode(
                rc,
                rb.OperationMode.Simulation if SIM_MODE else rb.OperationMode.Real
            )
            robot.set_speed_bar(rc, 0.3)
            robot.flush(rc)
            rc.error().throw_if_not_empty()
            debug("robot ready")
        else:
            debug("robot skipped")

        # Camera init
        if USE_CAMERA:
            calib = RealSenseCharucoHandEye(
                width=1280,
                height=720,
                fps=30,
                save_dir=str(SAVE_DIR),
            )
            debug("realsense ready")
        else:
            raise RuntimeError("USE_CAMERA=False이면 검증 불가")

        viewer_window = "HandEye Fixed Verification"
        cv2.namedWindow(viewer_window, cv2.WINDOW_NORMAL)

        # Initial pose
        initial_pose_vec = input_initial_pose_vec()

        if USE_ROBOT:
            print("\n[STEP 1] move to initial pose")
            arrived_pose = move_l_blocking(
                robot, rc, robot_data,
                initial_pose_vec,
                MOVE_SPEED,
                MOVE_ACCEL,
            )
            print("[STEP 1] arrived:", arrived_pose)

        # Board distance / radius
        print("\n[STEP 2] Check board detection. Press SPACE when ready.")
        measured_dist_mm = None

        while True:
            ok, rvec, tvec, msg = show_detection_viewer(
                calib,
                viewer_window,
                lines=[
                    "STEP 2: board detection check",
                    "SPACE: continue / q: quit",
                ]
            )

            if ok and tvec is not None:
                measured_dist_mm = float(np.linalg.norm(np.asarray(tvec).reshape(-1)) * 1000.0)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(" "):
                break
            elif key == ord("q") or key == 27:
                raise SystemExit("user quit")

        radius_default = measured_dist_mm if measured_dist_mm is not None else RADIUS_MM_DEFAULT
        radius_mm = input_radius_mm(radius_default)

        # Build poses
        pose_list = build_verify_pose_list(
            initial_pose_vec=initial_pose_vec,
            base_dist_mm=radius_mm,
            dist_offsets_mm=GRID_DIST_OFFSETS_MM,
            u_offsets_mm=GRID_U_OFFSETS_MM,
            v_offsets_mm=GRID_V_OFFSETS_MM,
        )
        pose_list = filter_unsafe_poses(pose_list)

        print("\n[VERIFY POSE LIST]")
        print(f"num poses = {len(pose_list)}")
        for i, p in enumerate(pose_list):
            print(f"{i:02d}: {np.array2string(p, precision=3)}")

        print("\nPress SPACE to start verification / q to quit")

        while True:
            show_detection_viewer(
                calib,
                viewer_window,
                lines=[
                    f"num verify poses: {len(pose_list)}",
                    "SPACE: start verification / q: quit",
                ]
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord(" "):
                break
            elif key == ord("q") or key == 27:
                raise SystemExit("user quit before verification")

        # Verification loop
        print("\n" + "=" * 70)
        print("START HAND-EYE FIXED VERIFICATION")
        print("=" * 70)

        for i, target_pose in enumerate(pose_list):
            print(f"\n[SAMPLE {i}/{len(pose_list)-1}]")

            # Move
            if USE_ROBOT:
                try:
                    arrived_tcp_pose = move_l_blocking(
                        robot, rc, robot_data,
                        target_pose,
                        MOVE_SPEED,
                        MOVE_ACCEL,
                    )
                except Exception as e:
                    print(f"[MOVE FAIL] sample={i}, err={e}")
                    continue
            else:
                arrived_tcp_pose = target_pose.copy()

            print(f"[ARRIVED TCP] {arrived_tcp_pose}")

            # Settling
            t0 = time.time()
            while time.time() - t0 < CAPTURE_DELAY_S:
                show_detection_viewer(
                    calib,
                    viewer_window,
                    lines=[
                        f"sample {i}/{len(pose_list)-1}",
                        "settling before capture...",
                    ]
                )
                cv2.waitKey(1)
                time.sleep(0.03)

            # Capture current TCP pose as base_T_ee
            if USE_ROBOT:
                cur_tcp_pose = get_current_tcp_pose_vec_mm(robot_data)
                if cur_tcp_pose is None:
                    print("[SKIP] cannot read current TCP")
                    continue
            else:
                cur_tcp_pose = arrived_tcp_pose.copy()

            base_T_ee = base_T_ee_from_tcp_pose_mm(cur_tcp_pose)

            # Detect and compute base_T_board
            sample, msg = capture_one_verification_sample(
                calib=calib,
                base_T_ee=base_T_ee,
                sample_idx=i,
            )

            if sample is None:
                print(f"[CAPTURE FAIL] sample={i}, {msg}")
                show_detection_viewer(
                    calib,
                    viewer_window,
                    lines=[
                        f"sample {i}: CAPTURE FAIL",
                        str(msg),
                    ]
                )
                time.sleep(0.5)
                continue

            samples.append(sample)

            board_pos_mm = np.asarray(sample["board_pos_base_mm"], dtype=np.float64)
            print(
                f"[CAPTURE OK] sample={i}, "
                f"base_board_xyz_mm = "
                f"[{board_pos_mm[0]:.3f}, {board_pos_mm[1]:.3f}, {board_pos_mm[2]:.3f}]"
            )

            show_detection_viewer(
                calib,
                viewer_window,
                lines=[
                    f"sample {i}: CAPTURE OK",
                    f"valid samples: {len(samples)}",
                    f"board base xyz mm: {board_pos_mm[0]:.1f}, {board_pos_mm[1]:.1f}, {board_pos_mm[2]:.1f}",
                ]
            )
            cv2.waitKey(1)
            time.sleep(0.4)

        # Summary
        summary = summarize_verification(samples)
        print_summary(summary)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = SAVE_DIR / f"handeye_fixed_verify_{timestamp}.json"

        result = {
            "summary": summary,
            "samples": samples,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] {out_json}")

        print("\nViewer 유지 중. q 누르면 종료.")
        while True:
            show_detection_viewer(
                calib,
                viewer_window,
                lines=[
                    "VERIFICATION DONE",
                    f"valid samples: {len(samples)}",
                    f"saved: {out_json.name}",
                    "q: quit",
                ]
            )

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break

    except SystemExit as e:
        print(f"[EXIT] {e}")

    except Exception as e:
        print("\n[EXCEPTION]")
        print(type(e).__name__, e)
        traceback.print_exc()

    finally:
        cv2.destroyAllWindows()

        if calib is not None:
            try:
                calib.stop()
            except Exception as e:
                print("[FINALLY] calib.stop error:", e)

        debug("program end")