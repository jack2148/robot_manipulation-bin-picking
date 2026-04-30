import time
from enum import Enum, auto

import numpy as np
import rbpodo as rb


class State(Enum):
    MOVE2PEG = auto()
    CAPTHER_PEG = auto()
    DOWN_PEG = auto()
    GRIP = auto()
    UP = auto()
    MOVE_HOLE = auto()
    CAPTHER_HOLE = auto()
    DOWN_HOLE = auto()
    DONE = auto()


ROBOT_IP = "192.168.1.10"
#ROBOT_MODE = rb.OperationMode.Real
ROBOT_MODE = rb.OperationMode.Simulation

"""
PEG_APPROACH_POSE = np.array([450.0, 0.0, 400.0, 90.0, 0.0, 44.0], dtype=float)
HOLE_APPROACH_POSE = np.array([-600.0, 0.0, 400.0, 90.0, 0.0, 44.0], dtype=float)
"""
# MOVE2PEG / MOVE_HOLE 에서 사용할 joint target
PEG_APPROACH_JOINT = np.array([14.27, -9.05, 96.26, 2.78, 90.0, 30.09], dtype=float)
HOLE_APPROACH_JOINT = np.array([-169.35, 11.76, 74.92, 3.32, 90.0, 35.42], dtype=float)

PEG_APPROACH_POSE = np.array([-120.0, -447.0, 369.0, 90.0, 0.0, 44.0], dtype=float)
HOLE_APPROACH_POSE = np.array([120.0, -447.0, 369.0, 90.0, 0.0, 44.0], dtype=float)

# 아래로 내려가는 양
PEG_DOWN_DZ = -80.0
HOLE_DOWN_DZ = -80.0

# 내려간 현재 위치 기준으로 위로 올리는 양
UP_AFTER_GRIP_DZ = 80.0
UP_AFTER_HOLE_DZ = 80.0

MOVE_J_SPEED = 60
MOVE_J_ACCEL = 80

MOVE_L_SPEED = 300
MOVE_L_ACCEL = 400


def set_mode(robot: rb.Cobot):
    rc = rb.ResponseCollector()
    robot.set_operation_mode(rc, ROBOT_MODE)
    rc.error().throw_if_not_empty()

    rc = rb.ResponseCollector()
    robot.set_speed_bar(rc, 0.5)
    rc.error().throw_if_not_empty()


def wait_motion(robot: rb.Cobot, timeout: float = 2.0):
    rc = rb.ResponseCollector()

    robot.flush(rc)
    rc.error().throw_if_not_empty()

    rc = rb.ResponseCollector()
    if robot.wait_for_move_started(rc, timeout).type() == rb.ReturnType.Success:
        rc.error().throw_if_not_empty()

        rc = rb.ResponseCollector()
        robot.wait_for_move_finished(rc)
        rc.error().throw_if_not_empty()
    else:
        rc.error().throw_if_not_empty()


def move_j_and_wait(robot: rb.Cobot, joint: np.ndarray, speed=MOVE_J_SPEED, accel=MOVE_J_ACCEL):
    print(f"[MOVE_J] target joint = {joint}")
    rc = rb.ResponseCollector()
    robot.move_j(rc, joint, speed, accel)
    rc.error().throw_if_not_empty()
    wait_motion(robot)


def move_l_and_wait(robot: rb.Cobot, pose: np.ndarray, speed=MOVE_L_SPEED, accel=MOVE_L_ACCEL,
                    ref: rb.ReferenceFrame = rb.ReferenceFrame.Base):
    print(f"[MOVE_L] target pose = {pose}")
    rc = rb.ResponseCollector()
    robot.move_l(rc, pose, speed, accel, ref)
    rc.error().throw_if_not_empty()
    wait_motion(robot)


def offset_pose_z(pose: np.ndarray, dz: float) -> np.ndarray:
    new_pose = pose.copy()
    new_pose[2] += dz
    return new_pose


def make_down_pose(base_pose: np.ndarray, down_dz: float) -> np.ndarray:
    return offset_pose_z(base_pose, down_dz)


def make_up_pose_from_current(current_pose: np.ndarray, up_dz: float) -> np.ndarray:
    return offset_pose_z(current_pose, up_dz)


def close_gripper():
    print("[GRIP] close gripper")
    time.sleep(0.5)


def _main():
    robot = rb.Cobot(ROBOT_IP)

    state = State.MOVE2PEG

    peg_approach_pose = PEG_APPROACH_POSE.copy()
    hole_approach_pose = HOLE_APPROACH_POSE.copy()

    peg_down_pose = None
    hole_down_pose = None

    up_target_pose = None
    next_state_after_up = None

    try:
        set_mode(robot)

        while state != State.DONE:
            print(f"[STATE] {state.name}")

            if state == State.MOVE2PEG:
                move_j_and_wait(robot, PEG_APPROACH_JOINT.copy())
                peg_approach_pose = PEG_APPROACH_POSE.copy()   # 이 pose가 반드시 위 joint와 같은 점이어야 함
                state = State.CAPTHER_PEG

            elif state == State.CAPTHER_PEG:
                peg_approach_pose = PEG_APPROACH_POSE.copy()
                state = State.DOWN_PEG

            elif state == State.DOWN_PEG:
                peg_down_pose = peg_approach_pose.copy()
                peg_down_pose[2] += PEG_DOWN_DZ
                move_l_and_wait(robot, peg_down_pose, speed=30, accel=60)
                state = State.GRIP

            elif state == State.GRIP:
                close_gripper()

                # 내려간 위치 기준으로 위로 올리기
                up_target_pose = make_up_pose_from_current(peg_down_pose, UP_AFTER_GRIP_DZ)
                next_state_after_up = State.MOVE_HOLE
                state = State.UP

            elif state == State.UP:
                if up_target_pose is None or next_state_after_up is None:
                    raise RuntimeError("UP 상태 진입 전에 up_target_pose / next_state_after_up 설정이 필요합니다.")

                move_l_and_wait(robot, up_target_pose)
                state = next_state_after_up
                up_target_pose = None
                next_state_after_up = None

            elif state == State.MOVE_HOLE:
                move_j_and_wait(robot, HOLE_APPROACH_JOINT.copy())
                state = State.CAPTHER_HOLE

            elif state == State.CAPTHER_HOLE:
                hole_approach_pose = HOLE_APPROACH_POSE.copy()
                state = State.DOWN_HOLE

            elif state == State.DOWN_HOLE:
                hole_down_pose = make_down_pose(hole_approach_pose, HOLE_DOWN_DZ)
                move_l_and_wait(robot, hole_down_pose)

                # 내려간 위치 기준으로 위로 올리기
                up_target_pose = make_up_pose_from_current(hole_down_pose, UP_AFTER_HOLE_DZ)
                next_state_after_up = State.DONE
                state = State.UP

            else:
                raise RuntimeError(f"정의되지 않은 상태입니다: {state}")

        print("[STATE] DONE")

    finally:
        print("Exit")


if __name__ == "__main__":
    _main()