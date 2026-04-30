from time import time, sleep
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

def rpy_to_rotmat(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # ZYX (yaw-pitch-roll) 순서
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1, 0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    # R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

# --------- desired ---------
desired_xpos_tcp = np.array([0.0, -0.5, 0.1])
desired_rpy = np.array([90.0, 0.0, 0.0])  # roll, pitch, yaw 입력
# -----------------------------

current_dir = Path(__file__).parent
model_path = str(current_dir / "rb5.xml")
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

d.qpos[:] = [-0.5, 0.0, 1.0, 0.0, 0.0, 0.0]

M = np.zeros((m.nv, m.nv), dtype=np.float64)
G = np.zeros((m.nv), dtype=np.float64)

jacp = np.zeros((3, m.nv), dtype=np.float64)
jacr = np.zeros((3, m.nv), dtype=np.float64)

C0 = np.zeros((6,6))

K_a = 50.0
zeta_a = 5.0

K_o = 30.0
zeta_o = 3.0

with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time()
    while viewer.is_running():
        t = time() - t0

        
        mujoco.mj_fullM(m, M, d.qM)

        qvel_backup = deepcopy(d.qvel)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, G) 
        d.qvel[:] = qvel_backup[:]
        mujoco.mj_forward(m, d)        

        np.fill_diagonal(C0, np.sum(np.abs(M[0:6, 0:6]), axis=1))

        tcp_site_id = m.site("tcp").id
        mujoco.mj_jacSite(m, d, jacp, jacr, tcp_site_id)
        jacp0 = deepcopy(jacp[:, 0:6])
        jacr0 = deepcopy(jacr[:, 0:6])
        # print(f"jr : {jacr0}")

        # Position Error
        #xpos_err0 = d.site("tcp").xpos - desired_xpos_tcp
        xpos_err0 = desired_xpos_tcp - d.site("tcp").xpos

        # Orientation Error (RPY 입력 반영)
        R_current = d.site(tcp_site_id).xmat.reshape(3, 3)
        # print(f"Ro : {R_current}")
        
        R_desired = rpy_to_rotmat(*desired_rpy)
        # print(f"Rd : {R_desired}")
        # R_desired = R_desired@R_desired
        ori_err0 = (
            np.cross(R_desired[:, 0], R_current[:, 0]) +
            np.cross(R_desired[:, 1], R_current[:, 1]) +
            np.cross(R_desired[:, 2], R_current[:, 2])
        ) # e  = (x X xd) + (y X yd) + (z X zd)
        
        # Angular Velocity
        w0 = jacr0 @ d.qvel[0:6]
        # print(f"wo : {w0}")

        # Orientation Force
        F_ori_0 = (K_o * ori_err0) #+ (zeta_o * np.sqrt(K_o) * w0)

        # Linear Velocity
        xpos_dot0 = jacp0 @ d.qvel[0:6]

        # Linear Force 
        #force0 = (K_a * xpos_err0) + (zeta_a * np.sqrt(K_a) * xpos_dot0)
        force0 = (K_a * xpos_err0) - (zeta_a * np.sqrt(K_a) * xpos_dot0)

        # Torque (PD + Gravity + Damping + Orientation)
        """
        torque0 = (- 0 * C0 @ d.qvel[0:6] 
                   - 1 * jacp0.T @ force0 
                   + 1 * G[0:6] 
                   - 0 * jacr0.T @ F_ori_0)
        """
        joint_damp = 0.05 * (C0 @ d.qvel[0:6])
        #torque0 = (jacp0.T @ force0 + G[0:6])
        torque0 = (jacp0.T @ force0 + G[0:6] - joint_damp)
        # print(f"torque 0 : {torque0}")
        
        max_torque = 50
        d.ctrl[0:6] = np.clip(torque0, -max_torque, max_torque)
        # d.qvel[:] = 0
        print(f"Taget torque : {d.ctrl[0:6]}")


        mujoco.mj_step(m, d)
        viewer.sync()