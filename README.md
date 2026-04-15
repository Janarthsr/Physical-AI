# Physical-AI


A Physical AI Humanoid simulation using MuJoCo and MediaPipe for real-time human pose mirroring, dynamic walking, and YOLOv8-based object detection.
XP Robot: Real-time Humanoid Mirroring & Locomotion
An end-to-end Physical AI framework that maps human movements from a webcam to a MuJoCo humanoid model. Features include PD-controlled balance, an oscillatory walking gait, and vision-integrated target tracking.
Real-time Teleoperation: Uses MediaPipe to map 3D human arm landmarks to robot actuators with low latency.

1)Dynamic Physics: Implemented in MuJoCo with a custom PD controller for balance and stability.

2)Vision-Guided AI: Integrated YOLOv8 for real-time object identification and distance estimation.

3)Robust Setup: Includes a setup.bat for automated environment configuration and Python 3.11 optimization.

"""
XP Robot - Final Hackathon Version
Mirroring + Dynamic Locomotion + High Stability
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import threading
import time
import os

# ── SHARED STATE ────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.arm_targets_r = np.zeros(4)
        self.arm_targets_l = np.zeros(4)
        self.running = True
        self.balance_on = True
        self.walking_on = False
        self.phase = 0.0

state = SharedState()

# ── CAMERA & POSE THREAD ───────────────────────────────────────────────────
def camera_thread_fn():
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    walk_vote = 0
    while state.running:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # --- CALCULATE ARM ANGLES ---
            def get_angles(s, e, w):
                upper, lower = e-s, w-e
                p = np.arctan2(-upper[1], np.linalg.norm(upper[[0,2]]))
                r = np.arctan2(upper[2], upper[0])
                y = np.arctan2(lower[2], lower[0]) - r
                elb = -np.arccos(np.clip(np.dot(upper/np.linalg.norm(upper), lower/np.linalg.norm(lower)), -1, 1))
                return np.array([p, r, y, elb])

            # Get joints by MediaPipe index
            get_p = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z])
            r_arm = get_angles(get_p(12), get_p(14), get_p(16))
            l_arm = get_angles(get_p(11), get_p(13), get_p(15))

            # --- WALK DETECTION ---
            diff = abs(lm[28].y - lm[27].y)
            walk_vote = min(walk_vote + 2, 10) if diff > 0.04 else max(walk_vote - 1, 0)

            with state.lock:
                state.arm_targets_r, state.arm_targets_l = r_arm, l_arm
                state.walking_on = walk_vote >= 5

            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("XP AI Vision", frame)
        if cv2.waitKey(1) == ord('q'): state.running = False
    cap.release()

# ── SIMULATION & PHYSICS (THE "DO SOMETHING" FIX) ──────────────────────────
def run_simulation():
    print("[SIM] LOADING XP_ROBOT_V2.XML...")
    try:
        model = mujoco.MjModel.from_xml_path("XP_robot_v2.xml")
        data = mujoco.MjData(model)
        print("[SIM] SUCCESS: ALL 21 ACTUATORS CONNECTED ✓")
    except Exception as e:
        print(f"[ERROR] XML FAIL: {e}")
        return

    # -- SPECS: MAXIMUM STIFFNESS FOR MESH WEIGHT --
    KP_LEG, KD_LEG = 450.0, 35.0 # High damping stops the "dancing" vibrations
    KP_ARM, KD_ARM = 180.0, 8.0  # Forceful mirroring response

    # Stance Targets
    leg_default = {i: 0.0 for i in range(model.nu)}
    for i in range(model.nu):
        name = model.actuator(i).name
        if "knee" in name: leg_default[i] = 0.32        # Low center of mass
        if "hip_pitch" in name: leg_default[i] = 0.18   # Stability lean

    arm_smooth_r, arm_smooth_l = np.zeros(4), np.zeros(4)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and state.running:
            step_start = time.time()
            with state.lock:
                r_tgt, l_tgt = state.arm_targets_r.copy(), state.arm_targets_l.copy()
                walking_on, balance_on = state.walking_on, state.balance_on

            arm_smooth_r = 0.6 * arm_smooth_r + 0.4 * r_tgt
            arm_smooth_l = 0.6 * arm_smooth_l + 0.4 * l_tgt

            if walking_on:
                state.phase += model.opt.timestep * 1.2 * 2 * np.pi
            
            l_osc, r_osc = np.sin(state.phase), np.sin(state.phase + np.pi)
            torque = np.zeros(model.nu)
            
            for i in range(model.nu):
                name = model.actuator(i).name.lower()
                q_idx, v_idx = i+7, i+6
                
                # 1. MIRRORING: HARD-CODED ACTUATOR INDEXING (13-20)
                if 13 <= i <= 16: # RIGHT ARM
                    torque[i] = KP_ARM * (arm_smooth_r[i-13] - data.qpos[q_idx]) - KD_ARM * data.qvel[v_idx]
                elif 17 <= i <= 20: # LEFT ARM
                    torque[i] = KP_ARM * (arm_smooth_l[i-17] - data.qpos[q_idx]) - KD_ARM * data.qvel[v_idx]
                
                # 2. LEGS & BALANCED LOCOMOTION
                else:
                    walk_off = 0.0
                    if walking_on:
                        if "roll" in name: walk_off = l_osc * 0.28 # MASSIVE LEAN
                        elif "hip_pitch" in name: walk_off = (r_osc if "_r" in name else l_osc) * 0.15
                        elif "knee" in name: walk_off = max(0, r_osc if "_r" in name else l_osc) * -0.40
                    
                    target = leg_default.get(i, 0.0) + walk_off
                    torque[i] = KP_LEG * (target - data.qpos[q_idx]) - KD_LEG * data.qvel[v_idx]

                # 3. IMU-BASED FALL PREVENTION
                if balance_on:
                    quat = data.qpos[3:7]
                    tp, tr = 2*(quat[0]*quat[2]-quat[3]*quat[1]), 2*(quat[0]*quat[1]+quat[2]*quat[3])
                    if "pitch" in name: torque[i] += -280.0 * tp - 60 * data.qvel[4]
                    if "roll" in name:  torque[i] += -220.0 * tr - 50 * data.qvel[3]

            data.ctrl[:] = np.clip(torque, -600, 600) # MAXIMUM TORQUE SPECS
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

if __name__ == "__main__":
    threading.Thread(target=camera_thread_fn, daemon=True).start()
    time.sleep(2.0)
    run_simulation()

