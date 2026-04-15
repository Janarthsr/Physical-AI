

https://github.com/user-attachments/assets/5c2168a1-b1e8-4b8e-859c-a7a846a37d84



https://github.com/user-attachments/assets/5024d673-9baa-4b88-a74e-270bf5efb269



# Physical-AI


A Physical AI Humanoid simulation using MuJoCo and MediaPipe for real-time human pose mirroring, dynamic walking, and YOLOv8-based object detection.
XP Robot: Real-time Humanoid Mirroring & Locomotion
An end-to-end Physical AI framework that maps human movements from a webcam to a MuJoCo humanoid model. Features include PD-controlled balance, an oscillatory walking gait, and vision-integrated target tracking.
Real-time Teleoperation: Uses MediaPipe to map 3D human arm landmarks to robot actuators with low latency.

1)Dynamic Physics: Implemented in MuJoCo with a custom PD controller for balance and stability.

2)Vision-Guided AI: Integrated YOLOv8 for real-time object identification and distance estimation.

3)Robust Setup: Includes a setup.bat for automated environment configuration and Python 3.11 optimization.

1. Header and Intro
Markdown
# XP-Robot-Physical-AI
**Task 4: Dynamic Humanoid Walking + Human Pose Mirroring**

---

## 🤖 Project Overview
This project implements a **Physical AI** system where a humanoid robot in a MuJoCo simulation mirrors human movements captured via webcam and performs dynamic tasks like object detection and balanced locomotion.
2. The Structured File List
Markdown
---

### 📂 Repository Structure
* **`main.py`** — The core engine managing Camera, Logic, and Simulation threads.
* **`XP_robot_v2.xml`** — High-fidelity robot model with $0.0005s$ physics timestep.
* **`setup.bat`** — One-click environment setup for Windows.
* **`requirements.txt`** — List of all Python dependencies.
* **`meshes/`** — Physical 3D components of the humanoid.
3. Key Features & Controls
Markdown
---

## 🚀 Key Features
* **Pose Mirroring:** Real-time mapping of human arm landmarks (MediaPipe) to robot actuators.
- **Dynamic Stability:** IMU-based fall prevention system and PD control for the legs.
- **Vision Integration:** YOLOv8-based object detection for identifying targets.

## 🎮 Controls
- `B`: Toggle IMU Balance Mode (prevents falling).
- `W`: Toggle Walking Mode (initiates gait cycle).
- `ESC`: Close the simulation safely.

