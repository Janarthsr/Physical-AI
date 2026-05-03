

https://github.com/user-attachments/assets/5c2168a1-b1e8-4b8e-859c-a7a846a37d84



https://github.com/user-attachments/assets/5024d673-9baa-4b88-a74e-270bf5efb269



# 🤖 XP-Humanoid: End-to-End Physical AI Framework
[![NVIDIA RTX 4050](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050-green.svg)](https://www.nvidia.com/en-in/geforce/laptops/rtx-4050/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-blue.svg)](https://mujoco.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow.svg)](https://www.python.org/)

An advanced **Physical AI** humanoid system that bridges the gap between human intent and robotic execution. This framework integrates **Computer Vision**, **Real-time Teleoperation**, and **Dynamic Physics Simulation** to achieve low-latency pose mirroring and autonomous locomotion.

## 🌟 Key Technical Features
*   **Real-time Human-to-Robot Mapping:** Leverages **MediaPipe** to extract 3D skeletal landmarks from a standard webcam, mapping them to MuJoCo actuators with optimized PD control loops.
*   **CUDA-Accelerated Vision:** Integrated **YOLOv8** for real-time spatial awareness, object classification, and distance estimation for autonomous target tracking.
*   **Dynamic Stability & Locomotion:** Features a custom **IMU-based fall prevention system** and oscillatory gait cycles for stable bipedal walking.
*   **High-Fidelity Simulation:** Optimized for a **0.0005s physics timestep** to ensure sub-millisecond precision in contact dynamics.

---

## 💻 Hardware & Performance
This project is optimized for high-compute workloads and is designed to be portable to **NVIDIA Jetson** edge platforms.
*   **Inference & Training GPU:** NVIDIA GeForce RTX 4050 Laptop GPU.
*   **Compute Stack:** CUDA-accelerated processing for simultaneous Vision (YOLOv8) and Physics (MuJoCo) pipelines.
*   **Simulation Environment:** Optimized for Windows 11 with Python 3.11 and high-frequency sensor polling.

---

## 📂 Repository Structure
*   **`main.py`** — Multithreaded core managing the vision-logic-simulation pipeline.
*   **`XP_robot_v2.xml`** — The Digital Twin; a high-fidelity robot model with comprehensive MJCF definitions.
*   **`setup.bat`** — Automated environment deployment for rapid testing.
*   **`meshes/`** — High-resolution STL components for the humanoid chassis.

---

## 🎮 Deployment & Controls
### One-Click Setup
```bash
./setup.bat
```
### Live Operation
| Key | Action |
| :--- | :--- |
| **B** | Toggle **IMU Balance Mode** (Active Fall Prevention) |
| **W** | Initiate **Gait Cycle** (Dynamic Walking) |
| **ESC** | Safe Shutdown of Simulation Threads |

---

## 🚀 Future Roadmap: Sim-to-Real
The architecture of this framework is designed for seamless deployment onto **NVIDIA Jetson Orin Nano**. Future iterations aim to replace the simulation environment with real-world **SO-101** or custom humanoid hardware using the same ACT-policy logic and vision-guided control.
