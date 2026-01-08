# Drone_project

# 🌊 SkyGuard: Autonomous Maritime Rescue Drone

**SkyGuard** is an advanced autonomous system designed for high-stakes search and rescue (SAR) missions in maritime environments. It integrates **PX4 Autopilot**, **ROS 2 Humble**, and **YOLOv8** to automate the detection of missing persons at sea.



## 🛠️ Key Technical Features
- **Autonomous Spiral Search:** Custom flight logic using MAVSDK to cover expanding radii.
- **AI Vision Pipeline:** Real-time human detection with YOLOv8n optimized for water backgrounds.
- **SDF Model Composition:** Integrated an **OakD-Lite camera** onto an **x500 drone** frame using fixed joints.
- **Web Reporting:** Automatic GPS and image upload to a Next.js dashboard via REST API.

## 🔧 Engineering Challenges & Solutions

### 🎨 1. Visual & Mesh Optimization (Blender)
- **Problem:** Incorrect textures and heavy model meshes caused rendering issues and simulation lag.
- **Solution:** Processed 3D models in **Blender** to fix texture paths and deleted non-essential elements to optimize physics performance in Gazebo.

### 🚀 2. Flight Stability & RTL Tuning
- **Problem:** Drone arming failures due to collision at spawn and excessive altitude gain during RTL.
- **Solution:** - Adjusted `PX4_GZ_MODEL_POSE` for safe spawning.
  - Programmatically set `RTL_RETURN_ALT` to **10m** to ensure a safe, low-altitude return mission.

### 🧵 3. System Synchronization (Multi-threading)
- **Problem:** AI inference caused delays in flight command execution (ACK lost).
- **Solution:** Implemented **Multi-threading** to isolate the computer vision node from the MAVSDK control loop, ensuring stable flight.

## 📂 Project Structure
- `/src`: Autonomous mission scripts (Python/MAVSDK).
- `/models`: Custom SDF and mesh files (x500 + OakD-Lite).
- `/worlds`: Custom maritime Gazebo environment.

## 🚀 How to Run
1. Launch PX4 SITL and Gazebo Harmonic.
2. Start the `ros_gz_bridge` for camera streaming.
3. Execute `python3 main_mission.py`.
