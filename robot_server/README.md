# Unitree G1 机器人端轻量部署指南 (Robot Server)

本目录包含了部署在 **Unitree G1 机器人本体（机载电脑 Jetson / 工控机）** 上的轻量化服务组件。

> [!NOTE]
> **设计特点**：
> - **极简零侵入**：机器人端**不需要**克隆完整的 LeRobot 庞大仓库，也**不需要**安装 PyTorch / CUDA 等庞大依赖。
> - **Python 3.8+ 原生兼容**：适配 G1 出厂的 Ubuntu 系统自带 Python 环境。
> - **双独立通信**：
>   1. **电机 DDS-ZMQ 桥接**（`motor_server.py`）：负责 6000 端口动作接收与 6001 端口关节状态广播。
>   2. **视觉推流**（`server.py`）：负责 5556 端口机载摄像头画面广播（RGB 640x480 @ 30fps）。
> - **一键启停**：通过 `start_server.sh` 脚本统一管理，按 `Ctrl + C` 自动清理所有后台进程。

---

## 1. 目录结构

```text
robot_server/
├── start_server.sh              # 机器人端一键启停主脚本
├── camera_server/
│   ├── motor_server.py          # 电机 DDS-to-ZMQ 状态/控制中转服务
│   └── server.py                # 摄像头 OpenCV-to-ZMQ base64 JPEG 推流服务
└── README.md                    # 本部署文档
```

---

## 2. 新机器人快速部署步骤（仅需 3 分钟）

### 步骤 1：同步文件到机器人
在上位机终端执行 `scp` 将本目录直接复制到机器人主目录下的 `~/lerobot`：
```bash
# 在上位机执行
scp -r robot_server unitree@192.168.123.164:~/lerobot
```

### 步骤 2：登录机器人并安装基础依赖
```bash
ssh unitree@192.168.123.164
# 密码: 123

# 安装 Python 基础通信库（仅 pyzmq 和 opencv）
python3 -m pip install pyzmq opencv-python

# 安装宇树 SDK2 (若机器人上未安装)
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
cd ~/unitree_sdk2_python
python3 -m pip install -e .
```

---

## 3. 运行服务命令

登录机器人后，进入 `~/lerobot` 目录：

### 模式 A：使用真实机载摄像头（默认模式）
同时启动 **电机状态桥接 (6000/6001)** 和 **机载摄像头推流 (5556)**：
```bash
cd ~/lerobot
./start_server.sh
```
> **视场角 (FOV) 与分辨率控制**：
> - 脚本默认以 **1280x720 (16:9 全视场角 @ 30FPS)** 进行硬件采集，并**原画质直接推流至电脑上位机**（不进行机载压缩与形变，具体分辨率变换和数据预处理由电脑上位机完成）。
> - 若需调整分辨率或在机器人端预处理：
>   ```bash
>   # 默认启动：1280x720 纯净全视场角推流
>   ./start_server.sh
>
>   # 可选：如果希望机器人端直接下采样到 640x480 并保留全 FOV
>   ./start_server.sh --capture-width=1280 --capture-height=720 --width=640 --height=480 --mode=resize
>
>   # 可选：如果希望使用极致画质 1080P 全画幅采集推流
>   ./start_server.sh --capture-width=1920 --capture-height=1080 --width=1920 --height=1080
>   ```
> - 若摄像头设备号变动（例如换到 `/dev/video4`），可带参数运行：`./start_server.sh --device=4`

---

### 模式 B：纯电机模式（用于上位机视频回放测试）
仅启动 **电机状态桥接 (6000/6001)**，不占用物理摄像头硬件：
```bash
cd ~/lerobot
./start_server.sh --no-camera
```

---

### 模式 C：纯机载视觉模式（只推流相机，不连接电机）
仅启动 **机载摄像头推流 (5556)**，完全不运行电机 DDS 桥接：
```bash
cd ~/lerobot
./start_server.sh --only-camera
# 或直接运行底层推流服务：
python3 camera_server/server.py --device 2 --port 5556
```

---

## 4. 退出服务
在运行 `start_server.sh` 的终端中按下 **`Ctrl + C`**，脚本将自动拦截退出信号并安全终止所有后台推流与电机通信进程。
