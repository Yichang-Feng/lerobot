# Unitree G1 机器人端轻量部署指南 (Robot Server)

本目录包含了部署在 **Unitree G1 机器人本体（机载电脑 Jetson / 工控机）** 上的轻量化服务组件。

> [!NOTE]
> **设计特点**：
> - **极简零侵入**：机器人端**不需要**克隆完整的 LeRobot 庞大仓库，也**不需要**安装 PyTorch / CUDA 等庞大依赖。
> - **Python 3.8+ 原生兼容**：适配 G1 出厂的 Ubuntu 系统自带 Python 环境。
> - **双独立通信**：
>   1. **电机 DDS-ZMQ 桥接**（`motor_server.py`）：负责 6000 端口动作接收与 6001 端口关节状态广播。
>   2. **视觉推流**（`server.py`）：负责 5555 端口机载摄像头画面广播（RGB 640x480 @ 30fps）。
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
同时启动 **电机状态桥接 (6000/6001)** 和 **机载摄像头推流 (5555)**：
```bash
cd ~/lerobot
./start_server.sh
```
> 若摄像头设备号变动（例如换到 `/dev/video4`），可带参数运行：`./start_server.sh --device=4`

---

### 模式 B：纯电机模式（用于上位机视频回放测试）
仅启动 **电机状态桥接 (6000/6001)**，不占用物理摄像头硬件：
```bash
cd ~/lerobot
./start_server.sh --no-camera
```

---

## 4. 退出服务
在运行 `start_server.sh` 的终端中按下 **`Ctrl + C`**，脚本将自动拦截退出信号并安全终止所有后台推流与电机通信进程。
