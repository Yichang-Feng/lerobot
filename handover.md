# LeRobot + Unitree G1 + PI0.5 实机与仿真部署交接文档 (HANDOVER.MD)

- **更新时间**: 2026-09-01
- **工作目录**: `/home/yichangfeng/lerobot`
- **上位机环境**: `/home/yichangfeng/miniforge3/envs/lerobot` (Python 3.12, GPU Workstation IP: `192.168.123.213`)
- **机器人端环境**: Unitree G1 机载电脑 (`unitree@192.168.123.164`, Ubuntu 20.04, Python 3.8)

---

## 1. 项目架构与通信拓扑

### 1.1 总体架构
本项目在 **LeRobot** 框架下，结合 **PI0.5 策略模型（`model/box_move_blue` / `model/box_pick`）** 与 **GrootLocomotionController / SonicWholeBodyController**，实现对 **Unitree G1 (29-DoF)** 人形机器人的控制。

- **输入**: 
  - `observation.images.global_view`: 480×640×3 RGB 图像（机载 RealSense 摄像头或 ZMQ 视频流）。
  - `observation.state`: 29 维关节位置状态向量。
- **输出**: 
  - 18 维 Action 向量（前 14 维为双臂关节目标角度，后 4 维为 `remote.lx`, `remote.ly`, `remote.rx`, `remote.ry` 遥控速度指令）。
- **底层控制**: 
  - 50Hz 独立控制线程运行平衡控制器（GR00T / SONIC），驱动下肢与腰部 15 个关节维持直立平衡，双臂 14 关节执行策略目标。

### 1.2 网络与通信拓扑
```text
┌──────────────────────────────────────────────┐                ┌──────────────────────────────────────────────┐
│         本机上位机 (GPU Workstation)          │                │            Unitree G1 机器人本体             │
│        IP: 192.168.123.213                   │                │          IP: 192.168.123.164                 │
│        网卡: enx6c1ff724495a                 │                │                                              │
│                                              │ 千兆以太网直连  │  【机载轻量服务 robot_server】                │
│  【lerobot-rollout 上位机推理】              │◄──────────────►│   ├─ ZMQ Port 6000 (PULL 接收 LowCmd)        │
│   ├─ PI0.5 Policy (双臂 14-DoF 目标)         │                │   ├─ ZMQ Port 6001 (PUB 广播 LowState)       │
│   ├─ GrootLocomotionController (50Hz 腰腿)   │                │   └─ ZMQ Port 5555 (PUB 广播机载摄像头帧)     │
│   └─ 键盘交互 ('s'=STAND / 'w'=WALK)          │                │  【底层 DDS】                                │
│                                              │                │   └─ 宇树原厂电机执行器 (29-DoF)             │
└──────────────────────────────────────────────┘                └──────────────────────────────────────────────┘
```

---

## 2. 仓库核心文件与目录结构

### 2.1 上位机核心文件
- **`REAL_Deploy.md`**: 实机多终端部署 SOP 与网络配置说明文档。
- **`deploy_real_g1.sh`**: 真实机载相机模式（模式 A）一键启动脚本。
- **`verify_video_rollout_real.sh`**: 实机动作 + 视频回放模式（模式 B）一键启动脚本。
- **`verify_video_rollout.sh`**: 纯仿真回放验证脚本。
- **`src/lerobot/robots/unitree_g1/`**: G1 机器人控制实现（`unitree_g1.py`, `unitree_sdk2_socket.py`, `config_unitree_g1.py`, 控制器目录 `controllers/`）。
- **`src/lerobot/cameras/zmq/`**: ZMQ 相机接收与推流模块（`camera_zmq.py`, `stream_dataset_video_zmq.py`, `image_server.py`）。

### 2.2 机器人端轻量服务文件 (`robot_server/`)
为避免在机器人端克隆完整的 LeRobot 仓库或安装 PyTorch/CUDA 庞大依赖，已将机器人机载服务独立解耦整理至 `robot_server/`：
- **`robot_server/start_server.sh`**: 机载一键启停总控制脚本（支持 `--no-camera`，带 `Ctrl+C` 信号捕获与进程清理）。
- **`robot_server/camera_server/motor_server.py`**: 独立电机 DDS-to-ZMQ 桥接服务（监听 6000 端口，广播 6001 端口）。
- **`robot_server/camera_server/server.py`**: 摄像头 OpenCV-to-ZMQ 推流服务（采集 `/dev/video2`，广播 5555 端口）。
- **`robot_server/README.md`**: 机器人机载端部署与运行说明文档。

---

## 3. 当前运行状态与待解决问题记录

### 3.1 当前执行命令
在上位机终端执行：
```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --interpolation_multiplier=2 \
    --policy.path=model/box_pick \
    --policy.device=cuda \
    --robot.type=unitree_g1 \
    --robot.is_simulation=true \
    --robot.controller=GrootLocomotionController \
    --robot.locomotion_mode=walk \
    --robot.cameras='{"global_view": {"type": "zmq", "server_address": "localhost", "port": 5556, "camera_name": "head_camera", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}}' \
    --task="move blue box" \
    --duration=1000 \
    --fps=25 \
    --display_data=false
```

### 3.2 运行现象与输出日志

#### 控制台实时输出日志
```text
INFO 2026-09-01 17:48:48 itree_g1.py:346 Controller actual rate: 49.5Hz (target: 50.0Hz)
INFO 2026-09-01 17:48:49 on_queue.py:272 Indexes diff is not equal to real delay. indexes_diff=5, real_delay=6
INFO 2026-09-01 17:48:51 on_queue.py:272 Indexes diff is not equal to real delay. indexes_diff=7, real_delay=8
INFO 2026-09-01 17:48:53 itree_g1.py:346 Controller actual rate: 49.5Hz (target: 50.0Hz)
INFO 2026-09-01 17:48:53 on_queue.py:272 Indexes diff is not equal to real delay. indexes_diff=5, real_delay=6
INFO 2026-09-01 17:48:55 on_queue.py:272 Indexes diff is not equal to real delay. indexes_diff=4, real_delay=5
INFO 2026-09-01 17:48:58 itree_g1.py:346 Controller actual rate: 49.6Hz (target: 50.0Hz)
```

#### 现象描述
1. **下肢与平衡状态**：
   - 下肢后台控制线程稳定运行在 49.5Hz ~ 49.6Hz（目标 50.0Hz），机器人全身直立与站立平衡维持正常。
2. **上肢手臂动作状态**：
   - 机器人的手臂动作未达到平滑连续状态，依然存在明显的周期性卡顿与小幅度抽动现象。
   - 控制台以约每 1~2 秒一次的频率持续输出 `on_queue.py:272 Indexes diff is not equal to real delay`，且每次输出时 `indexes_diff` 与 `real_delay` 均存在数值偏差（如 5 与 6、7 与 8、4 与 5）。
