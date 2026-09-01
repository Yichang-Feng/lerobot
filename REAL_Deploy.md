# Unitree G1 具身策略实机部署操作指南

本文档结合 **SonicStar 实机部署规范**（网卡 `enx6c1ff724495a`，本机 IP `192.168.123.213`，机器人 IP `192.168.123.164`）与 **LeRobot 项目架构**，详细指导如何将已在 MuJoCo 仿真中验证的 **Unitree G1 + GrootLocomotionController + PI0.5 策略（`model/box_move_blue`）** 部署到真实物理机器人上。

本文档特别支持 **两大视觉输入模式**：
1. **模式 A（真实机载视觉）**：连接 G1 机载 RealSense 摄像头实时闭环运行。
2. **模式 B（实机动作 + 训练视频回放）**：机器人本体执行真实物理动作，视觉输入使用训练集/采集视频（通过 ZMQ 循环推流），用于实机动作轨迹与控制响应的安全验证。

---

## 目录
1. [实机部署环境与网络拓扑（已确认配置）](#1-实机部署环境与网络拓扑已确认配置)
2. [实机双层控制架构与安全机制](#2-实机双层控制架构与安全机制)
   - [2.1 双层分层控制与零速度保护](#21-双层分层控制与零速度保护)
   - [2.2 运行中键盘交互（s / w 模式切换）](#22-运行中键盘交互s--w-模式切换)
   - [2.3 物理安全与吊架悬挂规范](#23-物理安全与吊架悬挂规范)
3. [多终端实机部署标准作业流程（SOP）](#3-多终端实机部署标准作业流程sop)
   - [【终端 0】（机器人端 SSH）启动机载服务与视觉推流](#终端-0机器人端-ssh启动机载服务与视觉推流)
   - [【终端 1】（上位机）网络连通性与网卡检查](#终端-1上位机网络连通性与网卡检查)
   - [【终端 2】（上位机-可选）启动训练视频 ZMQ 回放服务（模式 B 专用）](#终端-2上位机-可选启动训练视频-zmq-回放服务模式-b-专用)
   - [【终端 3】（上位机）执行实机 Rollout 推理（模式 A / 模式 B）](#终端-3上位机执行实机-rollout-推理模式-a--模式-b)
4. [核心实机执行命令详解与对比](#4-核心实机执行命令详解与对比)
5. [一键自动化实机启动脚本使用说明](#5-一键自动化实机启动脚本使用说明)
6. [常见故障排查（Troubleshooting）](#6-常见故障排查troubleshooting)

---

## 1. 实机部署环境与网络拓扑（已确认配置）

参考本套实机系统的物理配置：
- **机器人 IP**：`192.168.123.164`（G1 内部 Jetson / 工控电脑）
- **本机上位机 IP**：`192.168.123.213`
- **本机机器人直连网卡**：`enx6c1ff724495a`
- **Python 运行环境**：`/home/yichangfeng/miniforge3/envs/lerobot/bin/python`
- **Rollout CLI 路径**：`/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout`

```text
┌──────────────────────────────────────────────┐                ┌──────────────────────────────────────────────┐
│         本机上位机 (GPU Workstation)          │                │            Unitree G1 机器人本体             │
│        IP: 192.168.123.213                   │                │          IP: 192.168.123.164                 │
│        网卡: enx6c1ff724495a                 │                │                                              │
│                                              │ 千兆以太网直连  │  【机载服务 run_g1_server / composed_camera】 │
│  【lerobot-rollout 上位机推理】              │◄──────────────►│   ├─ ZMQ Port 6000 (PULL 接收 LowCmd)        │
│   ├─ PI0.5 Policy (双臂 14-DoF 目标)         │                │   ├─ ZMQ Port 6001 (PUB 广播 LowState)       │
│   ├─ GrootLocomotionController (50Hz 腰腿)   │                │   └─ ZMQ Port 5555 (PUB 广播机载 RealSense)  │
│   └─ 键盘监听器 ('s'=STAND / 'w'=WALK)        │                │  【底层 DDS】                                │
│                                              │                │   └─ 宇树原厂电机执行器 (29-DoF)             │
└──────────────────────────────────────────────┘                └──────────────────────────────────────────────┘
```

---

## 2. 实机双层控制架构与安全机制

### 2.1 双层分层控制与零速度保护

- **上层具身策略（PI0.5 模型）**：
  - 读取：`observation.state`（29 维关节位置） + `observation.images.global_view`（480×640×3 RGB 图像）。
  - 输出：18 维 Action 向量（前 14 维对应双臂各 7 个关节目标角度，后 4 维对应 `remote.lx`, `remote.ly`, `remote.rx`, `remote.ry` 遥控速度指令）。
- **下层平衡控制器（`GrootLocomotionController`）**：
  - 在后台以 **50Hz**（20ms）独立循环运行，驱动腰腿 15 个关节（`Joint 0 ~ 14`）保持直立平衡。
- **默认零速度保护（Zero Locomotion Command）**：
  - 启动参数 `--robot.zero_locomotion_cmd=true` 与 `--robot.locomotion_mode=stand` 会**强制拦截并将上层输出的移动速度置零**（`self.controller_input[remote_axes] = 0.0`）。
  - 这样即使策略输出移动指令，机器人也仅在原地维持全身稳定平衡并执行双臂动作，绝不会发生意外晃动或自主走动撞击物体。

### 2.2 运行中键盘交互（`s` / `w` 模式切换）

本项目在 `UnitreeG1` 类中内置了实时键盘监听器。在上位机运行 Rollout 的终端窗口中：

| 按键 | 模式名称 | 行为说明 |
| :---: | :---: | :--- |
| **`s`** / **`S`** | **`STAND` 原地保持模式** | **（默认启动模式）** 速度指令全部置零，下肢全力维持直立平衡，双臂执行动作。 |
| **`w`** / **`W`** | **`WALK` 允许行走模式** | 恢复策略输出的移动速度指令，下肢在平衡基础上跟随策略移动。 |
| **`Space`** / **`t`** | **`TOGGLE` 切换模式** | 在 `STAND` 和 `WALK` 之间快速翻转切换。 |

终端中将实时打印反馈：
```text
>>> [G1 Locomotion] Switched to STAND mode (原地保持: 速度指令已置零) <<<
>>> [G1 Locomotion] Switched to WALK mode (允许行走: 恢复策略速度指令) <<<
```

### 2.3 物理安全与吊架悬挂规范

> [!CAUTION]
> **安全守则：**
> 1. **必须使用龙门架/顶部弹性挂绳吊住 G1 机器人的吊环**，调节绳长使脚掌刚好平稳踩实地面，但若发生失衡能在倾倒前被拉住。
> 2. 清空机器人周围 2 米内的一切杂物和人员。
> 3. 操作员手指保持在键盘 `Ctrl+C` 上，或手持宇树遥控器（急停阻尼键）。

---

## 3. 多终端实机部署标准作业流程（SOP）

### 【终端 0】（机器人端 SSH）启动机载服务与视觉推流

通过网线 SSH 登录 G1 机载系统：
```bash
ssh unitree@192.168.123.164
# 密码: 123
```

根据您的机载环境，选择以下方式启动机载视觉与通信桥接：

- **方式 1（强烈推荐：使用轻量一体化一键脚本 `start_server.sh`，免克隆完整 LeRobot 仓库）**：
  在机器人端 `~/lerobot` 目录下执行：
  ```bash
  cd ~/lerobot

  # 模式 A：同时启动【电机 DDS 桥接 + 真实机载摄像头推流】
  ./start_server.sh

  # 模式 B：仅启动【电机 DDS 桥接】（不开启摄像头，用于上位机录像回放实机验证）
  ./start_server.sh --no-camera
  ```
  > 若摄像头设备号变动（默认 `--device=2`），可带参数指定：`./start_server.sh --device=4`
  > 在终端按下 `Ctrl + C` 即可一键安全退出所有后台子进程。

- **方式 2（使用 SonicStar 专有机载 Camera 服务）**：
  ```bash
  cd ~/camera_server
  source .venv_camera/bin/activate
  PYTHONPATH=$PWD python3 -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
  ```
  同时在机器人的另一个 SSH 窗口启动底层 DDS 桥接：
  ```bash
  cd ~/lerobot/camera_server
  python3 motor_server.py
  ```

---

### 【终端 1】（上位机）网络连通性与网卡检查

在上位机检查直连网卡 `enx6c1ff724495a` 的 IP 与连通性：

```bash
# 确认网卡已分配 192.168.123.213
ip a show enx6c1ff724495a

# 若未配置，可执行以下命令快速配置
sudo ip addr flush dev enx6c1ff724495a
sudo ip addr add 192.168.123.213/24 dev enx6c1ff724495a
sudo ip link set enx6c1ff724495a up

# 测试与 G1 机器人的通信
ping -c 3 192.168.123.164
```

---

### 【终端 2】（上位机-可选）启动训练视频 ZMQ 回放服务（模式 B 专用）

如果您希望在实机上测试机器人对**特定训练集视频**的动作响应（实机做物理动作，视觉吃录像数据）：

```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/python src/lerobot/cameras/zmq/stream_dataset_video_zmq.py \
    --video_path datasets/box_pick/videos/observation.images.global_view/chunk-000/file-003.mp4 \
    --port 5556 \
    --fps 30 \
    --loop
```

> [!NOTE]
> 该服务将在上位机本地 `5556` 端口循环广播视频帧。若采用**模式 A（真实机载相机）**，则无需启动终端 2。

---

### 【终端 3】（上位机）执行实机 Rollout 推理

在上位机打开主控制终端，根据选择的模式执行对应命令：

#### 模式 A：实机真实相机闭环部署（Real Camera Mode）
连接 G1 机器人机载推流端口 `192.168.123.164:5555`：

```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout \
    --strategy.type=base \
    --policy.path=model/box_move_blue \
    --policy.device=cuda \
    --robot.type=unitree_g1 \
    --robot.is_simulation=false \
    --robot.robot_ip=192.168.123.164 \
    --robot.controller=GrootLocomotionController \
    --robot.zero_locomotion_cmd=true \
    --robot.locomotion_mode=stand \
    --robot.cameras='{"global_view": {"type": "zmq", "server_address": "192.168.123.164", "port": 5555, "camera_name": "head_camera", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}}' \
    --task="move blue box back and forth between tables" \
    --duration=1000 \
    --fps=25 \
    --display_data=true
```

---

#### 模式 B：实机动作 + 训练视频回放模式（Video Replay Mode）
机器人本体为物理实机（`is_simulation=false`），但视觉连接上位机本地推流端口 `localhost:5556`：

```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout \
    --strategy.type=base \
    --policy.path=model/box_move_blue \
    --policy.device=cuda \
    --robot.type=unitree_g1 \
    --robot.is_simulation=false \
    --robot.robot_ip=192.168.123.164 \
    --robot.controller=GrootLocomotionController \
    --robot.zero_locomotion_cmd=true \
    --robot.locomotion_mode=stand \
    --robot.cameras='{"global_view": {"type": "zmq", "server_address": "localhost", "port": 5556, "camera_name": "head_camera", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}}' \
    --task="move blue box back and forth between tables" \
    --duration=1000 \
    --fps=25 \
    --display_data=true
```

---

## 4. 核心实机执行命令详解与对比

| 参数 | 真实相机模式 (模式 A) | 视频回放实机模式 (模式 B) | 作用原理 |
| :--- | :--- | :--- | :--- |
| `--robot.is_simulation` | `false` | `false` | **关闭仿真**，通过 `unitree_sdk2_socket.py` 连接物理电机 |
| `--robot.robot_ip` | `192.168.123.164` | `192.168.123.164` | G1 机载电脑 IP，建立 DDS 桥接通道（6000 动作 / 6001 状态） |
| `--robot.controller` | `GrootLocomotionController` | `GrootLocomotionController` | 启动 50Hz 神经网络下半身直立平衡控制器 |
| `--robot.zero_locomotion_cmd`| `true` | `true` | **默认置零移动速度**，防止机器人失稳移动 |
| `--robot.locomotion_mode` | `stand` | `stand` | 初始模式设为原地保持 |
| `--robot.cameras` | `server_address: 192.168.123.164`<br>`port: 5555` | `server_address: localhost`<br>`port: 5556` | 切换图像获取来源（机载 RealSense vs 本地 ZMQ 录像推流） |
| `--policy.path` | `model/box_move_blue` | `model/box_move_blue` | 训练好的 PI0.5 策略模型权重路径 |
| `--display_data` | `true` | `true` | 实时弹出可视化窗口监控相机输入与关节动作 |

---

## 5. 一键自动化实机启动脚本使用说明

项目根目录下已为您配置并优化了启动脚本 [**`deploy_real_g1.sh`**](file:///home/yichangfeng/lerobot/deploy_real_g1.sh) 和专门用于视频回放测试的 [**`verify_video_rollout_real.sh`**](file:///home/yichangfeng/lerobot/verify_video_rollout_real.sh)。

### 1. 真实机载相机模式启动：
```bash
./deploy_real_g1.sh
```

### 2. 实机动作 + 视频回放模式一键启动：
```bash
./verify_video_rollout_real.sh datasets/box_pick/videos/observation.images.global_view/chunk-000/file-003.mp4
```
该脚本会自动在后台拉起视频推流并在退出时自动清理进程。

---

## 6. 常见故障排查（Troubleshooting）

### Q1: 报错 `TimeoutError: Timed out waiting for robot state (10s)`
- **检查项**：
  1. 上位机网卡 `enx6c1ff724495a` 的 IP 是否为 `192.168.123.213`。
  2. G1 机载端是否已启动 `run_g1_server.py`。
  3. 执行 `ping 192.168.123.164` 确认千兆网线物理链路正常。

### Q2: 报错 `RuntimeError: Camera 'global_view' failed to connect`
- **检查项**：
  - 如果运行**模式 A**：检查 G1 机载相机的推流服务（端口 5555）是否正常运行。
  - 如果运行**模式 B**：检查本地 `stream_dataset_video_zmq.py` 是否已在端口 5556 启动。

### Q3: 运行中如何紧急停止？
- **正常停止**：在上位机终端按 **`Ctrl + C`**，`UnitreeG1` 将自动下发零力矩指令（`_send_zero_torque`），使各关节进入柔性阻尼状态，由吊架安全绳承托。
- **紧急急停**：使用宇树无线遥控器触发急停阻尼键，或直接断开上位机终端进程。

---

_文档更新时间：2026-09-01_
