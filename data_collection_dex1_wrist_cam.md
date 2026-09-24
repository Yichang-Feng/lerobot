# G1 三路摄像头 + Dex1 夹爪遥操作数据采集指南

本文档记录了使用 PICO XR 设备对宇树 G1-29 进行三路摄像头（头部 + 双腕部）+ Dex1 夹爪的遥操作数据采集完整流程。

---

## 1. 系统架构

### 1.1 硬件拓扑

```
┌──────────────────────────────────────────────────────┐
│  机器人 PC2 (192.168.123.164)                         │
│                                                      │
│  ┌─────────────┐  ┌───────────┐  ┌───────────┐      │
│  │ RealSense   │  │ 左腕相机   │  │ 右腕相机   │      │
│  │ D435i (头部) │  │ (JR0002)  │  │ (JR0001)  │      │
│  └──────┬──────┘  └─────┬─────┘  └─────┬─────┘      │
│         │               │               │            │
│         └───────┬───────┴───────┬───────┘            │
│                 ▼               ▼                    │
│         image_server (teleimager)                    │
│         ZMQ PUB :55555  :55556  :55557               │
│                                                      │
│  ┌──────────────────────────────┐                    │
│  │ dex1_1_service               │                    │
│  │ DDS: rt/dex1/{left,right}/   │                    │
│  │     {cmd, state}             │                    │
│  └──────────────────────────────┘                    │
└──────────────────────────────────────────────────────┘
                    │ 有线网卡
                    ▼
┌──────────────────────────────────────────────────────┐
│  上位机 Host (192.168.123.213)                        │
│                                                      │
│  teleop_hand_and_arm.py                              │
│  ├─ ImageClient ←── ZMQ SUB 三路图像                  │
│  ├─ G1_29_ArmController ←── DDS 关节控制              │
│  ├─ Dex1_1_Gripper_Controller ←── DDS 夹爪控制        │
│  ├─ LocoClientWrapper ←── 底盘速度 vx/vy/vyaw        │
│  ├─ TeleVuerWrapper ←── XR 手柄姿态/按键              │
│  └─ EpisodeWriter ──→ 异步落盘 data.json + colors/   │
└──────────────────────────────────────────────────────┘
                    │ WiFi
                    ▼
            ┌──────────────┐
            │ PICO 4 Ultra │
            │ (XR 头显)     │
            └──────────────┘
```

### 1.2 采集数据通道

| 数据通道 | 来源 | 传输方式 | 采集频率 |
| :--- | :--- | :--- | :--- |
| 头部 RGB 视频 | RealSense D435i | ZMQ :55555 | 30 FPS |
| 左腕部 RGB 视频 | USB 相机 JR0002 | ZMQ :55556 | 30 FPS |
| 右腕部 RGB 视频 | USB 相机 JR0001 | ZMQ :55557 | 30 FPS |
| 双臂 14 关节角 (qpos) | G1 低层 DDS | DDS `rt/lowstate` | 30 Hz |
| 左/右 Dex1 夹爪位置 | dex1_1_service | DDS `rt/dex1/*/state` | 30 Hz |
| 底盘速度指令 (vx, vy, vyaw) | PICO 摇杆输入 | 本地计算 | 30 Hz |
| 下肢关节 (leg_qpos, leg_qvel) | G1 低层 DDS | DDS `rt/lowstate` | 30 Hz |

---

## 2. 配置文件

### 2.1 机器人端相机配置

文件路径: `teleop/teleimager/cam_config_server.yaml`

关键配置项（腕部相机部分）:

```yaml
left_wrist_camera:
  enable_zmq: true
  zmq_port: 55556
  enable_webrtc: false
  type: uvc                    # pyuvc 驱动，MJPG 零拷贝
  image_shape: [480, 640]      # [height, width]
  fps: 30
  video_id: null               # 使用 serial_number 匹配，不依赖 /dev/video 编号
  serial_number: JR0002        # 左腕相机 USB 序列号
  physical_path: null

right_wrist_camera:
  enable_zmq: true
  zmq_port: 55557
  enable_webrtc: false
  type: uvc
  image_shape: [480, 640]
  fps: 30
  video_id: null
  serial_number: JR0001        # 右腕相机 USB 序列号
  physical_path: null
```

> **注意**: 头部相机保持现有 `type: uvc` 配置不变，已验证可正常采集。

### 2.2 相机驱动备选

如果 `type: uvc` (pyuvc) 无法识别腕部相机，可改为 `type: opencv`，此时 `image_server` 会使用 OpenCV V4L2 后端驱动。`image_server` 的 OpenCV 模式会自动设置 MJPG 格式以确保 30fps 帧率。

---

## 3. 部署步骤

### 3.1 机器人端 (PC2)

**终端 1 — 启动图像服务:**

```bash
ssh unitree@192.168.123.164    # 密码默认 123

cd ~/xr_teleoperate/teleop/teleimager   # 或 ~/teleimager
source ~/teleimager/env_teleimager/bin/activate
python3 -m teleimager.image_server --rs
```

成功标志:
```
[Image Server] head_camera is ready.
[Image Server] left_wrist_camera is ready.
[Image Server] right_wrist_camera is ready.
```

**终端 2 — 启动 Dex1 夹爪服务:**

```bash
# 根据 dex1_1_service 仓库的部署方式启动
# 参考: https://github.com/unitreerobotics/dex1_1_service
```

成功标志: DDS topic `rt/dex1/left/state` 和 `rt/dex1/right/state` 有数据输出。

### 3.2 上位机 (Host)

根据底层控制架构的不同，上位机支持以下启动模式：

#### 模式 A（强烈推荐，对齐 LeRobot VLA 部署流程）：ZMQ 全功能模式 (`--use-zmq`)
通过 ZeroMQ 与机载端通信，通信流程与 `src/lerobot/robots/unitree_g1/unitree_g1_client.py`（VLA 实机部署）及机器人端底层控制服务（`Groot / GripperBridge`）**完全 100% 严格一致**：
- **动作下发 (PUSH 6002)**：发送双臂 14 关节角、底盘摇杆速度，并在包内严格携带 `"gripper": {"right": {"q": r_grip}, "left": {"q": l_grip}}`：
  ```json
  {
    "cmd": "action",
    "action": {
      "kLeftShoulderPitch.q": -0.20, "kRightWristYaw.q": -0.01,
      "remote.lx": 0.0, "remote.ly": 0.0, "remote.rx": 0.0, "remote.ry": 0.0,
      "gripper": { "right": { "q": 0.0 }, "left": { "q": 5.0 } }
    },
    "timestamp": 1788514855.53
  }
  ```
- **状态接收 (SUB 6001)**：接收机载端 29-DoF 关节位置、速度与 IMU。
- **夹爪状态 (SUB 6004)**：从 6004 端口（`rt/dex1/state`）以 100Hz 接收实测夹爪开合状态并闭环录制：
  ```json
  {
    "topic": "rt/dex1/state",
    "data": {
      "right": { "q": 0.501, "dq": 0.0, "tau_est": 0.02 },
      "left":  { "q": 0.500, "dq": 0.0, "tau_est": 0.02 }
    }
  }
  ```
- **开合范围与速度配置**：
  - 默认开合范围为 `2.7 rad`（闭合）至 `5.0 rad`（张开），与硬件标定完全对齐。可通过 `--gripper-min`（默认 2.7）和 `--gripper-max`（默认 5.0）调节。
  - **非对称平滑速度控制**：开爪采用更轻快步长 `--gripper-open-step 0.15`（30Hz 下约 0.53s 全开，松开物品自然灵敏）；闭爪采用稳定防撞步长 `--gripper-close-step 0.08`（30Hz 下约 0.97s 闭合，平稳夹取）。
- **平滑介入插值 (防突跳)**：
  - 支持 `--smooth-duration` 参数（默认 `2.0` 秒）。
  - 当在初始位置按下右手 A 键（或从 Home 回零后恢复追踪）时，系统自动使用**余弦 S 曲线**（Cosine S-curve）从机械臂当前物理关节位置缓慢平滑过渡至 VR 目标位姿，起点与终点加速度为 0，彻底消除初始姿态差异导致的机械臂快速突跳与晃动。若需直接跳入追踪可设为 `--smooth-duration 0`。
- **图像流**：继续沿用当前稳定方案（`teleimager` 头部/双腕三路相机推流）。

```bash
conda activate tv
cd /home/yichangfeng/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex1 \
  --input-mode controller \
  --use-zmq \
  --robot-ip 192.168.123.164 \
  --zmq-state-port 6001 \
  --zmq-action-port 6002 \
  --zmq-gripper-port 6004 \
  --record \
  --task-name "g1_pick_put_dex1"
```

#### 模式 B（结合第三方/外部运控）：纯上肢 DDS 模式 (`--arm-only`)
当使用外部运控（如自研强化学习 RL 策略、WBC 等）控制机器人下肢站立行走，且通过 DDS (`rt/arm_sdk`) 独立下发手臂动作时使用：

```bash
conda activate tv
cd /home/yichangfeng/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex1 \
  --input-mode controller \
  --arm-only \
  --record \
  --task-name "g1_pick_put_dex1"
```

#### 模式 C：宇树官方运控 DDS 模式 (`--motion`)
当依赖宇树官方机载内置运控服务，并通过 PICO 手柄摇杆直接驱动官方底盘行走时使用：

```bash
conda activate tv
cd /home/yichangfeng/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex1 \
  --input-mode controller \
  --motion \
  --record \
  --task-name "dex1_wrist_cam_data"
```

> **注意**：若既不传 `--use-zmq`，也不传 `--motion` 或 `--arm-only`，脚本将进入调试锁定模式（自动切入 Debug 模式并将腿部以高刚度锁在初始位置，仅适用于上吊架单调手臂）。

启动成功标志:
```
[G1_29_ArmController] Subscribe dds ok.
EpisodeWriter initialized successfully.
🟢 Press [r] (or Pico Right [A]) to start syncing...
```

### 3.3 PICO 头显连接

1. 戴上 PICO 头显，确保连接到同一路由器局域网
2. 打开 PICO 内置浏览器，输入:
   - `https://vuer.ai?ws=wss://192.168.123.213:8012`
   - 或 `https://192.168.123.213:8012/?ws=wss://192.168.123.213:8012`
3. 点击 **"Virtual Reality"** 进入全沉浸 VR 视角

---

## 4. 操作流程

### 4.1 手柄按键说明

| 按键组合 | 功能 | 说明 |
| :--- | :--- | :--- |
| **右手 A 键** | 开始同步 / 恢复追踪 | 机械臂开始跟随 VR 手柄运动 |
| **左手扳机/Grip + 右手 A** | 开始录制 Episode | 开始录制当前 Episode 数据（若已在录制中再按保持录制，不改变状态） |
| **左手扳机/Grip + 右手 B** | 结束并保存录制 | 结束并保存当前 Episode（若未在录制中再按保持停止，不改变状态） |
| **左摇杆** | 行走控制 | 前/后=vx，左/右=vy，速度限制 0.3 m/s |
| **右摇杆** | 转向控制 | 左/右=vyaw，角速度限制 0.3 rad/s |
| **右手 A + 左手 X** | 双臂回零 + 暂停 | 自动保存当前录制，双臂平滑回预备位 |
| **右手 B 键 (单按)** | 安全退出 | 保存数据，关闭连接 |
| **左右摇杆同时按下** | 紧急停止 (Damp) | 机器人进入阻尼模式，软急停 |

### 4.2 典型采集工作流

```
1. 确认三路视频和夹爪服务均已启动
2. 上位机启动 teleop_hand_and_arm.py (含 --record --motion)
3. PICO 头显连接 VR 页面

4. [右手 A] → 开始遥操追踪（机械臂跟随人手运动）
5. [左手扳机/Grip + 右手 A] → 开始录制 Episode（开始后重复按不影响录制）
6. 操作机器人完成任务动作（抓取、放置等），同时可用摇杆控制行走
7. [左手扳机/Grip + 右手 B] → 结束并保存当前 Episode，自动递增编号（结束后重复按不影响状态）
8. 重复 5-7 采集多条轨迹

9. [右手 B (单按)] → 安全退出
```

---

## 5. 数据保存格式

### 5.1 目录结构

保存路径: `teleop/utils/data/<task-name>/`

```
dex1_wrist_cam_data/
├── episode_0001/
│   ├── colors/
│   │   ├── 000000_color_0.jpg      ← 头部相机 (RealSense D435i)
│   │   ├── 000000_color_1.jpg      ← 左腕部相机 (JR0002)
│   │   ├── 000000_color_2.jpg      ← 右腕部相机 (JR0001)
│   │   ├── 000001_color_0.jpg
│   │   ├── 000001_color_1.jpg
│   │   ├── 000001_color_2.jpg
│   │   └── ...
│   ├── depths/
│   ├── audios/
│   └── data.json
├── episode_0002/
│   └── ...
└── ...
```

### 5.2 data.json 格式

```json
{
  "info": {
    "version": "1.0.0",
    "date": "2026-09-22",
    "author": "unitree",
    "image": { "width": 640, "height": 480, "fps": 30 }
  },
  "text": {
    "goal": "task goal description",
    "desc": "task description",
    "steps": "step1: ...; step2: ..."
  },
  "data": [
    {
      "idx": 0,
      "colors": {
        "color_0": "colors/000000_color_0.jpg",
        "color_1": "colors/000000_color_1.jpg",
        "color_2": "colors/000000_color_2.jpg"
      },
      "depths": {},
      "states": {
        "left_arm":  { "qpos": ["7 个关节角 (rad)"], "qvel": [], "torque": [] },
        "right_arm": { "qpos": ["7 个关节角 (rad)"], "qvel": [], "torque": [] },
        "left_ee":   { "qpos": ["1 个夹爪位置 (rad)"], "qvel": [], "torque": [] },
        "right_ee":  { "qpos": ["1 个夹爪位置 (rad)"], "qvel": [], "torque": [] },
        "body": {
          "qpos": ["29 个全身关节角"],
          "qvel": ["29 个全身关节角速度"],
          "leg_qpos": ["12 个下肢关节角 (左右各6)"],
          "leg_qvel": ["12 个下肢关节角速度"]
        }
      },
      "actions": {
        "left_arm":  { "qpos": ["7 个 IK 目标角"], "qvel": [], "torque": [] },
        "right_arm": { "qpos": ["7 个 IK 目标角"], "qvel": [], "torque": [] },
        "left_ee":   { "qpos": ["1 个夹爪目标"], "qvel": [], "torque": [] },
        "right_ee":  { "qpos": ["1 个夹爪目标"], "qvel": [], "torque": [] },
        "body": { "qpos": ["vx, vy, vyaw (m/s, m/s, rad/s)"] }
      }
    }
  ]
}
```

### 5.3 数据字段说明

| 字段路径 | 维度 | 含义 |
| :--- | :---: | :--- |
| `colors.color_0` | 480×640 | 头部 RealSense RGB 图像 |
| `colors.color_1` | 480×640 | 左腕部相机 RGB 图像 |
| `colors.color_2` | 480×640 | 右腕部相机 RGB 图像 |
| `states.left_arm.qpos` | 7 | 左臂关节角度 (rad) |
| `states.right_arm.qpos` | 7 | 右臂关节角度 (rad) |
| `states.left_ee.qpos` | 1 | 左 Dex1 夹爪位置反馈 (rad) |
| `states.right_ee.qpos` | 1 | 右 Dex1 夹爪位置反馈 (rad) |
| `states.body.leg_qpos` | 12 | 下肢 12 关节角度 (左右各 6) |
| `states.body.leg_qvel` | 12 | 下肢 12 关节角速度 (rad/s) |
| `actions.left_arm.qpos` | 7 | 左臂 IK 解算目标关节角 |
| `actions.right_arm.qpos` | 7 | 右臂 IK 解算目标关节角 |
| `actions.left_ee.qpos` | 1 | 左 Dex1 夹爪目标指令 |
| `actions.right_ee.qpos` | 1 | 右 Dex1 夹爪目标指令 |
| `actions.body.qpos` | 3 | 底盘速度指令 [vx, vy, vyaw] (m/s, m/s, rad/s) |

---

## 6. 常见问题排查

### 6.1 腕部相机无法识别

**现象**: `image_server` 启动时报 `Cannot find UVCCamera for left_wrist_camera with serial number JR0002`

**解决**:
1. 检查相机 USB 是否正确连接到 PC2
2. 运行 `python3 -m teleimager.image_server --rs --verbose` 查看 CameraFinder 发现的所有相机及其序列号
3. 如果 pyuvc 无法识别，尝试将 `cam_config_server.yaml` 中腕部相机的 `type` 从 `uvc` 改为 `opencv`

### 6.2 腕部相机帧率低（5fps）

**原因**: 相机默认使用 YUYV 未压缩格式，USB 带宽不足

**解决**:
- `type: uvc` 模式: pyuvc 自动选择 MJPG 模式，通常不会出此问题
- `type: opencv` 模式: `image_server` 会设置 `cv2.CAP_PROP_FOURCC = MJPG`，如仍不生效可参考 `~/lerobot/robot_server/camera_server/multi_camera_server.py` 中的 `force_v4l2_mjpg()` 方法手动用 `v4l2-ctl` 强制设置

### 6.3 录制时部分图像为 None

**现象**: 终端频繁打印 `Left wrist image is None!` 或 `Right wrist image is None!`

**解决**:
1. 确认 PC2 上 `image_server` 三路相机均显示 ready
2. 确认上位机与 PC2 网络连通（`ping 192.168.123.164`）
3. 检查 ZMQ 端口是否被占用（`ss -tlnp | grep 5555`）

### 6.4 Dex1 夹爪无响应

**现象**: 夹爪不跟随手柄 trigger

**解决**:
1. 确认 `dex1_1_service` 已启动并运行正常
2. 检查 DDS 通信：确认 `rt/dex1/left/state` 和 `rt/dex1/right/state` 有数据
3. 确认启动命令包含 `--ee dex1`

### 6.5 PICO 浏览器显示 404

**解决**: 确保 URL 末尾不要加 `/index.html`，正确格式:
- `https://vuer.ai?ws=wss://192.168.123.213:8012`
- `https://192.168.123.213:8012/?ws=wss://192.168.123.213:8012`
