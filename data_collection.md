# Unitree G1 遥操作多模态数据采集部署与操作指南

本文档记录了使用 PICO XR 设备对宇树 Unitree G1（29 自由度，橡胶手版本）进行双臂动作同步、下肢运动遥控与多模态数据采集的完整部署与运行流程。

---

## 1. 系统架构与硬件拓扑

- **机器人下位机 (PC2)**: `192.168.123.164` (负责 RealSense D435i 相机流采集与传输)
- **电脑上位机 (Host)**: `192.168.123.213` (连接机器人内部有线网卡 `enx6c1ff724495a`，负责 IK 解算、DDS 控制与数据落盘)
- **XR 设备 (PICO)**: PICO 4 Ultra Enterprise（通过同一路由器 WiFi 连接上位机）
- **数据采集项**:
  1. 机器人头部视角彩色视频流（RealSense D435i RGB @ 30 FPS）
  2. 机器人双臂 14 关节角度（`qpos`，不记录上肢速度 `qvel`）
  3. 下肢行走与转向速度控制指令（`[vx, vy, vyaw]`）
  4. 下肢 12 个腿部关节角度与角速度状态（`leg_qpos`, `leg_qvel`）

---

## 2. 机器人端（PC2）启动步骤

机器人 PC2 负责读取头部 Intel RealSense D435i 相机并将图像流发布给上位机。

### 2.1 检查/生成相机配置文件
确保 `~/teleimager/cam_config_server.yaml` 已配置为使用 RealSense 的 RGB 设备（`video_id: 4`）并启用稳定低延时的 ZMQ 传输：

```bash
cat << 'EOF' > ~/teleimager/cam_config_server.yaml
head_camera:
  enable_zmq: true
  zmq_port: 55555
  enable_webrtc: false
  type: opencv
  image_shape: [480, 640]
  binocular: false
  fps: 30
  video_id: 4
  serial_number: null
  physical_path: "/sys/devices/platform/3610000.xhci/usb2/2-2/2-2.1/2-2.1:1.3"

left_wrist_camera:
  enable_zmq: false
  enable_webrtc: false

right_wrist_camera:
  enable_zmq: false
  enable_webrtc: false
EOF
```

### 2.2 启动图像服务
进入机器人 PC2 终端（可 SSH：`ssh unitree@192.168.123.164`，密码默认 `123`）：

```bash
cd ~/teleimager
source ~/teleimager/env_teleimager/bin/activate
python3 -m teleimager.image_server
```

> **成功标志**：终端输出 `[Image Server] head_camera is ready.`，并保持终端运行。

---

## 3. 电脑上位机启动步骤

确保上位机网线连接机器人网口，网卡（`enx6c1ff724495a`）已配置在 `192.168.123.x` 网段。

打开上位机终端，根据运控控制源选择运行模式：

#### 模式 A（结合第三方/外部运控）：纯上肢模式 (`--arm-only`)
当使用外部运控（如强化学习 RL 策略、WBC 等）控制下肢站立行走，本程序仅负责上肢遥操与全身多模态数据采集时：
- **优势**：不切 Debug 模式、不锁死腿部、上肢指令走 `rt/arm_sdk`，通信与数据采集完全正常。

```bash
conda activate tv
cd /home/yichangfeng/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm G1_29 \
  --input-mode controller \
  --arm-only \
  --record \
  --task-name "g1_loco_data"
```

#### 模式 B：官方全功能模式 (`--motion`)
当依赖宇树官方机载运控，并通过 PICO 手柄摇杆直接控制底盘行走与转向时：

```bash
conda activate tv
cd /home/yichangfeng/xr_teleoperate/teleop

python teleop_hand_and_arm.py \
  --arm G1_29 \
  --input-mode controller \
  --motion \
  --record \
  --task-name "g1_loco_data"
```

> **启动成功标志**：
> 1. 终端打印 `[G1_29_ArmController] Subscribe dds ok.` 和 `EpisodeWriter initialized successfully.`
> 2. 终端提示 `🟢 Press [r] (or Pico Right [A]) to start syncing...`，程序进入待命等待状态。

---

## 4. 浏览器与 PICO 头显连接

### 4.1 电脑端验证画面
在电脑浏览器（Chrome / Edge）打开：
👉 **`https://192.168.123.213:8012/?ws=wss://192.168.123.213:8012`**
- 若弹出“您的连接不是私密连接”，点击 **“高级” ➔ “继续前往 192.168.123.213（不安全）”**。
- 此时网页中将直接显示 RealSense D435i 的实时彩色画面。

### 4.2 PICO 头显进入
1. 戴上 PICO 头显，确保 PICO 连接到同一路由器局域网。
2. 打开 PICO 内置浏览器，输入以下网址（推荐使用官方托管链接，加载速度快且不会出现本地路径 404）：
   👉 **`https://vuer.ai?ws=wss://192.168.123.213:8012`**
   *(备用本地链接：`https://192.168.123.213:8012/?ws=wss://192.168.123.213:8012`，注意末尾不要加 `/index.html`)*
3. 进入网页后，画面中会显示机器人视角的实时视频。
4. 使用手柄光标点击右下角的 **“Virtual Reality”**（或 Enter VR）按钮，进入全沉浸式 VR 视角。

---

## 5. PICO 手柄操作与数据采集流程

本系统已实现**全流程纯手柄操控**，采集过程中无需触碰电脑键盘。

| 按键组合 | 功能说明 | 详细行为 |
| :--- | :--- | :--- |
| **右手 A 键** | **开始同步 / 恢复追踪** | 激活机械臂跟随模式，机器人双臂开始实时跟随 VR 手柄运动 |
| **左手扳机/握把 (Trigger/Grip) + 右手 A 键** | **开始录制** | 开启录制当前 Episode 数据（若已在录制中，再按保持录制，状态不改变） |
| **左手扳机/握把 (Trigger/Grip) + 右手 B 键** | **结束录制并保存** | 结束并保存当前 Episode，准备下一回合（若未在录制中，再按保持停止，状态不改变） |
| **左摇杆 (Thumbstick)** | **机器人行走控制** | <li>**推向前/后**：控制机器人前后行走速度 $v_x$</li><li>**推向左/右**：控制机器人左右横移平移速度 $v_y$</li> |
| **右摇杆 (Thumbstick)** | **机器人转向控制** | **推向左/右**：控制机器人原地旋转转向偏航角速度 $v_{yaw}$ |
| **右手 A 键 + 左手 X 键** | **双臂回零与暂停** | 双臂平滑回到安全预备位置，自动保存当前未完结的回合并暂停跟踪，进入安全休息状态（再次单按**右手 A 键**可恢复跟随） |
| **右手 B 键 (单按)** | **安全退出程序** | 停止上位机控制程序，自动保存数据并关闭连接 |

---

## 6. 数据保存结构与格式

录制的数据将自动保存在上位机的以下路径：
`teleop/utils/data/g1_loco_data/`

每次录制完成一个回合，将自动生成一个子文件夹（例如 `episode_0000`、`episode_0001` 等），包含：

- **`data.json`**: 记录完整的时间序列多模态数据，包含：
  ```json
  {
    "timestamp": 1726300000.123,
    "states": {
      "left_arm": { "qpos": [...] },          // 左臂 7 个关节角度 (rad)
      "right_arm": { "qpos": [...] },         // 右臂 7 个关节角度 (rad)
      "body": {
        "qpos": [...],                        // 全身 29 关节原始状态
        "leg_qpos": [...],                    // 下肢 12 个腿部关节角度 (左右各6)
        "leg_qvel": [...]                     // 下肢 12 个腿部关节角速度 (rad/s)
      }
    },
    "actions": {
      "left_arm": { "qpos": [...] },          // 左臂 7 关节目标指令
      "right_arm": { "qpos": [...] },         // 右臂 7 关节目标指令
      "body": {
        "qpos": [vx, vy, vyaw]                // 下肢线速度与角速度指令 [m/s, m/s, rad/s]
      }
    }
  }
  ```
- **`head_camera.mp4`**: 对应 Episode 的同步机器人视角 RGB 视频。
- **`rerun.rrd`**: Rerun 3D 轨迹回放与可视化日志文件。

---

## 7. 常见问题排查 (Troubleshooting)

1. **PICO 端打开网页显示 404**：
   - 检查输入的 URL 是否误加了 `/index.html`。正确格式应为 `https://192.168.123.213:8012/?ws=wss://192.168.123.213:8012` 或直接使用 `https://vuer.ai?ws=wss://192.168.123.213:8012`。
2. **电脑端或 PICO 端黑屏没有视频画面**：
   - 确认 PC2 上的 `cam_config_server.yaml` 中使用的是 `video_id: 4`，且 `enable_zmq: true, enable_webrtc: false`。
   - 确认 PC2 上的 `python3 -m teleimager.image_server` 正常运行中且未报 OpenCV 设备冲突。
3. **想要调整人手臂与机械臂映射的灵敏度或高低位置**：
   - **空间高低与前后偏移**：修改 `teleop/televuer/src/televuer/tv_wrapper.py` 中的 `transform_IPunitree_Brobot_world_arm_to_head_then_waist` 函数（调整 `+0.15` 和 `+0.45` 偏移量）。
   - **臂长比例缩放**：修改 `teleop/robot_control/robot_arm_ik.py` 中 `G1_29_ArmIK.scale_arms()` 并在 `solve_ik()` 中解除注释。
