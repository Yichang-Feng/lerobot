# Unitree G1 平滑移动操作（Loco-Manipulation）类 Psi0 数据采集与转换指南

本文档详细介绍了基于 **VR 动捕 + SONIC WBC（类 Psi0）** 体系的 Unitree G1 人形机器人平滑数据采集全流程。重点阐明了**不依赖手动推摇杆，由人体空间位移自动生成连续平滑速度指令**的底层机制，以及如何将采集到的数据转换并微调现有策略模型（如 `model/unitree_box_move`）。

---

## 目录
1. [设计理念与方案优势](#1-设计理念与方案优势)
2. [硬件拓扑与系统架构](#2-硬件拓扑与系统架构)
3. [速度指令自动提取机制（为什么不用摇杆）](#3-速度指令自动提取机制为什么不用摇杆)
4. [任务范式与短程微调流程](#4-任务范式与短程微调流程)
5. [数据采集标准作业程序（SOP）](#5-数据采集标准作业程序sop)
6. [采集数据结构与字段定义](#6-采集数据结构与字段定义)
7. [遥控速度指令的底层映射机制与真实数据分布](#7-遥控速度指令的底层映射机制与真实数据分布)
8. [微调模型推荐数据量与训练建议](#8-微调模型推荐数据量与训练建议)

---

## 1. 设计理念与方案优势

在人形机器人双臂搬运与移动操作（Loco-manipulation）任务中，传统的“**手柄摇杆控制底盘 + 外骨骼/遥操控制双臂**”方式存在严重弊端：

| 对比维度 | 传统手推摇杆方式 | 本方案（类 Psi0 身体动捕差分） |
| :--- | :--- | :--- |
| **速度连贯性** | 存在死区与阶跃突变（Bang-bang 开关效应），速度断断续续 | **天然具备人体物理惯性与加速度，属于 $C^2$ 连续平滑曲线** |
| **下肢平衡冲击** | 阶跃速度给 50Hz 底层平衡控制器带来巨大冲击，易失稳晃动 | **速度渐进升降，下肢步态极其平稳自然，彻底杜绝下肢晃动** |
| **操作协调度** | 操作员需要一手管手臂一手推摇杆，分心且动作极易割裂 | **操作员自然向前迈步、转身，全身协同直观，沉浸感极强** |
| **模型模仿学习质量** | 模型学习到剧烈抖动的速度指令，推理时机械臂与底盘容易抽动 | **训练出的 VLA 策略动作舒展连贯，实机部署成功率大幅提升** |

---

## 2. 硬件拓扑与系统架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                            上位机 (GPU Workstation)                          │
│                           IP: 192.168.123.213                                │
│                                                                              │
│  【VR 空间动捕追踪】                                                        │
│   ├─ 头显 + 手柄 6D 位姿追踪 (50Hz)                                          │
│   ├─ 身体空间位移时间差分 ──► 生成连续平滑 [vx, vy, vyaw] 与 height            │
│   └─ 双臂空间位姿逆运动学 ──► 解算双臂 14 关节目标角度                       │
│                                                                              │
│  【数据落盘 / Exporter】                                                     │
│   ├─ 自动打包并保存为标准 Episode (不含失败轨迹)                             │
│   └─ 格式转换器 ──► 生成 LeRobot v3.0 (29-D State + 18-D Action)             │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │ 千兆以太网直连 (网卡: enx6c1ff724495a)
┌───────────────────────▼──────────────────────────────────────────────────────┐
│                         Unitree G1 机器人本体                                │
│                           IP: 192.168.123.164                                │
│                                                                              │
│  【机载服务】                                                                │
│   ├─ ZMQ 5555: RealSense 头部机载摄像头实时推流 (480x640x3 RGB)              │
│   ├─ ZMQ 6000/6001: 底层电机 DDS 桥接                                        │
│   └─ 50Hz 全身平衡控制器 (SONIC / GR00T WBC)                                 │
│       ├─ 接收平滑速度指令 ──► 解算下肢与腰部 15 关节步态驱动迈步             │
│       └─ 接收双臂目标角度 ──► 驱动双臂 14 关节执行合抱搬运                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 速度指令自动提取机制（为什么不用摇杆）

当操作员佩戴 VR 设备在空间中移动时，系统每 $20\text{ ms}$（$50\text{ Hz}$）记录一次操作员躯干/骨盆位姿 $P(t) = [x(t), y(t), z(t)]$ 与旋转四元数 $Q(t)$：

1. **前后与横向线速度计算**：
   $$v_x(t) = \frac{x(t) - x(t-\Delta t)}{\Delta t}, \quad v_y(t) = \frac{y(t) - y(t-\Delta t)}{\Delta t}$$
2. **转向角速度计算**：
   $$\omega_z(t) = \frac{\text{yaw}(t) - \text{yaw}(t-\Delta t)}{\Delta t}$$
3. **骨盆高度计算**：
   $$h(t) = z(t)$$

这一组由身体真实位移微分得到的速度与高度数据：
* **在实时遥控时**：直接送入 SONIC WBC / Planner，带动 G1 机器人双足踏步前移或原地转体。
* **在数据落盘时**：作为真实的连续动作标签写入数据集，完全避免了摇杆死区和跳变。

---

## 4. 任务范式与短程微调流程

由于 G1 机载 RealSense 头部摄像头视场角（FOV 约 70°~80°）有限，**为了保证策略的视觉连续性，任务应当设计为“初始在视野内 $\rightarrow$ 短程微调步态 $\rightarrow$ 抓取搬运”**。

```
[阶段 1: 站立就绪] (0~3s)   ──► 机器人距桌台 0.6~1.0m，箱子位于头部视野中央
          │
[阶段 2: 逼近微调] (3~6s)   ──► 自然前移 1~2 步 (系统记录 vx ≈ 0.15~0.25 m/s 平滑速度)，贴近桌沿停步
          │
[阶段 3: 双手合抱] (6~12s)  ──► 速度自然归 0，双手下压抱紧箱子并起吊至胸前
          │
[阶段 4: 原地转身] (12~18s) ──► 身体原地旋转踏步 1~2 步 (系统记录 ωz ≈ 0.3 rad/s 旋转速度)，对准目标放置台
          │
[阶段 5: 放置复位] (18~25s) ──► 双臂展开将箱子放置在目标桌台，身体轻退半步，本条结束
```

---

## 5. 数据采集标准作业程序（SOP）

### 5.1 安全与物理准备
1. **吊架悬挂**：必须使用龙门架/弹性挂绳吊住 G1 机器人的安全吊环，调节绳长使脚掌平稳着地且具有跌落保护。
2. **环境布置**：将起始桌台与目标桌台放置在机器人周围，保证箱子在机器人初始站立时的机载视野正中偏下区域。

### 5.2 启动机载服务
SSH 登录 G1 机器人（`ssh unitree@192.168.123.164`）：
```bash
cd ~/lerobot
./start_server.sh
```

### 5.3 启动采集流
在上位机启动 VR 动捕与数据录制服务（通过 `psi_rtc_sonic_client` 或 `run_data_exporter.py`）：
* 操作员佩戴好 VR 头显和手柄。
* 观察机载图像画面，确认相机连接正常（端口 5555）。

### 5.4 录制交互与质量规范
* **一条数据（1 Episode）**：必须是一次**完整、连贯、成功**的“走近 $\rightarrow$ 抱起 $\rightarrow$ 转身 $\rightarrow$ 放下”全过程（耗时约 25~35 秒）。
* **按键管理**：
  * 若操作失误（如没抱稳、掉落、动作犹豫），**立即废弃重录（按 `←` 或手柄重录组合键）**。
  * 顺利完成后保存进入下一条（按 `→`），并在 10~15 秒复位时间内将箱子放回起始点。

---

## 6. 采集数据结构与字段定义

### 6.1 目标 LeRobot v3.0 格式（用于训练与微调 `model/box_move_blue`）

依据 [`datasets/unitree_box_move_blue_full/meta/info.json`](file:///home/yichangfeng/lerobot/datasets/unitree_box_move_blue_full/meta/info.json) 与 [`convert_rubberhand_to_g1_v30.py`](file:///home/yichangfeng/lerobot/convert_rubberhand_to_g1_v30.py)，数据集中各字段的定义如下：

| 字段名称 | 数据类型与维度 | 物理定义与关节分布 |
| :--- | :--- | :--- |
| **`observation.images.global_view`** | `(480, 640, 3)` MP4 视频流 | G1 头部机载 RealSense RGB 图像，采样帧率 **30 FPS** |
| **`observation.state`** | `float32 [29]` | **机器人全身 29 关节当前实际弧度 (q)**：<br>• `[0:12]`：双腿 12 关节（左髋Pitch/Roll/Yaw, 左膝, 左踝Pitch/Roll, 右髋Pitch/Roll/Yaw, 右膝, 右踝Pitch/Roll）<br>• `[12:15]`：腰部 3 关节（Yaw, Roll, Pitch）<br>• `[15:22]`：左臂 7 关节（肩Pitch/Roll/Yaw, 肘, 腕Roll/Pitch/Yaw）<br>• `[22:29]`：右臂 7 关节（肩Pitch/Roll/Yaw, 肘, 腕Roll/Pitch/Yaw） |
| **`action`** | `float32 [18]` | **模型预测与训练监督动作向量**：<br>• `[0:7]`：左臂 7 关节目标弧度<br>• `[7:14]`：右臂 7 关节目标弧度<br>• `[14:18]`：底盘 4 维遥控速度指令 `[remote.lx, remote.ly, remote.rx, remote.ry]` |

---

## 7. 遥控速度指令的底层映射机制与真实数据分布

### 7.1 采集端：遥控器/外骨骼数据读取与归一化实现

在 [`src/lerobot/teleoperators/unitree_g1/unitree_g1.py:L125-L155`](file:///home/yichangfeng/lerobot/src/lerobot/teleoperators/unitree_g1/unitree_g1.py#L125-L155) 中，手柄遥控输入被统一归一化为 `[-1.0, 1.0]` 范围：

```python
# 1. 外骨骼手柄 (12-bit ADC，采样范围 0~4095，ADC_HALF = 2047.5):
self.lx = (raw16[JOYSTICK_X_IDX] - self.left_center_x) / 2047.5
self.ly = (raw16[JOYSTICK_Y_IDX] - self.left_center_y) / 2047.5
self.rx = (raw16[JOYSTICK_X_IDX] - self.right_center_x) / 2047.5
self.ry = (raw16[JOYSTICK_Y_IDX] - self.right_center_y) / 2047.5

# 2. 宇树无线遥控器 (unitree_sdk2py):
self.lx = self._joystick.lx.data  # 范围 [-1.0, 1.0]
self.ly = self._joystick.ly.data  # 范围 [-1.0, 1.0]
```

### 7.2 控制器端：底盘速度转换公式与缩放系数 (CMD_SCALE)

在 [`src/lerobot/robots/unitree_g1/controllers/gr00t_locomotion.py:L47-L220`](file:///home/yichangfeng/lerobot/src/lerobot/robots/unitree_g1/controllers/gr00t_locomotion.py#L47-L220) 中，底层 Groot WBC 步态控制器接收并处理 `action[14:18]`：

```python
# 1. 轴向转换映射
self.cmd[0] = ly   # 前进/后退线速度 (m/s)
self.cmd[1] = -lx  # 左右横移线速度 (m/s)
self.cmd[2] = -rx  # 原地旋转角速度 (rad/s)

# 2. 底层步态网络输入缩放系数
CMD_SCALE = [2.0, 2.0, 0.25]
self.groot_obs_single[:3] = self.cmd * np.array(CMD_SCALE)

# 3. 步态模式切换判定
cmd_magnitude = np.linalg.norm(self.cmd)
# 当合速度模长 < 0.05 时切换为原地站立平衡 (balance)，>= 0.05 时切入双足行走 (walk)
selected_policy_name = "balance" if cmd_magnitude < 0.05 else "walk"
```

在 [`src/lerobot/robots/unitree_g1/controllers/holosoma_locomotion.py:L156-L163`](file:///home/yichangfeng/lerobot/src/lerobot/robots/unitree_g1/controllers/holosoma_locomotion.py#L156-L163) 中，控制器配置了速度死区与硬限幅：
```python
ly = ly if abs(ly) > 0.1 else 0.0
lx = lx if abs(lx) > 0.1 else 0.0
rx = rx if abs(rx) > 0.1 else 0.0
ly = np.clip(ly, -0.3, 0.3)
lx = np.clip(lx, -0.3, 0.3)
self.cmd[:] = [ly, -lx, -rx]
```

### 7.3 训练数据集真实数值统计（源自 `unitree_box_move_blue_full/meta/stats.json`）

依据训练集元数据 [`datasets/unitree_box_move_blue_full/meta/stats.json:L66-L250`](file:///home/yichangfeng/lerobot/datasets/unitree_box_move_blue_full/meta/stats.json#L66-L250)（共 475,206 帧有效样本），`action` 后 4 维遥控速度指令的真实统计指标如下：

| 字段 | 对应索引 | 最小值 (min) | 10% 分位数 (q10) | 50% 分位数 (q50) | 90% 分位数 (q90) | 99% 分位数 (q99) | 最大值 (max) | 均值 (mean) | 标准差 (std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`remote.lx`** | 14 | -0.7634 | -0.0246 | -0.0059 | **+0.0041** | +0.2483 | +1.2928 | -0.0134 | 0.1503 |
| **`remote.ly`** | 15 | -0.7707 | -0.1435 | -0.0221 | **+0.2730** | **+1.1797** | +1.2977 | +0.0285 | 0.3975 |
| **`remote.rx`** | 16 | -0.7775 | -0.0208 | -0.0048 | **+0.8547** | **+1.1984** | +1.2586 | +0.1279 | 0.4068 |
| **`remote.ry`** | 17 | -0.7726 | -0.2115 | -0.0435 | -0.0051 | +0.0185 | +1.3382 | -0.0749 | 0.1176 |

### 7.4 原始动捕数据到 LeRobot v3.0 的转换映射

在 [`convert_rubberhand_to_g1_v30.py:L170-L194`](file:///home/yichangfeng/lerobot/convert_rubberhand_to_g1_v30.py#L170-L194) 中执行的转换规则：
- **State (43-D $\rightarrow$ 29-D)**：
  ```python
  state_29 = np.concatenate([raw_state[0:22], raw_state[29:36]])
  # 包含: [0:12] 双腿 + [12:15] 腰部 + [15:22] 左臂 + [22:29] 右臂 (剔除了 14 维灵巧手)
  ```
- **Action (43-D $\rightarrow$ 18-D)**：
  ```python
  # 前 14 维为双臂目标关节角度，后 4 维为遥控速度指令
  action_18 = np.concatenate([raw_action[15:22], raw_action[29:36], [remote.lx, remote.ly, remote.rx, remote.ry]])
  ```
  在由空间人体动捕位移提取速度时，映射公式为：
  $$\text{remote.ly} = v_x, \quad \text{remote.lx} = -v_y, \quad \text{remote.rx} = -\omega_z, \quad \text{remote.ry} = 0.0$$

---

## 8. 微调模型推荐数据量与训练建议

### 8.1 推荐采集数据量
`model/unitree_box_move`（$\pi_{0.5}$ VLA 大模型）已具备强大的双臂运动学和下肢协调先验：

| 任务场景 | 推荐数据量 | 耗时估计 |
| :--- | :---: | :--- |
| **固定场景对齐（相同任务，新桌台、新背景光照）** | **30 ~ 50 条** | 约 30 ~ 45 分钟完成采集 |
| **空间轻度泛化（箱子初始位置 ±15cm 偏移，轻微角度偏差）** | **50 ~ 80 条** | 约 1 小时完成采集 |
| **多目标/新箱子形态（不同尺寸箱子，多个放置位置）** | **100 ~ 120 条** | 约 1.5 ~ 2 小时完成采集 |

### 8.2 微调训练命令示例
在上位机使用 `lerobot-train` 进行微调：
```bash
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="./datasets/unitree_box_move_finetune" \
    --policy.type=pi05 \
    --policy.pretrained_path="model/unitree_box_move" \
    --output_dir="./outputs/train/unitree_box_move_finetuned" \
    --job_name="g1_box_move_finetune" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --batch_size=32 \
    --steps=5000 \
    --policy.scheduler_decay_steps=5000 \
    --save_freq=2500 \
    --wandb.enable=false
```

### 8.3 实机部署闭环验证
```bash
./run_rollout.sh \
    --policy.path="./outputs/train/unitree_box_move_finetuned/checkpoints/005000/pretrained_model" \
    --task="pick up the box, turn right, and place it on the table" \
    --robot.is_simulation=false \
    --robot.zero_locomotion_cmd=false \
    --display_data=true
```
