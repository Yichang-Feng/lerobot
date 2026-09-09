# LeRobot + Unitree G1 + PI0.5 实机与仿真部署交接文档 (HANDOVER.MD)

- **更新时间**: 2026-09-02
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
- **`run_rollout.sh`**: **快捷调试与参数调优启动脚本**（顶部集中定义了模型、任务、仿真/实机、RTC 队列、插值倍率等全部常用参数，开箱即用）。
- **`rollout_config.yaml`**: **YAML 格式统一配置文件**（支持 `lerobot-rollout --config_path=rollout_config.yaml` 一键启动与调参）。
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

## 3. 当前运行状态与效果对比

### 3.1 快捷调试与执行命令（已全面精简）

#### 方式 1：使用快捷调试脚本 `run_rollout.sh`（推荐，最省心）
直接打开 [**`run_rollout.sh`**](file:///home/yichangfeng/lerobot/run_rollout.sh) 修改顶部参数，或在命令行直接传参覆盖：
```bash
# 默认启动（仿真 + 任务 "move blue box"）
./run_rollout.sh

# 命令行快速覆盖参数（例如切换任务、开启可视化或切换实机）
./run_rollout.sh --task="pick up blue box" --display_data=true
./run_rollout.sh --robot.is_simulation=false --robot.zero_locomotion_cmd=true
```

#### 方式 2：使用 YAML 配置文件 `rollout_config.yaml`
在 [**`rollout_config.yaml`**](file:///home/yichangfeng/lerobot/rollout_config.yaml) 中集中修改参数，然后运行：
```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout --config_path=rollout_config.yaml
```

#### 方式 3：精简 CLI 命令行直调
由于已将所有通用参数固化为默认配置，直接调用 CLI 时仅需指定必要参数：
```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout \
    --policy.path=model/box_move_blue \
    --task="move blue box" \
    --robot.is_simulation=true \
    --display_data=false
```

> **全量参数展开（系统内部默认等价于）：**
> ```bash
> lerobot-rollout \
>     --strategy.type=base \
>     --inference.type=rtc \
>     --inference.queue_threshold=40 \
>     --interpolation_multiplier=3 \
>     --policy.path=model/box_move_blue \
>     --policy.device=cuda \
>     --policy.dtype=bfloat16 \
>     --robot.type=unitree_g1 \
>     --robot.is_simulation=true \
>     --robot.controller=GrootLocomotionController \
>     --robot.locomotion_mode=stand \
>     --robot.cameras='{"global_view": {"type": "zmq", "server_address": "localhost", "port": 5556, "camera_name": "head_camera", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}}' \
>     --task="move blue box" \
>     --duration=1000 \
>     --fps=30 \
>     --display_data=false
> ```

---

## 4. 实机调试问题深度剖析与应对方案 (2026-09-07)

针对 Unitree G1 接入 PI0.5 策略实机联调中暴露的两个典型关键问题，进行系统的问题描述、底层机理剖析与应对方案总结。

---

### 4.1 问题 1：从 Locomotion 接入 VLA 瞬间手臂猛甩、抬起过快

#### 1. 问题描述
* 机器人在平衡控制器（Locomotion WBC）站立状态下，通过网络接入 VLA 动作流后，双臂会立刻去拟合抬手姿态。
* **现象对比**：
  * **5000 步模型**：抬手幅度相对适中，更接近示教采集的姿态，但动作依然偏突兀。
  * **1000 步模型**：抬手幅度极大且速度极快，电机出现剧烈机械冲击，易造成身体晃动失稳甚至跌倒意外。
* **核心诉求**：如何在既准确跟随 VLA 指令的前提下，消除开局骤突，让机械臂平缓、受控地过渡到初始作业姿态。

#### 2. 底层机理分析
1. **初始位置阶跃突变（Step-Input Discontinuity）**：
   * 机器人站立待命时，双臂通常处于自然下垂或默认归零状态（$q \approx 0$）。
   * 示教数据中，录制的第一帧往往已经是准备抱箱的预备抬手姿态（肩、肘关节已有明显角度偏差）。
   * VLA 客户端一旦接入，首帧输出的目标角度与机器人当前真实姿态存在数十度的跳变。
2. **欠拟合模型的方差放大**：
   * 1000 步模型处于欠拟合阶段，输出置信度低、方差大，首帧往往预测出幅度夸张的极端关节角；5000 步模型因在首帧附近拟合较好，输出相对温和。
3. **高刚度 PD 控制器的扭矩冲激**：
   * 底层服务端收到目标后直接写入 `motor_cmd.q`。瞬时位置偏差 $e = q_{target} - q_{current}$ 极大。
   * 电机端高刚度 PD 控制器输出扭矩 $\tau = k_p \cdot e$ 瞬间饱和，电机爆发最大加速度，导致手臂猛烈甩动。

#### 3. 应对与优化方案
1. **1.5 秒软启动余弦平滑过渡（Soft-Start Cosine S-Curve）**：
   * 当检测到 VLA 首次连接或刚执行完 Reset 时，捕捉当前机器人机械臂的真实实际角度 $q_{real\_start}$。
   * 在随后的 $T = 1.0 \sim 1.5$ 秒内，采用平滑余弦权重 $\alpha(t) = \frac{1}{2}\left(1 - \cos\left(\frac{\pi t}{T}\right)\right)$ 进行加权融合：
     $$q_{cmd}(t) = (1 - \alpha(t)) \cdot q_{real\_start} + \alpha(t) \cdot q_{vla\_target}(t)$$
   * 初始时刻误差为 0，扭矩为 0，随后 1.5 秒内优雅过渡到 VLA 控制姿态。
2. **关节角速度限幅器（Slew-Rate Limiter）**：
   * 限制机械臂各关节单步最大角速度不超过 **$1.2 \sim 1.5\text{ rad/s}$**。
   * 单步（20ms）最大变化量 $\Delta q_{max} = v_{max} \cdot \Delta t \approx 0.024\text{ rad}$。无论策略输出多大跳变，物理电机绝不超速。
3. **底盘速度淡入**：
   * 在软启动过渡期内，底盘移动/转向遥控速度也同步乘以 $\alpha(t)$，避免手臂未到指定开度底盘即急剧移动造成失衡。

---

### 4.2 问题 2：5000 步犹豫不前 vs 1000 步仅限首轮成功（为什么过拟合反而不敢动？）

#### 1. 问题描述
* **5000 步模型**：机器人看到箱子后表现为“犹犹豫豫”，有抱箱趋势但又停滞在半空不敢抱合。
* **1000 步模型**：能做出果断抱箱并转身的连贯动作，但**仅限第一次运行**；第二次把机器人和箱子转回原位后，动作变形失效。
* **核心疑问**：
  1. 为什么 5000 步效果反而不如 1000 步？按直觉理解，如果过拟合，不应该更加极致地跟随示教动作吗？
  2. 为什么 1000 步第二次就不行了？如何在显存不重新加载模型的前提下彻底清空上下文？

#### 2. 深度理论剖析：“为什么过拟合反而不敢动？”（静止吸引子问题）
直觉认为“过拟合 = 更激进地模仿人”，但在模仿学习与流匹配（Flow Matching / Diffusion Policy）中，过拟合往往导致**“静止瘫痪”（The Zero-Velocity Attractor / Policy Freezing）**：

1. **示范数据中的低速停顿偏差（Zero-Velocity Bias）**：
   * 人类在遥控采集抱箱任务时，在手爪靠近纸箱对准边缘的关键时刻，为了操作精确，动作通常极慢，甚至有数十毫秒的观察微停。
2. **协变量漂移（Covariate Shift）与速度矢量坍塌**：
   * 流匹配网络拟合的是动作的速度向量场 $v(x_t, t)$。
   * 5000 步微调时，Action Expert 被深度拘束在示范轨迹狭窄的超管流内。
   * 实机闭环中，环境光照轻微变化、箱子摆放偏移数厘米、或下肢站姿有微弱倾斜，当前的视觉与状态观测便落入了未见区域（Out of Distribution, OOD）。
   * 在 OOD 区域，过度拟合的网络无法泛化，各个模态预测的速度矢量在各方向上相互抵消，**输出的速度向量模长急剧萎缩接近于 0**。
   * 表现为：机器人“知道要抱（方向有轻微倾向），但速度场大小趋近于 0，双臂悬在半空打摆子、犹豫不前”。
3. **欠拟合（1000 步）为何反而动作果断？**：
   * 1000 步时，网络仅捕获了宏观动力学大趋势（“视野出现箱子 -> 双臂抱合 -> 转向”）。
   * 它未被局部微小停顿特征绑架，且大量保留了 $\pi_{0.5}$ 预训练底模原有的**物理动作流动先验（Action Flow Prior）**，因而动作大开大合，敢于向前扑击抱箱。
4. **黄金步数规律**：
   * 1000 步太粗糙（抗干扰差、首帧突变大），5000 步陷入静止陷阱；**最佳效果通常落在 2000 ~ 3000 步（如 Checkpoint 002000 / 003000）**，兼顾大动作的推进力与末端对准精度。

#### 3. 深度机理剖析：“为什么 1000 步第二次执行就失效？”
1. **RTC 残留前缀污染（Left-over Chunk & Prefix Contamination）**：
   * `run_vla.sh` 默认启用了实时分块推理（`--inference.type=rtc`）。
   * RTC 机制每次前向预测时，会抽取上一个 Chunk 尚未消费完的动作切片（`prev_chunk_left_over`）作为前缀约束，保障轨迹连续。
   * 当机器人完成第一次“抱箱+转身”后，若未做系统级重置，RTC 队列中仍填充着**上一轮“转身阶段/任务末尾”的大角速度与抱死手臂的前缀向量**。
   * 当操作员把机器人或箱子转回原处时，当前眼前的相机画面是“开局待命”，而 RTC 强行喂给策略模型的前缀却是“正在转身”，**视觉感知与历史前缀产生剧烈语义撕裂**，输出直接崩溃。
2. **动作插值器与底盘 WBC 状态残留**：
   * 上位机动作插值器与下肢控制器的积分状态未归零，上一轮残余的遥控转向偏置依然存在。

#### 4. 解决方案：显存常驻下的“0 耗时热重置”（Hot Reset）
避免每次测试都通过杀死脚本重新加载 9.35GB 权重（每次冷启动耗时 25~35 秒）：

1. **上位机上下文清空**：
   * 针对 `RTCInferenceEngine`：调用 `reset()`，立即清空 `ActionQueue`、清除 `prev_chunk_left_over`，重置预处理器与后处理器状态，并丢弃前一轮陈旧的 Observation。
   * 针对策略模型：调用 `policy.reset()`，清空动作缓冲队列，使流匹配从纯高斯白噪声开始去噪。
   * **9.35GB 模型常驻 GPU 显存，无需重新载入，0 秒完成逻辑重置！**
2. **机器人端状态回正**：
   * 向动作端口（6002）下发 `{"cmd": "reset"}`。
   * 服务端拦截该指令后，自动将底盘遥控指令清零，并驱动双臂通过 2~3 秒平滑插值返回默认待命姿态。
3. **实机推荐调试手段**：
   * **交互式模式（`--interactive`）**：在 VLA 启动脚本中开启交互式会话，完成一次任务后在终端输入 `/reset` 瞬间重置，重新摆放箱子后输入 `/start` 即可干净利落地开始下一轮。
   * **换用中间步数权重**：推荐实机重点评估 `aligned/003000` 或 `aligned/002000`，彻底摆脱 5000 步的犹豫停滞问题。

