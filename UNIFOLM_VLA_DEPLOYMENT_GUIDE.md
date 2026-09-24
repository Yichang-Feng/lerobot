# UnifoLM-VLA 嵌入 LeRobot 实机部署架构与运行指南

本文档介绍如何将 **UnifoLM-VLA-Base** 大模型无缝嵌入到您已有的 `~/lerobot` 项目中，并通过现有的 `./run_vla.sh` 脚本在 **Unitree G1** 人形机器人上完成实机部署与控制。

---

## 一、系统架构设计与复用机制

在您的现有 LeRobot 系统中，机器人控制采用 **ZMQ 双向异步通信** 与 **RTC (Real-Time Chunking) 异步推理** 架构：

```
                           【终端 1：实机驱动/仿真服务】
                         (unitree_real / mujoco_sim)
                                  │      ▲
              6001 端口 (ZMQ SUB) │      │ 6002 端口 (ZMQ PULL/PUSH)
             29-DoF 关节状态 & IMU │      │ 动作流: 18-DoF (无夹爪) 或 
              6004 端口 (ZMQ SUB) │      │        20-DoF (带夹爪 action.gripper: {right, left})
             夹爪当前角度 {left, right}│      │
              5556 端口 (ZMQ SUB) │      │
                  机载单相机视频流 │      │
                                  ▼      │
┌────────────────────────────────────────────────────────────────────────┐
│                   【终端 2：LeRobot 运行时 (./run_vla.sh)】               │
│                                                                        │
│   UnitreeG1Client ─────────────────────────────────────────────────┐   │
│   (负责 ZMQ 协议收发、{left,right}解析、动作封装与 S 曲线平滑接入)         │   │
│                                                                    ▼   │
│   ┌────────────────────────────────────────────────────────────┐       │
│   │            UnifolmVLA 策略适配器 (UnifolmVLAPolicy)         │       │
│   │                                                            │       │
│   │  1. 状态映射: 29 关节 + 实测夹爪角度 ──> 23 维末端 proprio  │       │
│   │  2. 视觉适配: 单相机图像 (480x640) ─> Resize (224, 224, 3) │       │
│   │  3. 提示词融合: "The task is '<instruction>'."             │       │
│   │  4. 核心推理: Qwen2.5-VL-7B + DiT 动作扩散模型             │       │
│   │  5. 动作重构: 23 维末端动作 ──IK解算──> 组装为 20 维控制块    │       │
│   │     (14臂关节 + 2夹爪[18右/19左] + 4摇杆轴)                 │       │
│   └────────────────────────────────────────────────────────────┘       │
│                                  │                                     │
│   RTC 引擎 (Real-Time Chunking): 30Hz 异步重规划 + 动作插值下发 ────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 核心设计优势
1. **协议向前兼容**：胶手（无夹爪）机器人依然默认使用 18-DoF 动作与 6001/6002 通信；带夹爪机器人开启 `--gripper` 即可自动切换为 20-DoF 与 6004 夹爪角度监听。
2. **夹爪角度闭环反馈**：从 6004 端口接收的 `{left, right}` 物理夹爪角度实时注入 23 维末端 proprio 的第 18（右爪）与 19（左爪）通道，输入大模型形成完整状态闭环。
3. **安全平滑下发**：无论单臂、双臂或夹爪动作，均经过余弦 S 曲线平滑接入，夹爪角度下发限制在安全量程 `[0.0, 5.0]` 内。
4. **启动命令零学习成本**：依然通过 `./run_vla.sh --gripper --policy.path=...` 一键启动。

---

## 二、关键技术差异适配点

### 1. 相机视点适配（单相机 vs 三相机多视角）
- **基线模型要求**：`~/unifolm-vla/model/UnifoLM-VLA-Base` 训练配置中 `use_wrist_image: true`，网络原生按三相机序列组织视觉 Token：
  1. **第 1 路（主全局视角）**：`observation.images.global_view`
  2. **第 2 路（左手腕视角）**：`observation.images.left_wrist`
  3. **第 3 路（右手腕视角）**：`observation.images.right_wrist`
- **UnifoLM-VLA 适配与容错机制**：
  - 策略适配器内部的 `_extract_images()` 严格按 `[全局主视角, 左手腕, 右手腕]` 顺序组织输入并统一调整尺寸为 `(224, 224, 3)`。
  - **占位安全补齐**：若现场仅接通单路全局相机而手腕相机未就绪，策略自动使用全黑图像进行 Padding 占位补齐，确保送入 Qwen2.5-VL 视觉编码器的图像 Token 结构与长度与模型微调权重严格一致，杜绝序列维度不匹配报错。
  - 实机运行时，只需在 `./run_vla.sh` 加上 `--wrist-cameras`（可选指定 `--left-wrist-port=5557`、`--right-wrist-port=5558`）即可直接拉取多路手腕画面。

### 2. 状态与动作维度映射（14 维双臂 vs 7 维单臂）
- **现有系统**：
  - 观测状态为 29 关节（或 14 臂关节，左臂 7 + 右臂 7）。
  - 下发动作为 18 维（左臂 7 + 右臂 7 + 4 轴虚拟摇杆）。
  - 无夹爪（使用胶手 / rubberhand）。
- **UnifoLM-VLA 适配**：
  - **输入提取**：根据配置项 `arm_side`（默认为 `right` 右臂）：
    - 若输入 29 维状态，右臂取 `state[22:29]`；若 14 维状态，右臂取 `state[7:14]`。
    - 该 7 维状态输入经过 UnifoLM-VLA 的 `norm_stats_proprio` 归一化。
  - **动作重构**：
    - 模型预测出 `(16, 7)` 的动作块，经 `norm_stats_action` 反归一化后：
    - **活动臂（右臂）**：赋值为模型预测的 7 维目标动作；
    - **非活动臂（左臂）**：自动锁存并维持当前实测位置，防止单臂操作时另一侧手臂下垂掉电；
    - **摇杆轴（14:18）**：补 0.0；
    - 组装成标准 `(16, 18)` 动作块直接交付给 LeRobot RTC 执行引擎。

---

## 三、文件组织与代码改动

我们已为您在 `~/lerobot` 项目中建立了以下文件：

```
~/lerobot/
├── src/lerobot/policies/unifolm_vla/
│   ├── __init__.py                       # 模块导出声明
│   ├── configuration_unifolm_vla.py       # UnifolmVLAConfig 策略配置类
│   ├── modeling_unifolm_vla.py            # UnifolmVLAPolicy 策略实现与维度适配
│   └── processor_unifolm_vla.py           # 前后处理流水线构建
├── src/lerobot/policies/__init__.py      # 已注册 UnifolmVLAConfig 导出
└── outputs/train/unifolm_vla_base/
    └── config.json                       # 部署策略配置文件
```

---

## 四、具体实机部署运行步骤

### 步骤 1：确认前置依赖环境
确保当前 Python 环境（如 `conda activate lerobot` 或 `unifolm-vla`）已安装 Qwen-VL 相关库：
```bash
pip install omegaconf qwen-vl-utils json_numpy
```

### 步骤 2：测试网络连通性（不载入大模型）
在启动控制前，建议先测试终端 1 与终端 2 之间的 ZMQ 端口是否通畅：
```bash
cd ~/lerobot
./run_vla.sh --check --real
```
如果看到各个端口（6001、6002、5556）连接测试通过，即可进入实车运行。

### 步骤 3：单臂纯视觉推断冒烟测试（无需机器人电机动作）
如果您想在不通电使能机器人关节的情况下，验证模型视觉前向传播与动作生成：
```bash
cd ~/lerobot
./run_vla.sh \
    --policy.path=outputs/train/unifolm_vla_base \
    --camera-only \
    --task="clean the table"
```
此时终端会读取真实相机视频流，关节使用零位虚拟状态，并在屏幕上打印动作块推理时延与数值。

### 步骤 4：实机完整闭环运行 (Real Deployment)
在机器人处于安全站立、挂架保护或有人看护的状态下：

```bash
cd ~/lerobot
./run_vla.sh \
    --policy.path=outputs/train/unifolm_vla_base \
    --real \
    --task="stack the red block on the blue block"
```

#### 常用控制参数说明：
- `--policy.path`：指定为 `outputs/train/unifolm_vla_base`。
- `--real`：自动将目标 IP 指向实机 `192.168.123.164`，相机端口为 `5556`。
- `--sim`：自动将目标 IP 指向本机 `localhost`（用于与 Mujoco 仿真环境联调）。
- `--task`：自然语言任务指令（如 `"clean the table"`、`"wipe the table"`、`"fold the towel"` 等）。
- `--auto`（默认开启）：自动监听 6000 端口，当手柄切入 VLA 模式时自动启动并带有余弦 S 曲线平滑介入，切出手柄时 0.2s 内自动释放控制权。
- `--record`：添加此参数可在部署运行时同步将相机视频、预测动作和关节状态全量落盘，便于后续复盘分析。
- `--gripper` 或 `--enable-gripper`：启用夹爪通信通道（接收 6004 夹爪角度并在 6002 action 中封装 `gripper: {right, left}`）。
- `--gripper-port=<端口>`：指定夹爪状态接收端口（默认 `6004`）。
- `--gripper-ip=<IP>`：指定夹爪发布端 IP（默认与 `--robot-ip` 相同）。

---

## 五、任务切换与参数定制

如果希望切换左臂控制，或更换不同的预训练操作任务，只需修改 [`outputs/train/unifolm_vla_base/config.json`](file:///home/yichangfeng/lerobot/outputs/train/unifolm_vla_base/config.json)：

```json
{
  "type": "unifolm_vla",
  "arm_side": "right",            // 切换为 "left" 可控制左臂，"both" 或 "dual" 为双臂
  "use_gripper": false,           // 设置为 true 启用夹爪 20-DoF 动作空间
  "task_name": "g1_clean_table",  // 切换对应的任务归一化统计量
  "chunk_size": 16,
  "n_action_steps": 16,
  "default_task": "clean the table"
}
```

支持的 `task_name` 包括：
- `g1_stack_block`（叠积木）
- `g1_clean_table`（清理桌面）
- `g1_wipe_table`（抹布擦桌子）
- `g1_pack_pencilbox`（装笔盒）
- `g1_erase_board`（擦黑板）
- `g1_pour_medicine`（倒药）
- `g1_pack_pingpong`（收乒乓球）
- `g1_fold_towel`（叠毛巾）

---

## 六、带夹爪机器人 (Gripper) 通信协议与实机操作指南

当使用配备夹爪（而非橡胶手）的人形机器人时，系统通过**双向独立信道**实现夹爪状态感知与动作下发：

### 1. 通信信道与数据协议规范

| 阶段 | 传输方向 | 传输端口 | 传输格式 | 字段与数值范围 |
| :--- | :--- | :--- | :--- | :--- |
| **夹爪状态接收** | 机器人/驱动端 ➔ VLA 客户端 | 默认 `6004` (ZMQ SUB, CONFLATE=1) | JSON 字典 或 `data` 广播前缀 | `data{"left": float, "right": float}` 或 `{"data": {"left": float, "right": float}}`<br>（兼容平铺 `{"left": float, "right": float}`）<br>开合角度数值范围: `0.0` (全闭) ~ `5.0` (全开) |
| **动作统一发布** | VLA 客户端 ➔ 机器人执行端 | 默认 `6002` (ZMQ PUSH) | JSON 字典 (`cmd: action`) | 在已有 `action` 字典中嵌套 `gripper`：<br>`"gripper": {"right": float, "left": float}`<br>数值范围严格限制在 `0.0 ~ 5.0` |

#### 接收 6004 报文示例 (Port 6004)
Unitree 灵巧手/夹爪驱动（`rt/dex1/state`）：
```json
{
  "topic": "rt/dex1/state",
  "data": {
    "right": {
      "q": 1.00369895,
      "dq": 0.00294524315,
      "tau_est": -0.048828125
    },
    "left": {
      "q": 0.0359565094,
      "dq": 0.0,
      "tau_est": -0.01953125
    }
  }
}
```
*注：客户端自动从 `right` 和 `left` 字典中提取 `q`（关节位置/角度），并同时向下兼容直接传标量 `{"data": {"left": 4.80, "right": 4.85}}` 或 `data {"left": ..., "right": ...}` 等格式。*

#### 下发 JSON 报文示例 (Port 6002)
```json
{
  "cmd": "action",
  "action": {
    "kLeftShoulderPitch.q": 0.25,
    "kLeftShoulderRoll.q": 0.12,
    "kLeftShoulderYaw.q": -0.05,
    "kLeftElbow.q": 0.88,
    "kLeftWristRoll.q": 0.0,
    "kLeftWristPitch.q": 0.0,
    "kLeftWristYaw.q": 0.0,
    "kRightShoulderPitch.q": -0.15,
    "kRightShoulderRoll.q": -0.30,
    "kRightShoulderYaw.q": 0.10,
    "kRightElbow.q": 0.95,
    "kRightWristRoll.q": 0.0,
    "kRightWristPitch.q": 0.0,
    "kRightWristYaw.q": 0.0,
    "remote.lx": 0.0,
    "remote.ly": 0.0,
    "remote.rx": 0.0,
    "remote.ry": 0.0,
    "gripper": {
      "right": { "q": 0.0 },
      "left": { "q": 5.0 }
    }
  },
  "timestamp": 1726818293.123
}
```

### 2. 内部维度映射关系

- **UnifoLM-VLA 23-DoF 末端空间**：
  - `0:3` 左手位置 (L_xyz), `3:9` 左手姿态 6D (L_rot6d)
  - `9:12` 右手位置 (R_xyz), `12:18` 右手姿态 6D (R_rot6d)
  - **Index 18 = 右夹爪目标** (`action_right_gripper`, 0.0~5.0)
  - **Index 19 = 左夹爪目标** (`action_left_gripper`, 0.0~5.0)
  - `20:23` 腰部与底盘 (Waist 3-DoF)
- **LeRobot 20-DoF 动作空间**：
  - `0:7` 左臂 7 关节，`7:14` 右臂 7 关节
  - `14` 右夹爪目标，`15` 左夹爪目标
  - `16:20` 4 轴虚拟遥控/摇杆

### 3. 带夹爪实机常用命令

#### ① 快速排查连通性 (2秒速查，包含夹爪 6004 端口)
```bash
./run_vla.sh --real --gripper --check
```
> 若夹爪驱动尚未启动，控制台会明确高亮提示 `[FAIL 连接被拒绝 Connection Refused]`，并指出 `目标机上没有程序监听端口 6004`。

#### ② 仿真环境 / 本机带夹爪测试
```bash
./run_vla.sh \
    --policy.path=outputs/train/unifolm_vla_base \
    --gripper \
    --task="pick up the block"
```

#### ③ 实机带夹爪自动闭环控制
```bash
./run_vla.sh \
    --policy.path=outputs/train/unifolm_vla_base \
    --real \
    --gripper \
    --auto \
    --task="pick up the block"
```

#### ④ 自定义夹爪端口与 IP (可选)
如果夹爪状态广播运行在特定端口或独立工控机上：
```bash
./run_vla.sh \
    --policy.path=outputs/train/unifolm_vla_base \
    --real \
    --gripper \
    --gripper-port=6004 \
    --gripper-ip=192.168.123.164
```
