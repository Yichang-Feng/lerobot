# LeRobot + Unitree G1 + PI0.5 实机与仿真部署交接文档 (HANDOVER.MD)

- **更新时间**: 2026-09-10
- **工作目录**: `/home/yichangfeng/lerobot`
- **上位机环境**: `/home/yichangfeng/miniforge3/envs/lerobot` (Python 3.12, GPU: NVIDIA GeForce RTX 4090 D 24GB, 上位机 IP: `192.168.123.213`)
- **多卡训练机环境**: `zhengwang@CNBESLPD260001` (Python 3.12, GPU: RTX 5090)
- **机器人端环境**: Unitree G1 机载电脑 (`unitree@192.168.123.164`, Ubuntu 20.04, Python 3.8)

---

## 1. 项目架构与通信拓扑

### 1.1 总体架构
本项目在 **LeRobot** 框架下，结合 **PI0.5 VLA 策略模型（23 亿参数）** 与 **GrootLocomotionController / SonicWholeBodyController / unitree_g1_client**，实现对 **Unitree G1 (29-DoF)** 人形机器人的闭环控制。

- **输入**: 
  - `observation.images.global_view`: 480×640×3 RGB 图像（机载 RealSense 摄像头或 ZMQ 视频流，默认端口 `5556` 或 `5555`）。
  - `observation.state`: 29 维关节位置状态向量（由 G1 机器人 `6001` 端口广播）。
- **输出**: 
  - 18 维 Action 向量（前 14 维为双臂关节目标角度，后 4 维为 `remote.lx`, `remote.ly`, `remote.rx`, `remote.ry` 遥控速度指令，下发至 `6002` 端口）。
- **RTC 异步推理**:
  - LeRobot 原生 RTC 机制，上位机 RTX 4090 D 以 ~30Hz 频率实时推理下发动作块（Action Chunk），队列满阈值与插值平滑处理。

### 1.2 网络与通信拓扑
```text
┌──────────────────────────────────────────────┐                ┌──────────────────────────────────────────────┐
│         本机上位机 (GPU Workstation)          │                │            Unitree G1 机器人本体             │
│        IP: 192.168.123.213                   │                │          IP: 192.168.123.164                 │
│        网卡: enx6c1ff724495a                 │                │                                              │
│                                              │ 千兆以太网直连  │  【机载轻量服务 robot_server】                │
│  【lerobot-rollout / run_vla 上位机推理】    │◄──────────────►│   ├─ ZMQ Port 6002 (PULL 接收 LowCmd)        │
│   ├─ PI0.5 Policy (RTX 4090 D 显存常驻)      │                │   ├─ ZMQ Port 6001 (PUB 广播 LowState)       │
│   ├─ unitree_g1_client (轻量网络客户端)       │                │   └─ ZMQ Port 5556/5555 (PUB 广播机载图像)   │
│   ├─ 原生 3-Subtask 状态自动流转引擎         │                │  【底层 DDS】                                │
│   └─ RTC 动作插值与流控                      │                │   └─ 宇树原厂电机执行器 (29-DoF)             │
└──────────────────────────────────────────────┘                └──────────────────────────────────────────────┘
```

---

## 2. 仓库核心文件与工具清单

### 2.1 推理部署与联调脚本
- **`run_vla.sh`**: **VLA 大模型主推理启动脚本（核心入口）**
  - 已完成**原生 3-Subtask 自动流转与动态环境自适应**集成。
  - 自动探测多用户/多机器 Python 虚拟环境与 `LD_LIBRARY_PATH`。
  - 支持单任务与 3 阶段子任务模式，采用单进程直接替换（`exec`），键盘 TTY 直达，`Ctrl+C` 显存瞬间释放清零。
- **`check_zmq_connection.py`**: 快速网络端口与 ZMQ 连通性测试脚本（毫秒级排查通信，无需加载 2.3B 大模型）。
- **`run_vla_subtask.py`**: （*已弃用*）旧版外部 subprocess supervisor 脚本。由于重定向管道破坏 TTY 且产生孤儿进程，已被 `run_vla.sh` 的原生集成方案彻底替代。

### 2.2 训练与微调脚本及配置
- **`run_train.sh`**: **通用跨机器模型微调训练脚本**
  - 动态检测并绑定运行环境（支持 `miniforge3`、`CONDA_PREFIX`、`VIRTUAL_ENV`）。
  - 自动链接 `libstdc++.so.6` 动态库，完美规避 Scipy / PyTorch CXXABI 报错。
  - 统一超参数配置：`batch_size=8`, `steps=5000`, `save_freq=1000`, `log_freq=50`。
- **`train_pi05_config.yaml`**: 4090 D（24GB 显存）微调配置文件（包含 LoRA/Full-FT 策略、图像分辨率与设备配置）。
- **`train_pi05_5090_config.yaml`**: 5090 机器微调配置文件（优化显存利用与加速编译）。

### 2.3 数据集处理工具链与 SOP
- **`Dataset_Cleaning_Guide.md`**: 数据集清洗、子任务切分与 State 代替 Action 处理 SOP 手册。
- **`dataset_tools/`**: 集中整理的数据处理工具包：
  - `align_velocity_dataset.py`: 速度指令与动作时间对齐工具。
  - `convert_rubberhand_to_g1_v30.py`: 原始灵巧手/手套数据转 G1 标准 29-DoF 格式。
  - `evaluate_state_action.py`: 动作抖动度、状态平滑度量化分析工具。
  - `merge_sonic_datasets.py`: 跨采集批次数据集融合脚本。
  - `replay_g1_dataset.py`: 动态回放演示数据，终端同步输出 HUD 指标。
  - `sanitize_sonic_dataset.py`: 异常值清理与丢帧修复。
  - `create_subtask_dataset.py`: 转向任务 3-Subtask 切分脚本（剔除 Episode 32, 48 脏数据，产出 85 条演示）。
  - `create_subtask_dataset_inplace.py`: 原地任务 2-Subtask 切分脚本（基于手腕高度与下放速度启动沿）。
  - `replace_action_with_state.py`: 使用机械臂低抖动真实 State 代替遥操抖动 Action（生成 shift=0 与 shift=1 数据集）。
  - `verify_subtask_splits.py`: 转向任务 3-Subtask 阶段切分曲线与时序可视化工具。
  - `verify_subtask_splits_inplace.py`: 原地任务 2-Subtask 阶段切分曲线与时序可视化工具。

---

## 3. 数据集体系与模型 Checkpoints 清单

### 3.1 数据集目录 (`datasets/`)
1. **`datasets/g1_box_pick_turn_v30`**:
   - 基础高质量完整长任务数据集（单任务指令："pick up the box, turn right, and place it on the table"）。
2. **`datasets/g1_box_pick_turn_v30_subtasks`**:
   - 3-Subtask 细粒度语言分段数据集（85 个 Episode，59,825 帧），包含三段式子任务文本。
3. **`datasets/g1_box_pick_turn_v30_subtasks_state_as_action_shift0`**:
   - 3-Subtask 数据集，上肢 14 个关节用无抖动的真实 State 代替原始 Action（`shift=0`，当前时刻状态对齐）。
4. **`datasets/g1_box_pick_turn_v30_subtasks_state_as_action_shift1`**:
   - 3-Subtask 数据集，上肢 14 个关节用下一时刻真实 State 代替原始 Action（`shift=1`，下一步目标对齐）。

### 3.2 训练产出模型权重 (`outputs/train/`)
1. **`outputs/train/pi05_box_pick_turn_v30_subtasks/checkpoints/005000/pretrained_model`**:
   - **【当前默认主力部署模型】**：基于 3-Subtask 分段数据集微调训练 5000 步的最新模型权重。
2. **`outputs/train/pi05_subtasks_state_as_action_shift1/checkpoints/005000/pretrained_model`**:
   - 基于 State-as-Action (shift=1) 数据集训练的模型，用于消除遥操固有手臂高频抖动。
3. **`outputs/train/pi05_subtasks_state_as_action_shift0/checkpoints/005000/pretrained_model`**:
   - 基于 State-as-Action (shift=0) 数据集训练的模型对比权重。

---

## 4. 历史部署问题根因排查与闭环总结

在之前的联调中，使用外置包装器 `run_vla_subtask.py` 暴露了 4 个致命问题。现已全部彻底根治，根因与解决方案对比如下：

| 故障现象 | 根因排查结论 | 根治修复方案 | 状态 |
| :--- | :--- | :--- | :---: |
| **1. 连接等待期输入新 IP 无法修改** | `run_vla_subtask.py` 采用了 `subprocess.Popen(..., stdin=PIPE)` 重定向标准输入，破坏了 TTY 控制台环境，导致底层 `unitree_g1_client.py` 的 `sys.stdin.isatty()` 为 `False`，所有终端键盘按键被静默丢弃。 | 废除外层管道，终端标准输入直接接管进程；同时解除 `unitree_g1_client.py` 中 `_check_stdin` 的 `isatty` 限制，并在热切换时联动更新 `action_ip` 和即时重连相机。 | **已解决** |
| **2. Ctrl+C 中断后显存泄露 (残留 16GB)** | 外部 Python 进程作为父进程启动底层推理进程，父子进程未归属同一进程组。用户 Ctrl+C 仅终止了外层脚本，底层的 PyTorch CUDA 引擎沦为孤儿进程继续滞留显存。 | `run_vla.sh` 采用 `exec "$PYTHON_BIN"` 单进程直接替换启动，**消除任何中间包装层**。Ctrl+C 信号直接直达 Python 解释器，优雅清理 ZMQ 并瞬间释放 GPU 显存。 | **已解决** |
| **3. IP 连通后不等待确认直接开动** | `run_vla_subtask.py` 内部检测到 Interactive banner 打印后，在代码第 398 行硬编码执行了 `send_cmd("/start")`，跳过了人机确认环节。 | 恢复 LeRobot 标准交互握手：模型就绪并连通后，机器人保持在零动作安全待命姿态，打印操作菜单，**必须由操作员主动输入 `s`（或 `/s`）才会开跑**。 | **已解决** |
| **4. 3 阶段执行完毕后闪退退出** | `run_vla_subtask.py` 在第 3 阶段（放箱）达到 8.5s 后，直接触发 `break` 并强行调用了 `proc.terminate()` 杀死进程。 | 采用长效交互会话：第 3 阶段完成后，系统打印任务完成通知，**机器人平稳保持当前姿态，绝不闪退**！操作员键入 `r` 即可平滑复位重置，随时可键入 `s` 再次测试。 | **已解决** |

---

## 5. 3-Subtask 原生深度集成方案 (核心技术实现)

我们彻底抛弃外部 supervisor 脚本，直接将子任务流转引擎原生嵌入 LeRobot 体系：

### 5.1 配置扩展 (`src/lerobot/rollout/configs.py`)
在 `RolloutConfig` 中注册 `--subtasks bool = False` 参数，CLI 原生支持 `--subtasks=true/false`。

### 5.2 状态监测与自动流转引擎 (`src/lerobot/rollout/interactive.py`)
- **智能自激活**:
  - 当 CLI 传入 `--subtasks=true` 或模型路径中包含 `subtask` 字符时，自动激活 3 阶段引擎，并将初始任务设为 `"clamp and lift the box"`。
- **轻量零开销状态感知 (`_get_robot_metrics`)**:
  - 直接读取已有机器人实例的 `ctx.robot_wrapper.inner._latest_state`，无需开启额外 ZMQ 连接或占用额外端口，内存级无锁读取。
- **3 阶段精准流转条件 (`_subtask_tracker_loop`)**:
  - **阶段 1**: `"clamp and lift the box"` (夹持并抬箱)
    - 触发条件: 双肩俯仰角 `l_pitch <= -0.35` 且 `r_pitch <= -0.35` rad 持续 1.0s，或经验保底超时 12.0s。
  - **阶段 2**: `"hold the box and turn right"` (抱箱右转)
    - 触发条件: 机载 IMU 航向偏航角 $|\Delta \text{yaw}| \ge 78^\circ$（右转负偏航角到位），或经验保底超时 6.5s。
  - **阶段 3**: `"place the box on the table and release"` (俯身放箱并松开)
    - 持续执行 8.5s 后触发完成提醒，**机器人保持就绪姿态，保持在交互控制台中**。
- **状态 HUD 实时打印**:
  - 运行过程中每 2.0s 在终端打印一行富文本物理监控信息（耗时、双肩俯仰角度数、累计偏航角旋转度数）。

### 5.3 控制台快捷指令集
在 `InteractiveSession` 中注入单键/斜杠通用别名映射：

| 快捷键 | 完整指令 | 功能说明 |
| :---: | :---: | :--- |
| **`s`** | `/s` / `/start` | 启动策略推理循环（确认现场安全后开跑） |
| **`r`** | `/r` / `/reset` | 停止运动，机器人平滑返回安全初始姿态，子任务重置回阶段 1 |
| **`n`** | `/n` / `/next` | 任何时刻手动提前跳至下一个子任务 |
| **`1`** | `/1` / `/phase1` | 直接跳转至阶段 1 (`clamp and lift the box`) |
| **`2`** | `/2` / `/phase2` | 直接跳转至阶段 2 (`hold the box and turn right`) |
| **`3`** | `/3` / `/phase3` | 直接跳转至阶段 3 (`place the box on the table and release`) |
| - | `/subtask <text>` | 临时手动更换自定义任务语言指令 |
| **`q`** | `/q` / `/stop` | 安全断开连接并退出程序，显存瞬间清零 |
| **`h`** | `/h` / `/help` | 查看交互控制台帮助指南 |

---

## 6. 标准化使用指南 (SOP)

### 6.1 实机部署运行 (推荐)
进入项目主目录，直接执行：
```bash
./run_vla.sh --real
```
- 默认自动载入 3-Subtask 最优模型：`outputs/train/pi05_box_pick_turn_v30_subtasks/checkpoints/005000/pretrained_model`。
- 自动连接默认机器人 IP (`192.168.123.164`)、状态端口 `6001`、动作端口 `6002`、相机端口 `5556`。

> **IP 热切换提示**:
> 若机器人 IP 有变化，无需退出程序：
> 1. 可启动时指定：`./run_vla.sh --real --robot_ip=192.168.123.xxx`；
> 2. 或在启动后的连接等待提示 `⏳ [等待接入]` 时，**直接在终端键盘键入新 IP 并按回车**（例如 `192.168.123.200`），系统将在 0.1 秒内自动重连。

### 6.2 本地/仿真纯视觉测试
在没有连接实体机器人电机时，测试相机与策略推流：
```bash
./run_vla.sh --sim --camera_only
```

### 6.3 运行过程操作流程
1. **启动与就绪**: 脚本加载 2.3B 模型至显存，连通机器人后打印控制台 Banner，机器人待命静止。
2. **确认开跑**: 确认周边环境安全后，在终端键入 `s` 并回车，机器人开始执行抱箱动作。
3. **观察流转**:
   - 抱起箱子稳定 1 秒后，系统自动流转至阶段 2 并打印通知，机器人开始踏步右转；
   - 右转达到 ~80 度后，系统自动流转至阶段 3 并打印通知，机器人俯身放箱并松手；
   - 放箱完成后，终端提示任务结束，机器人平稳保持在最后姿态。
4. **复位与重新测试**:
   - 键入 `r` 并回车：机器人平稳返回初始待命姿势，子任务自动重置回阶段 1。
   - 随时再次键入 `s` 即可开始下一轮测试。
5. **退出**:
   - 键入 `q` 或直接按 `Ctrl+C`：程序优雅退出，显存即刻清零。

---

## 7. 应急与排错指南 (Troubleshooting)

1. **显存被占满处理**:
   若因其他非标准脚本异常退出导致 GPU 显存残留，可执行：
   ```bash
   pkill -9 -f lerobot-rollout
   pkill -9 -f rerun
   ```
2. **ZMQ 端口连通性排查**:
   无需加载模型，毫秒级快速测试机载端口：
   ```bash
   ./run_vla.sh --check --real
   ```
   可直观查看 6001（状态）、6002（动作）、5556（相机）的收发帧率与连通状态。
