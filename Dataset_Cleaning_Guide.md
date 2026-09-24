# Unitree G1 具身数据集清洗、Sub-task 切片与消抖处理标准操作规程 (SOP)
## Dataset Cleaning, Sub-task Segmentation & State-as-Action Guide (`Dataset_Cleaning_Guide.md`)

- **更新时间**: 2026-09-10
- **工作目录**: `/home/yichangfeng/lerobot`
- **适用硬件**: Unitree G1 (29-DoF) 人形机器人 + 固定橡胶手 (Rubber Hand)
- **核心工具目录**: [`dataset_tools/`](./dataset_tools/)

---

## 目录
1. [核心背景与数据缺陷剖析](#一核心背景与数据缺陷剖析)
2. [全链路数据拓扑与处理架构图](#二全链路数据拓扑与处理架构图)
3. [SOP 阶段 1：原始采集数据转换与末尾截断](#三sop-阶段-1原始采集数据转换与末尾截断)
4. [SOP 阶段 2：失败示范数据审计与剔除](#四sop-阶段-2失败示范数据审计与剔除)
5. [SOP 阶段 3：3 阶段 Sub-task 语言切片与时序融合判定](#五sop-阶段-33-阶段-sub-task-语言切片与时序融合判定)
6. [SOP 阶段 4：State-as-Action 物理状态替代动作与消抖处理](#六sop-阶段-4state-as-action-物理状态替代动作与消抖处理)
7. [SOP 阶段 5：数据集质量验证、消抖评估与仿真回放](#七sop-阶段-5数据集质量验证消抖评估与仿真回放)
8. [SOP 阶段 6：下游模型微调训练启动指南](#八sop-阶段-6下游模型微调训练启动指南)
9. [数据处理工具集与文件索引](#九数据处理工具集与文件索引)

---

## 一、核心背景与数据缺陷剖析

在针对 Unitree G1 机器人搬运箱子任务进行遥操采集与策略微调的过程中，系统实测暴露了三大底层数据缺陷：

1. **控制动作严重抖动（高频震颤达 12 mrad）**：
   - **成因 1（VR 光学追踪遮挡）**：PICO 4 VR 手柄在双臂合抱箱体时被箱体和人体遮挡，算法频繁在红外定位与 IMU 积分间跳变，引入高频空间噪声。
   - **成因 2（逆运动学非线性放大）**：WBC（全身运控）逆运动学在抱箱构型下接近奇异值（Singularity），末端微小的位姿波动被雅可比伪逆矩阵数学放大为机械臂各关节角度的剧烈阶跃。
   - **对比基准**：物理电机编码器测得的实测状态 $q_{\text{state}}$ 受机械臂转动惯量与阻尼物理滤波，抖动仅 **$1.24\text{ mrad}$**；而控制目标 $q_{\text{action}}$ 抖动高达 **$12.01\text{ mrad}$**（膨胀近 10 倍）。
2. **长程单一 Prompt 导致的因果混淆（Causal Confusion）**：
   - 原始数据仅使用单一粗粒度指令：`"pick up the box, turn right, and place it on the table"`。
   - 模型在复杂时空长程序列下难以准确把握动作意图切换，常出现“双手未碰触箱子便提前举手外展转身”的因果错乱。
3. **任务末尾无效待机与“举手投降”姿态**：
   - 操作员在放箱后为了防刮碰本能抬手悬空，且按下停止键有 2~3 秒反应延迟，导致尾部记录了多余的悬空姿态并被策略误学为终局必选行为。

为此，本项目制定并落地了完整的**数据清洗、3-Subtask 细粒度标注与 State-as-Action 消抖处理 SOP**。

---

## 二、全链路数据拓扑与处理架构图

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 原始采集数据 (SonicStar 43D Parquet)                                                    │
│ 路径: ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn (87 Episodes, 60,809 帧)        │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                        [步骤 1: 格式转换与清洗截断]
                        • convert_rubberhand_to_g1_v30.py (43D -> 29D/18D, 计算偏航速度)
                        • sanitize_sonic_dataset.py (截断末尾 2~3s 反应延迟帧)
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LeRobot v3.0 基准数据集                                                                │
│ 路径: datasets/g1_box_pick_turn_v30 (87 Episodes, 单一 Prompt)                         │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                        [步骤 2 & 3: 剔除失败样本 + 3-Subtask 物理切分]
                        • 审计剔除未完成转向的 Episode 32, 48
                        • create_subtask_dataset.py (偏航角积分 + 双臂姿态判定)
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3-Subtask 细粒度子任务数据集 (保留原始 Action)                                            │
│ 路径: datasets/g1_box_pick_turn_v30_subtasks (85 Episodes, 59,825 帧, 3 Tasks)          │
│ Task 0: clamp and lift the box                                                         │
│ Task 1: hold the box and turn right                                                    │
│ Task 2: place the box on the table and release                                         │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                        [步骤 4: State 替代 Action 消抖处理]
                        • replace_action_with_state.py
                        • action[0:14] = state[15:29], 保留 action[14:18] 转向速度
                        • 注入基模分位数, 彻底消除高频控制振颤
                                          ▼
                ┌─────────────────────────┴─────────────────────────┐
                ▼ (时序因果迁移 a_t = s_{t+1})                        ▼ (当前观测复制 a_t = s_t)
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│ 推荐版本: shift=1                            │    │ 消融对比版本: shift=0                         │
│ datasets/g1_box_pick_turn_v30_subtasks_      │    │ datasets/g1_box_pick_turn_v30_subtasks_      │
│ state_as_action                              │    │ state_as_action_shift0                       │
│ (动作作为下一时刻状态转移目标，消除抽搐)       │    │ (作为基线对比组)                              │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

---

## 三、SOP 阶段 1：原始采集数据转换与末尾截断

### 1.1 核心原理
原始数据由 SonicStar 记录为 43 维结构（包含了未配备电机的灵巧手空槽位）。需执行：
1. **维度重排与去冗余**：剔除 14 个虚拟手部关节，组织为 29 维状态（`observation.state`）与 18 维动作（`action`：双臂 14 关节角 + 4 轴底盘速度）。
2. **底盘速度提取**：通过机身 IMU 四元数差分计算瞬时偏航角速度：$\text{remote.rx} = -\frac{\Delta \text{yaw}}{\Delta t}$。
3. **尾部待机帧截断**：放箱着地瞬间后截除冗余的双手悬空帧。

### 1.2 执行命令
```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

# 1. 转换原始数据为 LeRobot v3.0 格式
$PYTHON dataset_tools/convert_rubberhand_to_g1_v30.py \
    --src-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn \
    --dst-dir datasets/g1_box_pick_turn_v30

# 2. 验证转化前后数值无损（绝对误差 < 1e-7）
$PYTHON dataset_tools/verify_raw_vs_converted.py
```

---

## 四、SOP 阶段 2：失败示范数据审计与剔除

### 2.1 审计结论
对全量 87 条 Episode 进行转向角积分 $\psi_{\text{total}} = \int (-\text{remote.rx}) dt$ 扫描，发现两条异常数据：
- **Episode 32**: 累计转向角度仅 **$-4.58^\circ$**（操作员未执行踏步转向便放箱）。
- **Episode 48**: 累计转向角度仅 **$-7.19^\circ$**（操作员未执行踏步转向便放箱）。

### 2.2 处理规约
这两条样本属于严重的“未按规范执行任务”的失败示范，在构建下游任务数据集时必须**硬性彻底剔除**：
- 剔除前：87 个 Episode，60,809 帧。
- 剔除后：**85 个高质量 Episode，59,825 帧**。

---

## 五、SOP 阶段 3：3 阶段 Sub-task 语言切片与时序融合判定

### 5.1 物理切分算法机理
单纯依赖瞬时速度切分会导致开局重心摆动误触发与踏步过零点频繁跳变。算法采用**多信号融合滤波**：
1. **偏航积分角**：$\psi(t) = \int (-\text{remote.rx}) dt$，右转单调累积至 $-90^\circ$。
2. **转弯起点 $t_{\text{split1}}$（阶段 0 $\to$ 阶段 1）**：
   - 偏航角开始单调下穿 $-13^\circ$（目标角度 15% 处）；
   - 向前沿回溯至平滑角速度上升沿，确保此时箱子已被双臂夹稳并离开桌面；
   - 标注为：`"clamp and lift the box"`。
3. **转弯终点 $t_{\text{split2}}$（阶段 1 $\to$ 阶段 2）**：
   - 偏航角累积达到目标转角 88%（约 $-78^\circ \sim -88^\circ$）；
   - 且平滑角速度平稳回落至 $< 0.15\text{ rad/s}$（原地踏步已稳妥到位）；
   - 阶段 1 标注为：`"hold the box and turn right"`。
4. **放箱完成 $t_{\text{end}}$（阶段 2）**：
   - 机器人正对目标桌，双臂下俯将箱子平稳放置并脱开双手；
   - 阶段 2 标注为：`"place the box on the table and release"`。

### 5.2 生成 Subtask 数据集命令
```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

$PYTHON dataset_tools/create_subtask_dataset.py \
    --src-dir datasets/g1_box_pick_turn_v30 \
    --dst-dir datasets/g1_box_pick_turn_v30_subtasks \
    --exclude-episodes 32 48
```

### 5.3 切片报告可视化校验
运行切片可视化脚本，在 `outputs/subtask_reports/` 下生成切片诊断波形图与 $t_0, t_{\text{split1}}, t_{\text{split2}}, t_{\text{end}}$ 四张相机真机画面拼图：
```bash
$PYTHON dataset_tools/verify_subtask_splits.py
```

---

## 六、SOP 阶段 4：State-as-Action 物理状态替代动作与消抖处理

### 6.1 核心理论与因果对齐设计
为彻底根除训练数据注入策略模型带来的高频机械抖动，采用物理编码器实测关节角替换动作：

1. **双臂前 14 维替换**：`action[0:14] = observation.state[15:29]`。
2. **因果时序映射对比（`shift=1` vs `shift=0`）**：
   - **`shift = 1`（推荐，因果状态转移目标）**：
     $$a_t^{\text{arm}} = s_{t+1}^{\text{arm}} \quad (\forall t < T-1); \quad a_{T-1}^{\text{arm}} = s_{T-1}^{\text{arm}}$$
     *物理内涵*：在行为克隆（BC）中，$a_t$ 的本质是“由当前状态 $s_t$ 驱动系统转移到下一状态 $s_{t+1}$ 的控制律”。以 $s_{t+1}$ 为目标具有明确的方向性前馈引导，机械臂运行平滑且响应灵敏。
   - **`shift = 0`（对比组，当前步恒等复制）**：
     $$a_t^{\text{arm}} = s_t^{\text{arm}}$$
     *物理内涵*：动作直接等于当前测量状态。在实机闭环时容易由于误差为零而陷入迟钝、停滞，仅用于消融对比。
3. **底盘速度保留不变**：
   - 后 4 维遥控速度 `action[14:18]`（尤其是 `remote.rx`）必须完整保留，否则底盘将丧失原地踏步转向能力。
4. **统计量重算与基模分位数对齐 (`align_velocity`)**：
   - 替换双臂动作后，自动重算前 14 维的 `min`、`max`、`mean`、`std` 与 `q01..q99` 分位数；
   - 数据集内横移速度恒为 0，若直接取经验分位数会导致 `q99 - q01 ≈ 0`，引发 QUANTILES 归一化除零与数值爆炸。脚本自动注入 `model/box_pick` 基模的标准底盘速度分位数，确保归一化严密稳定。

### 6.2 两个版本的生成命令

#### 版本 A：生成推荐的 `shift=1` 数据集
```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

$PYTHON dataset_tools/replace_action_with_state.py \
    --src-dir datasets/g1_box_pick_turn_v30_subtasks \
    --dst-dir datasets/g1_box_pick_turn_v30_subtasks_state_as_action \
    --shift 1
```

#### 版本 B：生成消融对比的 `shift=0` 数据集
```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

$PYTHON dataset_tools/replace_action_with_state.py \
    --src-dir datasets/g1_box_pick_turn_v30_subtasks \
    --dst-dir datasets/g1_box_pick_turn_v30_subtasks_state_as_action_shift0 \
    --shift 0
```

---

## 七、SOP 阶段 5：数据集质量验证、消抖评估与仿真回放

### 7.1 数值健康自检
运行自检确认两套数据集的帧数、分词及归一化表现：
```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

$PYTHON -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
for name in ['g1_box_pick_turn_v30_subtasks_state_as_action', 'g1_box_pick_turn_v30_subtasks_state_as_action_shift0']:
    ds = LeRobotDataset(name, root=f'datasets/{name}')
    print(f'[{name}] Episodes: {ds.num_episodes}, Frames: {len(ds)}, Task0: {ds[0][\"task\"]}')
"
```

### 7.2 二阶抖动量化指标对比
根据离散二阶加速度差分：$\text{Jitter} = \frac{1}{N}\sum |q_{t+1} - 2q_t + q_{t-1}|$：

| 数据集版本 | 双臂 Action 抖动均值 | 相比原始 Action 改善幅度 |
| :--- | :--- | :--- |
| 原始采集数据 (`g1_box_pick_turn_v30`) | **$12.007\text{ mrad}$** | 基准（剧烈高频震颤） |
| **新生成数据集 (`state_as_action`, shift=1)** | **$1.655\text{ mrad}$** | **大幅降低 86.2%（极度平滑）** |
| 物理电机实测状态 (`state`) | **$1.652\text{ mrad}$** | 物理机械基线（完全重合） |

### 7.3 MuJoCo 仿真 3D 回放验证
在终端启动 3D 仿真回放工具，验证机械臂平滑轨迹与底盘转向，界面抬头显示器（HUD）将动态显示当前子任务名称：
```bash
# 回放新数据集动作 (Action 模式)
$PYTHON dataset_tools/replay_g1_dataset.py \
    --dataset-dir datasets/g1_box_pick_turn_v30_subtasks_state_as_action \
    --mode action \
    --episode 0
```

---

## 八、SOP 阶段 6：下游模型微调训练启动指南

### 8.1 统一配置文件快捷启动 (推荐)
已在 `train_pi05_config.yaml` 中将全部超参数统一对齐（5000 步、余弦衰减至 5000 步、每 1000 步存盘、关闭在线 WandB、显存占用 ~16GB）：

```bash
cd /home/yichangfeng/lerobot

# 方式 1: 直接使用默认配置启动推荐组 (shift=1 数据集)
./run_train.sh

# 方式 2: 一键启动消融对照组 (shift=0 数据集)
./run_train.sh \
    --dataset.repo_id="g1_box_pick_turn_v30_subtasks_state_as_action_shift0" \
    --dataset.root="datasets/g1_box_pick_turn_v30_subtasks_state_as_action_shift0" \
    --output_dir="outputs/train/pi05_subtasks_state_as_action_shift0" \
    --job_name="pi05_subtasks_state_as_action_shift0"

# 方式 3: 后台持久化挂起运行并实时看日志
nohup ./run_train.sh > train.log 2>&1 &
tail -f train.log
```

### 8.2 等价的全量显式 CLI 启动命令
上述 `./run_train.sh` 脚本背后加载 `train_pi05_config.yaml`，其完整行为与以下显式命令 100% 严格等价：

```bash
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

$PYTHON -m lerobot.scripts.lerobot_train \
    --config_path="train_pi05_config.yaml" \
    --dataset.repo_id="g1_box_pick_turn_v30_subtasks_state_as_action" \
    --dataset.root="datasets/g1_box_pick_turn_v30_subtasks_state_as_action" \
    --policy.path="model/box_pick" \
    --policy.train_expert_only=true \
    --policy.compile_model=false \
    --output_dir="outputs/train/pi05_subtasks_state_as_action_shift1" \
    --job_name="pi05_subtasks_state_as_action_shift1" \
    --batch_size=8 \
    --steps=5000 \
    --save_freq=1000 \
    --log_freq=50 \
    --policy.device="cuda" \
    --wandb.enable=false
```

---

## 九、数据处理工具集与文件索引

数据处理相关脚本已全部收归至专属目录 [`dataset_tools/`](./dataset_tools/) 统一定位与维护：

```text
dataset_tools/
├── convert_rubberhand_to_g1_v30.py     # 格式转换：Sonic 43D -> LeRobot 29D/18D
├── sanitize_sonic_dataset.py           # 样本清洗：截断任务终局多余悬停帧
├── merge_sonic_datasets.py             # 数据合并：多批次 Parquet 拼接
├── align_velocity_dataset.py           # 速度对齐：底盘遥控速度分位数注入
├── verify_raw_vs_converted.py          # 精度校验：逐帧核验转换绝对误差
├── create_subtask_dataset.py           # 转向任务 3-Subtask 切片：剔除失败 Episode，注入 3-Subtask 标签
├── create_subtask_dataset_inplace.py   # 原地任务 2-Subtask 切片：基于手腕高度与下放速度启动沿自动切分
├── verify_subtask_splits.py            # 转向任务切片可视化：生成时序分割曲线与相机关键帧拼图
├── verify_subtask_splits_inplace.py    # 原地任务切片可视化：手腕高度、下放速度与实景拼图
├── replace_action_with_state.py        # State 替代 Action：消抖处理，支持 shift=1 与 shift=0
├── evaluate_state_action.py            # 评估工具：计算抖动度与位置偏差学术报表
├── replay_g1_dataset.py                # 仿真回放：MuJoCo 3D 动态回放与 Subtask HUD 显示
└── README.md                           # 工具包索引与说明
```

> **总结**：
> 通过本 SOP 规程处理后的两个数据集已全部就绪。新数据集完全继承了清洗后的 85 条演示与 3-Subtask 标注，同时将机械臂高频动作抖动降低了 **86.2%**，为 $\pi_{0.5}$ 模型在 MuJoCo 仿真与物理 G1 实机上的平滑、鲁棒部署奠定了高质量的数据基础。
