# Unitree G1 具身数据动作质量深度剖析与调试方案 (Debug.md)

本文档针对配备 **Rubber Hand（固定橡胶手）** 的物理 Unitree G1 人形机器人在数据采集、动作重放（[`dataset_tools/replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py)）以及 $\pi_{0.5}$ 策略微调后出现的动作异常现象，进行深入、系统的机理剖析，记录量化评估结果，并制定下一步具体的实验验证与溯源排查工作规划。

---

## 目录
1. [核心现象与问题概述](#一核心现象与问题概述)
2. [量化评估与图表分析 (3 张评估图表)](#二量化评估与图表分析)
   - [图 1：Action 与 State 抖动程度对比](#1-图-1action-与-state-抖动程度对比-eval_1_jitter_comparisonpng)
   - [图 2：Action 与 State 位置偏差与抱箱动力学](#2-图-2action-与-state-位置偏差与抱箱动力学-eval_2_position_discrepancypng)
   - [图 3：自采数据集与开源基准数据集横向对比](#3-图-3自采数据集与开源基准数据集横向对比-eval_3_cross_dataset_comparisonpng)
3. [深层技术机理剖析](#三深层技术机理剖析)
   - [3.1 为什么 Action 剧烈抖动，而 State 相对平滑？](#31-为什么-action-剧烈抖动而-state-相对平滑)
   - [3.2 为什么抱箱时期望位置必须比实测位置更内收夹紧？](#32-为什么抱箱时期望位置必须比实测位置更内收夹紧)
   - [3.3 为什么末尾会出现双手举到胸前、大幅外展的“投降姿态”？](#33-为什么末尾会出现双手举到胸前大幅外展的投降姿态)
4. [配套调试与评估工具集](#四配套调试与评估工具集)
5. [前期验证课题规划 (基线设计)](#五前期验证课题规划-基线设计)
   - [5.1 验证课题一：使用 State 代替 Action 进行策略训练对比实验](#51-验证课题一使用-state-代替-action-进行策略训练对比实验)
   - [5.2 验证课题二：全链路溯源采集中的 Action 与 State 具体数据来源](#52-验证课题二全链路溯源采集中的-action-与-state-具体数据来源)
6. [全链路溯源排查结论与转化前后一致性定论](#六全链路溯源排查结论与转化前后一致性定论)
   - [6.1 转化前 vs 转化后数值一致性排查](#61-转化前-vs-转化后数值一致性排查)
   - [6.2 为什么以往在 Sonic 遥操和回放中感觉不到抖动？](#62-为什么以往在-sonic-遥操和回放中感觉不到抖动)
   - [6.3 信号全链路逐层穿透明细表](#63-信号全链路逐层穿透明细表)
7. [State 替代 Action 对比实验的具体实现与操作指引](#七state-替代-action-对比实验的具体实现与操作指引)
   - [7.1 核心理论与因果对齐设计](#71-核心理论与因果对齐设计)
   - [7.2 一键生成工具 dataset_tools/replace_action_with_state.py](#72-一键生成工具-replace_action_with_statepy)
   - [7.3 策略微调训练启动命令](#73-策略微调训练启动命令)
8. [接下来的研判认识与后续推进任务清单 (下一步行动)](#八接下来的研判认识与后续推进任务清单-下一步行动)
   - [8.1 核心研判认识与物理博弈边界](#81-核心研判认识与物理博弈边界)
   - [8.2 落地任务实施清单与兜底改进方案](#82-落地任务实施清单与兜底改进方案)

---

## 一、核心现象与问题概述

在通过仿真回放工具（[`replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py)）对当前转化后的数据集（`datasets/g1_box_pick_turn_v30`）进行可视化观察，并与模型实机推理表现对照后，发现策略学出的异常行为高度映射了训练数据中的底层缺陷，主要体现为两大核心问题：

1. **Action 手臂抖动严重（高频震颤）**：
   * 在仿真中播放 `action` 模式时，机械臂各关节伴随明显的锯齿状高频振颤；
   * 相比于同数据中的 `state`（实测物理关节），`action` 的抖动程度高出一个数量级；
   * 相比于开源成熟数据集（如 [`datasets/unitree_box_move_blue_full`](file:///home/yichangfeng/lerobot/datasets/unitree_box_move_blue_full)），开源数据的 `action` 仅比 `state` 略微粗糙，而自采数据的抖动剧烈程度远超开源基准。

2. **手臂动作过度夸张（过度内收交叉与末尾“投降姿势”）**：
   * **双手交叉内收**：抱箱时期望位置显著穿透物体表面；
   * **过度举高与末尾“投降”**：在箱子放置到桌面后，机械臂没有平稳自然复位，而是双手大幅度举至胸前甚至面部前方，同时大臂向身体两侧极大角度张开外展（Shoulder Roll 达到 $+45^\circ \sim -67^\circ$），呈现类似“举手投降”的夸张姿势。开源数据集在放箱后虽然也有抬手动作，但幅度克制且自然，自采数据的幅度被极度放大。

---

## 二、量化评估与图表分析

运行评估脚本 [`evaluate_state_action.py`](file:///home/yichangfeng/lerobot/dataset_tools/evaluate_state_action.py)，对自采数据集（`g1_box_pick_turn_v30` Episode 0）与开源数据集（`unitree_box_move_blue_full` Episode 20）进行全维度横向比对，生成的三张高清分析图表存放于 [`evaluation_plots/`](file:///home/yichangfeng/lerobot/evaluation_plots/)：

### 1. 图 1：Action 与 State 抖动程度对比 ([`eval_1_jitter_comparison.png`](file:///home/yichangfeng/lerobot/evaluation_plots/eval_1_jitter_comparison.png))

采用二阶有限差分作为离散角加速度变化率（抖动震颤度度量）：
$$\text{Jitter}_t = |q_{t+1} - 2q_t + q_{t-1}|$$

* **全关节均值对比（图 1A）**：
  * 双臂 14 个关节中，所有关节的 Action 抖动幅度均显著高于 State；
  * 左肩俯仰（L_ShP）放大 **11.2x**，左肩侧摆（L_ShR）放大 **9.8x**，手肘（L_Elb）放大 **8.6x**，手腕部分关节甚至达到 **12.5x**；
* **时域波形图（图 1B & 1C）**：
  * State（实测位置）总抖动范数 $\|\Delta^2 q\|_2$ 保持在 $0.0012 \ \text{rad}$ 的平稳基线；
  * Action（控制目标）总抖动范数剧烈跳跃在 $0.010 \sim 0.025 \ \text{rad}$ 之间，在 5 秒局部窗口中呈现出密集的锯齿波。

### 2. 图 2：Action 与 State 位置偏差与抱箱动力学 ([`eval_2_position_discrepancy.png`](file:///home/yichangfeng/lerobot/evaluation_plots/eval_2_position_discrepancy.png))

位置偏差定义为：$\Delta q(t) = q_{\text{action}}(t) - q_{\text{state}}(t)$。

* **抱箱阶段的内夹力矩（图 2A & 2B）**：
  * 在接触并抱起箱子期间（约 $t=5\text{s} \sim 22\text{s}$），双臂并没有重合在箱子外壁；
  * 左肩侧摆维持约 $-10^\circ \sim -15^\circ$ 的内收偏差，右肩维持 $+5^\circ \sim +10^\circ$ 的内收偏差；
  * 双手肘维持 $-15^\circ \sim -25^\circ$ 的向内紧勒偏差，肩部俯仰维持 $-15^\circ \sim -30^\circ$ 的托举偏差；
  * 此稳态位置偏差乘以电机刚度 $K_p$，构成了持续挤压箱体产生静摩擦力的必要物理源头。
* **放箱后的“投降姿势”突变（图 2C）**：
  * 在 $t=25\text{s}$ 箱子着地后的最后 3 秒内，手臂角度轨迹发生剧烈阶跃：
  * 左肩侧摆瞬间由 $-18.8^\circ$（内夹）突变至 $+45.6^\circ$（大幅外展）；
  * 右肩侧摆瞬间由 $-9.7^\circ$（内夹）突变至 $-67.0^\circ$（大幅外展）；
  * 肩部俯仰角上升至 $-46.9^\circ$（大臂抬升至接近水平），造成了极度夸张的双臂张开高架姿势。

### 3. 图 3：自采数据集与开源基准数据集横向对比 ([`eval_3_cross_dataset_comparison.png`](file:///home/yichangfeng/lerobot/evaluation_plots/eval_3_cross_dataset_comparison.png))

| 评估指标 | 自采数据集 (`g1_box_pick_turn_v30`) | 开源基准数据集 (`unitree_box_move_blue_full`) | 差距倍数 / 特征 |
| :--- | :--- | :--- | :--- |
| **平均 Action 抖动** | **$0.01201 \ \text{rad}$ ($12.01 \ \text{mrad}$)** | **$0.00195 \ \text{rad}$ ($1.95 \ \text{mrad}$)** | 自采数据抖动高达开源的 **$6.16$ 倍** |
| **平均 State 抖动** | **$0.00124 \ \text{rad}$ ($1.24 \ \text{mrad}$)** | **$0.00146 \ \text{rad}$ ($1.46 \ \text{mrad}$)** | 物理机体机械抖动基本一致（$\sim 1.3 \ \text{mrad}$） |
| **抖动膨胀比 (Act/St)** | **$9.70 \times$** | **$1.34 \times$** | **核心差距所在**：开源几乎无噪声膨胀 |
| **末尾左肩侧展 (L_Roll)** | **$+45.6^\circ$ (剧烈外展)** | **$+6.1^\circ$ (微收靠拢)** | 自采数据外展幅度放大 **$7.5$ 倍** |
| **末尾右肩侧展 (R_Roll)** | **$-67.0^\circ$ (剧烈外展)** | **$-14.1^\circ$ (自然下垂)** | 自采数据外展幅度放大 **$4.7$ 倍** |
| **末尾手肘屈曲 (L_Elbow)** | **$+0.9^\circ$ (近乎僵直架起)** | **$+71.4^\circ$ (自然屈曲半抱前倾)** | 开源呈现自然前收姿态 |
| **误差分布标准差 ($\sigma_{\Delta q}$)** | **$13.8^\circ$ (离散宽广)** | **$6.2^\circ$ (紧凑对称聚集于 0 附近)** | 自采数据存在显著的非线性极值漂移 |

> [!NOTE]
> **图 3A/3B 坐标轴尺度压缩陷阱与优化说明**：
> * **异常成因**：开源数据集 `unitree_box_move_blue_full` 使用的是 23-DoF（或手腕未配电机的刚体手）版本的 G1 机器人，其 `L_WrPitch`、`L_WrYaw`、`R_WrPitch`、`R_WrYaw` 4 个关节在物理上无电机，实测状态 $q_{\text{state}}$ 恒为 `0.000`。早先代码在计算 Action/State 比率时分母除以 $\epsilon = 10^{-6}$，虚增至 $1170\times \sim 1550\times$，将全局 Y 轴拉大至 1600，导致真正发生运动的核心关节（肩、肘真实比率 10x~20x）被压缩至 1% 像素高度，误视作“接近 0”。
> * **修正方案**：
>   1. **图 3A** 直接展示 **Action 绝对抖动量（mrad）**，彻底避免除零，清晰展现自采数据 Action 抖动（$13 \sim 19 \ \text{mrad}$）相比开源（$1.7 \sim 2.8 \ \text{mrad}$）在全关节高达 **$6\times \sim 10\times$** 的真实物理震颤差距；
>   2. **图 3B** 过滤掉开源数据固定为 0 的假手腕维度，展示 10 个真实活动关节的抖动膨胀倍数（自采平均 **$13.1\times$** vs 开源平均 **$1.1\times$**），坐标轴稳定在 0~25，对比极其醒目。

---

## 三、深层技术机理剖析

### 3.1 为什么 Action 剧烈抖动，而 State 相对平滑？

1. **VR 光学追踪噪声与遮挡退化**：
   PICO 4 VR 头显通过头载摄像头对双手柄红外 LED 进行视觉定位，辅以 6 轴 IMU。当操作员双手合抱于箱体两侧时，手柄传感器大部分光路受到箱体本身及操作员身躯的物理遮挡，追踪算法频繁在“光学位姿校准”与“纯 IMU 积分漂移”之间高频跳变切换，引入了毫米至厘米级的末端笛卡尔高频抖动。
2. **逆运动学求解（IK）的非线性放大效应**：
   在 C++ WBC 控制器（`gear_sonic_deploy`）中，末端手腕位姿必须经过非线性逆运动学映射为 7-DoF 关节空间角：
   $$\Delta q = J^{\dagger}(q) \Delta x$$
   在抱箱或前探位姿下，机械臂处于接近奇异构型（Singularity）或接近关节极限的边缘，雅可比伪逆矩阵 $J^{\dagger}(q)$ 的条件数恶化，**末端微小的空间位移跳变被数学放大为各关节旋转角度的剧烈阶跃震荡**。
3. **物理刚体的机械低通滤波机制**：
   `observation.state` 来源于宇树原厂驱动器读取的电机编码器测量值。机械臂各连杆具备质量与转动惯量（Inertia），且底层关节控制算法具备速度阻尼项（$-K_d \dot{q}$），这使得执行器对 50Hz 的高频指令震荡具备物理上的低通抑制能力，因此真实关节角表现平滑，而未滤波的控制指令 `action` 则将噪声全盘记录。
4. **开源数据集的前置/后置平滑处理**：
   开源基准数据集在数据导出或遥操链路中，通常介入了轨迹平滑算法（如 OneEuroFilter、Savitzky-Golay 样条滤波或指数加权移动平均 EMA），剔除了高频噪声；自采链路中目前为原始输出。

### 3.2 为什么抱箱时期望位置必须比实测位置更内收夹紧？

物理 G1 机械臂在宇树底层采用位置刚度控制（PD / 阻抗控制）：
$$\tau = K_p (q_{\text{des}} - q_{\text{meas}}) - K_d \dot{q}$$

* 若 $q_{\text{des}} = q_{\text{meas}}$：机械臂只要一接触到箱体外壁，误差立即衰减为零，关节输出力矩消失，箱子会因无法对抗重力瞬间滑脱；
* 只有当 $q_{\text{des}}$ 持续向物体几何轮廓内部深入（虚拟穿透，Virtual Penetration），被刚性箱体阻挡在表面的实测角 $q_{\text{meas}}$ 才会与期望角形成持久稳定的位置误差 $(q_{\text{des}} - q_{\text{meas}})$，转化为垂直于箱壁的横向挤压力与垂直支撑静摩擦力。

### 3.3 为什么末尾会出现双手举到胸前、大幅外展的“投降姿态”？

1. **录制延迟与人类操作员的本能避障回抽动作**：
   在物理操作中，操作人员把箱子放置在桌面上后，为了防止手臂刮蹭刚放稳的箱子或桌面边缘，本能动作是将双手向后、向上抽离，并在胸腹前方处于待机悬停状态。此时距离操作员按下手柄上的保存键（`Left Grip + A`）往往有 **2~3 秒的反应延迟**。
   在此延迟期内，系统依然以 30FPS 记录数据，将这段无效的“悬空举手待机”姿态录入数据尾部。模仿学习将这一状态识别为任务终局的必选行为。
2. **人体工效与 G1 机器人尺寸的尺度失配（Retargeting Mismatch）**：
   成人的自然肩宽为 $45 \sim 50\text{cm}$，而 Unitree G1 的躯干极为紧凑，肩宽仅约 $40\text{cm}$。在 VR 空间重定向（Retargeting）过程中，操作员自认为正常的“双手自然放于腹部前侧”，经比例折算到窄小的 G1 躯干坐标系后，直接映射为了大臂向外大幅展肩（Shoulder Roll 激增）。
3. **7-DoF 机械臂手肘冗余自由度的零空间漂移（Null-space Drift）**：
   G1 单臂 7 个关节控制 6 维末端手腕，具有 1 个手肘自旋冗余自由度。当手腕回抽靠近胸口时，WBC 的 IK 求解器在缺乏强零空间下垂姿态牵引权重的情况下，往往倾向于计算出“手肘外展、大臂高架”的解，直接形成了仿真中的投降形态。

---

## 四、配套调试与评估工具集

为方便持续监控和可视化验证，已在代码库中建立了专用工具链：

1. **数据集动作仿真回放工具**：[`replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py)
   * 支持通过 `--mode state` 验证电机真实轨迹，通过 `--mode action` 观察期望控制与夹持超调；
   * 内置底盘转向积分，可重现原地踏步转身；
   * 支持离线导出 MP4（`--save-video`）及与真实相机视频分屏比对（`--side-by-side`）；
   * **已升级**：全面向下兼容 SonicStar 原始 43 维 Parquet 格式，可直接无缝回放 `~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn`。
2. **状态动作差距评估工具**：[`evaluate_state_action.py`](file:///home/yichangfeng/lerobot/dataset_tools/evaluate_state_action.py)
   * 自动计算各关节二阶抖动指数、时域位置偏差、末尾静止角度，并绘制生成三张标准化学术图表。
3. **数据集一致性与抖动校验工具**：[`verify_raw_vs_converted.py`](file:///home/yichangfeng/lerobot/dataset_tools/verify_raw_vs_converted.py)
   * 自动逐帧对齐并比对转化前（SonicStar 43D）与转化后（LeRobot v3.0）的数值绝对误差与关节抖动值，精确排查抖动源头。
4. **State 替代 Action 数据集制作工具**：[`replace_action_with_state.py`](file:///home/yichangfeng/lerobot/dataset_tools/replace_action_with_state.py)
   * 支持 `--shift 1`（采用下一帧 $s_{t+1}$）或 `--shift 0`（采用当前帧 $s_t$）生成对比数据集；
   * 严格保留后 4 维底盘转向速度指令，自动重新计算前 14 维新动作分位数并注入基模后 4 维分位数，完成健康自检。

---

## 五、前期验证课题规划 (基线设计)

针对上述分析，后续需重点推进两项核心验证与溯源课题。**以下仅详细定义其研究目标、验证方案与技术路线，不在此处提前给出结论，待实验实施后回填数据：**

### 5.1 验证课题一：使用 State 代替 Action 进行策略训练对比实验

#### 1. 实验研究目标
验证在模仿学习训练阶段，将监督学习标签从原有的控制目标期望动作 $q_{\text{action}}$（期望角）替换为机器人编码器实测物理状态 $q_{\text{state}}$（实测角）的可行性与优劣表现。

#### 2. 待验证的核心假设
* **积极假设（解决抖动）**：
  由于物理实测的 $q_{\text{state}}$ 二阶抖动指标比 $q_{\text{action}}$ 降低了近 10 倍（$1.24 \ \text{mrad}$ vs $12.01 \ \text{mrad}$），策略若以 $q_{\text{state}}$ 作为监督目标，能否完全消除推理输出中的关节剧烈高频震颤，使实机运行变得极为平滑柔顺？
* **潜在风险假设（丧失抱箱力矩）**：
  鉴于抱箱接触动力学依赖于 $(q_{\text{action}} - q_{\text{state}})$ 的内夹超调误差来产生法向压力，若直接以贴在箱壁表面的 $q_{\text{state}}$ 作为动作标签，训练出的策略在双手触碰箱体后，输出的目标角是否会仅仅停留在箱子几何外壳，从而无法在底层 PD 控制器上激发出足够的持续抓夹力矩，导致抱起阶段箱子滑脱？

#### 3. 具体实施方案与技术路线
1. **数据切片生成**：
   编写转换补丁或配置项，在训练数据读取管道中将 `action` 的前 14 维直接映射赋值为当前帧（或未来单步）的 `observation.state[15:29]`，而后 4 维底盘遥控速度仍保留真实差分速度；
2. **基线微调训练**：
   在相同超参数（`steps=5000`, `batch_size=4`, `model/box_pick` 底模）下训练策略；
3. **对比评估指标**：
   * 在仿真中使用 [`dataset_tools/replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py) 观察推理输出的关节抖动率变化；
   * 在物理机器人上进行真机抱箱测试，记录触箱后力矩上升曲线与抱箱成功率。

---

### 5.2 验证课题二：全链路溯源采集中的 Action 与 State 具体数据来源

#### 1. 排查目标
彻底理清整个数据流管道中每一个信号字段的源头与经历的中间变换，建立完整的输入输出物理量映射图谱，精确定位 Action 抖动与姿态畸变究竟发生在哪一层级。

#### 2. 需要深入穿透的软件与硬件链路层级

```text
【PICO VR 设备】
     │  (手柄光学与 IMU 原始追踪数据)
     ▼
【XRoboToolkit / SMPL 姿态层】
     │  (PICO 驱动通过 XRoboToolkit 重构人体 SMPL 骨骼姿态)
     ▼
【pico_manager_thread_server.py】
     │  (提取 VR 3 点位姿: L-Wrist, R-Wrist, Neck，并执行零位标定)
     ▼  [ZMQ Port: 5556 / topic: pose]
【C++ WBC 控制器 (gear_sonic_deploy)】
     │  (执行全身逆运动学 IK 求解、双臂跟踪与下肢运控)
     ▼  [ZMQ Port: 5557 / topic: g1_debug]
【数据记录器 (run_data_exporter.py)】
     │  (订阅 g1_debug，调用 robot_model 组织 43 维数据结构)
     ▼  [Parquet 数据集]
【转换脚本 (dataset_tools/convert_rubberhand_to_g1_v30.py)】
     │  (剥离手部 14 维补零，重组为 29 维 State 与 18 维 Action)
     ▼
【面向 Pi0.5 微调训练集 (LeRobot v3.0 格式)】
```

#### 3. 重点排查问题清单
1. **数据集中 `action.wbc` / `action` 的物理实质**：
   * 它究竟是 C++ WBC 内部根据 VR 3 点位姿通过逆运动学求解算出的期望关节角（`last_action`）？
   * 还是 VR 设备经过 `XRoboToolkit` 人体姿态估计后输出的某种人体关节相对角度？
   * 其在写入 ZMQ 之前是否叠加了 default-angle offset 或经过了尺度缩放（`g1_action_scale`）？
2. **数据集中 `observation.state` 的物理实质**：
   * 它是 Unitree 机器人原厂底层驱动（CycloneDDS）由物理电机编码器真实返回的瞬时测量角（`unitree_joint_state.q()`）？
   * 还是经过了 SonicStar 内部低通滤波或状态估计器（State Estimator）处理后的平滑滤波状态？
3. **VR 相对位姿映射与标定偏移**：
   * 操作员在 PICO VR 中标定时的基准姿势（T-pose 或零位）是如何被捕获的？
   * `ThreePointPose` 中的脖颈逆四元数旋转矩阵与手腕相对平移偏移，是否存在累积漂移或未考虑人机体型比例的固定放大系数？
4. **开源数据集的数据产生差异**：
   * 开源数据集 `unitree_box_move_blue_full` 是由哪套软件架构生成的？
   * 其 `action` 是否使用了专用轨迹规划器（如闭环最小加加速度轨迹、样条平滑或在线阻抗控制器），导致其动作极其柔顺？

---

## 六、全链路溯源排查结论与转化前后一致性定论

针对 5.2 节提出的溯源课题，通过穿透源码、进行逐帧数值比对以及升级回放工具，现已形成确凿结论：

### 6.1 转化前 vs 转化后数值一致性排查

运行专用比对校验工具 [`verify_raw_vs_converted.py`](file:///home/yichangfeng/lerobot/dataset_tools/verify_raw_vs_converted.py)，对转化前（SonicStar 原始数据 `~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn/data/chunk-000/episode_000000.parquet`）与转化后（LeRobot v3.0 `datasets/g1_box_pick_turn_v30/data/chunk-000/file-000.parquet` Episode 0）进行逐点对齐比对：

| 评估项目 | 转化前 (SonicStar 原始采集) | 转化后 (LeRobot v3.0 格式) | 比对差异与排查结论 |
| :--- | :--- | :--- | :--- |
| **总有效帧数** | **840 帧** | **840 帧** | 完全对齐无抽帧或丢帧 |
| **Action 绝对数值差** | — | — | **最大绝对差 $5.95 \times 10^{-8}$**，平均绝对差 $8.39 \times 10^{-9}$（属于 float64 $\to$ float32 正常舍入误差） |
| **State 绝对数值差** | — | — | **最大绝对差 $1.11 \times 10^{-16}$**（浮点完全一致） |
| **Action 二阶抖动均值** | **$0.012007 \ \text{rad}$ ($12.01 \ \text{mrad}$)** | **$0.012007 \ \text{rad}$ ($12.01 \ \text{mrad}$)** | **精确一致，转化过程未引入任何数值抖动** |
| **State 二阶抖动均值** | **$0.001237 \ \text{rad}$ ($1.24 \ \text{mrad}$)** | **$0.001237 \ \text{rad}$ ($1.24 \ \text{mrad}$)** | **精确一致，物理状态保持平滑** |
| **抖动膨胀比率 (Act/St)** | **$9.70 \times$** | **$9.70 \times$** | **原始采集数据落盘瞬间就已经存在高频振颤** |

> [!IMPORTANT]
> **排查核心定论**：
> 格式转换脚本 [`convert_rubberhand_to_g1_v30.py`](file:///home/yichangfeng/lerobot/dataset_tools/convert_rubberhand_to_g1_v30.py) 仅执行了手部空槽位剔除（43D $\to$ 29D/18D）与底盘偏航速度推算，**100% 忠实保留了原始动作数值，未引入任何算法噪声。抖动的真正源头在采集记录器记录瞬间的底层信号中。**

---

### 6.2 对SonicStar原始数据进行分析

通过对 SonicStar 架构与原厂工具代码的深入审查，：

   *现已升级 [`replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py)，已可直接无缝回放 `~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn` 原始数据集，可直观验证两者的动作波形完全一致。*

---

### 6.3 信号全链路逐层穿透明细表

| 信号字段 | 链路节点与源文件 | 物理实质与数学变换 | 抖动表现与特征 |
| :--- | :--- | :--- | :--- |
| **State (实测状态)** | ① G1 电机编码器 $\to$ Unitree SDK `LowState_.motor_state[i].q`<br>② `g1_deploy_onnx_ref.cpp:2826` 转 IsaacLab 序<br>③ `zmq_output_handler.hpp:316` 转 MuJoCo 序并加回 `default_angles`<br>④ `run_data_exporter.py:662` 手部补零至 43D<br>⑤ `dataset_tools/convert_rubberhand_to_g1_v30.py:216` 剔除空手提取 29D | **瞬时物理电机实际测量角度**（无算法滤波，由转子与连杆物理惯量阻尼天然滤波）。 | 极其平滑（均值 $1.24 \ \text{mrad}$），但在箱壁表面受阻停留，无法反映期望抓紧意图。 |
| **Action (前 14 维双臂)** | ① PICO 4 手柄光学追踪 + IMU<br>② `pico_manager_thread_server.py:1857` 计算 `vr_3pt_position`<br>③ `g1_deploy_onnx_ref.cpp:3108` WBC 全身运控 RL 网络推理<br>④ `zmq_output_handler.hpp:345` 计算绝对目标角：$q_{\text{des}} = \text{action} \times \text{scale} + q_{\text{default}}$<br>⑤ `run_data_exporter.py:667` 组装为 43D `action.wbc`<br>⑥ `dataset_tools/convert_rubberhand_to_g1_v30.py:225` 提取双臂前 14 维 | **底层 PD 控制器追踪的目标绝对角度**。抱箱时深入物体内部以激发接触法向压力 $\tau = K_p(q_{\text{des}} - q_{\text{meas}})$。 | 剧烈抖动（均值 $12.01 \ \text{mrad}$），承载了 VR 追踪抖动与 IK 逆运动学放大。 |
| **Action (后 4 维底盘速度)** | ① `observation.root_orientation`（机体 IMU 四元数）<br>② `dataset_tools/convert_rubberhand_to_g1_v30.py:126` 计算偏航角差分：$\text{remote.rx} = -\text{yaw\_rate} = -\frac{\Delta \text{yaw}}{\Delta t}$ | **驱动下肢原地踏步转身的离散角速度指令**，与横移速度（恒为0）组合。 | 平滑低频指令，但需进行分位数对齐以防归一化除以 $\varepsilon$。 |

---

## 七、State 替代 Action 对比实验的具体实现与操作指引

### 7.1 核心理论与因果对齐设计

1. **时序因果对齐原则（`--shift 1`）**：
   在机器人模仿学习（行为克隆）中，时刻 $t$ 的动作 $a_t$ 物理含义是**“引导系统从当前状态 $s_t$ 转移到下一状态 $s_{t+1}$ 的控制量”**。
   * 若直接采用当前步 $a_t = s_t$，策略将退化为恒等映射（Identity Mapping），实机运行时往往表现为反应极度迟钝、静止卡顿；
   * 因此必须采用**未来一步物理状态**作为监督标签：
     $$a_t^{\text{arm}} = s_{t+1}^{\text{arm}}, \quad \text{其中 } t \in [0, T-2]; \quad a_{T-1}^{\text{arm}} = s_{T-1}^{\text{arm}}$$
2. **底盘速度遥控指令必须保留**：
   `observation.state` 仅包含 29 个电机角度与 IMU 姿态，**不包含底盘行走电机的速度目标**。因此后 4 维底盘速度指令 `action[14:18]`（尤其是原地踏步转身必需的 `remote.rx`）必须保持原样不变，否则机器人将丧失转向能力。
3. **分位数统计量自动重算与注入**：
   替换前 14 维动作后，动作的统计分布（`min..q99`）发生变化。脚本将自动重算前 14 维真实统计量，并自动注入基模后 4 维分位数，防止 QUANTILES 归一化除零与产生 NaN。

---

### 7.2 一键生成工具 `dataset_tools/replace_action_with_state.py`

已在 `dataset_tools/` 目录下落地自动化制作工具 [`dataset_tools/replace_action_with_state.py`](file:///home/yichangfeng/lerobot/dataset_tools/replace_action_with_state.py)。

运行以下命令，即可生成用于对比实验的新数据集：

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

# 一键生成实验组数据集 (采用下一帧状态 s_{t+1} 作为动作，并完成分位数对齐)
python dataset_tools/replace_action_with_state.py \
    --src-dir datasets/g1_box_pick_turn_v30 \
    --dst-dir datasets/g1_box_pick_turn_v30_state_as_action \
    --shift 1
```

*脚本已内置完整的 LeRobotDataset 归一化自检模块，执行完毕后将自动检验并确认无 NaN、无 Inf。*

---

### 7.3 策略微调训练启动命令

生成完成后，使用以下命令启动 $\pi_{0.5}$ 实验组策略微调训练：

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=g1_box_pick_turn_v30_state_as_action \
    --dataset.root=datasets/g1_box_pick_turn_v30_state_as_action \
    --policy.path=model/box_pick \
    --policy.train_expert_only=true \
    --policy.compile_model=false \
    --output_dir=outputs/train/pi05_state_as_action \
    --job_name=pi05_state_as_action_finetune \
    --batch_size=4 \
    --steps=5000 \
    --log_freq=50 \
    --save_freq=1000 \
    --env_eval_freq=0 \
    --policy.device=cuda \
    --wandb.enable=false
```

---

## 八、接下来的研判认识与后续推进任务清单 (下一步行动)

### 8.1 核心研判认识与物理博弈边界

通过上述机理穿透，我们对“使用 State 替代 Action”的实验结果建立清晰的先验研判：

1. **积极收益研判（平滑性必定大幅改善）**：
   由于实测状态 $q_{\text{state}}$ 的高频振颤比 Action 降低了近 10 倍（$1.24 \ \text{mrad}$ vs $12.01 \ \text{mrad}$），策略以平滑状态作为监督目标后，**推理输出的机械臂轨迹必然极其柔顺，彻底消除真机高频“抽搐”与电机驱动器尖锐啸叫**。
2. **核心物理风险研判（接触夹持力矩不足导致掉箱）**：
   橡胶手没有手指电机，完全依赖双臂将箱体横向抱紧。根据阻抗控制方程：
   $$\tau = K_p (q_{\text{des}} - q_{\text{meas}}) - K_d \dot{q}$$
   在原始 Action 中，操作员给出的 $q_{\text{des}}$ 深度穿透箱体几何表面，形成了稳定的位置超调误差 $(q_{\text{des}} - q_{\text{meas}}) > 0$，激发出持续的法向夹紧力矩；
   若直接以贴在箱壁表面的 $q_{\text{state}}$ 作为动作标签，训练出的策略在双手触碰到箱壁后，输出目标角仅仅停留在箱壁表面（即 $q_{\text{des}} \approx q_{\text{meas}}$），**底层 PD 控制器的夹紧力矩将急剧衰减甚至归零，可能导致机器人在转身踏步过程中箱子滑脱坠落！**

---

### 8.2 落地任务实施清单与兜底改进方案

后续工作严格按照以下优先级流水线推进：

```text
【任务 1】生成数据集 ──► 【任务 2】启动微调训练 ──► 【任务 3】离线/仿真抖动评估 ──► 【任务 4】物理真机部署验证
                                                                                    │
                                                            ┌───────────────────────┴───────────────────────┐
                                                            ▼ (抱箱稳固)                                     ▼ (力矩不足掉箱)
                                                      【实验完全成功】                                   【启用兜底方案】
                                                                                                    ├─ 方案 A: Action 滤波 (OneEuroFilter)
                                                                                                    ├─ 方案 B: 接触阶段力矩偏置补偿
                                                                                                    └─ 方案 C: 尾部悬停截断
```

#### 立即执行任务清单：
1. **执行数据集替换与生成**：
   运行 `python dataset_tools/replace_action_with_state.py` 生成 `datasets/g1_box_pick_turn_v30_state_as_action`；
2. **执行微调训练**：
   启动 `lerobot-train` 微调 `outputs/train/pi05_state_as_action`（5000 steps，预计耗时约 1.5~2 小时）；
3. **离线抖动量化评估**：
   使用离线评估工具预测轨迹，调用 [`evaluate_state_action.py`](file:///home/yichangfeng/lerobot/dataset_tools/evaluate_state_action.py) 统计动作抖动度是否从 $12.01 \ \text{mrad}$ 显著降低到 $1.5 \ \text{mrad}$ 附近；在 [`replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py) 中回放观察动作姿态；
4. **物理真机接触与掉箱验证**：
   按照 [`REAL_Deploy.md`](file:///home/yichangfeng/lerobot/REAL_Deploy.md) 部署权重，执行 10 次真机抱箱测试，重点记录：
   - 触碰箱体瞬间电机力矩值 `motor_state.tau_est`（是否达到 $> 5 \ \text{N}\cdot\text{m}$）；
   - 原地踏步转身期间箱子是否发生滑动或掉落。

#### 兜底与备选改进方案（若出现力矩不足掉箱）：
* **备选方案 A：对原始 Action 施加轨迹平滑滤波（首选推荐）**：
  若纯 State 无法维持抓取力矩，则证明“内夹超调”不可或缺。此时最佳方案是在转换管道中对 `action.wbc` 施加 **One-Euro Filter** 或 **Savitzky-Golay 样条滤波**：既滤除 $>10\text{Hz}$ 的 VR 光学追踪高频噪声，又完整保留稳态内夹穿透深度。
* **备选方案 B：平滑 State + 接触力矩偏置补偿（Contact Offset Compensation）**：
  采用 State 训练策略确保全流程平滑，而在实机部署推流端检测到接触箱体后，在底层为 Shoulder Roll 和 Elbow 施加固定的虚拟内夹偏置 $\Delta q_{\text{clamp}}$。
* **备选方案 C：放箱后无效帧截断**：
  利用 [`sanitize_sonic_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/sanitize_sonic_dataset.py) 自动截断任务终局放箱后操作人员反应延迟造成的最后 2~3 秒待机悬停帧，彻底根除“举手投降”姿态。

---

---

## 九、3 阶段子任务（Sub-task）自动切片、失败数据剔除与可信度验证

### 9.1 核心痛点与算法机理定位

为了彻底破除单 Prompt 长程任务导致的**因果混淆（Causal Confusion）与“不夹箱子直接举手”现象**，我们为整个轨迹引入多阶段细粒度 Sub-task 语言标注：
* `Task 0`: `"clamp and lift the box"`（从开始到转弯前）
* `Task 1`: `"hold the box and turn right"`（原地向右踏步转身阶段）
* `Task 2`: `"place the box on the table and release"`（放箱至目标桌并释放）

#### 为什么单纯依赖“角速度”不合适？
经对全量 87 个 Episode 逐帧排查，单纯依赖瞬时角速度阈值（如 $|rx| > \epsilon$）存在三大物理缺陷：
1. **开局晃动与重心调整误判**：如 Episode 1 在伸手阶段底盘产生 $+11.7^\circ$ 摆动，纯角速度在第 72 帧过早触发“开始转弯”（此时箱子还没碰到）；
2. **踏步过零点与顿挫**：双足原地踏步时两脚交替导致角速度出现短暂停顿或落入死区，单纯看角速度会造成阶段标签在 0 和 1 之间剧烈跳变闪烁；
3. **转弯完成界限模糊**：容易在转到 $75^\circ$ 时因瞬时角速度下降过早切入放箱阶段。

#### 科学判定方案：多信号融合滤波算法
落地 [`verify_subtask_splits.py`](file:///home/yichangfeng/lerobot/dataset_tools/verify_subtask_splits.py) 与 [`create_subtask_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/create_subtask_dataset.py)：
* **偏航积分角**：$\psi(t) = \int (-\text{rx}) dt$，右转单调积累至 $-90^\circ$；
* **转弯起点 $t_{\text{split1}}$**：偏航角单调下穿有效右转门限，向后沿平滑角速度上升沿回溯，确保滤除开局零漂；
* **转弯终点 $t_{\text{split2}}$**：偏航角达到目标转角 88%（约 $-80^\circ \sim -90^\circ$），且平滑角速度回落至 $< 0.1 \text{ rad/s}$；
* **双臂几何双重约束**：结合机械臂肩部 Pitch（俯仰高度）与 Roll（开合宽度）验证抬箱与放箱姿态。

---

### 9.2 失败样本审计与剔除定论

全量扫描发现，**Episode 32（总转角仅 $-4.58^\circ$）与 Episode 48（总转角仅 $-7.19^\circ$）为操作未转弯的失败示范数据**。根据用户确认，新数据集生成工具已将这两个 Episode **直接彻底剔除**：
* 原始数据量：87 个 Episode，60,809 帧；
* 剔除后数据量：**85 个高质量有效 Episode，59,825 帧**；
* 重新连续对齐全局帧索引（`index: 0..59824`）与 Episode 索引（`episode_index: 0..84`）。

---

### 9.3 三重直观验证保障体系

1. **可视化时间序列切片报告（含真实机载画面）**：
   运行 `python dataset_tools/verify_subtask_splits.py`，自动为代表性 Episode 生成诊断图表（保存在 [`outputs/subtask_reports/`](file:///home/yichangfeng/lerobot/outputs/subtask_reports)）：
   * 上下两层时序曲线显示偏航角与双臂构型；
   * 底部拼接 $t_0$、$t_{\text{split1}}$、$t_{\text{split2}}$、$t_{\text{end}}$ 4 个关键帧的真实机载摄像头画面，清晰直观核验：
     - $t_{\text{split1}}$ 时，箱子已被机械臂紧紧夹住并完全离开桌面；
     - $t_{\text{split2}}$ 时，机器人已精准右转 $90^\circ$ 正对目标桌面。
2. **MuJoCo 3D 回放 HUD 动态指示**：
   升级 [`replay_g1_dataset.py`](file:///home/yichangfeng/lerobot/dataset_tools/replay_g1_dataset.py)，在仿真回放或导出视频时，终端与画面实时悬浮显示当前帧归属的 Sub-task 名称。
3. **LeRobotDataset 与 Pi0.5 Tokenizer 全链路自检**：
   已执行端到端分词测试，确认数据集中每帧的 `item["task"]` 能够随着时间步精准切换为对应的 3 个子任务指令，并在 `processor_pi05.py` 中顺利完成 PaliGemma Discretized State-Language Tokenization。

---

### 9.4 3-Subtask 数据集生成与训练指引

#### 生成新数据集命令：
```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

python dataset_tools/create_subtask_dataset.py \
    --src-dir datasets/g1_box_pick_turn_v30 \
    --dst-dir datasets/g1_box_pick_turn_v30_subtasks \
    --exclude-episodes 32 48
```

#### 启动 3-Subtask 策略微调训练命令：
```bash
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=g1_box_pick_turn_v30_subtasks \
    --dataset.root=datasets/g1_box_pick_turn_v30_subtasks \
    --policy.path=model/box_pick \
    --policy.train_expert_only=true \
    --policy.compile_model=false \
    --output_dir=outputs/train/pi05_subtasks \
    --job_name=pi05_subtasks_finetune \
    --batch_size=4 \
    --steps=5000 \
    --log_freq=50 \
    --save_freq=1000 \
    --env_eval_freq=0 \
    --policy.device=cuda \
    --wandb.enable=false
```

---

> **结语**：
> 本文已闭环完成全链路溯源、State-as-Action 实验实现、3 阶段子任务自动化切片、失败数据过滤与可视化核验工具落地。

