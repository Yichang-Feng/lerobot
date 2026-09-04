# Unitree G1 (Rubber Hand) 实机数据采集与 Pi0.5 微调全流程指南
# (Data Collection to Pi0.5 Fine-tuning Guide for Unitree G1 with Rubber Hands)

本文档针对配备 **Rubber Hand（固定橡胶手，29-DoF 刚体关节，无灵巧手手指电机）** 的物理 Unitree G1 人形机器人，结合机载实际部署的 **`robot_server`（真实视觉与电机通信桥接）** 与 **`REAL_Deploy.md`** 规范，详细指导如何完成：
1. **机载视觉服务（`robot_server/camera_server`）启动**；
2. **SonicStar / GEAR-SONIC 实机遥操作与自动视频录制**；
3. **Rubber Hand 43维原始数据解析与清洗**；
4. **面向 `~/lerobot/model/box_pick`（π₀.₅ 策略模型）的数据格式转化（含原地抱箱转身动力学处理）**；
5. **启动 Pi0.5 快速微调训练**。

---

## 目录
1. [核心疑问解答 (FAQ)](#一核心疑问解答-faq)
   - 实机采集与仿真采集是否一致？
   - 是否需要像 VLA 实机推理一样启动视觉节点？
   - 怎么录制视频数据？
2. [硬件拓扑与机载 `robot_server` 架构](#二硬件拓扑与机载-robot_server-架构)
   - 网络拓扑与端口划分 (5555 / 6000 / 6001)
   - 机载相机服务 (`server.py`) 原理与键名兼容性 (`head_camera` vs `ego_view`)
   - Rubber Hand 采集数据的具体维度 (43维结构)
3. [实机遥操作采集全流程操作指引 (SOP)](#三实机遥操作采集全流程操作指引-sop)
   - 准备工作与安全规范
   - 【终端 0】机器人端启动机载服务 (`start_server.sh`)
   - 【终端 1】上位机启动 C++ WBC 底层部署 (`deploy.sh`)
   - 【终端 2】上位机启动 PICO VR 遥控串流 (`pico_manager`)
   - 【终端 3】上位机启动相机监视画面 (`camera_viewer`)
   - 【终端 4】上位机启动数据自动记录器 (`data_exporter`)
   - 录制与标记交互按键 (PICO 手柄 & 键盘)
4. [数据转化：面向 Pi0.5 (model/box_pick)](#四数据转化面向-pi05-modelbox_pick)
   - 维度映射关系 (43D $\to$ 29D State, 18D Action)
   - 原地抱箱转身任务的关键动力学处理 (`remote.rx`)
   - 转换脚本执行 (`convert_rubberhand_to_g1_v30.py` / `Psi0`)
5. [启动 Pi0.5 快速微调训练](#五启动-pi05-快速微调训练)

---

## 一、核心疑问解答 (FAQ)

### Q1: 实机采集的流程和仿真中是否一致？
**核心结论：上层遥控操作与数据记录逻辑完全一致，主要区别在物理底层与相机源。**

| 比较项 | 物理真机采集 (Real Robot) | MuJoCo 仿真采集 (Sim) |
| :--- | :--- | :--- |
| **相机画面来源** | G1 机器人机载轻量服务 (`robot_server/camera_server/server.py`) | 仿真器离屏渲染 (`run_sim_loop.py --enable-image-publish`) |
| **底层运控节点** | `deploy.sh` 绑定物理直连网卡 (如 `enx6c1ff724495a`) | `deploy.sh` 传入 `sim` 参数与本地模拟器通信 |
| **仿真节点 (`run_sim_loop.py`)** | **严禁运行 / 跳过**（由物理机器人和物理世界充当实体） | **必须运行** |
| **遥操推流 (PICO VR)** | 完全一致 (`pico_manager_thread_server.py`) | 完全一致 |
| **数据记录器 (`run_data_exporter`)** | 完全一致，指向机器人机载 IP (`192.168.123.164:5555`) | 完全一致，指向 `localhost:5555` |
| **手柄录制按键控制** | 完全一致 (`Left Grip + A` 启停/保存) | 完全一致 |
| **保存文件格式** | 完全一致 (LeRobot v2.1 Parquet + MP4) | 完全一致 |

### Q2: 怎么录制视频数据？
**视频是由上位机数据记录器 (`run_data_exporter.py`) 在后台自动订阅截帧、队列缓存并编码合成。**
1. 上位机运行数据记录器时指定机器人的相机地址 `--camera-host 192.168.123.164 --camera-port 5555`。
2. 记录器通过 ZMQ 持续接收机载相机推流。
3. 当操作人员按下录制快捷键（PICO 手柄 `Left Grip + A` 或上位机键盘 `c`）触发 **START RECORDING** 时，视频帧与当前时刻的关节状态严格对齐并压入缓存队列。
4. 当完成一次“抱箱+转身+放置”再次按下快捷键触发 **STOP RECORDING** 时，系统自动调用底层 `LeRobotDataset`（基于 PyAV / H.264 编码），将该 Episode 的图像序列直接编码写入为标准 `.mp4` 文件：
   ```text
   videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
   ```
   同时同步写入该段动作的表格式数据 `data/chunk-000/episode_XXXXXX.parquet`。

---

## 二、硬件拓扑与机载 `robot_server` 架构

### 1. 网络拓扑与端口划分

```text
       【机器人机载端 (Jetson Orin)】                           【上位机工作站 (PC / 4090)】
           IP: 192.168.123.164                                  IP: 192.168.123.213
 ┌───────────────────────────────────────┐              ┌─────────────────────────────────────┐
 │ 机载服务 (robot_server)                │              │ PICO VR Teleop (pico_manager)       │
 │   ├─ camera_server/server.py          │              │   └─ 发布 pose (Port:5556)          │
 │   │    └─ ZMQ PUB 广播画面 (Port:5555) ─(ZMQ Video)─►│                                     │
 │   └─ camera_server/motor_server.py    │              ├─────────────────────────────────────┤
 │        ├─ ZMQ PULL 接收指令 (Port:6000)│◄─────────────│ C++ WBC 控制器 (gear_sonic_deploy)   │
 │        └─ ZMQ PUB 广播状态 (Port:6001)─┼─────────────►│   └─ 发布 g1_debug 状态 (Port:5557)  │
 ├───────────────────────────────────────┤              ├─────────────────────────────────────┤
 │ Unitree 电机底层驱动 (CycloneDDS)     │              │ 数据收集器 (run_data_exporter.py)     │
 │   └─ 宇树原厂电机执行器 (29-DoF 刚体)  │              │   └─ 接收 5555/5556/5557 自动生成     │
 └───────────────────────────────────────┘              │      LeRobot Parquet + MP4          │
                                                        └─────────────────────────────────────┘
```

### 2. 机载相机服务 (`server.py`) 原理与键名兼容性技巧

机载视觉推流脚本位于 `robot_server/camera_server/server.py`：
* 采用 OpenCV 打开指定视频设备（默认 `/dev/video2`，640×480 @ 30fps）；
* 转为 RGB 并进行 JPEG 压缩，通过 base64 编码后以 JSON 格式在 ZMQ 5555 端口广播：
  ```python
  payload = {
      "timestamps": {camera_name: timestamp},
      "images": {camera_name: base64_image_string}
  }
  ```

> [!IMPORTANT]
> **相机键名（Camera Name）关键兼容性：**
> * **LeRobot 实机推理 (`REAL_Deploy.md`)**：默认寻找 `"head_camera"` 并映射给 `global_view`。
> * **Sonic 采集记录器 (`run_data_exporter.py`)**：默认寻找与特征名一致的 `"ego_view"`（由 `observation.images.ego_view` 拆解）。
>
> **解决方案（二选一）**：
> 1. **命令行显式指定（最简单）**：在机器人端启动时显式传入 `--name ego_view`：
>    ```bash
>    python3 ~/lerobot/camera_server/server.py --device 2 --port 5555 --name ego_view
>    ```
> 2. **代码双向兼容（最推荐，一次修改两处通用）**：
>    在机器人端 `~/lerobot/camera_server/server.py` 的第 72~75 行将 payload 构造改为同时包含两个键名：
>    ```python
>    payload = {
>        "timestamps": {"head_camera": timestamp, "ego_view": timestamp},
>        "images": {"head_camera": encoded_image, "ego_view": encoded_image}
>    }
>    ```
>    这样不管是 LeRobot Rollout 还是 Sonic 数据采集均可免改参无缝连接！

---

### 3. Rubber Hand 采集数据的具体维度 (43维结构)

虽然物理 G1 机器人只配备了 29 个刚体电机（12腿 + 3腰 + 14双臂），但 SonicStar 采集系统（Pinocchio 运动学模型）保持统一样本结构，将手部空缺槽位用**固定常数（0值）**补全：

1. **`observation.images.ego_view`**：`[480, 640, 3]` (RGB 视频，50 FPS)。
2. **`observation.state` (43维 float64)**：
   * `[0:12]`：双腿 12 关节测量角度（左6 + 右6）。
   * `[12:15]`：腰部 3 关节测量角度（Yaw, Roll, Pitch）。
   * `[15:22]`：左臂 7 关节测量角度（Shoulder P/R/Y, Elbow, Wrist R/P/Y）。
   * `[22:29]`：左手 7 自由度（**橡胶手无电机，系统自动填充常数 0**）。
   * `[29:36]`：右臂 7 关节测量角度（Shoulder P/R/Y, Elbow, Wrist R/P/Y）。
   * `[36:43]`：右手 7 自由度（**橡胶手无电机，系统自动填充常数 0**）。
3. **`action.wbc` (43维 float64)**：与 `observation.state` 结构一一对应的期望目标关节角度。
4. **`teleop.*` 辅助特征**：
   * `teleop.delta_heading` (1维): 机体偏航旋转增量。
   * `teleop.planner_movement` (3维): 速度方向向量。
   * `teleop.planner_facing` (3维): 躯干朝向向量。
   * `teleop.planner_speed` (1维): 规划移动线速度。
   * `teleop.planner_height` (1维): 离地高度目标。
   * `observation.root_orientation` (4维): 基座 IMU 四元数 `[qw, qx, qy, qz]`。

---

## 三、实机遥操作采集全流程操作指引 (SOP)

### 1. 准备工作与安全规范
1. **吊架安全第一**：必须使用龙门架/顶部弹性安全绳吊住 G1 机器人的吊环，调节绳长使脚掌平稳踩实地面，在倾倒前能被拉住。
2. **网络连通性检查**：在上位机确认网卡 IP 并 ping 通机器人：
   ```bash
   ping -c 3 192.168.123.164
   ```
3. **佩戴 PICO 4 头显**：打开串流助手，进入 VR 串流界面并校准站立高度。

---

### 2. 多终端标准启动序列

#### 【终端 0】(机载计算机 SSH) 启动机载视觉与电机服务
通过网线 SSH 登录 G1 机载系统（密码通常为 `123`）：
```bash
ssh unitree@192.168.123.164
cd ~/lerobot

# 启动机载服务（支持 --device 指定摄像头设备号，默认 2）
./start_server.sh
```
> 若需要单独以 `ego_view` 键名推流相机：
> ```bash
> python3 ~/lerobot/camera_server/server.py --device 2 --port 5555 --name ego_view
> ```

#### 【终端 1】(上位机) 启动 C++ WBC 控制器
在上位机连接物理直连网卡（如 `enx6c1ff724495a`）：
```bash
cd ~/SonicStar/wbc/gear_sonic_deploy
conda activate starVLA

bash deploy.sh --input-type zmq_manager enx6c1ff724495a
```
等待打印 `Init done`，底层控制器进入就绪状态。

#### 【终端 2】(上位机) 启动 PICO 遥操串流服务
```bash
cd ~/SonicStar/wbc
conda activate starVLA

python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

#### 【终端 3】(上位机) 启动相机监视窗口 (验证图像传输与光照)
```bash
cd ~/SonicStar/wbc
conda activate starVLA

python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```
此时应能看到清晰的第一视角机载相机画面。

#### 【终端 4】(上位机) 启动数据记录器 (Data Exporter)
```bash
# 必须使用专门预装了 datasets 和 lerobot 录制依赖的环境
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
cd ~/SonicStar/wbc

python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box, turn right, and place it on the table" \
    --dataset-name g1_rubberhand_pick_turn \
    --camera-host 192.168.123.164 \
    --camera-port 5555 \
    --data-collection-frequency 50
```

> [!TIP]
> **Prompt 规范设计说明**：
> * **推荐标准 Prompt**：`"pick up the box, turn right, and place it on the table"`
> * **测试评估与泛化变体族（Evaluation Prompts）**：
>   微调完成后，可以测试以下同义变体检验模型的语言泛化能力：
>   - `"pick up the box, turn right, and put it on the table"`
>   - `"pick up the box and turn right to place it on the table"`
>   - `"lift the box, turn right, and place it on the table"`

---

### 3. 录制与标记交互按键

遥操人员在控制机器人抱起箱子、转身放置的过程中，通过以下按键控制数据录制：

| 控制设备 | 按键动作 | 功能行为 | 语音反馈 |
| :--- | :--- | :--- | :--- |
| **PICO VR 手柄** | **Left Grip + A** | **开始录制 / 结束并保存** 当前 Episode | *"Recording started"* / *"Episode saved"* |
| **PICO VR 手柄** | **Left Grip + B** | **放弃当前 Episode** (标记为 Discarded，丢弃失误动作) | *"Episode discarded"* |
| **上位机键盘** | `c` 键 | 切换录制 / 保存 | 同上 |
| **上位机键盘** | `x` 键 | 放弃当前 Episode | 同上 |

每次录制完毕后，数据会自动累积保存在：
`~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn/`

### 4. 数据整理与异常样本剔除 (通用清洗工具)

在批量录制过程中，若发现某些 Episode 操作失误（掉箱、绊倒）或前置等待时间过长，请使用专用的全自动清洗工具 [`sanitize_sonic_dataset.py`](file:///home/yichangfeng/lerobot/sanitize_sonic_dataset.py)：

```bash
# 示例 1: 删除失败的第 5 和第 14 条数据
python ~/lerobot/sanitize_sonic_dataset.py --delete-episodes 5 14

# 示例 2: 将第 2 条数据从第 13.0 秒起步（裁剪掉前 13 秒等待期）
python ~/lerobot/sanitize_sonic_dataset.py --trim-start 2:13.0

# 示例 3: 仅重新排版与编号对齐（自动检查音画同步）
python ~/lerobot/sanitize_sonic_dataset.py
```
> 详细数据要素规范与跨 Agent 复用说明请参阅：[`Dataset_Cleaning_Guide.md`](file:///home/yichangfeng/lerobot/Dataset_Cleaning_Guide.md)。

---

## 四、数据转化：面向 Pi0.5 (model/box_pick)

### 1. 维度对应与映射法则

`~/lerobot/model/box_pick` 是基于 Physical Intelligence 的 **$\pi_{0.5}$ (pi05)** 模型，其输入输出规范为：
* **Observation State (29维)**：12腿 + 3腰 + 7左臂 + 7右臂（**完美对应 Rubber Hand 实机的 29 个真实物理电机**）。
* **Action (18维)**：7左臂期望角度 + 7右臂期望角度 + 4维底盘遥控速度 `[remote.lx, remote.ly, remote.rx, remote.ry]`。
* **Images**: 键名必须命名为 `observation.images.global_view`。

### 2. “原地抱箱转身放桌子”的关键动力学处理
在 LeRobot 的 G1 底层控制器中：
* `cmd_vel[0]`（前进/后退速度）= `remote.ly`
* `cmd_vel[1]`（横向移动速度）= `-remote.lx`
* `cmd_vel[2]`（**原地旋转偏航角速度 Yaw Rate**）= **`-remote.rx`**

> [!IMPORTANT]
> 针对您的任务**“原地抱起一个箱子然后转身放在另一个桌子上”**：
> 不能将这 4 维遥控速度全赋 0！由于不涉及大范围平移走动，`lx` 和 `ly` 设为 0，但 **`remote.rx` 必须注入转向角速度**：
> $$r_x = -\text{yaw\_rate} = -\frac{\Delta \text{yaw}}{\Delta t}$$
> 这样模型微调后，才能学会“在双臂保持抱箱托举姿态的同时，输出底盘旋转指令驱动下肢原地踏步转向”。

### 3. 执行数据转换

在 `~/lerobot` 项目中直接运行转换脚本 [`convert_rubberhand_to_g1_v30.py`](file:///home/yichangfeng/lerobot/convert_rubberhand_to_g1_v30.py)：

```bash
cd ~/lerobot
conda activate lerobot
python convert_rubberhand_to_g1_v30.py \
    --src-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn \
    --dst-dir ~/lerobot/datasets/g1_box_pick_turn_v30 \
    --fps 30 \
    --task "pick up the box, turn right, and place it on the table"
```

*（若想基于 `~/Psi0` 的 36 维标准全身动作头进行训练，可调用 [`~/Psi0/scripts/data/convert_sonic_to_psi36.py`](file:///home/yichangfeng/Psi0/scripts/data/convert_sonic_to_psi36.py) 进行转码）。*

---

## 五、启动 Pi0.5 快速微调训练

> [!TIP]
> 完整训练参数表、后台运行指南及部署说明已整理于专属文档：[`Train_Pi05_Guide.md`](file:///home/yichangfeng/lerobot/Train_Pi05_Guide.md)。

转换完成后，加载已有预训练权重作为底模启动轻量微调（微调模式下显存仅占用约 8~9 GB，跳过耗时编译，5秒内启动）：

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=g1_box_pick_turn_v30 \
    --dataset.root=datasets/g1_box_pick_turn_v30 \
    --policy.path=model/box_pick \
    --policy.train_expert_only=true \
    --policy.compile_model=false \
    --output_dir=outputs/train/pi05_box_pick_turn_final \
    --job_name=pi05_box_turn_finetune \
    --batch_size=4 \
    --steps=5000 \
    --log_freq=50 \
    --save_freq=1000 \
    --env_eval_freq=0 \
    --policy.device=cuda \
    --wandb.enable=false
```

微调生成的权重将保存至 `outputs/train/pi05_box_pick_turn_final/checkpoints/`，随后即可按照 [`REAL_Deploy.md`](file:///home/yichangfeng/lerobot/REAL_Deploy.md) 的实机部署流程进行在线 Rollout 验证。

