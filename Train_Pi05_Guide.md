# Unitree G1 (Rubber Hand) Pi0.5 策略模型微调训练指南

本文档专门整理 Unitree G1 橡胶手（29-DoF 刚体关节）针对**“原地抱箱、转身、放桌上”**任务的 $\pi_{0.5}$ 策略模型微调训练全流程与可直接复制执行的命令。

---

## 一、硬件与环境实测结论

* **GPU 规格**：NVIDIA GeForce RTX 4090 D（24,564 MiB 显存）
* **基础底模**：`~/lerobot/model/box_pick`（Physical Intelligence $\pi_{0.5}$ 2.3B 参数多模态策略模型，含 SigLIP 视觉骨干 + Flow Matching 动作扩散头）
* **显存实测情况**：
  * `batch_size=2`：显存占用约 **10.1 GB / 24.5 GB**（剩余 14.4 GB，非常轻松）
  * `batch_size=4`（**推荐**）：显存占用约 **14.5 GB ~ 16.0 GB**，兼顾梯度稳定性和显存安全性，完全不会 OOM。
* **首次编译说明**：PyTorch Inductor 和 Triton 在 Step 1 时会自动针对 RTX 4090 D 进行 Kernel 自动寻优编译（耗时约 1~2 分钟）。编译完成后，后续每个 Step 训练耗时约 **0.3 ~ 0.5 秒**。

---

## 二、训练前环境配置

由于系统底层的 `libstdc++.so.6` 动态链接库版本可能低于 Conda 环境中的版本，在启动训练前**务必设置 `LD_LIBRARY_PATH`**，否则可能触发 Triton / Inductor 编译时的 CXXABI 报错：

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH
```

---

## 三、训练前数据集同步（可选但建议）

若原始采集目录 `~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn` 中有新增的录制 Episode，执行以下命令同步转换为 LeRobot v3.0 格式（已包含转向角速度动力学解耦）：

```bash
cd ~/lerobot
conda activate lerobot

python convert_rubberhand_to_g1_v30.py \
    --src-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn \
    --dst-dir ~/lerobot/datasets/g1_box_pick_turn_v30 \
    --fps 30 \
    --task "pick up the box, turn right, and place it on the table"
```

---

## 四、训练核心命令

### 1. 试水快速验证命令（100 步，验证数据加载与 Loss 计算）

此命令用于验证显存占用、Triton 编译与 Loss 下降曲线是否符合预期：

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=g1_box_pick_turn_v30 \
    --dataset.root=datasets/g1_box_pick_turn_v30 \
    --policy.path=model/box_pick \
    --output_dir=outputs/train/test_smoke_run \
    --job_name=test_smoke_pi05 \
    --batch_size=2 \
    --steps=100 \
    --log_freq=10 \
    --save_freq=0 \
    --env_eval_freq=0 \
    --policy.device=cuda \
    --wandb.enable=false
```

---

### 2. 正式微调训练命令（推荐 5000 步）

> [!TIP]
> **关于首次启动耗时**：
> 底模 `model/box_pick` 默认配置了 `compile_mode: "max-autotune"`。
> - **默认模式**：首次启动 Step 1 时，Triton 会对 2.3B 模型的每一层矩阵乘法和注意力算子进行详尽硬件测速（耗时约 5~8 分钟），编译完成后之后 5000 步达到最快。
> - **秒开模式（强烈推荐）**：加上 `--policy.compile_model=false`，将**完全跳过 Triton 编译测试，5 秒内直接启动训练**，在 RTX 4090 D 强大的纯算力下整体耗时几乎没有明显差异！

#### 方案 A：秒开微调模式（强烈推荐，5秒内启动，显存仅约 8~9 GB）

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

#### 方案 B：后台持久化运行（nohup，秒开模式）

```bash
cd ~/lerobot
conda activate lerobot
export LD_LIBRARY_PATH=/home/yichangfeng/miniforge3/envs/lerobot/lib:$LD_LIBRARY_PATH

nohup python -m lerobot.scripts.lerobot_train \
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
    --wandb.enable=false > train.log 2>&1 &
```

* **实时查看训练日志输出**：
  ```bash
  tail -f ~/lerobot/train.log
  ```
* **监控 GPU 显存与核心温度**：
  ```bash
  watch -n 1 nvidia-smi
  ```
* **若需中途停止训练**：
  ```bash
  pkill -f lerobot_train
  ```

---

## 五、参数详细说明表

| 参数名 | 填报值 | 作用与关键注意事项 |
| :--- | :--- | :--- |
| `--policy.train_expert_only` | `true` | **核心微调设置**。冻结 2B PaliGemma 大视觉语言模型，仅微调 300M Action Expert。大幅削减显存（从 30GB+ 降至 8GB），防止小样本过拟合并杜绝 OOM。 |
| `--policy.compile_model` | `false` | **秒开设置**。跳过漫长的 Triton Autotune 编译测试，5 秒内直接启动训练，同时节省 3GB+ CUDA Graph 显存。 |
| `--dataset.repo_id` | `g1_box_pick_turn_v30` | **必填项**。LeRobot 数据集标识名，不填会报 `Missing required field repo_id` 错误。 |
| `--dataset.root` | `datasets/g1_box_pick_turn_v30` | 本地转换后的 LeRobot v3.0 数据集目录路径。 |
| `--policy.path` | `model/box_pick` | 基础底模权重目录（内置预训练参数与 `config.json`）。 |
| `--output_dir` | `outputs/train/...` | 训练产物（日志、Checkpoints、最终模型）的存放路径。 |
| `--batch_size` | `4`（正式） / `2`（试水） | RTX 4090 D 推荐为 `4`；微调模式下显存仅占用约 8~9 GB，极为安全。 |
| `--steps` | `5000` | 迭代总步数。5000 步对于 20~30 条示教轨迹的微调收敛效果最佳。 |
| `--log_freq` | `50` | 每 50 步打印一次当前 Step、训练 Loss、采样耗时与吞吐量。 |
| `--save_freq` | `1000` | 每 1000 步自动存盘一份 Checkpoint，防止训练中断前功尽弃。 |
| `--env_eval_freq` | `0` | **设为 0 关闭仿真评估**。由于真机任务无 MuJoCo 仿真环境，必须关闭该项以避免报错。 |
| `--policy.device` | `cuda` | 强制使用 GPU 加速。 |
| `--wandb.enable` | `false` | 关闭远端 WandB 云端看板上传，保证离线稳定运行。 |

---

## 六、训练产物与实机部署衔接

训练完成后，权重将保存在：
```text
outputs/train/pi05_box_pick_turn_final/
├── checkpoints/
│   ├── 001000/pretrained_model/
│   ├── 002000/pretrained_model/
│   ├── 003000/pretrained_model/
│   ├── 004000/pretrained_model/
│   └── 005000/pretrained_model/   <- [最优微调权重]
```

后续如需实机推理部署，直接在部署命令（详见 [`REAL_Deploy.md`](file:///home/yichangfeng/lerobot/REAL_Deploy.md)）中将 `--policy.path` 替换为最新产物即可：

```bash
--policy.path=outputs/train/pi05_box_pick_turn_final/checkpoints/005000/pretrained_model
```

---

## 七、查看训练指标与损失曲线

### 方式 1：终端实时输出（最直观）
在正在训练的终端窗口中，系统每隔 50 步（`--log_freq=50`）会实时打印一行结构化指标：
```text
step:50 smpl:200 ep:0.3 epch:0.01 loss:0.184 grdn:0.85 lr:2.5e-05 data_s:0.01 prep_s:0.00 updt_s:0.21 step_s:0.23 smp/s:17 mem_gb:8.91
step:100 smpl:400 ep:0.6 epch:0.02 loss:0.132 grdn:0.79 lr:2.5e-05 data_s:0.01 prep_s:0.00 updt_s:0.21 step_s:0.23 smp/s:17 mem_gb:8.91
```
* **`loss`**：策略流匹配（Flow Matching）的动作拟合损失。正常情况下会从最开始的 **0.20 左右逐步震荡下降到 0.03 ~ 0.05 左右**。
* **`step`**：当前步数（从 0 走到 5000）。
* **`mem_gb`**：当前实测显存占用（约 8.9 GB）。
* **`step_s`**：单步迭代耗时（约 0.23 秒，非常快）。

### 方式 2：一键绘制离线折线图（保存为高清图片）
我们提供了配套的绘图小工具 [`plot_train_curve.py`](file:///home/yichangfeng/lerobot/plot_train_curve.py)。
若你是使用 `nohup ... > train.log 2>&1 &` 运行的，或者将终端输出保存为了日志文件，只需执行：
```bash
python plot_train_curve.py train.log
```
即可在当前目录下生成高分辨率的 **`training_loss_curve.png`** 折线图（包含 Loss 下降趋势与学习率变化）。

### 方式 3：WandB 在线网页动态看板（可选）
如果希望在浏览器中看到类似 TensorBoard 的交互式动态实时曲线：
只需在训练命令中将 `--wandb.enable=false` 改为：
```bash
--wandb.enable=true
```
启动时终端会输出一个专属网页链接，用浏览器打开即可实时查看带平滑滤波、缩放的交互式 Loss 曲线。

