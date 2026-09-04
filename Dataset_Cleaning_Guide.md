# Unitree G1 遥操数据整理与清洗复用指南 (SonicStar / WBC)

本指南针对在 Unitree G1 上使用 **SonicStar / GR00T-WBC** 进行 VR 遥控采集时产生的数据，详细说明**底层数据结构、需要整理哪些文件、为何不能手动单点删除**，并提供**一键自动化清洗脚本**，方便在切换 Agent、多终端或后续长期项目中快速复用。

---

## 一、为什么需要数据整理？

在真机遥控采集（如抱箱转身放桌子）过程中，必然会遇到以下几类常见情况：
1. **起手等待期过长**：VR 佩戴者在按下开始录制后，前几秒还在调整站姿或等待机器人站稳，导致轨迹前段有数秒是无用的原地静止（如 Episode 2 前 13 秒未抱箱）；
2. **操作失误/脱手掉箱**：某些轨迹中途箱子滑落或步态绊倒，需要将整条失败轨迹废弃；
3. **索引空洞**：若直接手动用 `rm` 删除了某个 `episode_000005.mp4`，会导致编号断层（0, 1, 2, 3, 4, 6...），使得后续 `run_data_exporter.py` 断点续采混乱，甚至使下游 LeRobot DataLoader 抛出找不到文件的异常。

> [!CAUTION]
> **绝对不能仅在文件夹里手动删视频或删单张表格！**
> 采集系统是一个闭环，包含 Parquet 动作表、MP4 视频流以及 3 个索引元数据文件，各文件之间存在严格的帧数、时间戳与全局索引映射。一旦单点删除导致不一致，会导致**数据导出器崩溃**或**模型训练报错退出**。

---

## 二、数据集必须同步整理的 5 大核心要素

原始采集数据存储于 `~/SonicStar/wbc/outputs/<dataset_name>/`（例如 `outputs/g1_rubberhand_pick_turn`）：

```text
g1_rubberhand_pick_turn/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  <- [要素1] 动作与关节时序表 (43维状态/动作、四元数等)
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.ego_view/
│           ├── episode_000000.mp4  <- [要素2] 第一视角 50Hz RGB 视频流 (帧数必须与 Parquet 严格 1:1)
│           ├── episode_000001.mp4
│           └── ...
└── meta/
    ├── info.json                   <- [要素3] 总样本数、总帧数、视频总数、train split
    ├── episodes.jsonl              <- [要素4] 每条 Episode 的编号、任务 Prompt、帧数长度
    └── episodes_stats.jsonl        <- [要素5] 每条 Episode 所有维度的 min/max/mean/std 归一化统计
```

### 整理规范要求：
1. **帧数严格一致**：每条 Episode 的 `Parquet 行数` 与 `MP4 视频总帧数` 必须 `Diff = 0`；
2. **序号严格连续**：所有样本必须从 `episode_000000` 开始连续递增至 `episode_{N-1}`，中间不得跳号；
3. **时序严格归零**：若做了时间裁剪，裁剪后的 `timestamp` 必须重新以 `0.0s` 起步，`frame_index` 必须重新以 `0` 起步；
4. **全局索引自增**：Parquet 中的 `index` 字段记录全数据集的累加帧号，必须连续递增无重叠。

---

## 三、一键自动化整理工具：`sanitize_sonic_dataset.py`

为了避免繁琐的人工核算与出错，本仓库提供了开箱即用的通用清洗工具：
[`sanitize_sonic_dataset.py`](file:///home/yichangfeng/lerobot/sanitize_sonic_dataset.py)。

### 工具特性：
* **全自动安全备份**：运行前自动生成带时间戳的完整备份，误操作可随时秒级还原；
* **支持指定删除**：一键剔除失败的 Episode；
* **毫秒级时间裁剪**：裁剪视频与表格（精确到单帧重编码），音画严格对齐；
* **全量自动重编排**：自动消灭断层序号，重新生成 `info.json`、`episodes.jsonl` 与 `episodes_stats.jsonl`；
* **终局严密自检**：处理完毕后自动逐条校验 Parquet 与 MP4 帧数，确保 100% 完美对齐。

---

## 四、高频使用场景与命令速查

在上位机执行清洗命令（推荐在 `miniforge3/envs/lerobot` 或 `.venv_data_collection` 环境下运行）：

### 场景 1：删除指定失败的 Episode
若采集过程中第 5 条和第 14 条操作失误，需将其废弃：
```bash
python ~/lerobot/sanitize_sonic_dataset.py --delete-episodes 5 14
```

### 场景 2：对某个 Episode 进行起手裁剪（剪掉前置等待期）
若第 2 条数据前 13 秒是在原地等待，希望从第 13.0 秒起步保存：
```bash
python ~/lerobot/sanitize_sonic_dataset.py --trim-start 2:13.0
```

### 场景 3：复合操作（一边裁剪、一边删除失败样本）
```bash
python ~/lerobot/sanitize_sonic_dataset.py \
    --trim-start 2:13.0 \
    --delete-episodes 5 14
```

### 场景 4：仅检查与连续重排（消除断层空洞）
如果您手动移动了某些文件或想要重新校准元数据，直接无参运行即可：
```bash
python ~/lerobot/sanitize_sonic_dataset.py
```

### 场景 5：演练模式（Dry-Run，不修改任何文件）
在执行任何改动前，加上 `--dry-run` 预览清洗规划：
```bash
python ~/lerobot/sanitize_sonic_dataset.py \
    --trim-start 2:13.0 \
    --delete-episodes 5 14 \
    --dry-run
```

---

## 五、跨 Agent / 跨会话完整工作流（SOP）

采集与整理支持两种标准工作流：
* **模式 A（连续增量模式）**：所有数据直接追加到同一主目录中，中途暂停清洗；
* **模式 B（分批模块化模式，强烈推荐）**：每次启动录制生成独立的带时间戳文件夹，单独整理质检后，一键合并入主数据集。

```mermaid
flowchart TD
    subgraph 模式B: 分批独立采集与合并（推荐）
        B1["终端4: 启动录制 (自动带时间戳)\ng1_rubberhand_pick_turn_$(date +%Y%m%d_%H%M%S)"] --> B2["检查本批次视频质量\n手动删除失败视频或指定裁剪"]
        B2 --> B3["运行 sanitize_sonic_dataset.py\n--dataset-dir <本批次目录>"]
        B3 --> B4["本批次质检通过 (Diff=0)"]
        B4 --> B5["运行 merge_sonic_datasets.py\n将多个批次一键合并入统一主目录"]
    end
    B5 --> M["最终转码: convert_rubberhand_to_g1_v30.py"]
    M --> T["启动 Pi0.5 策略微调"]
```

---

### 【模式 B】分批带时间戳采集与整合全流程操作（推荐）

#### 步骤 1：在终端 4 录制新批次（另起独立时间戳目录）
在启动采集器时，利用 `$(date +%Y%m%d_%H%M%S)` 自动生成唯一的批次文件夹：
```bash
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
cd ~/SonicStar/wbc

python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the box, turn right, and place it on the table" \
    --dataset-name "g1_rubberhand_pick_turn_$(date +%Y%m%d_%H%M%S)" \
    --camera-host 192.168.123.164 \
    --camera-port 5555 \
    --data-collection-frequency 50
```
> 例如本次录制将保存在：`~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_20260904_113000`

#### 步骤 2：对该批次进行独立整理与质检
录制完成后，按 `Ctrl+C` 退出。进入该批次的 `videos/...` 目录浏览视频：
- 如果某条轨迹失误，直接删除该 MP4 视频即可；
- 然后运行独立清洗工具（自动识别被删视频并重排，无需手动算序号）：
```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/python ~/lerobot/sanitize_sonic_dataset.py \
    --dataset-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_<时间戳>
```
> 该批次将立即重排为干净、连续、Diff = 0 的完美独立数据集。

#### 步骤 3：一键整合所有批次为一个完整主数据集
当采集整理好若干个独立批次后，运行合并工具一键整合：
```bash
/home/yichangfeng/miniforge3/envs/lerobot/bin/python ~/lerobot/merge_sonic_datasets.py \
    --src-dirs ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_* \
    --output-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn
```
* **自动按时间排序**：按批次顺序依次累加全局帧号与连续 Episode 序号；
* **自动隔离安全备份**：如果目标主目录已存在，自动生成带时间戳备份；
* **终局自检**：逐条校验所有合并轨迹的音视频与动作表对齐情况。

#### 步骤 4：转码为 LeRobot v3.0 进行微调
```bash
cd ~/lerobot

/home/yichangfeng/miniforge3/envs/lerobot/bin/python convert_rubberhand_to_g1_v30.py \
    --src-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn \
    --dst-dir ~/lerobot/datasets/g1_box_pick_turn_v30 \
    --fps 30 \
    --task "pick up the box, turn right, and place it on the table"
```

---

## 六、安全回滚机制

所有脚本（包括清洗与合并）在每次修改前，都会在 `outputs/` 下自动生成时间戳备份目录，例如：
`~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_backup_20260904_112058`

若在操作中有任何误操作或需要还原，只需秒级恢复：
```bash
# 还原示例
rm -rf ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn
cp -r ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_backup_<时间戳> ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn
```
安全无忧，放心使用！
