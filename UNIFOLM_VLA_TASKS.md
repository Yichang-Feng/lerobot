# UnifoLM-VLA 任务提示词与单双手切换指南

本文档汇总了 **UnifoLM-VLA-Base** 模型原生支持的所有预训练任务、对应推荐的自然语言提示词（Prompt）、推荐控制臂以及在实机部署中如何便捷切换单臂/双臂模式。

---

## 一、核心任务与推荐 Prompt 映射表

UnifoLM-VLA 内部通过自然语言指令与视觉信息联合驱动动作预测。运行脚本会根据 `--task="..."` 中的关键词**自动匹配最适宜的任务统计量（Norm Stats）**，无需手动硬编码任务 ID。

### 1. 用户核心 8 项任务

| 任务标识（`task_name`） | 任务说明 | 推荐控制臂 | 推荐 Prompt（英文） | 常用替代 Prompt / 中文含义 |
| :--- | :--- | :--- | :--- | :--- |
| **`g1_stack_block`** | 叠积木 | 右臂 / 双臂 | `"stack the red block on the blue block"` | `"stack the blocks"`<br>（叠积木 / 把红积木叠在蓝积木上） |
| **`g1_clean_table`** | 清理桌面 | 右臂 | `"clean the table"` | `"tidy up the tabletop"`<br>（清理桌面杂物） |
| **`g1_wipe_table`** | 抹布擦桌子 | 右臂 | `"wipe the table with a cloth"` | `"wipe the table"`<br>（手持抹布擦拭桌面） |
| **`g1_pack_pencilbox`** | 装笔盒 | 右臂 | `"put the pencil into the pencil box"` | `"pack the pencil box"`<br>（拾取文具装入笔盒） |
| **`g1_erase_board`** | 擦黑板/白板 | 右臂 | `"erase the board"` | `"clean the board with the eraser"`<br>（抬手使用板擦擦黑板） |
| **`g1_pour_medicine`** | 倒药 | 右臂 / 双臂 | `"pour the medicine into the cup"` | `"pour medicine"`<br>（倾斜药瓶倒药） |
| **`g1_pack_pingpong`** | 收乒乓球 | 右臂 | `"pick up the ping-pong ball and put it in the box"` | `"pack pingpong balls"`<br>（拾取乒乓球放入收纳盒） |
| **`g1_fold_towel`** | 叠毛巾 | **双臂（Both）** | `"fold the towel"` | `"fold the towel on the table"`<br>（双手抓取毛巾对折平铺） |

---

### 2. 预训练内置拓展 4 项任务

模型权重内置统计量还完整支持以下 4 个官方任务：

| 任务标识（`task_name`） | 任务说明 | 推荐控制臂 | 推荐 Prompt（英文） | 中文含义 |
| :--- | :--- | :--- | :--- | :--- |
| **`g1_dual_clean_table`** | 双臂协同清桌 | **双臂（Both）** | `"collaborate to clean the table"` | 双臂协作扫除桌面 |
| **`g1_bag_insert`** | 插袋/装袋子 | 双臂 / 右臂 | `"insert the item into the bag"` | 把物体塞入收纳袋 |
| **`g1_organize_tools`** | 整理工具 | 右臂 | `"organize the tools on the table"` | 摆正整理桌上工具 |
| **`g1_prepare_fruit`** | 水果摆盘 | 右臂 | `"pick up the fruit and place it on the plate"` | 抓取水果放入盘中 |

---

## 二、如何便捷进行单双手任务切换

在真实机器人控制中，单臂任务（如擦黑板、装笔盒）与双臂任务（如叠毛巾、双臂清桌）对非受控臂的处理截然不同：
* **单臂模式（Left / Right）**：受控臂执行 VLA 模型解算出的目标轨迹；**非受控臂自动锁定在初始静止位姿**，彻底防止因重力下沉导致手臂慢慢坠落。
* **双臂模式（Both / Dual）**：两只手臂同时由 Pinocchio + CasADi 全身逆运动学闭环求解，两臂均跟随 VLA 输出的笛卡尔末端 6D 位姿运动。

### 方式 1：终端启动时直接添加快捷标志（最推荐，零配置）

在执行 `run_vla.sh` 时，直接通过命令行快捷参数指定手臂模式：

#### ① 右臂单臂任务（默认模式）
```bash
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="clean the table"
# 或：
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --arm=right --task="wipe the table with a cloth"
```

#### ② 双臂协同任务（如叠毛巾、双臂整理）
```bash
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --both --task="fold the towel"
# 也可以使用 --dual：
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --dual --task="collaborate to clean the table"
```

#### ③ 左臂单臂任务
```bash
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --left --task="clean the table"
# 或：
./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --arm=left --task="put the pencil into the pencil box"
```

---

### 方式 2：修改模型默认配置文件 `config.json`

若希望固定某个任务或手臂作为每次启动的默认配置，可直接修改：
[`lerobot/outputs/train/unifolm_vla_base/config.json`](file:///home/yichangfeng/lerobot/outputs/train/unifolm_vla_base/config.json)

```json
{
  "type": "unifolm_vla",
  "arm_side": "both",               // 可选: "right" (右臂), "left" (左臂), "both" (双臂)
  "task_name": "g1_fold_towel",     // 默认任务名
  "default_task": "fold the towel", // 默认 Prompt
  "chunk_size": 16,
  "n_action_steps": 16,
  "device": "cuda",
  "dtype": "bfloat16"
}
```

---

## 三、常用任务一键运行命令速查

| 需求场景 | 一键启动命令 |
| :--- | :--- |
| **叠毛巾（双臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --both --task="fold the towel"` |
| **清理桌面（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="clean the table"` |
| **抹布擦桌（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="wipe the table with a cloth"` |
| **装笔盒（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="put the pencil into the pencil box"` |
| **擦黑板（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="erase the board"` |
| **倒药（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="pour the medicine into the cup"` |
| **收乒乓球（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="pick up the ping-pong ball and put it in the box"` |
| **叠积木（右臂）** | `./run_vla.sh --policy.path=outputs/train/unifolm_vla_base --real --right --task="stack the red block on the blue block"` |

> [!TIP]
> 每次执行前若添加 `--record` 参数（例如 `./run_vla.sh --real --both --task="fold the towel" --record`），系统会在运行结束后自动输出 30Hz 高清视频、全量 18 维执行动作切片与 29-DoF 关节实测角度至 `outputs/diagnostics/`，便于复盘回放。
