# VLA 演示控制台 · 交接文档

给后续接手调整界面的人或 agent 看。读完这份文档，应该能在**不连真机**的情况下改界面、验证效果、导出视频。

---

## 1. 这是什么

一个 PySide6 桌面程序。它给 Unitree G1 Dex-1 的 VLA + JEV 闭环系统做演示界面，有三个用途：

1. **实时**：显示三路相机、机器人状态、JEV 判断和状态机，并从界面启动、控制主程序 `run_dex1_water_bottle_transfer_hold.sh`。
2. **录制**：点「启动」后，自动把所有显示用的原始数据录到 `~/vla_recordings/session_*/`。
3. **回放 / 导出**：用录制驱动同一套界面代码；可以交互回放，也可以离线导出 1440×1080 的 mp4。改完界面后直接重新导出，不用再做一遍真机演示。

成片是 16:9：右侧 1/4 放机器人实拍，左侧 3/4 放这个界面，所以界面默认是 12:9（1440×1080）。

---

## 2. 文件清单与部署

| 文件 | 放在哪 | 作用 | 必需 |
|---|---|---|---|
| `vla_viewer.py` | 上位机工作站（和主程序同一台机器） | 界面、解析、回放控制、导出 | ✅ |
| `vla_io.py` | 和 `vla_viewer.py` 同一目录 | 数据层：ZMQ 接收、Hub、录制、回放 | ✅ |
| `valen_remote_service.py` | 替换 Valen 仓库里的 `scripts/valen_remote_service.py` | 打过补丁的 JEV 服务，多开一个 5560 PUB 广播 | ✅ |
| `valen_viz.patch` | — | 上面那个补丁的 diff，和替换文件二选一，留作记录 | 二选一 |
| `mock_robot.py` | 开发机 | 模拟 Orin：三路相机 + 6001 状态 | 开发用 |
| `mock_main.py` | 开发机 | 模拟主程序日志 + 模拟打过补丁的 JEV 广播 | 开发用 |

依赖：`pip install PySide6 pyzmq`。导出视频需要 `ffmpeg`，没有时退回 `cv2.VideoWriter`。模拟脚本还需要 `numpy pillow`。

**界面必须和主程序跑在同一台机器上**，因为主程序是界面用 `QProcess` 拉起来的，输出和输入都走管道。

---

## 3. 运行

```bash
python vla_viewer.py                                  # 真机；点「▶ 启动」后自动录制
python vla_viewer.py --frameless                      # 无边框、固定 1440×1080、贴左上角，录屏用
python vla_viewer.py --no-record                      # 不自动录制
python vla_viewer.py --replay ~/vla_recordings/session_xxx            # 交互回放
python vla_viewer.py --replay DIR --export out.mp4 --start 10 --end 95  # 离线导出（自动 offscreen）
python vla_viewer.py --size 1200x900                  # 其他尺寸

# 不连真机开发：
python mock_robot.py &              # 终端 1
python vla_viewer.py --mock         # 终端 2：连 127.0.0.1，主程序换成 mock_main.py --viz
```

其他参数：`--robot-ip`、`--cmd`（覆盖主程序命令）、`--jev-viz ""`（不订阅 JEV 广播，强制走日志解析）、`--record-dir`、`--fps`。

快捷键：
- `F9` 手动开始或停止录制
- 回放时：`空格` 播放/暂停，`← / →` 跳 5 秒，`Home` 回到开头

回放控制条在一个**单独的窗口**里，所以不会出现在录屏区域。

---

## 4. 架构

```
live:
  ZMQ SubThread ×5 ─┐  (相机 ×3 / 6001 状态 / 5560 JEV)
  QProcess stdout ──┼──► Hub ──► MainWindow.refresh()（30Hz 定时器）──► 各个控件
  按钮 / 输入框  ───┘     └──► Recorder（后台线程写盘）

replay / export:
  Session(文件) ──► Player ──► Hub ──► MainWindow.refresh()   （与 live 完全同一条路径）
```

**核心原则：录原始数据，显示时再解析。**录下来的是原始日志行、原始 JPEG、原始 JSON，而不是界面上的文字或状态。解析规则和控件都在显示端，所以改了解析或界面之后，旧录制回放出来就是新效果。新增功能时请保持这个原则：不要把界面的派生状态写进录制。

### Hub（`vla_io.py`）

所有数据的唯一入口：

| 方法 | 用途 |
|---|---|
| `frame(i, jpeg, t=None)` | 相机帧：解码成 QImage 存进 `cams[i]` slot；有 recorder 时同时录 JPEG |
| `state_msg(obj, t=None)` | 6001 状态：存进 `state` slot；录制时最多 30Hz |
| `event(kind, data, t=None, img=None)` | 离散事件：进队列 `q`，并录制 |
| `drain()` | 界面每次 refresh 时取走全部事件 |
| `latest(slot)` / `rate(slot)` / `alive(slot, 1.0)` | 最新值、近 1 秒频率、近 1 秒内是否有数据 |
| `reset()` | 回放跳转时清空 |
| `clock` | live 时是 `time.time`；回放时 Player 会替换成媒体时间，所以「无信号」「Hz」在回放和暂停时都正确 |
| `recorder` | 非 None 时，所有进入 Hub 的数据都会被录制 |

### 事件类型（`kind`）

| kind | data | 来源 | 界面行为 |
|---|---|---|---|
| `log` | 一行原始 stdout（含 ANSI 颜色） | `on_output` | 解析，然后显示到程序输出 |
| `jev` | JEV 广播 JSON（见 §5） | 5560 SubThread | `on_jev_svc` → JEV 面板 |
| `input` | 写入 stdin 的一行 | `send_line` | 在输出里显示 `› xxx`；长度 > 1 的当作「任务」；回放时闪一下对应按钮 |
| `typing` | 输入框当前文本 | `textChanged` | 只在回放时使用，用来还原打字过程 |
| `proc` | `{"running": bool, "code": int}` | 启动 / 退出 | 按钮状态；启动后状态机默认「待命」 |

`state` 不走事件队列，而是存在 slot 里，但录制和回放时与事件放在同一个 `events.jsonl`。

---

## 5. 数据源与协议

| 来源 | 地址 | 格式 | 处理 |
|---|---|---|---|
| 左腕 / 头部 / 右腕相机 | `tcp://robot_ip:5557 / 5555 / 5558`，SUB | 原始 JPEG、base64 JPEG、JSON 包裹、multipart 都兼容（`find_image_bytes`） | 只取最新帧 |
| 机器人状态 | `tcp://robot_ip:6001`，SUB | JSON；**真实字段还没确认** | `update_robot` 按 key 猜：模式 `mode/robot_mode/state/control_mode`；夹爪 `right_gripper/r_gripper`；关节 `q/joint_pos[right_shoulder_pitch_idx]` |
| JEV 判定 | `tcp://127.0.0.1:5560`，SUB | multipart `[JSON, JPEG]` | 逐条处理，不丢 |
| 主程序输出 | QProcess stdout+stderr 合并 | 文本行 | `\r` 当作换行；不带换行的提示符 200ms 后补刷 |
| 指令 | QProcess stdin | 一行文本加 `\n` | 发送 = 写入；重置 = `r`；停止 = `q`，3 秒后 terminate，5 秒后 kill |

子进程环境变量会设置 `PYTHONUNBUFFERED=1`、`FORCE_COLOR=1`。如果以后换成 C++ 程序，需要做行缓冲：`stdbuf -oL -eL ./prog`，或者在代码里 `setvbuf(stdout, nullptr, _IOLBF, 0)`。否则界面会一直收不到输出。

### JEV 补丁（`valen_remote_service.py`）

- 新增参数 `--viz-port`，默认 5560，设为 0 关闭。
- 每次 REP 应答**发出之后**，非阻塞地 PUB 一条 `[json, 输入 JPEG]`。所以对主程序零影响，没有订阅者时消息直接丢弃。
- JSON 字段：`status, phase(int), choice, confidence, probabilities{label: p}, cost_ms, t, seq, instructions, criteria[]`。
- 新增的代码都标了 `# [VIZ]`。

如果服务没打补丁、或 5560 没有数据，界面会自动退回到**解析主程序日志**（见 §7）。

---

## 6. 界面结构（`MainWindow._build`）

```
root (QVBoxLayout, 边距 10, 间距 10)
├── cam_row：3 × CameraView         高度在 resizeEvent 中按 (W-40)/3 × 3/4 计算，保证 4:3
└── bottom (QGridLayout 2×2，列宽 1:1，行高 1:1)
    ├── (0,0) 指令卡
    │   ├── 标题行：「指令」 + rec_label(● REC，只在 live 录制时显示) + dot(运行指示)
    │   ├── input: PromptEdit       大字号输入框，回车发送
    │   └── 按钮行（等宽）：btn_proc(▶ 启动 / ■ 停止) · btn_reset(⟲ 重置) · btn_send(发送)
    ├── (1,0) 程序输出卡：output (QPlainTextEdit，只读，最多 3000 行)
    └── (0,1)-(1,1) 右列 (QVBoxLayout)
        ├── 状态机卡：fsm_chip(右上角当前阶段) · flow: PhaseFlow · 「任务」 prompt
        ├── JEV 卡 (stretch=1)：jev_meta(右上角：连续 x/n · ms) · jev_q(问题) · options: OptionBars · banner: Banner
        └── 机器人卡：link(相机 n/3 · 状态 Hz) · mode: Chip · grip_bar/grip_v · pitch_bar/pitch_v
```

### 自绘控件

| 类 | 说明 |
|---|---|
| `CameraView` | 保持比例绘制；左上角画名字标签；1 秒内没有新帧就盖上「无信号」；只在 seq 或 alive 变化时重绘 |
| `PromptEdit(QLineEdit)` | 字号自适应：取 `PROMPT_SAMPLES` 里最长的那条，算出单行撑满宽度的字号，上限是高度 × 0.6；输入更长文本时自动缩小，最小 18px。演示用的提示词改这个列表就行 |
| `PhaseFlow` | 状态机节点和连线：当前节点蓝色并带光圈，已过的节点绿色，未到的灰色；节点来自 `PHASES` |
| `OptionBars` | JEV 每个候选一行：名称、概率条、百分比；被选中的一行加粗并用横幅颜色；目标选项上画阈值刻度；行高 `RH=48` |
| `Banner` | 结论横幅：颜色 + 左侧色条 + 25px 粗体 |
| `Bar` / `Chip` / `card()` / `kv()` | 通用小部件 |

### 样式

- 颜色常量：`BG, CARD, LINE, TEXT, MUTED, DIM, ACCENT(蓝), OK(绿), WARN(黄), BAD(红)`。
- 字体：`UI_FONTS`、`MONO_FONTS` 是回退列表。用 `font(px, bold, mono)` 生成。
- 全局样式表：文件末尾的 `STYLE`，用 objectName（`#card #input #prompt #question #taskKey #console`）和动态属性 `[big="true"]` 选择控件。

### 刷新流程（`refresh`，live / 回放 / 导出共用）

1. `hub.drain()`，逐个 `on_event`。
2. 程序输出：把这次攒的 HTML 一次性追加，最多 300 行，用于回放跳转时的追赶；只在原本就在底部时自动滚动。
3. 如果 `_jev_dirty`，调用 `render_jev()`，每次刷新最多一次。
4. 相机 `feed`。
5. `update_robot()`。

---

## 7. 解析规则（`vla_viewer.py` 顶部的 PARSE 区）

主程序日志格式变了，**只改这里**。

| 名称 | 作用 |
|---|---|
| `PHASES` | 状态机节点 `(key, 中文名)`。key 是 `"idle"` 或 `"0"`–`"4"` |
| `FSM_RULES` | 按顺序匹配，第一条命中就 `set_fsm(key)`。现在的规则：`Clean Reset/Zero-Reset/彻底重置 → idle`、`桌1…感知中/静止待命/待命 → idle`、`桌2…感知中 → 2`、`Phase N → N` |
| `PASS_RE` / `WAIT_RE` | 主程序明确的放行或等待（「判定通过」/「暂未检测到…感知中」），会覆盖 JEV 横幅的结论 |
| `HIDE_RE` | 纯分隔线（`=====`）不显示，但仍然录制 |
| `P_RE` / `MS_RE` | 从日志行提取置信度（支持 `P=97.8%` 和 `P=0.978`）和耗时 |
| `GRIP_RE` / `PITCH_RE` | 6001 没提供夹爪或肩部字段时，用日志里的数值兜底 |
| `JEV` | 每个 JEV phase 的展示配置：`q` 问题、`target` 放行候选、`thr` 阈值、`need` 需要连续几帧、`labels` 候选→中文名、`yes/no` 横幅文案、`bad` 异常候选→文案 |

`JEV` 的 phase 编号跟 `valen_remote_service.py` 一致：4 = 桌1有无水瓶，3 = 离地，2 / 5 = 桌2收纳盒，0 = 入盒完成，1 = 放回桌面。阈值和 need 来自系统手册 §3.2 / §3.5。

### 已经对上的真实日志格式

用户截图里桌1的格式如下：

```
⏳ [桌1 视觉感知中] 手柄处于 VLA 模式，桌面上暂未检测到水瓶。
  ├─ 识别选项: no_water_bottle (置信度 P=97.8%, 耗时: 389ms)
  └─ 机械臂保持静止待命，持续检测中... 请将水瓶摆放到桌面视野内。
🤖 [桌1 JEV 判定通过] 视觉确认桌面上存在水瓶!
  ├─ 识别选项: water_bottle_present (置信度 P=67.6%, 耗时: 424ms)
  └─ 自动切入 VLA 控制模式 (启动 Phase 0 水瓶夹取, 和按 's' 效果一致)...
```

离地判定、桌2、入盒这几个阶段的真实日志**还没见过**。`mock_main.py` 是按同样的风格猜的。拿到真机录制后，要用 `events.jsonl` 里的 `log` 行核对一遍。

### JEV 面板的判定逻辑

- **数据优先级**：5 秒内收到过 5560 的广播，就只用广播数据，日志里的「识别选项」行会被忽略（但仍然参与 PASS / WAIT 提示）。否则从日志行解析，二分类的另一项按 `1 - p` 补齐。
- `show_jev`：`hit = (choice == target) 且 P(target) ≥ thr`。`streak` 按 phase 累计，phase 一变就清零。如果此刻之前 1.0 秒内出现过 PASS / WAIT 提示，就作为 `verdict` 覆盖。
- `set_hint`：每一行日志都检查 PASS_RE / WAIT_RE。如果和当前 JEV 判定相差不到 1.5 秒，直接改写它的 verdict。这是因为广播先到、日志后到。
- `render_jev` 的横幅：
  - 选中 `bad` 候选 → 红色
  - verdict 为真，或者（hit 且 streak ≥ need）→ 绿色 `yes`
  - hit 但 streak 还不够 → 蓝色「确认中 x/n」
  - 其他 → 黄色 `no`

---

## 8. 录制格式（`vla_io.Recorder`）

```
~/vla_recordings/session_YYYYmmdd_HHMMSS/
├── meta.json        cameras, jev_views, robot_ip, main_cmd, window, version,
│                    start_wall(epoch), start_local, end_wall, duration, dropped_items
├── events.jsonl     每行 {"t": 相对秒, "k": kind, "d": data[, "img": jev 图序号]}
│                    k ∈ log | state | jev | input | typing | proc
├── cam0.mjpg/.idx   cam 序号 = CONFIG["cameras"] 的显示顺序（0 左腕、1 头部、2 右腕）
├── cam1.mjpg/.idx
├── cam2.mjpg/.idx
└── jev.mjpg/.idx    JEV 服务实际收到的输入图（目前界面不显示，但仍在录制）
```

- `.mjpg` 是原始 JPEG 依次拼接，可以直接 `ffplay -f mjpeg cam1.mjpg`。
- `.idx` 每条记录是 `struct "<dQI"`，即 `(t: f64, offset: u64, len: u32)`。
- 写盘在后台线程完成：队列 900 条，满了就丢弃并计入 `dropped_items`；每 0.5 秒 flush 一次，所以异常退出最多丢 0.5 秒。读取时能容忍被截断的 idx。
- 体积：真实相机大约 200 MB/分钟。`CONFIG["record_fps"]` 可以给相机录制降帧。
- 生命周期：主程序启动 → 开始录制；主程序退出 → 后台关闭录制；关窗时会等待写完。F9 可手动开关。
- `start_wall` 是绝对时间，用来和实拍视频对时。

---

## 9. 回放与导出（`vla_io.Player`、`vla_viewer.export_video`）

- 构造 Player 时会把 `hub.clock` 替换为 `lambda: player.t`（媒体时间）。
- `tick()`：按墙钟 × `speed` 推进；`set_time(t)`：导出时逐帧推进，只能往前。
- `_advance(t)`：把 `t` 之前的事件推进 Hub；每路相机按 idx 二分查找当前帧，帧号变化才解码。
- `seek(t)`：先 `MainWindow.reset_view()`，再 `hub.reset()`，然后从 0 追赶到 `t`。追赶时**只读取最后一张 JEV 图**，界面也只追加最后 300 行日志。所以跳转一般在 50ms 以内。
- `ReplayBar`：一个独立窗口，包含播放、滑条（拖完松手才跳转）、时间和倍速。
- `export_video`：窗口固定 W×H，然后逐帧执行 `set_time → refresh → processEvents → grab()`，得到 RGB888 原始帧，通过管道交给 `ffmpeg libx264 crf18 yuv420p`。速度大约和实时相当。
- 导出结束和正常退出都用 `os._exit`，用来绕开 PySide 退出析构时偶发的段错误。退出前录制已经同步写完了。

---

## 10. 常见修改

| 想改什么 | 改哪里 | 怎么验证 |
|---|---|---|
| 布局比例、卡片顺序 | `_build`（bottom 网格的 stretch、右列的 addWidget 顺序）、`resizeEvent` | 回放截图 |
| 字号、颜色 | `STYLE`、颜色常量、各自绘控件 `paintEvent` 里的 `font(px)` | 同上 |
| 输入框字号基准 | `PROMPT_SAMPLES`；高度上限是 `PromptEdit._px` 里的 `h * 0.6` | 回放到打字时刻 |
| 增删按钮 | `_build` 的按钮行；需要在回放时闪烁的，加进 `key_btns`；`set_running` 控制启用状态 | live + 回放 |
| 主程序日志格式变了 | PARSE 区：`FSM_RULES / PASS_RE / WAIT_RE / P_RE` | **拿真实录制回放**，不用真机 |
| 新增或修改 JEV phase | `JEV` 字典，同时核对 `valen_remote_service.get_default_questions_for_phase` | `mock_main.py` |
| 6001 真实字段 | `update_robot` | 真机或录制 |
| 新数据源 | `SubThread(handler)` → 在 handler 里 `hub.event("新kind", data)` → 在 `on_event` 加分支。录制和回放自动支持 | mock |
| 恢复 JEV 输入图 | 图仍在录制，并作为 `on_jev_svc` 的 `img_bytes` 传进来。在 `show_jev` 里存下，在 `render_jev` 里用 `QImage.fromData` 显示即可（早期版本有个 `JevImage` 控件） | 回放 |
| 换窗口尺寸 | `--size`；相机行和输入框字号会自适应，自绘控件内部的像素值可能需要调整 | 导出检查 |

---

## 11. 测试方法（不需要真机）

```bash
# 1) 起模拟机器人
python mock_robot.py &

# 2) 离屏跑界面并截图（脚本里用 QTimer.singleShot 驱动操作）
QT_QPA_PLATFORM=offscreen python your_test.py
#   核心写法：
#   hub = Hub(3); w = MainWindow(cfg, hub); w.setFixedSize(1440,1080); w.show(); w.start()
#   QTimer.singleShot(300, w.toggle_program)
#   QTimer.singleShot(9000, lambda: w.grab().save("shot.png"))
#   cfg = dict(CONFIG, robot_ip="127.0.0.1", main_cmd="python3 -u mock_main.py --viz")

# 3) 回放跳转 + 截图
#   s = Session(dir); hub = Hub(3); pl = Player(s, hub); w = MainWindow(CONFIG, hub, pl)
#   bar = ReplayBar(pl, w); bar.seek(12.0); app.processEvents(); w.grab().save(...)

# 4) 导出并检查
python vla_viewer.py --replay DIR --export t.mp4 --start 0 --end 10
ffprobe -v error -show_entries stream=width,height,nb_frames -of csv=p=0 t.mp4
```

`mock_main.py` 大约 25 秒走完一轮：无水瓶等待 6 秒 → 判定通过 → P0 → 离地连续 2/2 → P1 → P2 → 桌2连续 2/2 → P4 → 入盒 → 待命，然后循环。不加 `--viz` 时只输出日志，可以用来测日志解析这条路径。

改完界面后，**最可靠的回归测试**是：拿一个真实录制，回放到几个关键时刻截图，再导出一小段视频看一遍。

---

## 12. 踩过的坑

- 全局样式表 `QWidget { font-size:14px }` 会**覆盖 `setFont`**。要单独改某个控件的字号，得给它设置自己的样式表（`PromptEdit` 就是这么做的），或者在 `paintEvent` 里自己画。
- Qt 样式表里的 `#RRGGBBAA` 会被当成 `#AARRGGBB`。半透明颜色一律用 `rgba()` 辅助函数。
- 程序输出如果设成 `NoWrap`，再 `moveCursor(End)`，会横向滚到最右边。现在用的是自动换行。
- 离屏渲染用 `QT_QPA_PLATFORM=offscreen` 加 `widget.grab()` 就行，截图和导出都依赖它。
- PySide6 退出时析构顺序会导致偶发段错误，见 §9 的 `os._exit`。
- 中文字体回退：没装 Noto CJK 的机器会显示成别的字体，但不会乱码。emoji（⏳🤖）依赖系统的 emoji 字体。

---

## 13. 待确认和已知限制

1. **6001 的消息格式**还没确认。模式、夹爪、肩部 Pitch 目前靠猜 key，加上日志兜底。
2. 离地、桌2、入盒三个阶段的**真实日志措辞**没见过，`FSM_RULES` 和 `PASS_RE/WAIT_RE` 可能需要补。
3. 主程序必须**按行读取 stdin**。如果它用的是 termios 单键读取，「发送」和「重置」都不会生效。
4. 「任务」只取发送出去的、长度大于 1 的文本，单字母视为控制键。主程序如果不处理自由文本，这行文字也只用于展示。
5. 界面上**没有急停按钮**，这是有意的：`q` 只是退出程序，不等于电机阻尼。硬件急停用手柄双摇杆（Damp）。
6. `CONFIG["jev_views"]` 目前只写进 meta，界面没有用到。
