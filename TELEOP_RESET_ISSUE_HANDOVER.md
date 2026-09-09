# Unitree G1 遥操作手臂归位与跟随状态交接文档

- **记录时间**: 2026-09-08
- **目标机器人**: Unitree G1 (29-DoF 人形机器人)
- **代码仓库**: `GR00T-WholeBodyControl`
- **文档用途**: 记录当前仿真环境中 VR 遥操作手臂状态切换与归位异常问题，用于技术交接与排查。

---

## 1. 当前系统配置与运行环境

### 1.1 软件与硬件拓扑
- **底层仿真与全身控制**: `GR00T-WholeBodyControl` (Decoupled WBC 架构)
  - 下肢/全身平衡控制器: `G1GearWbcPolicy` (基于 ONNX 模型 `GR00T-WholeBodyControl-Balance.onnx`)
  - 上身控制器: `InterpolationPolicy` (接收上身 14/17 维关节位姿指令并进行轨迹插值)
- **遥操作交互设备**:
  - PICO VR 头显与手柄（仅使用手柄控制器，无额外的手肘/躯干 Tracker）
- **通信方式**:
  - ROS2 主题发布与订阅（`ControlPolicy/upper_body_pose`）
  - CycloneDDS / ROS2 进程间通信

### 1.2 启动方式与运行指令
当前测试在 Docker 容器或本地仿真环境中运行：

```bash
# 终端 1: 启动仿真与全身控制回路 (Sim WBC Loop)
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface sim \
    --no-with_hands

# 终端 2: 启动 VR 遥操作策略节点 (Teleop Policy Loop)
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --hand_control_device=pico \
    --body_control_device=pico
```

---

## 2. 遥操作交互逻辑与按键映射

### 2.1 PICO 手柄控制映射
- **触发按键**: 左手柄菜单键 + 右手柄扳机
  - 条件表达式: `pico_data["left_menu_button"] and (pico_data["right_trigger"] > 0.5)`
  - 对应信号: `toggle_activation`
- **功能定义**:
  - 开启跟随: 标定手柄初始位姿，机器人手臂进入跟随模式。
  - 停止跟随: 退出跟随模式，手臂应当收回至机器人默认姿态（`default_upper_body_pose`）。

### 2.2 辅助键盘按键（若启用键盘监听）
- 键盘键 `l`: 翻转切换遥操作激活状态 (`is_active = not is_active`)。
- 键盘键 `k`: 触发遥操作策略复位 (`teleop_policy.reset()`)。

---

## 3. 当前问题现象与复现步骤

### 3.1 现象复现流程
在仿真环境启动后，按顺序按下「左菜单 + 右扳机」，系统表现如下：

| 触发次序 | 操作 | 实际表现 | 是否符合预期 |
| :--- | :--- | :--- | :--- |
| **第 1 次按下** | 左菜单 + 右扳机 | 机器人手臂开始跟随手柄运动 | 符合预期 |
| **第 2 次按下** | 左菜单 + 右扳机 | 机器人手臂停止跟随，但**停留在当前姿势，没有归位** | **异常**（未回到默认姿势） |
| **第 3 次按下** | 左菜单 + 右扳机 | 机器人手臂**先归位**回到默认姿势，然后**开始跟随**手柄 | **异常**（归位发生在了开启阶段） |
| **第 4 次按下** | 左菜单 + 右扳机 | 机器人手臂停止跟随 | 符合停止逻辑，但仍未归位 |
| **第 5 次及之后** | 左菜单 + 右扳机 | 机器人手臂重新开始跟随，但**不再进行归位** | **异常**（归位机制彻底失效） |

### 3.2 预期行为对比
- **预期行为**: 
  - 任何时候按下快捷键退出遥操作（停止跟随），手臂均应平滑、稳定地回到初始默认姿态；
  - 再次按下快捷键进入遥操作（开始跟随），手臂应以当前姿势或默认姿势重新校准后进行动作跟随；
  - 状态切换应具备确定性与可重复性，不随触发次数发生行为偏离。

---

## 4. 涉及的核心代码模块与路径

以下为控制回路与遥操作链路中直接相关的核心文件与方法列表：

### 4.1 手柄信号捕获与处理
- **文件路径**: `decoupled_wbc/control/teleop/streamers/pico_streamer.py`
  - `_generate_unified_raw_data()`: 负责读取 `left_menu_button` 与 `right_trigger` 边沿信号，生成 `toggle_activation`。
  - `reset()` / `reset_status()`: 手柄状态复位逻辑。

### 4.2 遥操作策略管理
- **文件路径**: `decoupled_wbc/control/policy/teleop_policy.py`
  - `check_activation()`: 处理 `toggle_activation` 信号并切换 `self.is_active` 状态。
  - `get_action()`: 根据 `is_active` 与 `ik_data` 计算并输出 `target_upper_body_pose`。
  - `trigger_reset_to_default()`: 手臂平滑过渡回位与重置接口。
  - `reset()`: 策略全局重置逻辑。

### 4.3 逆运动学求解器
- **文件路径**: `decoupled_wbc/control/teleop/teleop_retargeting_ik.py`
  - `compute_joint_positions()`: 求解双臂目标关节角。
  - `reset()`: 重置前向运动学与 QP/Pink 求解器内部状态。
  - `get_action()`: 返回当前解算得到的上身关节角度向量。

### 4.4 消息发布与主循环
- **文件路径**: `decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py`
  - 主循环中的 `get_action()` 调用、`target_time` 设置与 ROS2 话题发布逻辑。

### 4.5 WBC 仿真与指令接收端
- **文件路径**: `decoupled_wbc/control/main/teleop/run_g1_control_loop.py`
  - `upper_body_policy_subscriber`: 接收目标位姿并传入 `wbc_policy.set_goal()`。
- **文件路径**: `decoupled_wbc/control/policy/interpolation_policy.py`
  - `schedule_waypoint()`: 对接收到的 `target_upper_body_pose` 和 `target_time` 进行插值执行。

---

## 5. 交接需求与待解决事项

1. **状态机逻辑梳理**:
   - 梳理手柄边沿触发、`is_active` 状态切换、以及回位插值流程之间的时序逻辑，确保每次停止触发时均能稳定激活回位流程。
2. **重入与循环稳定性**:
   - 保证在反复进行「跟随 -> 停止 -> 跟随 -> 停止」操作时，状态机变量（如时间戳、初始插值起点、求解器状态）能够被干净重置，避免在第 3 次及后续调用时状态失锁或失效。
