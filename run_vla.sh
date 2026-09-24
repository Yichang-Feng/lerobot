#!/usr/bin/env bash
# ==============================================================================
# 终端 2 启动脚本: VLA 大模型推理客户端
# ==============================================================================
# 功能:
#   1. 载入 VLA 策略模型到 GPU 显存
#   2. 作为轻量网络客户端 (unitree_g1_client) 连接终端 1
#   3. 从 6001 端口拉取 29-DoF 关节状态，从 5556/5555 端口拉取机载视觉图像
#   4. 运行 RTC 异步推理 (~30Hz)，向 6002 端口发送 18-DoF 动作流
#   5. 支持 3-Subtask 自动/手动流转 (抱箱抬起 -> 右转 -> 放箱)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/home/yichangfeng/lerobot/paligemma_tokenizer}"
export HF_HUB_OFFLINE=1

# 默认策略路径与任务指令 (优先选择最新的模型检查点)
if [ -d "outputs/train/pi05_lora_5090_g1_rubberhand_pick_put_v30_slow/merged_model_10000" ]; then
    POLICY_PATH="outputs/train/pi05_lora_5090_g1_rubberhand_pick_put_v30_slow/merged_model_10000"
    TASK="pick up the box then put it in the blue area"
elif [ -d "outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000" ]; then
    POLICY_PATH="outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000"
    TASK="clamp and lift the box"
elif [ -d "outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model" ]; then
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
    TASK="pick up the box, turn right, and place it on the table"
else
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
fi

TASK="pick up the box then put it in the blue area"
SUBTASKS=""

ROBOT_IP="localhost"
ACTION_IP=""
CAMERA_IP=""
CAMERA_PORT=5556
STATE_PORT=6001
ACTION_PORT=6002
DISPLAY_DATA=true
QUEUE_THRESHOLD=35
INTERPOLATION_MULTIPLIER=3
FPS=30
DURATION=1000
CONNECT_TIMEOUT=60
WAIT_UNTIL_READY=true
CHECK_ONLY=false
MOCK_STATE=false
INTERACTIVE=true
RECORD=false
DIAGNOSTICS_DIR=""
AUTO_MODE=true
MODE_PORT=6000
MODE_IP=""
ENGAGEMENT_DURATION=1.2
ARM_SIDE=""
ENABLE_GRIPPER=true
GRIPPER_PORT=6004
GRIPPER_IP=""
ENABLE_WRIST_CAMERAS=false
LEFT_WRIST_PORT=5557
RIGHT_WRIST_PORT=5558
LEFT_WRIST_IP=""
RIGHT_WRIST_IP=""
ARM_ONLY=""
USER_RENAME_MAP=""

CLI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --arm=*|--policy.arm_side=*|--arm_side=*|--arm-side=*)
            ARM_SIDE="${arg#*=}"
            ;;
        --right|--right-arm|--right_arm)
            ARM_SIDE="right"
            ;;
        --left|--left-arm|--left_arm)
            ARM_SIDE="left"
            ;;
        --both|--dual|--dual-arm|--both-arms|--both_arms)
            ARM_SIDE="both"
            ;;
        --policy.path=*)
            POLICY_PATH="${arg#*=}"
            ;;
        --task=*)
            TASK="${arg#*=}"
            ;;
        --record|--record=true|--record_chunks|--record-chunks|--record_diagnostics|--record-diagnostics)
            RECORD=true
            ;;
        --no-record|--record=false)
            RECORD=false
            ;;
        --diagnostics_dir=*|--diagnostics-dir=*)
            DIAGNOSTICS_DIR="${arg#*=}"
            RECORD=true
            ;;
        --auto|--auto-mode|--auto_mode|--auto_mode=true)
            AUTO_MODE=true
            ;;
        --manual|--manual-mode|--manual_mode|--auto_mode=false)
            AUTO_MODE=false
            ;;
        --mode_port=*|--robot.mode_port=*)
            MODE_PORT="${arg#*=}"
            ;;
        --mode_ip=*|--robot.mode_ip=*)
            MODE_IP="${arg#*=}"
            ;;
        --gripper|--enable-gripper|--enable_gripper|--with-gripper)
            ENABLE_GRIPPER=true
            ;;
        --no-gripper|--disable-gripper)
            ENABLE_GRIPPER=false
            ;;
        --gripper_port=*|--gripper-port=*|--robot.gripper_port=*)
            GRIPPER_PORT="${arg#*=}"
            ENABLE_GRIPPER=true
            ;;
        --gripper_ip=*|--gripper-ip=*|--robot.gripper_ip=*)
            GRIPPER_IP="${arg#*=}"
            ENABLE_GRIPPER=true
            ;;
        --engagement_duration=*|--robot.engagement_duration=*)
            ENGAGEMENT_DURATION="${arg#*=}"
            ;;
        --subtasks)
            SUBTASKS=true
            ;;
        --no-subtasks|--no_subtasks)
            SUBTASKS=false
            ;;
        --subtasks=*)
            SUBTASKS="${arg#*=}"
            ;;
        --robot_ip=*|--robot.robot_ip=*)
            ROBOT_IP="${arg#*=}"
            ;;
        --action_ip=*|--robot.action_ip=*)
            ACTION_IP="${arg#*=}"
            ;;
        --camera_ip=*|--robot.camera_ip=*)
            CAMERA_IP="${arg#*=}"
            ;;
        --local_camera|--local-camera)
            CAMERA_IP="127.0.0.1"
            ;;
        --camera_port=*|--robot.camera_port=*)
            CAMERA_PORT="${arg#*=}"
            ;;
        --wrist-cameras|--wrist_cameras|--enable-wrist-cameras|--multi-camera|--3cams)
            ENABLE_WRIST_CAMERAS=true
            ;;
        --no-wrist-cameras|--no_wrist_cameras|--single-camera)
            ENABLE_WRIST_CAMERAS=false
            ;;
        --left_wrist_port=*|--left-wrist-port=*|--robot.left_wrist_port=*)
            LEFT_WRIST_PORT="${arg#*=}"
            ENABLE_WRIST_CAMERAS=true
            ;;
        --right_wrist_port=*|--right-wrist-port=*|--robot.right_wrist_port=*)
            RIGHT_WRIST_PORT="${arg#*=}"
            ENABLE_WRIST_CAMERAS=true
            ;;
        --left_wrist_ip=*|--left-wrist-ip=*|--robot.left_wrist_ip=*)
            LEFT_WRIST_IP="${arg#*=}"
            ENABLE_WRIST_CAMERAS=true
            ;;
        --right_wrist_ip=*|--right-wrist-ip=*|--robot.right_wrist_ip=*)
            RIGHT_WRIST_IP="${arg#*=}"
            ENABLE_WRIST_CAMERAS=true
            ;;
        --state_port=*|--robot.state_port=*)
            STATE_PORT="${arg#*=}"
            ;;
        --action_port=*|--robot.action_port=*)
            ACTION_PORT="${arg#*=}"
            ;;
        --display_data=*)
            DISPLAY_DATA="${arg#*=}"
            ;;
        --queue_threshold=*)
            QUEUE_THRESHOLD="${arg#*=}"
            ;;
        --interpolation_multiplier=*)
            INTERPOLATION_MULTIPLIER="${arg#*=}"
            ;;
        --connect_timeout=*|--connect-timeout=*|--robot.connect_timeout=*)
            CONNECT_TIMEOUT="${arg#*=}"
            ;;
        --no-wait|--strict-timeout)
            WAIT_UNTIL_READY=false
            ;;
        --wait|--wait-until-ready)
            WAIT_UNTIL_READY=true
            ;;
        --check|--test|--test-connection)
            CHECK_ONLY=true
            ;;
        --camera_only|--camera-only|--mock_state|--mock-state)
            MOCK_STATE=true
            ;;
        --real)
            ROBOT_IP="10.3.42.221"
            CAMERA_PORT=5556
            ;;
        --sim)
            ROBOT_IP="localhost"
            CAMERA_PORT=5556
            ;;
        --interactive=*)
            INTERACTIVE="${arg#*=}"
            ;;
        --interactive)
            INTERACTIVE=true
            ;;
        --no-interactive|--no_interactive|--batch)
            INTERACTIVE=false
            ;;
        --arm_only|--arm-only)
            ARM_ONLY=true
            ;;
        --arm_only=*|--arm-only=*|--robot.arm_only=*)
            ARM_ONLY="${arg#*=}"
            ;;
        --rename_map=*|--rename-map=*)
            USER_RENAME_MAP="${arg#*=}"
            ;;
        *)
            CLI_ARGS+=("$arg")
            ;;
    esac
done

# ==============================================================================
# 动态检测 Python 解释器与运行环境
# ==============================================================================
check_python_lerobot() {
    local py_bin="$1"
    if [ -n "$py_bin" ] && [ -x "$py_bin" ] && "$py_bin" -c "import lerobot" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

PYTHON_BIN=""

if [ -n "$PYTHON" ] && [ -x "$PYTHON" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -n "$VIRTUAL_ENV" ] && check_python_lerobot "$VIRTUAL_ENV/bin/python"; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ] && check_python_lerobot "$CONDA_PREFIX/bin/python"; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif check_python_lerobot "$(command -v python 2>/dev/null)"; then
    PYTHON_BIN="$(command -v python)"
elif check_python_lerobot "$(command -v python3 2>/dev/null)"; then
    PYTHON_BIN="$(command -v python3)"
elif [ -x "/home/yichangfeng/miniforge3/envs/lerobot/bin/python" ] && check_python_lerobot "/home/yichangfeng/miniforge3/envs/lerobot/bin/python"; then
    PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[Error] 未找到可用的 Python 解释器！请先激活虚拟环境 (例如: conda activate env_lerobot 或 source env_lerobot/bin/activate)"
    exit 1
fi

PY_ENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"
if [ -d "$PY_ENV_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$PY_ENV_DIR/lib:${LD_LIBRARY_PATH}"
fi

# 自动判断是否开启子任务流转模式
if [ -z "$SUBTASKS" ]; then
    if [[ "$POLICY_PATH" == *"subtask"* ]]; then
        SUBTASKS=true
    else
        SUBTASKS=false
    fi
fi

if [ "$SUBTASKS" = true ] && [ "$TASK" = "pick up the box, turn right, and place it on the table" ]; then
    TASK="clamp and lift the box"
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "================================================================================"
    echo " [快速网络与 ZMQ 连通性测试 (不加载模型)]"
    echo " 解释器路径 : ${PYTHON_BIN}"
    echo " 状态源主机 : ${ROBOT_IP}:${STATE_PORT}"
    echo " 动作目标机 : ${ACTION_IP:-$ROBOT_IP}:${ACTION_PORT}"
    echo " 相机源主机 : ${CAMERA_IP:-$ROBOT_IP}:${CAMERA_PORT}"
    echo " 模式源主机 : ${MODE_IP:-$ROBOT_IP}:${MODE_PORT}"
    if [ "$ENABLE_GRIPPER" = true ]; then
        echo " 夹爪源主机 : ${GRIPPER_IP:-$ROBOT_IP}:${GRIPPER_PORT}"
    fi
    echo "================================================================================"
    CHECK_CMD=(
        "$PYTHON_BIN" check_zmq_connection.py
        --robot-ip="${ROBOT_IP}"
        --action-ip="${ACTION_IP:-$ROBOT_IP}"
        --camera-ip="${CAMERA_IP:-$ROBOT_IP}"
        --mode-ip="${MODE_IP:-$ROBOT_IP}"
        --state-port="${STATE_PORT}"
        --action-port="${ACTION_PORT}"
        --camera-port="${CAMERA_PORT}"
        --mode-port="${MODE_PORT}"
    )
    if [ "$ENABLE_GRIPPER" = true ]; then
        CHECK_CMD+=(--enable-gripper --gripper-port="${GRIPPER_PORT}" --gripper-ip="${GRIPPER_IP:-$ROBOT_IP}")
    fi
    if [ "$ENABLE_WRIST_CAMERAS" = true ]; then
        CHECK_CMD+=(
            --wrist-cameras
            --left-wrist-port="${LEFT_WRIST_PORT}"
            --right-wrist-port="${RIGHT_WRIST_PORT}"
        )
        if [ -n "$LEFT_WRIST_IP" ]; then
            CHECK_CMD+=(--left-wrist-ip="${LEFT_WRIST_IP}")
        fi
        if [ -n "$RIGHT_WRIST_IP" ]; then
            CHECK_CMD+=(--right-wrist-ip="${RIGHT_WRIST_IP}")
        fi
    fi
    exec "${CHECK_CMD[@]}"
fi

# 检查本地策略路径是否存在，防止由于拼写错误误触发 HuggingFace Hub 校验报错
if [[ "$POLICY_PATH" == outputs/* || "$POLICY_PATH" == ./* || "$POLICY_PATH" == /* ]] && [ ! -d "$POLICY_PATH" ]; then
    echo "❌ 错误: 指定的本地模型路径不存在: '${POLICY_PATH}'"
    echo "💡 提示: outputs/train/ 下可用的模型目录包括:"
    for d in outputs/train/*; do
        if [ -d "$d" ]; then
            echo "   - $d"
        fi
    done
    exit 1
fi

# 自动判断是否开启 16-DoF 纯双臂+双夹爪模式 (如 Dex-1，无下肢/无底盘摇杆轴)
if [ -z "$ARM_ONLY" ]; then
    if [ -f "${POLICY_PATH}/config.json" ]; then
        if grep -q '"shape": *\[ *16 *\]' "${POLICY_PATH}/config.json" || grep -q 'kLeftGripper' "${POLICY_PATH}/config.json"; then
            ARM_ONLY=true
        else
            ARM_ONLY=false
        fi
    else
        ARM_ONLY=false
    fi
fi

# 自动判断三相机重命名映射 (如果模型包含 base_0_rgb，自动对齐机载 global_view / left_wrist / right_wrist)
RENAME_MAP_ARGS=()
if [ -n "$USER_RENAME_MAP" ]; then
    RENAME_MAP_ARGS+=("--rename_map=${USER_RENAME_MAP}")
elif [ -f "${POLICY_PATH}/config.json" ] && grep -q 'base_0_rgb' "${POLICY_PATH}/config.json"; then
    RENAME_MAP_ARGS+=('--rename_map={"observation.images.global_view":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}')
fi

echo "================================================================================"
echo " [终端 2: VLA 大模型推理客户端]"
echo " Python 解释器: ${PYTHON_BIN}"
echo " 模型路径     : ${POLICY_PATH}"
echo " 初始任务指令 : ${TASK}"
if [ "$AUTO_MODE" = true ]; then
    echo " 部署控制模式 : ★ 自动模式 (Auto Mode: 监听 ${ROBOT_IP}:${MODE_PORT}，手柄切VLA自动启动+S曲线平滑/切出手柄0.2s自动复位)"
else
    echo " 部署控制模式 : 手动模式 (Manual Mode: 键盘输入 's' 启动，'r' 复位，'q' 退出)"
fi
if [ "$ARM_ONLY" = true ]; then
    echo " 动作控制空间 : ★ 16-DoF 纯双臂+双夹爪模式 (Dex-1 专用: 14 臂关节 + 2 夹爪，无下肢与摇杆轴)"
elif [ "$ARM_SIDE" = "both" ] || [ "$ARM_SIDE" = "dual" ]; then
    echo " 控制手臂     : ★ 双臂协同模式 (Both / Dual: 左右臂均执行逆运动学规划)"
elif [ "$ARM_SIDE" = "left" ]; then
    echo " 控制手臂     : ★ 左臂单臂模式 (Left: 控制左臂，右臂锁定静止)"
elif [ "$ARM_SIDE" = "right" ]; then
    echo " 控制手臂     : ★ 右臂单臂模式 (Right: 控制右臂，左臂锁定静止)"
else
    echo " 控制手臂     : 默认配置 (跟随模型配置: outputs/train/unifolm_vla_base/config.json)"
fi
if [ "$ENABLE_GRIPPER" = true ]; then
    echo " 夹爪控制     : ★ 夹爪已启用 (接收端口: ${GRIPPER_IP:-$ROBOT_IP}:${GRIPPER_PORT} -> 下发至 6002 action.gripper {right, left})"
fi
if [ ${#RENAME_MAP_ARGS[@]} -gt 0 ]; then
    echo " 视角重映射   : ★ 机载视角 1:1 对齐大模型 (${RENAME_MAP_ARGS[*]})"
fi
if [ "$SUBTASKS" = true ]; then
    echo " 子任务模式   : ★ 3 阶段流转已启用 (1:抱箱抬起 -> 2:右转 -> 3:放箱)"
    echo " 快捷指令     : n (跳下一阶段) | 1/2/3 (直达阶段) | s (开始) | r (复位) | q (退出)"
fi
if [ "$MOCK_STATE" = true ]; then
    echo " 运行模式     : ★ 纯视觉推理测试 (只接收相机，关节使用虚拟零位，不依赖机器人电机)"
else
    if [ "$ARM_ONLY" = true ]; then
        echo " 运行模式     : 完整闭环控制 (接收 16-DoF 臂+夹爪状态与 3 路相机，下发 16-DoF 动作流)"
    elif [ "$ENABLE_GRIPPER" = true ]; then
        echo " 运行模式     : 完整闭环控制 (接收 29 关节 + 夹爪 + 相机，下发 20-DoF 动作)"
    else
        echo " 运行模式     : 完整闭环控制 (接收关节与相机，下发 18-DoF 动作)"
    fi
    echo " 状态源主机   : ${ROBOT_IP}:${STATE_PORT}"
    echo " 动作目标机   : ${ACTION_IP:-$ROBOT_IP}:${ACTION_PORT}"
    if [ "$AUTO_MODE" = true ]; then
        echo " 模式源主机   : ${ROBOT_IP}:${MODE_PORT}"
    fi
    if [ "$ENABLE_GRIPPER" = true ]; then
        echo " 夹爪源主机   : ${GRIPPER_IP:-$ROBOT_IP}:${GRIPPER_PORT}"
    fi
fi
if [ "$ENABLE_WRIST_CAMERAS" = true ]; then
    echo " 视觉输入模式 : ★ 三相机多视角模式 (全局主摄: ${CAMERA_IP:-$ROBOT_IP}:${CAMERA_PORT} | 左手腕: ${LEFT_WRIST_IP:-${CAMERA_IP:-$ROBOT_IP}}:${LEFT_WRIST_PORT} | 右手腕: ${RIGHT_WRIST_IP:-${CAMERA_IP:-$ROBOT_IP}}:${RIGHT_WRIST_PORT})"
else
    echo " 相机源主机   : ${CAMERA_IP:-$ROBOT_IP}:${CAMERA_PORT} (单路全局主视角)"
fi
echo " 平滑介入时长 : ${ENGAGEMENT_DURATION} 秒 (余弦 S 曲线过渡)"
echo " 连接机制     : 显存常驻 + 后台弹性接入 (连接未就绪不中断退出)"
echo " 交互模式     : ${INTERACTIVE} (提示: 终端输入 /s 开始推理，/r 0秒热重置，/q 退出)"
echo " 超时预警     : ${CONNECT_TIMEOUT} 秒"
echo " RTC 队列     : queue_threshold=${QUEUE_THRESHOLD} | 插值倍率=${INTERPOLATION_MULTIPLIER}"
echo " 决策帧率     : ${FPS} Hz | 可视化: ${DISPLAY_DATA}"
if [ "$RECORD" = true ] && [ -z "$DIAGNOSTICS_DIR" ]; then
    DIAGNOSTICS_DIR="outputs/diagnostics/session_$(date +%Y%m%d_%H%M%S)"
fi

if [ "$RECORD" = true ]; then
    mkdir -p "$DIAGNOSTICS_DIR"
    echo " 诊断数据集录制: 🔴 已启用 -> ${DIAGNOSTICS_DIR}"
    echo "                 (每次推理 50 步预测 Chunk、RTC 实际执行切片、关节状态与视频全量落盘)"
else
    echo " 诊断数据集录制: ⚪ 未开启 (命令行添加 --record 即可一键开启 50 步 Chunk 与切片记录)"
fi
echo "================================================================================"

EXTRA_RECORD_ARGS=()
if [ "$RECORD" = true ]; then
    EXTRA_RECORD_ARGS+=("--record=true" "--diagnostics_dir=${DIAGNOSTICS_DIR}")
fi

# 智能检测模型策略类型 (仅当模型为早期 UnifoLM-VLA 且显式包含 arm_side 配置时传递 policy 级参数)
# 对于 Pi0.5 / Pi0 等现代多任务策略，其动作空间(16-DoF) 已固化在 config.json 中，不支持也不需要这些参数
IS_UNIFOLM=false
if [ -f "${POLICY_PATH}/config.json" ]; then
    if grep -q '"type": *"unifolm_vla"' "${POLICY_PATH}/config.json" || grep -q '"arm_side"' "${POLICY_PATH}/config.json"; then
        IS_UNIFOLM=true
    fi
fi

EXTRA_POLICY_ARGS=()
if [ "$IS_UNIFOLM" = true ]; then
    if [ -n "$ARM_SIDE" ]; then
        EXTRA_POLICY_ARGS+=("--policy.arm_side=${ARM_SIDE}")
    fi
    if [ "$ENABLE_GRIPPER" = true ]; then
        EXTRA_POLICY_ARGS+=("--policy.use_gripper=true")
    fi
fi

EXTRA_ROBOT_ARGS=()
if [ "$ARM_ONLY" = true ]; then
    EXTRA_ROBOT_ARGS+=("--robot.arm_only=true")
fi

if [ "$ENABLE_GRIPPER" = true ]; then
    EXTRA_ROBOT_ARGS+=(
        "--robot.enable_gripper=true"
        "--robot.gripper_port=${GRIPPER_PORT}"
    )
    if [ -n "$GRIPPER_IP" ]; then
        EXTRA_ROBOT_ARGS+=("--robot.gripper_ip=${GRIPPER_IP}")
    fi
fi

if [ "$ENABLE_WRIST_CAMERAS" = true ]; then
    EXTRA_ROBOT_ARGS+=(
        "--robot.enable_wrist_cameras=true"
        "--robot.left_wrist_port=${LEFT_WRIST_PORT}"
        "--robot.right_wrist_port=${RIGHT_WRIST_PORT}"
    )
    if [ -n "$LEFT_WRIST_IP" ]; then
        EXTRA_ROBOT_ARGS+=("--robot.left_wrist_ip=${LEFT_WRIST_IP}")
    fi
    if [ -n "$RIGHT_WRIST_IP" ]; then
        EXTRA_ROBOT_ARGS+=("--robot.right_wrist_ip=${RIGHT_WRIST_IP}")
    fi
fi

"$PYTHON_BIN" -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.queue_threshold="${QUEUE_THRESHOLD}" \
    --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    "${EXTRA_POLICY_ARGS[@]}" \
    "${RENAME_MAP_ARGS[@]}" \
    --robot.type=unitree_g1_client \
    "${EXTRA_ROBOT_ARGS[@]}" \
    --robot.robot_ip="${ROBOT_IP}" \
    --robot.action_ip="${ACTION_IP:-$ROBOT_IP}" \
    --robot.camera_ip="${CAMERA_IP:-$ROBOT_IP}" \
    --robot.camera_port="${CAMERA_PORT}" \
    --robot.state_port="${STATE_PORT}" \
    --robot.action_port="${ACTION_PORT}" \
    --robot.mode_port="${MODE_PORT}" \
    --robot.mode_ip="${MODE_IP:-$ROBOT_IP}" \
    --robot.engagement_duration="${ENGAGEMENT_DURATION}" \
    --robot.connect_timeout="${CONNECT_TIMEOUT}" \
    --robot.wait_until_ready="${WAIT_UNTIL_READY}" \
    --robot.mock_state="${MOCK_STATE}" \
    --auto_mode="${AUTO_MODE}" \
    --interactive="${INTERACTIVE}" \
    --subtasks="${SUBTASKS}" \
    --task="${TASK}" \
    --duration="${DURATION}" \
    --fps="${FPS}" \
    --display_data="${DISPLAY_DATA}" \
    "${EXTRA_RECORD_ARGS[@]}" \
    "${CLI_ARGS[@]}"

ROLLOUT_EXIT=$?

if [ "$RECORD" = true ] && [ -d "$DIAGNOSTICS_DIR" ]; then
    echo ""
    echo "================================================================================"
    echo "📊 [VLA 部署诊断录制完成] 本次部署数据已成功保存！"
    echo "   数据路径: ${DIAGNOSTICS_DIR}"
    echo "   一键分析: python dataset_tools/analyze_recorded_chunks.py ${DIAGNOSTICS_DIR}"
    echo "================================================================================"
fi

exit $ROLLOUT_EXIT
