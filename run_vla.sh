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

# 默认策略路径与任务指令 (优先选择最新的 3-subtasks 检查点)
if [ -d "outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model" ]; then
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
elif [ -d "outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model" ]; then
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
elif [ -d "outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model" ]; then
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
else
    POLICY_PATH="outputs/train/pi05_box_pick_99_unified_lora_4090/merged_model"
fi

TASK="pick up the box, turn right, and place it on the table"
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

CLI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --policy.path=*)
            POLICY_PATH="${arg#*=}"
            ;;
        --task=*)
            TASK="${arg#*=}"
            ;;
        --record|--record=true)
            RECORD=true
            ;;
        --no-record|--record=false)
            RECORD=false
            ;;
        --record_diagnostics|--record-diagnostics|--record_diagnostics=true|--record-diagnostics=true)
            RECORD=true
            ;;
        --diagnostics_dir=*|--diagnostics-dir=*)
            DIAGNOSTICS_DIR="${arg#*=}"
            ;;
        --subtasks|--auto)
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
            ROBOT_IP="192.168.123.164"
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
    echo "================================================================================"
    exec "$PYTHON_BIN" check_zmq_connection.py \
        --robot-ip="${ROBOT_IP}" \
        --action-ip="${ACTION_IP:-$ROBOT_IP}" \
        --camera-ip="${CAMERA_IP:-$ROBOT_IP}" \
        --state-port="${STATE_PORT}" \
        --action-port="${ACTION_PORT}" \
        --camera-port="${CAMERA_PORT}"
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

echo "================================================================================"
echo " [终端 2: VLA 大模型推理客户端]"
echo " Python 解释器: ${PYTHON_BIN}"
echo " 模型路径     : ${POLICY_PATH}"
echo " 初始任务指令 : ${TASK}"
if [ "$SUBTASKS" = true ]; then
    echo " 子任务模式   : ★ 3 阶段流转已启用 (1:抱箱抬起 -> 2:右转 -> 3:放箱)"
    echo " 快捷指令     : n (跳下一阶段) | 1/2/3 (直达阶段) | s (开始) | r (复位) | q (退出)"
fi
if [ "$MOCK_STATE" = true ]; then
    echo " 运行模式     : ★ 纯视觉推理测试 (只接收相机，关节使用虚拟零位，不依赖机器人电机)"
else
    echo " 运行模式     : 完整闭环控制 (接收关节与相机，下发 18-DoF 动作)"
    echo " 状态源主机   : ${ROBOT_IP}:${STATE_PORT}"
    echo " 动作目标机   : ${ACTION_IP:-$ROBOT_IP}:${ACTION_PORT}"
fi
echo " 相机源主机   : ${CAMERA_IP:-$ROBOT_IP}:${CAMERA_PORT}"
echo " 连接机制     : 显存常驻 + 后台弹性接入 (连接未就绪不中断退出)"
echo " 交互模式     : ${INTERACTIVE} (提示: 终端输入 /s 开始推理，/r 0秒热重置，/q 退出)"
echo " 超时预警     : ${CONNECT_TIMEOUT} 秒"
echo " RTC 队列     : queue_threshold=${QUEUE_THRESHOLD} | 插值倍率=${INTERPOLATION_MULTIPLIER}"
echo " 决策帧率     : ${FPS} Hz | 可视化: ${DISPLAY_DATA}"
if [ "$RECORD" = true ]; then
    echo " 诊断数据集录制: ★ 已启用 (视频: video.mp4, 动作: actions.npy, 关节: robot_states.npy, Token: tokens.npy)"
    if [ -n "$DIAGNOSTICS_DIR" ]; then
        echo " 录制目标目录 : ${DIAGNOSTICS_DIR}"
    else
        echo " 录制目标目录 : outputs/diagnostics/session_<时间戳>/"
    fi
fi
echo "================================================================================"

EXTRA_RECORD_ARGS=()
if [ "$RECORD" = true ]; then
    EXTRA_RECORD_ARGS+=("--record=true")
    if [ -n "$DIAGNOSTICS_DIR" ]; then
        EXTRA_RECORD_ARGS+=("--diagnostics_dir=${DIAGNOSTICS_DIR}")
    fi
fi

exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.queue_threshold="${QUEUE_THRESHOLD}" \
    --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --robot.type=unitree_g1_client \
    --robot.robot_ip="${ROBOT_IP}" \
    --robot.action_ip="${ACTION_IP:-$ROBOT_IP}" \
    --robot.camera_ip="${CAMERA_IP:-$ROBOT_IP}" \
    --robot.camera_port="${CAMERA_PORT}" \
    --robot.state_port="${STATE_PORT}" \
    --robot.action_port="${ACTION_PORT}" \
    --robot.connect_timeout="${CONNECT_TIMEOUT}" \
    --robot.wait_until_ready="${WAIT_UNTIL_READY}" \
    --robot.mock_state="${MOCK_STATE}" \
    --interactive="${INTERACTIVE}" \
    --subtasks="${SUBTASKS}" \
    --task="${TASK}" \
    --duration="${DURATION}" \
    --fps="${FPS}" \
    --display_data="${DISPLAY_DATA}" \
    "${EXTRA_RECORD_ARGS[@]}" \
    "${CLI_ARGS[@]}"
