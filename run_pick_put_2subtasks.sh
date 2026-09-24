#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 橡胶手抱箱与前伸放置 2-Subtask 实机部署一键启动脚本
# ==============================================================================
# 对应模型: outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000
# 阶段 1: "clamp and lift the box"
# 阶段 2: "reach forward and place the box in the blue area"
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 运行环境与动态库配置
PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
PY_LIB="/home/yichangfeng/miniforge3/envs/lerobot/lib"
if [ -d "$PY_LIB" ]; then
    export LD_LIBRARY_PATH="$PY_LIB:${LD_LIBRARY_PATH:-}"
fi

export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/home/yichangfeng/lerobot/paligemma_tokenizer}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib_cache
export PYTHONUNBUFFERED=1

# 2. 默认部署配置
POLICY_PATH="outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000"
ROBOT_IP="192.168.123.164"
MODE="auto"
QUEUE_THRESHOLD=35
INTERPOLATION_MULTIPLIER=3
HOLD_TIME=1.8
MAX_CLAMP_TIME=5.0
REACH_TIME=8.5
CAMERA_PORT=5556
STATE_PORT=6001
ACTION_PORT=6002
DISPLAY_DATA=false
RECORD=false
DIAGNOSTICS_DIR=""

# 3. 解析外部命令行参数
CLI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --manual)
            MODE="manual"
            ;;
        --auto)
            MODE="auto"
            ;;
        --mode=*)
            MODE="${arg#*=}"
            ;;
        --robot_ip=*)
            ROBOT_IP="${arg#*=}"
            ;;
        --policy.path=*)
            POLICY_PATH="${arg#*=}"
            ;;
        --queue_threshold=*)
            QUEUE_THRESHOLD="${arg#*=}"
            ;;
        --hold_time=*)
            HOLD_TIME="${arg#*=}"
            ;;
        --max_clamp_time=*)
            MAX_CLAMP_TIME="${arg#*=}"
            ;;
        --display_data=*)
            DISPLAY_DATA="${arg#*=}"
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
        *)
            CLI_ARGS+=("$arg")
            ;;
    esac
done

if [ "$RECORD" = true ]; then
    CLI_ARGS+=("--record")
    if [ -n "$DIAGNOSTICS_DIR" ]; then
        CLI_ARGS+=("--diagnostics_dir=${DIAGNOSTICS_DIR}")
    fi
fi

echo "================================================================================"
echo " 🤖 [Unitree G1 橡胶手抱箱前伸 2-Subtask 部署控制器启动]"
echo "--------------------------------------------------------------------------------"
echo " 策略模型     : ${POLICY_PATH}"
echo " 控制模式     : ${MODE} (auto: 自动抱稳计时/姿态流转, manual: 空格单键手动)"
echo " 机器人 IP   : ${ROBOT_IP} (状态端口: ${STATE_PORT}, 动作端口: ${ACTION_PORT})"
echo " 机载相机端口 : ${CAMERA_PORT}"
echo " RTC 打断队列 : queue_threshold=${QUEUE_THRESHOLD} | 插值倍率=${INTERPOLATION_MULTIPLIER}"
echo " 抱稳流转门限 : 维持抱持 >= ${HOLD_TIME}s 或 超时保底 >= ${MAX_CLAMP_TIME}s"
echo " 前伸放置时限 : ${REACH_TIME}s"
if [ "$RECORD" = true ]; then
    if [ -n "$DIAGNOSTICS_DIR" ]; then
        echo " 诊断数据录制 : 🔴 已开启 -> ${DIAGNOSTICS_DIR}"
    else
        echo " 诊断数据录制 : 🔴 已开启 -> 记录每次50步预测Chunk与RTC执行切片到 outputs/diagnostics/"
    fi
else
    echo " 诊断数据录制 : ⚪ 未开启 (添加 --record 即可自动记录大模型50步Chunk与切片数据)"
fi
echo "================================================================================"
echo " 快捷操作提示:"
echo "   - [Space] / [Enter] / 'n' : 无论自动或手动模式，均可一键强制切入 Phase 2 前伸"
echo "   - '1' / '2'               : 手动跳转到对应阶段"
echo "   - 'r'                     : 重置回到初始位姿 (/r)"
echo "   - 'q'                     : 安全停止退出 (/stop)"
echo "   - 诊断数据录制            : 启动时带上 --record 即可完整抓取实机 50 步与流转切片"
echo "================================================================================"

# 4. 执行监管调度器
exec "$PYTHON_BIN" dataset_tools/../run_vla_pick_put_2subtasks.py \
    --mode="${MODE}" \
    --robot_ip="${ROBOT_IP}" \
    --policy.path="${POLICY_PATH}" \
    --queue_threshold="${QUEUE_THRESHOLD}" \
    --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
    --hold_time="${HOLD_TIME}" \
    --max_clamp_time="${MAX_CLAMP_TIME}" \
    --reach_time="${REACH_TIME}" \
    --camera_port="${CAMERA_PORT}" \
    --state_port="${STATE_PORT}" \
    --action_port="${ACTION_PORT}" \
    --display_data="${DISPLAY_DATA}" \
    "${CLI_ARGS[@]}"
