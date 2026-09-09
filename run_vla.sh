#!/usr/bin/env bash
# ==============================================================================
# 终端 2 启动脚本: VLA 大模型推理客户端
# ==============================================================================
# 功能:
#   1. 载入 VLA 策略模型到 GPU 显存
#   2. 作为轻量网络客户端 (unitree_g1_client) 连接终端 1
#   3. 从 6001 端口拉取 29-DoF 关节状态，从 5556/5555 端口拉取机载视觉图像
#   4. 运行 RTC 异步推理 (~30Hz)，向 6002 端口发送 18-DoF 动作流
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认策略路径与任务指令
POLICY_PATH="outputs/train/pi05_box_pick_turn_final/checkpoints/002000/pretrained_model"
TASK="pick up the box, turn right, and place it on the table"

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

CLI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --policy.path=*)
            POLICY_PATH="${arg#*=}"
            ;;
        --task=*)
            TASK="${arg#*=}"
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

export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
LEROBOT_ROLLOUT="/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"

if [ "$CHECK_ONLY" = true ]; then
    echo "================================================================================"
    echo " [快速网络与 ZMQ 连通性测试 (不加载模型)]"
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
echo " 模型路径 : ${POLICY_PATH}"
echo " 任务指令 : ${TASK}"
if [ "$MOCK_STATE" = true ]; then
    echo " 运行模式 : ★ 纯视觉推理测试 (只接收相机，关节使用虚拟零位，不依赖机器人电机)"
else
    echo " 运行模式 : 完整闭环控制 (接收关节与相机，下发 18-DoF 动作)"
    echo " 状态源主机 : ${ROBOT_IP}:${STATE_PORT}"
    echo " 动作目标机 : ${ACTION_IP:-$ROBOT_IP}:${ACTION_PORT}"
fi
echo " 相机源主机 : ${CAMERA_IP:-$ROBOT_IP}:${CAMERA_PORT}"
echo " 连接机制   : 显存常驻 + 后台弹性接入 (连接未就绪不中断退出)"
echo " 交互模式   : ${INTERACTIVE} (提示: 终端输入 /s 开始推理，/r 0秒热重置，/q 退出)"
echo " 超时预警   : ${CONNECT_TIMEOUT} 秒"
echo " RTC 队列   : queue_threshold=${QUEUE_THRESHOLD} | 插值倍率=${INTERPOLATION_MULTIPLIER}"
echo " 决策帧率   : ${FPS} Hz | 可视化: ${DISPLAY_DATA}"
echo "================================================================================"

exec "$LEROBOT_ROLLOUT" \
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
    --task="${TASK}" \
    --duration="${DURATION}" \
    --fps="${FPS}" \
    --display_data="${DISPLAY_DATA}" \
    "${CLI_ARGS[@]}"
