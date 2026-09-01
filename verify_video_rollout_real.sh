#!/usr/bin/env bash
# ==============================================================================
# Script to verify PI0.5 policy on Physical G1 Robot with training video replay
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
LEROBOT_ROLLOUT="/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"

VIDEO_PATH="${1:-datasets/box_pick/videos/observation.images.global_view/chunk-000/file-003.mp4}"
START_TIME="${2:-0:00}"
CONTROLLER="${3:-GrootLocomotionController}"
LOCOMOTION_MODE="${4:-stand}"
ROBOT_IP="${5:-192.168.123.164}"
PORT=5556
FPS=30

echo "================================================================================"
echo " [Unitree G1 实机动作 + 视频回放 Rollout 验证]"
echo " 机器人 IP   : ${ROBOT_IP} (实机模式 is_simulation=false)"
echo " 视频路径    : ${VIDEO_PATH}"
echo " 起始时间    : ${START_TIME}"
echo " 控制器      : ${CONTROLLER}"
echo " 初始模式    : ${LOCOMOTION_MODE} (按 's' 原地保持 / 'w' 允许行走)"
echo " 视频推流端口: ${PORT} | 帧率: ${FPS}"
echo "================================================================================"

# 1. 检查物理网络连接
echo -n "正在检测与 G1 (${ROBOT_IP}) 的网络连接..."
if ping -c 1 -W 2 "${ROBOT_IP}" > /dev/null 2>&1; then
    echo " [OK]"
else
    echo " [FAIL]"
    echo "错误: 无法 ping 通 ${ROBOT_IP}，请确认网卡 enx6c1ff724495a 已配置为 192.168.123.213 并已连接网线。"
    exit 1
fi

# 2. 启动本地训练视频 ZMQ 推流
echo "================================================================================"
echo " [1/2] 启动训练视频推流服务 (Port: ${PORT})"
echo "================================================================================"

$PYTHON_BIN src/lerobot/cameras/zmq/stream_dataset_video_zmq.py \
    --video_path "$VIDEO_PATH" \
    --start_time "$START_TIME" \
    --port "$PORT" \
    --camera_names head_camera global_view \
    --fps "$FPS" \
    --loop &
STREAMER_PID=$!

cleanup() {
    echo ""
    echo "正在终止视频推流服务 (PID: $STREAMER_PID)..."
    kill -TERM "$STREAMER_PID" 2>/dev/null || true
    wait "$STREAMER_PID" 2>/dev/null || true
    echo "已安全清理退出。"
}
trap cleanup EXIT INT TERM

# 等待推流服务绑定端口
sleep 2

# 3. 启动实机 Rollout
echo "================================================================================"
echo " [2/2] 启动上位机实机 Rollout 推理 (Unitree G1 + Controller: ${CONTROLLER} + PI0.5)"
echo " 提示: 默认处于零速度原地平衡模式，在终端中输入 's' 保持原地，输入 'w' 开启行走"
echo "================================================================================"

$LEROBOT_ROLLOUT \
    --strategy.type=base \
    --policy.path=model/box_move_blue \
    --policy.device=cuda \
    --robot.type=unitree_g1 \
    --robot.is_simulation=false \
    --robot.robot_ip="${ROBOT_IP}" \
    --robot.controller="${CONTROLLER}" \
    --robot.locomotion_mode="${LOCOMOTION_MODE}" \
    --robot.zero_locomotion_cmd=true \
    --robot.cameras="{\"global_view\": {\"type\": \"zmq\", \"server_address\": \"localhost\", \"port\": ${PORT}, \"camera_name\": \"head_camera\", \"width\": 640, \"height\": 480, \"fps\": 30, \"warmup_s\": 5}}" \
    --task="move blue box back and forth between tables" \
    --duration=1000 \
    --fps=25 \
    --display_data=true
