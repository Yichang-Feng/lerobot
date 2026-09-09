#!/usr/bin/env bash
# ==============================================================================
# 终端 1 启动脚本: G1 Locomotion + 物理实机控制服务端
# ==============================================================================
# 功能:
#   1. 连接 Unitree G1 物理实机电机底层 DDS
#   2. 启动机载 RealSense 相机推流服务 (默认端口 5555, ZMQ PUB)
#   3. 启动 50Hz GROOT Locomotion WBC 自平衡控制器 (Balance / Walk)
#   4. 广播 29-DoF 关节状态到端口 6001 (ZMQ PUB)
#   5. 监听来自终端 2 (VLA) 的 18-DoF 动作并驱动机器人 (端口 6002, ZMQ PULL)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
CONTROLLER="${CONTROLLER:-GrootLocomotionController}"
STATE_PORT="${STATE_PORT:-6001}"
ACTION_PORT="${ACTION_PORT:-6002}"
CAMERA_PORT="${CAMERA_PORT:-5555}"

echo "================================================================================"
echo " [终端 1: G1 Locomotion + 物理实机控制服务端]"
echo " 运行模式     : G1 物理实机"
echo " 机器人 IP   : ${ROBOT_IP}"
echo " 平衡控制器   : ${CONTROLLER} (50Hz)"
echo " 状态广播端口 : tcp://0.0.0.0:${STATE_PORT} (ZMQ PUB)"
echo " 动作接收端口 : tcp://0.0.0.0:${ACTION_PORT} (ZMQ PULL)"
echo " 相机推流端口 : ${CAMERA_PORT} (ZMQ PUB)"
echo "================================================================================"

exec "$PYTHON_BIN" -m lerobot.robots.unitree_g1.run_g1_locomotion_server \
    --real \
    --robot-ip="${ROBOT_IP}" \
    --controller="${CONTROLLER}" \
    --state-port="${STATE_PORT}" \
    --action-port="${ACTION_PORT}" \
    --camera-port="${CAMERA_PORT}" \
    "$@"
