#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 实机部署一键启动脚本 (模式 A: 真实机载相机闭环)
# ==============================================================================
set -e

ROBOT_IP="${1:-192.168.123.164}"
POLICY_PATH="${2:-model/box_move_blue}"
TASK_DESC="${3:-move blue box back and forth between tables}"
CAMERA_PORT="${4:-5555}"

PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
LEROBOT_ROLLOUT="/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"

echo "================================================================================"
echo " [Unitree G1 实机真实相机闭环 Rollout 启动]"
echo " 机器人 IP   : ${ROBOT_IP} (实机模式 is_simulation=false)"
echo " 策略模型   : ${POLICY_PATH}"
echo " 任务描述   : ${TASK_DESC}"
echo " 相机推流端口: ${CAMERA_PORT}"
echo " 提示: 默认处于零速度原地平衡模式，在终端中输入 's' 保持原地，输入 'w' 开启行走"
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

# 2. 执行实机 Rollout
echo "启动实机 Rollout 进程..."
$LEROBOT_ROLLOUT \
    --policy.path="${POLICY_PATH}" \
    --task="${TASK_DESC}" \
    --robot.is_simulation=false \
    --robot.robot_ip="${ROBOT_IP}" \
    --robot.zero_locomotion_cmd=true \
    --robot.cameras="{\"global_view\": {\"type\": \"zmq\", \"server_address\": \"${ROBOT_IP}\", \"port\": ${CAMERA_PORT}, \"camera_name\": \"head_camera\", \"width\": 640, \"height\": 480, \"fps\": 30, \"warmup_s\": 5}}" \
    --display_data=true
