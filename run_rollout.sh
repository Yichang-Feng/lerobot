#!/usr/bin/env bash
# ==============================================================================
# LeRobot + Unitree G1 + PI0.5 Rollout 快捷调试启动脚本
# ==============================================================================
# 使用方式:
#   1. 直接运行 (使用本文件内配置的默认参数):
#      ./run_rollout.sh
#   2. 命令行传参快速覆盖任意参数:
#      ./run_rollout.sh --task="pick up blue box" --display_data=true
#      ./run_rollout.sh --robot.is_simulation=false
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# 【常用默认值配置区】
# ------------------------------------------------------------------------------

# 1. 策略模型与任务目标
POLICY_PATH="model/box_move_blue"               # 策略路径 (如 model/box_move_blue, model/box_pick)
TASK="move blue box"                           # 任务指令文本

# 2. 运行环境与可视化
IS_SIMULATION=true                             # true: 仿真模式 (MuJoCo) | false: 实机模式 (连接物理 G1)
DISPLAY_DATA=false                             # 是否开启 Rerun 实时数据推流与视图 (true / false)

# 3. 异步推理与控制队列参数 (RTC 调优)
QUEUE_THRESHOLD=35                             # RTC 队列触发阈值 (推荐 35~40)
INTERPOLATION_MULTIPLIER=3                     # 指令插值倍率 (25Hz * 3 = 75Hz 指令流)
FPS=25                                         # Policy 决策帧率 (Hz)
DURATION=1000                                  # 运行最长时间 (秒)

# 4. 机器人与平衡控制器
ROBOT_TYPE="unitree_g1"                        # 机器人类型
CONTROLLER="GrootLocomotionController"         # 下肢平衡控制器 (GrootLocomotionController / SonicWholeBodyController)
LOCOMOTION_MODE="stand"                        # 初始移动模式 (stand: 原地保持 / walk: 允许行走)
ZERO_LOCOMOTION_CMD=false                      # 是否强制截断遥控移动速度 (实机安全建议设为 true)
ROBOT_IP="192.168.123.164"                     # G1 机载工控机 IP

# 5. 自定义相机配置（留空则根据仿真/实机自动推导）
CAMERA_SERVER=""                               # 相机推流地址
CAMERA_PORT=""                                 # 相机推流端口
CAMERA_NAME="head_camera"                      # 相机名称

# ------------------------------------------------------------------------------
# 动态解析命令行传入参数（实现参数智能联动与覆盖）
# ------------------------------------------------------------------------------
CLI_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --robot.is_simulation=false|--is_simulation=false|--real)
            IS_SIMULATION=false
            ;;
        --robot.is_simulation=true|--is_simulation=true|--sim)
            IS_SIMULATION=true
            ;;
        --task=*)
            TASK="${arg#*=}"
            ;;
        --policy.path=*)
            POLICY_PATH="${arg#*=}"
            ;;
        --display_data=*)
            DISPLAY_DATA="${arg#*=}"
            ;;
        --queue_threshold=*|--inference.queue_threshold=*)
            QUEUE_THRESHOLD="${arg#*=}"
            ;;
        --interpolation_multiplier=*)
            INTERPOLATION_MULTIPLIER="${arg#*=}"
            ;;
        --camera_server=*)
            CAMERA_SERVER="${arg#*=}"
            ;;
        --camera_port=*)
            CAMERA_PORT="${arg#*=}"
            ;;
        --robot.robot_ip=*)
            ROBOT_IP="${arg#*=}"
            ;;
        --robot.zero_locomotion_cmd=*|--zero_locomotion_cmd=*)
            ZERO_LOCOMOTION_CMD="${arg#*=}"
            ;;
        --robot.controller=*)
            CONTROLLER="${arg#*=}"
            ;;
        *)
            CLI_ARGS+=("$arg")
            ;;
    esac
done

# 根据是否为仿真模式自动推导相机推流地址与安全保护
if [ "$IS_SIMULATION" = true ]; then
    CAMERA_SERVER="${CAMERA_SERVER:-localhost}"
    CAMERA_PORT="${CAMERA_PORT:-5556}"
else
    CAMERA_SERVER="${CAMERA_SERVER:-$ROBOT_IP}"
    CAMERA_PORT="${CAMERA_PORT:-5555}"
    # 实机模式下默认开启零速度原地平衡保护（防止意外移动）
    ZERO_LOCOMOTION_CMD=true
fi

# ------------------------------------------------------------------------------
# 执行命令生成与启动
# ------------------------------------------------------------------------------
export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
LEROBOT_ROLLOUT="/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"

echo "================================================================================"
echo " [LeRobot G1 Rollout 快捷调试启动]"
echo " 模型路径 : ${POLICY_PATH}"
echo " 任务指令 : ${TASK}"
echo " 运行模式 : $( [ "$IS_SIMULATION" = true ] && echo "MuJoCo 仿真" || echo "G1 物理实机 (${ROBOT_IP})" )"
echo " 平衡控制 : ${CONTROLLER} (模式: ${LOCOMOTION_MODE})"
echo " 相机连接 : ${CAMERA_SERVER}:${CAMERA_PORT} (${CAMERA_NAME})"
echo " RTC 队列 : queue_threshold=${QUEUE_THRESHOLD} | 插值倍率=${INTERPOLATION_MULTIPLIER}"
echo " 决策帧率 : ${FPS} Hz | 可视化: ${DISPLAY_DATA}"
echo " 速度保护 : zero_locomotion_cmd=${ZERO_LOCOMOTION_CMD}"
echo "================================================================================"

exec "$LEROBOT_ROLLOUT" \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.queue_threshold="${QUEUE_THRESHOLD}" \
    --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}" \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --robot.type="${ROBOT_TYPE}" \
    --robot.is_simulation="${IS_SIMULATION}" \
    --robot.robot_ip="${ROBOT_IP}" \
    --robot.controller="${CONTROLLER}" \
    --robot.locomotion_mode="${LOCOMOTION_MODE}" \
    --robot.zero_locomotion_cmd="${ZERO_LOCOMOTION_CMD}" \
    --robot.cameras="{\"global_view\": {\"type\": \"zmq\", \"server_address\": \"${CAMERA_SERVER}\", \"port\": ${CAMERA_PORT}, \"camera_name\": \"${CAMERA_NAME}\", \"width\": 640, \"height\": 480, \"fps\": 30, \"warmup_s\": 5}}" \
    --task="${TASK}" \
    --duration="${DURATION}" \
    --fps="${FPS}" \
    --display_data="${DISPLAY_DATA}" \
    "${CLI_ARGS[@]}"
