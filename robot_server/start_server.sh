#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 机器人端一键启动脚本 (电机桥接 + 可选机载相机推流)
# ==============================================================================

# 1. 自动设置 CycloneDDS 环境变量
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}"
export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="${SCRIPT_DIR}/camera_server"

# 2. 参数解析
USE_CAMERA=true
CAMERA_DEVICE=2
CAMERA_PORT=5555

for arg in "$@"; do
    case $arg in
        --no-camera|--no-cam|no_cam|-n)
            USE_CAMERA=false
            shift
            ;;
        --camera|--cam)
            USE_CAMERA=true
            shift
            ;;
        --device=*|-d=*)
            CAMERA_DEVICE="${arg#*=}"
            shift
            ;;
    esac
done

echo "================================================================================"
echo " [Unitree G1 机载服务启动器]"
echo " 电机 DDS 桥接 (ZMQ) : [启用] (LowCmd: 6000, LowState: 6001)"
if [ "$USE_CAMERA" = true ]; then
    echo " 机载视觉推流 (ZMQ) : [启用] (设备: /dev/video${CAMERA_DEVICE}, 端口: ${CAMERA_PORT})"
else
    echo " 机载视觉推流 (ZMQ) : [已禁用] (仅提供电机状态，视觉由上位机视频回放提供)"
fi
echo " 提示: 在终端按下 [Ctrl + C] 即可一键安全退出所有服务"
echo "================================================================================"

PIDS=()

# 捕获退出信号，一键关闭所有子进程
cleanup() {
    echo ""
    echo "[*] 正在停止所有机载服务..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    wait 2>/dev/null
    echo "[+] 所有服务已安全退出。"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 3. 启动电机 DDS 桥接服务
echo "[1/2] 正在启动电机 DDS 桥接服务..."
python3 "${SERVER_DIR}/motor_server.py" &
PIDS+=($!)

# 4. 根据选项决定是否启动相机服务
if [ "$USE_CAMERA" = true ]; then
    sleep 1
    echo "[2/2] 正在启动机载视觉推流服务..."
    python3 "${SERVER_DIR}/server.py" --device "${CAMERA_DEVICE}" --port "${CAMERA_PORT}" &
    PIDS+=($!)
fi

# 保持前台运行并监听子进程
wait
