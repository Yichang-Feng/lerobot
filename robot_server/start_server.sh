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
USE_MOTOR=true
USE_CAMERA=true
CAMERA_DEVICE=2
CAMERA_PORT=5556
CAPTURE_WIDTH=1280
CAPTURE_HEIGHT=720
OUTPUT_WIDTH=1280
OUTPUT_HEIGHT=720
RESIZE_MODE="resize"
CAMERA_NAME="ego_view"
EXTRA_CAMERA_ARGS=()

for arg in "$@"; do
    case $arg in
        --no-camera|--no-cam|no_cam|-nc|-n)
            USE_CAMERA=false
            shift
            ;;
        --no-motor|--no-motors|no_motor|-nm)
            USE_MOTOR=false
            shift
            ;;
        --only-camera|--camera-only|only_camera|-c)
            USE_CAMERA=true
            USE_MOTOR=false
            shift
            ;;
        --only-motor|--motor-only|only_motor|-m)
            USE_CAMERA=false
            USE_MOTOR=true
            shift
            ;;
        --camera|--cam)
            USE_CAMERA=true
            shift
            ;;
        --motor)
            USE_MOTOR=true
            shift
            ;;
        --device=*|-d=*)
            CAMERA_DEVICE="${arg#*=}"
            shift
            ;;
        --port=*|-p=*)
            CAMERA_PORT="${arg#*=}"
            shift
            ;;
        --capture-width=*|-cw=*)
            CAPTURE_WIDTH="${arg#*=}"
            shift
            ;;
        --capture-height=*|-ch=*)
            CAPTURE_HEIGHT="${arg#*=}"
            shift
            ;;
        --width=*|-w=*)
            OUTPUT_WIDTH="${arg#*=}"
            shift
            ;;
        --height=*|-h=*)
            OUTPUT_HEIGHT="${arg#*=}"
            shift
            ;;
        --mode=*|--resize-mode=*)
            RESIZE_MODE="${arg#*=}"
            shift
            ;;
        --name=*)
            CAMERA_NAME="${arg#*=}"
            shift
            ;;
        --dual-names|--dual)
            EXTRA_CAMERA_ARGS+=("--dual-names")
            shift
            ;;
        --no-realsense)
            EXTRA_CAMERA_ARGS+=("--no-realsense")
            shift
            ;;
    esac
done

echo "================================================================================"
echo " [Unitree G1 机载服务启动器]"
if [ "$USE_MOTOR" = true ]; then
    echo " 电机 DDS 桥接 (ZMQ) : [启用] (LowCmd: 6000, LowState: 6001)"
else
    echo " 电机 DDS 桥接 (ZMQ) : [已禁用] (仅推流相机画面，不占用电机 DDS)"
fi

if [ "$USE_CAMERA" = true ]; then
    echo " 机载视觉推流 (ZMQ) : [启用] (设备: /dev/video${CAMERA_DEVICE}, 端口: ${CAMERA_PORT})"
    echo "                      底层采集: ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} (全视场角)"
    echo "                      目标输出: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT} (模式: ${RESIZE_MODE})"
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
if [ "$USE_MOTOR" = true ]; then
    echo "[1/2] 正在启动电机 DDS 桥接服务..."
    python3 "${SERVER_DIR}/motor_server.py" &
    PIDS+=($!)
else
    echo "[1/2] 电机 DDS 桥接服务 : [已跳过]"
fi

# 4. 根据选项决定是否启动相机服务
if [ "$USE_CAMERA" = true ]; then
    sleep 1
    echo "[2/2] 正在启动机载视觉推流服务..."
    python3 "${SERVER_DIR}/server.py" \
        --device "${CAMERA_DEVICE}" \
        --port "${CAMERA_PORT}" \
        --capture-width "${CAPTURE_WIDTH}" \
        --capture-height "${CAPTURE_HEIGHT}" \
        --width "${OUTPUT_WIDTH}" \
        --height "${OUTPUT_HEIGHT}" \
        --resize-mode "${RESIZE_MODE}" \
        --name "${CAMERA_NAME}" \
        "${EXTRA_CAMERA_ARGS[@]}" &
    PIDS+=($!)
else
    echo "[2/2] 机载视觉推流服务 : [已跳过]"
fi

# 保持前台运行并监听子进程
wait
