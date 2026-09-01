#!/usr/bin/env bash
# ==============================================================================
# Script to verify PI0.5 policy response to training video replay in simulation
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
LEROBOT_ROLLOUT="/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"

VIDEO_PATH="${1:-datasets/box_pick/videos/observation.images.global_view/chunk-000/file-002.mp4}"
START_TIME="${2:-3:10}"
CONTROLLER="${3:-GrootLocomotionController}"
LOCOMOTION_MODE="${4:-stand}"
PORT=5556
FPS=30

echo "================================================================================"
echo " [1/2] Launching ZMQ Video Streamer for Training Video Replay"
echo " Video      : ${VIDEO_PATH}"
echo " Start Time : ${START_TIME} (Task start segment)"
echo " Port       : ${PORT} | FPS: ${FPS}"
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
    echo "Terminating video streamer (PID: $STREAMER_PID)..."
    kill -TERM "$STREAMER_PID" 2>/dev/null || true
    wait "$STREAMER_PID" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Wait for video streamer to bind socket and warm up
sleep 2

echo "================================================================================"
echo " [2/2] Launching lerobot-rollout (Unitree G1 + Controller: ${CONTROLLER} + PI0.5)"
echo " Diagnostic logs written to outputs/sonic_io_dump.txt"
echo "================================================================================"

$LEROBOT_ROLLOUT \
    --strategy.type=base \
    --policy.path=model/box_pick \
    --policy.device=cuda \
    --robot.type=unitree_g1 \
    --robot.is_simulation=true \
    --robot.controller="${CONTROLLER}" \
    --robot.locomotion_mode="${LOCOMOTION_MODE}" \
    --robot.cameras='{"global_view": {"type": "zmq", "server_address": "localhost", "port": 5556, "camera_name": "head_camera", "width": 640, "height": 480, "fps": 30, "warmup_s": 5}}' \
    --task="move blue box" \
    --duration=1000 \
    --fps=25 \
    --display_data=true
