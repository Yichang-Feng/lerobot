#!/usr/bin/env bash
# ==============================================================================
# G1 PI0.5 策略模型离线综合定量评测脚本
# ==============================================================================
# 使用方式:
#   1. 评测全部模型 (默认快速抽样 60 帧，各模型横向排行):
#      ./run_eval_offline.sh
#   2. 评测指定 runs 或 steps:
#      ./run_eval_offline.sh --runs=pi05_box_pick_turn_aligned --steps=003000,004000,005000
#   3. 全量验证样本评估:
#      ./run_eval_offline.sh --max-samples=-1
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/yichangfeng/miniforge3/envs/lerobot/lib:${LD_LIBRARY_PATH}"
PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"

exec "$PYTHON_BIN" eval_checkpoints_offline.py "$@"
