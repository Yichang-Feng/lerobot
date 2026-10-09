#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 Dex-1 跨桌水瓶转运 (抓取位保持防碰桌模式) 一键启动脚本
# ==============================================================================
# 核心机制:
#   1. 桌 A 夹水瓶后: 保持在抓取位置 (跳过回身侧默认位置，杜绝水瓶刮碰桌面)
#   2. 导航转移期间: 外部项目代码将机器人手臂安全收回身侧默认位置
#   3. 到达桌 B 后: 键盘输入 'd' 回车，机械臂在默认位置直接启动 VLA 放置推理！
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LEROBOT_LIB="/home/yichangfeng/miniforge3/envs/lerobot/lib"
if [ -d "$LEROBOT_LIB" ]; then
    export LD_LIBRARY_PATH="$LEROBOT_LIB:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="/home/yichangfeng/miniforge3/envs/lerobot/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_vla_water_bottle_transfer_hold.py" "$@"
