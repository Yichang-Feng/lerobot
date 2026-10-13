#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 Dex-1 跨桌水瓶转运 (抓取位保持防碰桌模式) 一键启动脚本
# ==============================================================================
# 核心机制:
#   1. 桌 A 夹水瓶前: 6000 端口手柄切入 VLA (或按'd'/'s')，JEV 视觉确认桌上有水瓶后自动开始夹取！
#   2. 桌 A 夹水瓶后: 保持在抓取位置 (跳过回身侧默认位置，杜绝水瓶刮碰桌面)
#   3. 导航转移期间: 外部项目代码将机器人手臂安全收回身侧默认位置
#   4. 到达桌 B 后: 6000 端口手柄切入 VLA (或输入'd')，JEV 视觉确认桌上有盒子后直接启动 VLA 放置！
#
# 推荐全流程高精度录制 (无 GUI 干扰，全流程 3 路视频+关节+手柄+JEV 决策对齐保存):
#   ./run_dex1_water_bottle_transfer_hold.sh \
#       --robot_ip=10.3.42.221 \
#       --policy.path=outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model \
#       --valen_evaluator=true \
#       --valen_ip=0.0.0.0 \
#       --valen_port=5559 \
#       --record=true \
#       --display_data=false
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
