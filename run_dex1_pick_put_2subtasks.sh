#!/usr/bin/env bash
# ==============================================================================
# Unitree G1 Dex-1 抓水瓶双子任务实机部署脚本 (完全兼容 run_vla.sh 所有参数)
# ==============================================================================
# 子任务对应:
#   Phase 1 (1): "pick up the water bottle from the table and place it into the blue box"
#   Phase 2 (2): "take the water bottle out of the blue box and place it back on the table"
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_POLICY="outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model"
DEFAULT_SUBTASKS="pick up the water bottle from the table and place it into the blue box,take the water bottle out of the blue box and place it back on the table"

HAS_POLICY=false
HAS_SUBTASKS=false

for arg in "$@"; do
    case "$arg" in
        --policy.path=*|--policy=*)
            HAS_POLICY=true
            ;;
        --subtasks=*)
            HAS_SUBTASKS=true
            ;;
    esac
done

PREPEND_ARGS=()
if [ "$HAS_POLICY" = false ]; then
    PREPEND_ARGS+=("--policy.path=${DEFAULT_POLICY}")
fi
if [ "$HAS_SUBTASKS" = false ]; then
    PREPEND_ARGS+=("--subtasks=${DEFAULT_SUBTASKS}")
fi

exec "$SCRIPT_DIR/run_vla.sh" "${PREPEND_ARGS[@]}" "$@"
