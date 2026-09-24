#!/bin/bash
# 一键启动数据采集脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/config.yaml"

echo "=========================================="
echo "XR Teleoperate 数据采集系统"
echo "=========================================="

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：找不到配置文件 $CONFIG_FILE"
    exit 1
fi

python3 "${SCRIPT_DIR}/main.py" --config "$CONFIG_FILE" "$@"
