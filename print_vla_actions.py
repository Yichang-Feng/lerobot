#!/usr/bin/env python3
"""
命令行实时打印 VLA 下发动作监听工具
用法:
    python print_vla_actions.py [--port=6002] [--raw]
"""

import argparse
import json
import time
import zmq


def main():
    parser = argparse.ArgumentParser(description="监听并打印 VLA 下发的动作数据")
    parser.add_argument("--port", type=int, default=6002, help="监听的 ZMQ 端口 (默认 6002)")
    parser.add_argument("--raw", action="store_true", help="打印完整原始 JSON，而不是格式化表格")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # 仅保留最新一帧，避免终端打印延迟
    sock.bind(f"tcp://0.0.0.0:{args.port}")

    print("=" * 75)
    print(f" [VLA 动作实时监听器 - 正在监听 tcp://0.0.0.0:{args.port}]")
    print(" 等待上位机 VLA 下发动作包... (按 Ctrl+C 退出)")
    print("=" * 75)

    last_time = time.time()
    count = 0

    try:
        while True:
            payload = sock.recv()
            recv_time = time.time()
            count += 1

            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception as e:
                print(f"[!] 收到非 JSON 数据: {payload[:50]}... ({e})")
                continue

            if args.raw:
                # 打印原始缩进 JSON
                print(json.dumps(data, indent=2, ensure_ascii=False))
                continue

            cmd = data.get("cmd", "unknown")
            ts = data.get("timestamp", recv_time)
            latency_ms = (recv_time - ts) * 1000.0
            action = data.get("action", {})

            # 计算接收频率
            dt = recv_time - last_time
            fps = 1.0 / dt if dt > 0 else 0.0
            last_time = recv_time

            # 提取底盘速度
            lx = action.get("remote.lx", 0.0)
            ly = action.get("remote.ly", 0.0)
            rx = action.get("remote.rx", 0.0)

            # 格式化输出
            print(
                f"\r\033[K"  # 清除当前行
                f"[{count:05d}帧 | {fps:4.1f}Hz | 网络延时:{latency_ms:5.1f}ms] "
                f"模式:{cmd:6s} | "
                f"速度: 前进={ly:+.3f} 横移={-lx:+.3f} 转向={-rx:+.3f} | "
                f"左肘={action.get('kLeftElbow.q', 0.0):+.3f} "
                f"右肘={action.get('kRightElbow.q', 0.0):+.3f}",
                end="",
                flush=True,
            )

            # 如果收到 reset 或 stop 等特殊指令，单独换行打印
            if cmd in ("reset", "stop"):
                print(f"\n>>> [系统事件] 收到指令: {cmd.upper()} <<<")

    except KeyboardInterrupt:
        print("\n\n[*] 退出动作监听器。")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
