#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6001 端口 (LowState) 实时数据包监听与诊断工具
用法:
    python monitor_state_6001.py                      # 默认连接 10.3.42.221:6001 优雅面板刷新
    python monitor_state_6001.py --robot-ip 127.0.0.1  # 指定 IP
    python monitor_state_6001.py --raw                 # 打印完整原始 JSON 字符串
    python monitor_state_6001.py --once                # 打印首个接收到的包后直接退出
    python monitor_state_6001.py --every-packet        # 不限流，每收到一个包就打印一次
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zmq

# ANSI 颜色定义
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 6001 端口 (LowState) 数据包监听工具")
    parser.add_argument("--robot-ip", "-ip", type=str, default="10.3.42.221", help="目标机器人 IP (默认: 10.3.42.221)")
    parser.add_argument("--port", "-p", type=int, default=6001, help="ZMQ PUB 端口 (默认: 6001)")
    parser.add_argument("--raw", "-r", action="store_true", help="打印原始 JSON 字符串")
    parser.add_argument("--once", "-1", action="store_true", help="仅打印接收到的第一个包后退出")
    parser.add_argument("--every-packet", "-a", action="store_true", help="打印每一个收到的数据包 (默认每秒限制刷新 2 次避免刷屏)")
    parser.add_argument("--interval", type=float, default=0.5, help="面板打印间隔秒数 (默认: 0.5s)")
    args = parser.parse_args()

    addr = f"tcp://{args.robot_ip}:{args.port}"
    print("=" * 75)
    print(f"{C_BOLD}📡 [6001 端口 LowState 监听器]{C_RESET}")
    print(f"目标地址: {C_GREEN}{addr}{C_RESET}")
    print(f"模式: {'打印原始 JSON' if args.raw else '结构化面板 (每0.5s刷新)'} | 退出: 按 Ctrl+C")
    print("=" * 75)
    print(f"⏳ 正在连接并等待首个数据包推送...")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # 保持最新一帧
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(addr)

    packet_count = 0
    start_time = time.time()
    last_print_time = 0.0

    try:
        while True:
            # 轮询等待数据包 (每次超时 500ms)
            if sock.poll(timeout=500):
                payload = sock.recv(zmq.NOBLOCK)
                now = time.time()
                packet_count += 1

                # 计算接收帧率 (Hz)
                elapsed = now - start_time
                freq = packet_count / elapsed if elapsed > 0 else 0.0

                # 限流检查 (除非指定 --every-packet 或 --once)
                if not args.every_packet and not args.once and (now - last_print_time < args.interval):
                    continue
                last_print_time = now

                # 1. 原始 JSON 模式
                if args.raw:
                    print(f"\n{C_BOLD}[Packet #{packet_count} | 长度: {len(payload)}B | 频率: {freq:.1f}Hz]{C_RESET}")
                    try:
                        raw_json = json.loads(payload.decode("utf-8"))
                        print(json.dumps(raw_json, indent=2, ensure_ascii=False))
                    except Exception:
                        print(payload.decode("utf-8", errors="replace"))
                    if args.once:
                        break
                    continue

                # 2. 结构化解析与面板展示
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception as e:
                    print(f"{C_RED}❌ JSON 解码错误: {e}{C_RESET}")
                    continue

                topic = msg.get("topic", "N/A")
                data = msg.get("data", msg) if isinstance(msg, dict) else {}
                motor_state = data.get("motor_state", []) if isinstance(data, dict) else []
                imu_state = data.get("imu_state", {}) if isinstance(data, dict) else {}

                num_motors = len(motor_state)
                non_zero_motors = sum(1 for m in motor_state if abs(m.get("q", 0.0)) > 1e-4)

                status_color = C_GREEN if non_zero_motors > 0 else C_YELLOW
                print("\n" + "-" * 75)
                print(
                    f"📦 {C_BOLD}收到数据包 #{packet_count}{C_RESET} | 来源: {addr} | "
                    f"实时接收率: {C_GREEN}{freq:.1f} Hz{C_RESET} | 报文大小: {len(payload)} 字节"
                )
                print(
                    f"   ├─ Topic 字段: {C_CYAN}{topic}{C_RESET}\n"
                    f"   ├─ 电机总数: {C_BOLD}{num_motors}{C_RESET} 个 | "
                    f"非零关节数: {status_color}{non_zero_motors} / {num_motors}{C_RESET} "
                    f"{'(✅ 真实物理关节角度)' if non_zero_motors > 0 else '(⚠️ 关节全为0)'}"
                )

                # 提取关键双臂关节值 (重点关注下探、夹紧与前伸)
                key_joints = {
                    15: "左肩俯仰 (L_ShoulderPitch)",
                    16: "左肩外展 (L_ShoulderRoll)",
                    18: "左肘弯曲 (L_Elbow)",
                    19: "左腕翻转 (L_WristRoll)",
                    22: "右肩俯仰 (R_ShoulderPitch)",
                    23: "右肩外展 (R_ShoulderRoll)",
                    25: "右肘弯曲 (R_Elbow)",
                    26: "右腕翻转 (R_WristRoll)",
                    12: "腰部航向 (WaistYaw)",
                }

                print(f"\n   {C_BOLD}🦾 关键动作关节实时物理值 (q, dq, tau_est):{C_RESET}")
                for idx, label in key_joints.items():
                    if idx < len(motor_state):
                        m = motor_state[idx]
                        q = float(m.get("q", 0.0))
                        dq = float(m.get("dq", 0.0))
                        tau = float(m.get("tau_est", m.get("tau", 0.0)))
                        temp = float(m.get("temperature", 0.0))
                        q_deg = math.degrees(q)
                        print(
                            f"      • [{idx:02d}] {label:<28s}: "
                            f"q={C_GREEN}{q:+7.3f} rad{C_RESET} ({q_deg:+6.1f}°) | "
                            f"dq={dq:+6.2f} rad/s | "
                            f"tau={tau:+6.2f} Nm | {temp:.0f}°C"
                        )

                # 提取 IMU 状态
                if imu_state:
                    rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])
                    gyro = imu_state.get("gyroscope", [0.0, 0.0, 0.0])
                    acc = imu_state.get("accelerometer", [0.0, 0.0, 0.0])
                    print(f"\n   🧭 {C_BOLD}机载 IMU 状态:{C_RESET}")
                    print(f"      • 欧拉角 (RPY) : Roll={rpy[0]:+6.2f}°, Pitch={rpy[1]:+6.2f}°, Yaw={rpy[2]:+6.2f}°")
                    print(f"      • 陀螺仪 (Gyro): x={gyro[0]:+6.2f}, y={gyro[1]:+6.2f}, z={gyro[2]:+6.2f}")
                    print(f"      • 加速度 (Acc) : x={acc[0]:+6.2f}, y={acc[1]:+6.2f}, z={acc[2]:+6.2f}")

                if args.once:
                    print(f"\n{C_GREEN}✔ 单次检查完成，程序退出。{C_RESET}")
                    break
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}用户中断，已停止监听。{C_RESET}")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
