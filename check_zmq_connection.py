#!/usr/bin/env python3
"""
快速网络与 ZMQ 连通性排查脚本 (无需加载大模型，2秒出结果)
用法示例:
    python check_zmq_connection.py --robot-ip=10.8.8.118 --action-ip=10.8.8.118 --camera-ip=10.3.42.138 --camera-port=5555
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import zmq


def check_ping(host: str) -> bool:
    print(f"[*] 1. 测试基础网络 ICMP Ping [{host}] ...", end="", flush=True)
    try:
        ret = subprocess.run(
            ["ping", "-c", "1", "-W", "1", host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ret.returncode == 0:
            print(" \033[92m[PASS 通过]\033[0m")
            return True
        else:
            print(" \033[91m[FAIL 失败]\033[0m (ping 不通，可能处于不同隔离子网或被防火墙禁 ping)")
            return False
    except Exception as e:
        print(f" \033[91m[ERROR 异常: {e}]\033[0m")
        return False


def check_tcp_port(host: str, port: int, desc: str) -> bool:
    print(f"[*] 2. 测试 TCP 端口握手 [{desc}: {host}:{port}] ...", end="", flush=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        s.connect((host, port))
        s.close()
        print(" \033[92m[PASS 端口开放]\033[0m")
        return True
    except ConnectionRefusedError:
        print(" \033[91m[FAIL 连接被拒绝 Connection Refused]\033[0m")
        print(f"     -> 原因: 目标机 {host} 上没有程序监听端口 {port}，请确认服务端是否已启动！")
        return False
    except socket.timeout:
        print(" \033[91m[FAIL 超时 Timeout]\033[0m")
        print(f"     -> 原因: 端口 {port} 被防火墙 (ufw/iptables) 拦截，或 Wi-Fi 路由隔离！")
        return False
    except Exception as e:
        print(f" \033[91m[FAIL 失败: {e}]\033[0m")
        return False


def check_zmq_state(host: str, port: int) -> bool:
    print(f"[*] 3. 测试 ZMQ 状态订阅 (从 {host}:{port} 拉取 29-DoF 关节数据) ...", end="", flush=True)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    # 最多等待 2.5 秒
    socks = dict(poller.poll(2500))
    if sub in socks and socks[sub] == zmq.POLLIN:
        try:
            payload = sub.recv(zmq.NOBLOCK)
            data = json.loads(payload.decode("utf-8"))
            motors = data.get("motors", {})
            print(" \033[92m[PASS 成功接收]\033[0m")
            print(f"     -> 成功解码: 包含 {len(motors)} 个电机状态, 时间戳: {data.get('timestamp')}")
            sub.close(linger=0)
            ctx.term()
            return True
        except Exception as e:
            print(f" \033[93m[WARN 接收到数据但解析错误: {e}]\033[0m")
            sub.close(linger=0)
            ctx.term()
            return False
    else:
        print(" \033[91m[FAIL 超时未收到数据 (2.5s)]\033[0m")
        print(f"     -> 原因: 端口连接成功，但 {host}:{port} 没有广播数据。请确认服务端的 state_publisher_loop 线程是否正常工作。")
        sub.close(linger=0)
        ctx.term()
        return False


def check_zmq_camera(host: str, port: int) -> bool:
    print(f"[*] 4. 测试 ZMQ 相机推流 (从 {host}:{port} 拉取图像帧) ...", end="", flush=True)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    socks = dict(poller.poll(2500))
    if sub in socks and socks[sub] == zmq.POLLIN:
        try:
            payload = sub.recv(zmq.NOBLOCK)
            data = json.loads(payload.decode("utf-8"))
            images = data.get("images", {})
            cam_name = list(images.keys())[0] if images else "none"
            print(" \033[92m[PASS 成功接收]\033[0m")
            print(f"     -> 成功解码相机画面: 相机名 '{cam_name}', 帧字节数: {len(images.get(cam_name, ''))}")
            sub.close(linger=0)
            ctx.term()
            return True
        except Exception as e:
            print(f" \033[93m[WARN 接收到数据但解析错误: {e}]\033[0m")
            sub.close(linger=0)
            ctx.term()
            return False
    else:
        print(" \033[91m[FAIL 超时未收到图像帧 (2.5s)]\033[0m")
        print(f"     -> 原因: 相机服务未在 {host}:{port} 推流，请检查相机服务端是否启动！")
        sub.close(linger=0)
        ctx.term()
        return False


def main():
    parser = argparse.ArgumentParser(description="快速网络与 ZMQ 连通性排查工具")
    parser.add_argument("--robot-ip", type=str, default="10.8.8.118", help="状态源机器人 IP")
    parser.add_argument("--action-ip", type=str, default="10.8.8.118", help="动作目标 IP")
    parser.add_argument("--camera-ip", type=str, default="10.3.42.138", help="相机源 IP")
    parser.add_argument("--state-port", type=int, default=6001, help="状态端口 (默认 6001)")
    parser.add_argument("--action-port", type=int, default=6002, help="动作端口 (默认 6002)")
    parser.add_argument("--camera-port", type=int, default=5556, help="相机端口 (默认 5556)")
    args = parser.parse_args()

    print("=" * 70)
    print(" [ZMQ 快速连通性诊断 - 2秒速查]")
    print(f" 状态源主机 : {args.robot_ip}:{args.state_port}")
    print(f" 动作目标机 : {args.action_ip}:{args.action_port}")
    print(f" 相机源主机 : {args.camera_ip}:{args.camera_port}")
    print("=" * 70)

    # 1. Ping
    ping_ok = check_ping(args.robot_ip)

    # 2. 状态端口 TCP
    state_port_ok = check_tcp_port(args.robot_ip, args.state_port, "状态端口 (PUB)")

    # 3. 动作端口 TCP
    action_port_ok = check_tcp_port(args.action_ip, args.action_port, "动作端口 (PULL)")

    # 4. 接收状态数据
    if state_port_ok:
        state_data_ok = check_zmq_state(args.robot_ip, args.state_port)
    else:
        state_data_ok = False

    # 5. 相机端口与推流
    cam_port_ok = check_tcp_port(args.camera_ip, args.camera_port, "相机推流端口")
    if cam_port_ok:
        cam_data_ok = check_zmq_camera(args.camera_ip, args.camera_port)
    else:
        cam_data_ok = False

    print("=" * 70)
    print(" [诊断汇总]")
    if state_data_ok and action_port_ok and cam_data_ok:
        print(" \033[92m★ 全部连接正常！你可以直接启动 ./run_vla.sh 进行大模型推理。\033[0m")
    else:
        print(" \033[91m✖ 存在连接阻塞项，请根据上述详细提示修复后再启动 VLA。\033[0m")
    print("=" * 70)


if __name__ == "__main__":
    main()
