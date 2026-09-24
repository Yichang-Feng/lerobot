#!/usr/bin/env python3
"""
快速网络与 ZMQ 连通性排查脚本 (无需加载大模型，2秒出结果)
用法示例:
    python check_zmq_connection.py --robot-ip=10.8.8.118 --action-ip=10.8.8.118 --camera-ip=10.3.42.138 --camera-port=5555
"""

import argparse
import ast
import json
import re
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
            raw = json.loads(payload.decode("utf-8"))
            data = raw.get("data", raw) if isinstance(raw, dict) else {}
            motors = data.get("motors") or raw.get("motors") or data.get("motor_state") or raw.get("motor_state") or {}
            num_motors = len(motors)
            ts = data.get("timestamp") or raw.get("timestamp")
            print(" \033[92m[PASS 成功接收]\033[0m")
            print(f"     -> 成功解码: 包含 {num_motors} 个电机状态, 时间戳: {ts}")
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


def check_zmq_camera(host: str, port: int, desc: str = "头部全局相机") -> bool:
    print(f"[*] 测试 ZMQ 相机推流 [{desc}: 从 {host}:{port} 拉取图像帧] ...", end="", flush=True)
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
            print(f"     -> 成功解码相机画面 [{desc}]: 相机名 '{cam_name}', 帧字节数: {len(images.get(cam_name, ''))}")
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
        print(f"     -> 原因: [{desc}] 相机服务未在 {host}:{port} 推流，请检查相机服务端是否启动！")
        sub.close(linger=0)
        ctx.term()
        return False


def check_zmq_mode(host: str, port: int) -> bool:
    print(f"[*] 6. 测试模式流 (从 {host}:{port} 监听手柄/导航/VLA广播) ...", end="", flush=True)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 10)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    # 等待最多 2.5 秒
    socks = dict(poller.poll(2500))
    if sub in socks and socks[sub] == zmq.POLLIN:
        try:
            parts = sub.recv_multipart(zmq.NOBLOCK)
            raw_text = " | ".join(p.decode("utf-8", errors="ignore").strip() for p in parts)
            
            # 统计接下来的频率
            t0 = time.time()
            extra_count = 1
            while time.time() - t0 < 0.3:
                try:
                    sub.recv_multipart(zmq.NOBLOCK)
                    extra_count += 1
                except zmq.Again:
                    time.sleep(0.01)
            hz_est = extra_count / (time.time() - t0)

            # 解析模式
            val_lower = raw_text.lower()
            if any(k in val_lower for k in ("vla", "2")):
                mode_str = "\033[92mVLA 控制模式 (vla)\033[0m"
            elif any(k in val_lower for k in ("nav", "0")):
                mode_str = "\033[96m导航模式 (nav)\033[0m"
            elif any(k in val_lower for k in ("gamepad", "manual", "shoubing", "loco", "1")):
                mode_str = "\033[93m手柄控制模式 (gamepad)\033[0m"
            else:
                mode_str = f"未知模式 ({raw_text})"

            print(" \033[92m[PASS 成功接收]\033[0m")
            print(f"     -> 原始接收: {raw_text!r}")
            print(f"     -> 当前解析模式: {mode_str}")
            print(f"     -> 广播频率估算: ~{hz_est:.1f} Hz (持续发送正常)")
            sub.close(linger=0)
            ctx.term()
            return True
        except Exception as e:
            print(f" \033[93m[WARN 接收到数据但解析异常: {e}]\033[0m")
            sub.close(linger=0)
            ctx.term()
            return False
    else:
        print(f" \033[93m[WARN 超时未收到模式数据 (2.5s)]\033[0m")
        print(f"     -> 原因: 端口连接无响应或未收到数据包。请确认手柄/导航节点是否已启动并绑定 tcp://*:{port} 进行 PUB 广播。")
        print(f"     -> 提示: 若暂不使用自动模式，可传 --manual 仅使用键盘 's' 启动。")
        sub.close(linger=0)
        ctx.term()
        return False


def _parse_gripper_data(raw_data):
    """Parse raw bytes, string, multipart list, or dict into (left, right)."""
    try:
        if isinstance(raw_data, (list, tuple)):
            for part in reversed(raw_data):
                parsed = _parse_gripper_data(part)
                if parsed is not None:
                    return parsed
            joined = b" ".join(part if isinstance(part, bytes) else str(part).encode() for part in raw_data)
            return _parse_gripper_data(joined)

        if isinstance(raw_data, bytes):
            s = raw_data.decode("utf-8", errors="ignore").strip().strip("\x00\r\n\t ")
        else:
            s = str(raw_data).strip().strip("\x00\r\n\t ")
        if not s:
            return None

        # Look for JSON payload in string
        obj = None
        if "{" in s and "}" in s:
            first_b = s.find("{")
            last_b = s.rfind("}")
            candidate = s[first_b : last_b + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                try:
                    obj = ast.literal_eval(candidate)
                except Exception:
                    obj = None

        def _extract_val(v):
            if isinstance(v, dict):
                for k in ("q", "position", "pos", "angle", "rad", "val", "value"):
                    if k in v:
                        return float(v[k])
                if v:
                    first_v = next(iter(v.values()))
                    if isinstance(first_v, (int, float)):
                        return float(first_v)
                return 0.0
            elif isinstance(v, (list, tuple)):
                return float(v[0])
            return float(v)

        def extract_from_dict(d):
            if not isinstance(d, dict):
                return None
            if "data" in d:
                sub = extract_from_dict(d["data"])
                if sub is not None:
                    return sub
                if isinstance(d["data"], (list, tuple)) and len(d["data"]) >= 2:
                    return float(d["data"][0]), float(d["data"][1])
            if "gripper" in d:
                sub = extract_from_dict(d["gripper"])
                if sub is not None:
                    return sub
            if "left" in d or "right" in d:
                l = _extract_val(d.get("left", d.get("kLeftGripper", 5.0)))
                r = _extract_val(d.get("right", d.get("kRightGripper", 5.0)))
                return l, r
            return None

        if isinstance(obj, dict):
            res = extract_from_dict(obj)
            if res is not None:
                return res

        # Regex fallback
        m_left = re.search(r"left[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        m_right = re.search(r"right[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        if m_left or m_right:
            l = float(m_left.group(1)) if m_left else 5.0
            r = float(m_right.group(1)) if m_right else 5.0
            return l, r

    except Exception as e:
        pass
    return None


def check_zmq_gripper(host: str, port: int) -> bool:
    print(f"[*] 7. 测试 ZMQ 夹爪角度订阅 (从 {host}:{port} 拉取 data{{left, right}} 数据) ...", end="", flush=True)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 2500)
    sub.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    socks = dict(poller.poll(2500))
    if sub in socks and socks[sub] == zmq.POLLIN:
        try:
            msg = sub.recv_multipart(zmq.NOBLOCK)
            raw_text = " | ".join(p.decode("utf-8", errors="ignore").strip() for p in msg)

            # 统计频率
            t0 = time.time()
            extra_count = 1
            while time.time() - t0 < 0.3:
                try:
                    sub.recv_multipart(zmq.NOBLOCK)
                    extra_count += 1
                except zmq.Again:
                    time.sleep(0.01)
            hz_est = extra_count / (time.time() - t0)

            parsed = _parse_gripper_data(msg)
            if parsed is not None:
                left_val, right_val = parsed
                print(" \033[92m[PASS 成功接收]\033[0m")
                print(f"     -> 原始接收: {raw_text!r}")
                print(f"     -> 夹爪角度: left={left_val:.2f}, right={right_val:.2f} (范围: 0.0~5.0)")
                print(f"     -> 广播频率估算: ~{hz_est:.1f} Hz (持续发送正常)")
                sub.close(linger=0)
                ctx.term()
                return True
            else:
                print(f" \033[93m[WARN 接收到数据但解析异常: 未能提取 left 和 right]\033[0m")
                print(f"     -> 原始接收: {raw_text!r}")
                sub.close(linger=0)
                ctx.term()
                return False
        except Exception as e:
            print(f" \033[93m[WARN 接收到数据但解析异常: {e}]\033[0m")
            sub.close(linger=0)
            ctx.term()
            return False
    else:
        print(f" \033[93m[WARN 超时未收到夹爪数据 (2.5s)]\033[0m")
        print(f"     -> 原因: 端口连接无响应或未收到数据包。请确认夹爪驱动节点是否已启动并绑定 tcp://*:{port} 进行 PUB 广播。")
        sub.close(linger=0)
        ctx.term()
        return False


def main():
    parser = argparse.ArgumentParser(description="快速网络与 ZMQ 连通性排查工具")
    parser.add_argument("--robot-ip", type=str, default="10.3.42.221", help="状态源机器人 IP")
    parser.add_argument("--action-ip", type=str, default="", help="动作目标 IP (默认与 robot-ip 相同)")
    parser.add_argument("--camera-ip", type=str, default="", help="相机源 IP (默认与 robot-ip 相同)")
    parser.add_argument("--mode-ip", type=str, default="", help="模式源 IP (默认与 robot-ip 相同)")
    parser.add_argument("--gripper-ip", type=str, default="", help="夹爪源 IP (默认与 robot-ip 相同)")
    parser.add_argument("--state-port", type=int, default=6001, help="状态端口 (默认 6001)")
    parser.add_argument("--action-port", type=int, default=6002, help="动作端口 (默认 6002)")
    parser.add_argument("--camera-port", type=int, default=5556, help="相机端口 (默认 5556)")
    parser.add_argument("--mode-port", type=int, default=6000, help="模式端口 (默认 6000)")
    parser.add_argument("--gripper-port", type=int, default=6004, help="夹爪端口 (默认 6004)")
    parser.add_argument("--enable-gripper", "--gripper", action="store_true", help="是否启用夹爪端口检测")
    parser.add_argument("--wrist-cameras", "--wrist_cameras", "--enable-wrist-cameras", action="store_true", help="是否启用手腕双相机检测")
    parser.add_argument("--left-wrist-port", type=int, default=5557, help="左手腕相机端口 (默认 5557)")
    parser.add_argument("--right-wrist-port", type=int, default=5558, help="右手腕相机端口 (默认 5558)")
    parser.add_argument("--left-wrist-ip", type=str, default="", help="左手腕相机 IP (默认同 camera-ip)")
    parser.add_argument("--right-wrist-ip", type=str, default="", help="右手腕相机 IP (默认同 camera-ip)")
    args = parser.parse_args()

    action_host = args.action_ip if args.action_ip else args.robot_ip
    camera_host = args.camera_ip if args.camera_ip else args.robot_ip
    mode_host = args.mode_ip if args.mode_ip else args.robot_ip
    gripper_host = args.gripper_ip if args.gripper_ip else args.robot_ip
    left_wrist_host = args.left_wrist_ip if args.left_wrist_ip else camera_host
    right_wrist_host = args.right_wrist_ip if args.right_wrist_ip else camera_host

    print("=" * 70)
    print(" [ZMQ 快速连通性诊断 - 2秒速查]")
    print(f" 状态源主机 : {args.robot_ip}:{args.state_port}")
    print(f" 动作目标机 : {action_host}:{args.action_port}")
    print(f" 相机源主机 : {camera_host}:{args.camera_port} (全局主摄)")
    if args.wrist_cameras:
        print(f" 左腕相机机 : {left_wrist_host}:{args.left_wrist_port}")
        print(f" 右腕相机机 : {right_wrist_host}:{args.right_wrist_port}")
    print(f" 模式源主机 : {mode_host}:{args.mode_port}")
    if args.enable_gripper:
        print(f" 夹爪源主机 : {gripper_host}:{args.gripper_port}")
    print("=" * 70)

    # 1. Ping
    ping_ok = check_ping(args.robot_ip)

    # 2. 状态端口 TCP
    state_port_ok = check_tcp_port(args.robot_ip, args.state_port, "状态端口 (PUB)")

    # 3. 动作端口 TCP
    action_port_ok = check_tcp_port(action_host, args.action_port, "动作端口 (PULL)")

    # 4. 接收状态数据
    if state_port_ok:
        state_data_ok = check_zmq_state(args.robot_ip, args.state_port)
    else:
        state_data_ok = False

    # 5. 相机端口与推流
    cam_port_ok = check_tcp_port(camera_host, args.camera_port, "全局主相机端口")
    if cam_port_ok:
        cam_data_ok = check_zmq_camera(camera_host, args.camera_port, "全局主视角")
    else:
        cam_data_ok = False

    # 5.1 手腕相机检测 (如果启用)
    wrist_cam_ok = True
    if args.wrist_cameras:
        lw_port_ok = check_tcp_port(left_wrist_host, args.left_wrist_port, "左手腕相机端口")
        rw_port_ok = check_tcp_port(right_wrist_host, args.right_wrist_port, "右手腕相机端口")
        lw_data_ok = check_zmq_camera(left_wrist_host, args.left_wrist_port, "左手腕相机") if lw_port_ok else False
        rw_data_ok = check_zmq_camera(right_wrist_host, args.right_wrist_port, "右手腕相机") if rw_port_ok else False
        wrist_cam_ok = lw_data_ok and rw_data_ok

    # 6. 模式端口与推流
    mode_port_ok = check_tcp_port(mode_host, args.mode_port, f"手柄/模式广播端口 (PUB: {args.mode_port})")
    if mode_port_ok:
        mode_data_ok = check_zmq_mode(mode_host, args.mode_port)
    else:
        mode_data_ok = False

    # 7. 夹爪端口与推流 (如果启用)
    gripper_data_ok = True
    if args.enable_gripper:
        gripper_port_ok = check_tcp_port(gripper_host, args.gripper_port, f"夹爪角度广播端口 (PUB: {args.gripper_port})")
        if gripper_port_ok:
            gripper_data_ok = check_zmq_gripper(gripper_host, args.gripper_port)
        else:
            gripper_data_ok = False

    print("=" * 70)
    print(" [诊断汇总]")
    if state_data_ok and action_port_ok and cam_data_ok and wrist_cam_ok and gripper_data_ok:
        if mode_data_ok:
            print(" \033[92m★ 全部连接正常 (包含所有相机流、模式流与夹爪流)！可以直接启动大模型推理。\033[0m")
        else:
            print(f" \033[93m▲ 机器人基础连接正常，但 {args.mode_port} 模式流未就绪。\033[0m")
            print("   -> 若现在启动，请使用 ./run_vla.sh --manual 手动按 's' 运行。")
    else:
        print(" \033[91m✖ 存在连接阻塞项，请根据上述详细提示修复后再启动 VLA。\033[0m")
    print("=" * 70)


if __name__ == "__main__":
    main()
