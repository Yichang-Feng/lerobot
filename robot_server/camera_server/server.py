#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree G1 极简视觉推流服务 (Python 3.8+ 兼容，无需克隆 LeRobot 完整仓库)
功能：
- 采集指定摄像头（默认 /dev/video2）RGB 图像
- 编码为 base64 JPEG 格式
- 通过 ZMQ PUB (端口 5555) 广播给上位机
"""
import argparse
import base64
import json
import time
import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 摄像头推流服务")
    parser.add_argument("--device", type=int, default=2, help="摄像头设备号 (默认: 2)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ 推流端口 (默认: 5555)")
    parser.add_argument("--width", type=int, default=640, help="图像宽度 (默认: 640)")
    parser.add_argument("--height", type=int, default=480, help="图像高度 (默认: 480)")
    parser.add_argument("--fps", type=int, default=30, help="推流帧率 (默认: 30)")
    parser.add_argument("--name", type=str, default="head_camera", help="相机名称 (默认: head_camera)")
    args = parser.parse_args()

    print(f"[*] 正在打开摄像头 /dev/video{args.device}...")
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"[!] 错误: 无法打开摄像头 /dev/video{args.device}，请检查设备号或 USB 权限！")
        return

    # 测试读取一帧
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"[!] 错误: 打开了 /dev/video{args.device} 但无法捕获图像帧！")
        cap.release()
        return

    print(f"[+] 摄像头启动成功！实际捕获分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")

    # 初始化 ZMQ PUB 服务
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 10)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[+] ZMQ 推流服务已在 tcp://0.0.0.0:{args.port} 启动 (帧率: {args.fps} FPS, 名称: {args.name})")

    interval = 1.0 / args.fps
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # 转换为 RGB 格式并编码为 base64 JPEG
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
            encoded_image = base64.b64encode(buffer).decode("ascii")

            # 构造上位机期望的标准 JSON 数据包
            timestamp = time.time()
            payload = {
                "timestamps": {args.name: timestamp},
                "images": {args.name: encoded_image}
            }

            socket.send_string(json.dumps(payload), zmq.NOBLOCK)

            # 精准控制推流帧率
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] 正在关闭视觉推流服务...")
    finally:
        cap.release()
        socket.close()
        context.term()
        print("[+] 视觉服务已安全退出。")


if __name__ == "__main__":
    main()
