#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree G1 极简视觉推流服务 (Python 3.8+ 兼容，无需克隆 LeRobot 完整仓库)
功能：
- 优先采用 RealSense (pyrealsense2) 获取 100% 真彩色 RGB 图像
- 若 RealSense 不可用则自动回退至 OpenCV (/dev/video*) 设备
- 支持底层全视场角高分辨率采集 (如 1920x1080 / 1280x720)，在机载端高效处理为目标尺寸 (640x480)
- 编码为 base64 JPEG 格式
- 通过 ZMQ PUB (默认端口 5556) 广播给上位机
- 支持同时推流 head_camera 和 ego_view 两个键名，兼容 LeRobot 与 SonicStar/GR00T
"""
import argparse
import base64
import json
import time
import cv2
import zmq

try:
    import numpy as np
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


def process_frame(frame, target_w, target_h, mode="resize"):
    """
    将采集到的高分辨率图像处理为目标尺寸 (target_w, target_h)

    参数:
        frame: 输入图像 (BGR 格式)
        target_w: 目标宽度 (默认 640)
        target_h: 目标高度 (默认 480)
        mode:
            - 'resize': 全视野直接缩放 (默认)。完全保留水平和垂直 100% 物理视场角 (FOV)，画面会有轻微比例压缩 (16:9 压缩为 4:3)。对机器人策略模型泛化效果最好，盲区最小。
            - 'crop': 保持比例中心裁剪为 4:3 后缩放。物理无畸变，但左右两侧会损失约 25% 视野。
            - 'letterbox': 等比缩放后上下填充黑边 (Padding)。物理无畸变且全视野，但黑边浪费部分像素。
    """
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame

    if mode == "resize":
        # cv2.INTER_AREA 是最适合图像下采样缩小且抗锯齿的插值算法
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    target_aspect = target_w / target_h
    src_aspect = w / h

    if mode == "crop":
        if src_aspect > target_aspect:
            # 原始画面更宽，中心裁剪左右两侧
            new_w = int(h * target_aspect)
            x_start = (w - new_w) // 2
            cropped = frame[:, x_start:x_start + new_w]
        else:
            # 原始画面更高，中心裁剪上下两侧
            new_h = int(w / target_aspect)
            y_start = (h - new_h) // 2
            cropped = frame[y_start:y_start + new_h, :]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)

    elif mode == "letterbox":
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 摄像头推流服务")
    parser.add_argument("--device", type=int, default=2, help="OpenCV 摄像头设备号 (默认: 2)")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ 推流端口 (默认: 5556)")
    parser.add_argument("--capture-width", type=int, default=1280, help="底层硬件采集宽度 (默认: 1280，16:9 全视场角)")
    parser.add_argument("--capture-height", type=int, default=720, help="底层硬件采集高度 (默认: 720，16:9 全视场角)")
    parser.add_argument("--width", type=int, default=1280, help="推流目标图像宽度 (默认: 1280，直接推流给电脑)")
    parser.add_argument("--height", type=int, default=720, help="推流目标图像高度 (默认: 720，直接推流给电脑)")
    parser.add_argument("--resize-mode", type=str, default="resize", choices=["resize", "crop", "letterbox"],
                        help="分辨率变换模式: resize (全视野直接压缩), crop (保持比例中心裁剪), letterbox (等比缩放加黑边)")
    parser.add_argument("--fps", type=int, default=30, help="推流帧率 (默认: 30)")
    parser.add_argument("--name", type=str, default="ego_view", help="相机名称 (默认: ego_view)")
    parser.add_argument("--dual-names", action="store_true", help="同时推流 head_camera 和 ego_view 两个键名以兼容 LeRobot 与 Sonic")
    parser.add_argument("--no-realsense", action="store_true", help="强制禁用 RealSense 原生驱动，改用 OpenCV")
    args = parser.parse_args()

    use_realsense = False
    pipe = None
    cap = None

    if HAS_REALSENSE and not args.no_realsense:
        # 优先使用指定的全画幅采集分辨率，若硬件受限（如 USB 2.0 带宽或特定固件）则多级优雅回退
        resolutions_to_try = [
            (args.capture_width, args.capture_height),
            (1280, 720),
            (args.width, args.height),
        ]
        # 去重保持原有优先级
        seen = set()
        resolutions_to_try = [r for r in resolutions_to_try if not (r in seen or seen.add(r))]

        for cap_w, cap_h in resolutions_to_try:
            try:
                print(f"[*] 正在尝试 RealSense 原生 RGB 彩色管道 ({cap_w}x{cap_h} @ {args.fps} FPS)...")
                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, cap_w, cap_h, rs.format.bgr8, args.fps)
                pipe.start(cfg)
                # 测试预热抓取一帧
                test_frames = pipe.wait_for_frames(timeout_ms=5000)
                if test_frames.get_color_frame():
                    use_realsense = True
                    print(f"[+] RealSense 原生 RGB 启动成功！硬件采集: {cap_w}x{cap_h} -> 处理后输出: {args.width}x{args.height} (模式: {args.resize_mode})")
                    break
                else:
                    pipe.stop()
                    pipe = None
            except Exception as e:
                print(f"[!] RealSense 尝试 {cap_w}x{cap_h} 失败 ({e})，尝试下一级候选分辨率...")
                if pipe:
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    pipe = None

    if not use_realsense:
        print(f"[*] 正在打开摄像头 /dev/video{args.device} (OpenCV 模式)...")
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

        if not cap.isOpened():
            print(f"[!] 错误: 无法打开摄像头 /dev/video{args.device}，请检查设备号或 USB 权限！")
            return

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"[!] 错误: 打开了 /dev/video{args.device} 但无法捕获图像帧！")
            cap.release()
            return
        actual_h, actual_w = test_frame.shape[:2]
        print(f"[+] 摄像头启动成功！硬件实际采集分辨率: {actual_w}x{actual_h} -> 处理后输出: {args.width}x{args.height} (模式: {args.resize_mode})")

    # 初始化 ZMQ PUB 服务
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 10)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    mode_str = "RealSense 原生 RGB" if use_realsense else f"OpenCV /dev/video{args.device}"
    names_str = "head_camera + ego_view (双键名兼容)" if args.dual_names else args.name
    print(f"[+] ZMQ 推流服务已在 tcp://0.0.0.0:{args.port} 启动 [{mode_str}] (帧率: {args.fps} FPS, 相机名: {names_str})")

    interval = 1.0 / args.fps
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    try:
        while True:
            t_start = time.time()

            if use_realsense:
                frames = pipe.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_bgr = np.asanyarray(color_frame.get_data())
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            # 处理分辨率转换 (全视场角图像 -> 目标尺寸如 640x480)
            frame_processed = process_frame(frame_bgr, args.width, args.height, mode=args.resize_mode)

            # 转换为 RGB 格式并编码为 base64 JPEG
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
            encoded_image = base64.b64encode(buffer).decode("ascii")

            timestamp = time.time()
            if args.dual_names:
                payload = {
                    "timestamps": {"head_camera": timestamp, "ego_view": timestamp},
                    "images": {"head_camera": encoded_image, "ego_view": encoded_image},
                }
            else:
                payload = {
                    "timestamps": {args.name: timestamp},
                    "images": {args.name: encoded_image},
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
        if pipe:
            pipe.stop()
        if cap:
            cap.release()
        socket.close()
        context.term()
        print("[+] 视觉服务已安全退出。")


if __name__ == "__main__":
    main()
