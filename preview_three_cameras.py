#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路相机实时监控与预览工具 (支持 OpenCV 拼屏 / Rerun 仪表板)
支持同时监听:
  - 头部相机 (tcp://<ip>:5556, observation.images.global_view)
  - 左手腕相机 (tcp://<ip>:5557, observation.images.left_wrist)
  - 右手腕相机 (tcp://<ip>:5558, observation.images.right_wrist)

用法示例:
  # 1. 在桌面弹出 OpenCV 三联屏窗口查看:
  python preview_three_cameras.py --ip 10.3.42.221

  # 2. 通过 Rerun 查看 (本地启动 GUI):
  python preview_three_cameras.py --ip 10.3.42.221 --rerun

  # 3. 通过 Rerun 浏览器网页版查看 (无桌面/远程环境推荐):
  python preview_three_cameras.py --ip 10.3.42.221 --rerun --web
"""

import argparse
import base64
import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import zmq


class ZMQCameraReceiver:
    def __init__(self, name: str, ip: str, port: int):
        self.name = name
        self.ip = ip
        self.port = port
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_time: float = 0.0
        self.fps: float = 0.0
        self.frame_count: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _worker(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.RCVTIMEO, 1000)
        sub.connect(f"tcp://{self.ip}:{self.port}")

        last_fps_calc = time.time()
        fps_frames = 0

        while self._running:
            try:
                msg = sub.recv_string()
                data = json.loads(msg)
                images = data.get("images", {})
                if not images:
                    continue

                img_b64 = next(iter(images.values()))
                img_bytes = base64.b64decode(img_b64)
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    self.latest_frame = frame
                    self.latest_time = time.time()
                    self.frame_count += 1
                    fps_frames += 1

                now = time.time()
                if now - last_fps_calc >= 1.0:
                    self.fps = fps_frames / (now - last_fps_calc)
                    fps_frames = 0
                    last_fps_calc = now

            except zmq.Again:
                continue
            except Exception:
                time.sleep(0.01)

        sub.close(linger=0)
        ctx.term()


def parse_args():
    parser = argparse.ArgumentParser(description="Unitree G1 三路相机实时预览工具")
    parser.add_argument("--ip", type=str, default="10.3.42.221", help="相机服务 IP 地址")
    parser.add_argument("--head-port", type=int, default=5556, help="头部相机端口 (默认 5556)")
    parser.add_argument("--left-port", type=int, default=5557, help="左手腕相机端口 (默认 5557)")
    parser.add_argument("--right-port", type=int, default=5558, help="右手腕相机端口 (默认 5558)")
    parser.add_argument("--rerun", action="store_true", help="使用 Rerun 仪表板展示画面")
    parser.add_argument("--web", action="store_true", help="开启 Rerun Web 网页端查看 (适用于远程/无桌面)")
    parser.add_argument("--web-port", type=int, default=9090, help="Rerun Web 端口 (默认 9090)")
    parser.add_argument("--no-gui", action="store_true", help="纯控制台打印帧率统计，不弹窗")
    return parser.parse_args()


def main():
    args = parse_args()

    cams = [
        ("头部全局 (Head)", ZMQCameraReceiver("global_view", args.ip, args.head_port)),
        ("左手腕 (Left Wrist)", ZMQCameraReceiver("left_wrist", args.ip, args.left_port)),
        ("右手腕 (Right Wrist)", ZMQCameraReceiver("right_wrist", args.ip, args.right_port)),
    ]

    print("=" * 70)
    print(" 正在启动三路相机 ZMQ 接收监听...")
    for label, cam in cams:
        print(f"  - [{label}]: tcp://{cam.ip}:{cam.port}")
        cam.start()
    print("=" * 70)

    rr_module = None
    if args.rerun:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
            rr_module = rr

            rr.init("g1_three_cameras_preview", spawn=not args.web)
            if args.web:
                rr.serve_web(open_browser=True, web_port=args.web_port)
                print(f"[*] Rerun Web Viewer 已开启: http://localhost:{args.web_port}")

            views = [
                rrb.Spatial2DView(origin="observation.images.global_view", name="头部主摄 (Global View)"),
                rrb.Spatial2DView(origin="observation.images.left_wrist", name="左手腕 (Left Wrist)"),
                rrb.Spatial2DView(origin="observation.images.right_wrist", name="右手腕 (Right Wrist)"),
            ]
            rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views)))
            print("[+] Rerun 蓝图已加载 (3 个相机网格视图)")
        except ImportError:
            print("[!] 未安装 rerun-sdk，降级为 OpenCV/终端模式。可运行 `pip install rerun-sdk` 安装。")
            args.rerun = False

    try:
        placeholder = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(placeholder, "WAITING...", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        while True:
            t0 = time.time()
            display_tiles = []

            for label, cam in cams:
                frame = cam.latest_frame
                is_fresh = (time.time() - cam.latest_time) < 1.0 if cam.latest_time > 0 else False

                if frame is not None and is_fresh:
                    # 统一缩放至 480x360 便于拼屏
                    tile = cv2.resize(frame, (480, 360))
                    # 在画面上标注文案与 FPS
                    status_text = f"{label} | {cam.fps:.1f} FPS"
                    color = (0, 255, 0)
                else:
                    tile = placeholder.copy()
                    status_text = f"{label} | NO SIGNAL"
                    color = (0, 0, 255)

                cv2.putText(tile, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                display_tiles.append(tile)

                # 如果开启了 Rerun，同步向 Rerun 推送 RGB 图像
                if args.rerun and rr_module is not None and frame is not None and is_fresh:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    entity_path = f"observation.images.{cam.name}"
                    rr_module.log(entity_path, rr_module.Image(rgb_frame))

            if not args.no_gui and not args.rerun:
                # 横向拼接 3 路画面: [头部 | 左腕 | 右腕]
                combo = np.hstack(display_tiles)
                cv2.imshow("Unitree G1 Three Cameras Preview (Press 'q' to quit)", combo)
                key = cv2.waitKey(20) & 0xFF
                if key == ord("q"):
                    break
            else:
                # 终端模式或 Rerun 模式：每 2 秒打印一次简报
                fps_summary = " | ".join([f"{label}: {cam.fps:.1f}fps" for label, cam in cams])
                print(f"\r[监控] {fps_summary}  (Ctrl+C 退出)", end="", flush=True)
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[*] 正在退出监控...")
    finally:
        for _, cam in cams:
            cam.stop()
        if not args.no_gui and not args.rerun:
            cv2.destroyAllWindows()
        print("[+] 监控已结束")


if __name__ == "__main__":
    main()
