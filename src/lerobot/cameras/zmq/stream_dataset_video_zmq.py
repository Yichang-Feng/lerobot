#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
High-Performance Universal ZMQ Multi-Camera Video Streamer for LeRobot Rollout.
Supports:
1. Native LeRobotDataset multi-camera streaming (--dataset-root <path> --episode <idx>)
   with perfect time-lockstep across cam_high (5556), cam_left_wrist (5557), cam_right_wrist (5558).
2. Explicit multi-video streaming (--video-path <high> --left-video-path <left> --right-video-path <right>).
3. Single video broadcasting across all ports as fallback.
4. Auto-reconnect and infinite looping at accurate target FPS.
"""

import os
import sys

# Ensure conda env C++ libs are prioritized to prevent CXXABI errors
py_lib = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "lib"))
if os.path.exists(os.path.join(py_lib, "libstdc++.so.6")):
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if py_lib not in cur_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{py_lib}:{cur_ld}" if cur_ld else py_lib

import argparse
import base64
import json
import logging
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ZMQVideoStreamer")


def parse_time_str(val: str | float | None) -> float | None:
    """Parses time specifications like '190', '3:10', '03:10', '01:05:30' into seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if ":" in val_str:
        parts = val_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(val_str)


def encode_bgr_frame(frame_bgr: np.ndarray, quality: int = 80) -> str:
    """Encodes an OpenCV BGR numpy array into a Base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


class ZMQVideoStreamer:
    def __init__(
        self,
        video_path: str | Path | None = None,
        left_video_path: str | Path | None = None,
        right_video_path: str | Path | None = None,
        dataset_root: str | Path | None = None,
        episode_idx: int = 0,
        port: int = 5556,
        left_wrist_port: int = 5557,
        right_wrist_port: int = 5558,
        multi_port: bool = True,
        fps: float = 30.0,
        loop: bool = True,
        quality: int = 80,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else None
        self.episode_idx = episode_idx
        self.video_path = Path(video_path).expanduser().resolve() if video_path else None
        self.left_video_path = Path(left_video_path).expanduser().resolve() if left_video_path else None
        self.right_video_path = Path(right_video_path).expanduser().resolve() if right_video_path else None

        self.port = port
        self.left_wrist_port = left_wrist_port
        self.right_wrist_port = right_wrist_port
        self.multi_port = multi_port

        self.fps = fps
        self.interval = 1.0 / fps if fps > 0 else 0.0333
        self.loop = loop
        self.quality = quality

        self.running = False
        self.context = None
        self.sockets = {}

    def _init_zmq(self):
        """Initializes the ZMQ PUB sockets for all camera streams."""
        self.context = zmq.Context()
        self.sockets = {}

        # 1. Main / Head camera socket (Port 5556)
        s_main = self.context.socket(zmq.PUB)
        s_main.setsockopt(zmq.SNDHWM, 20)
        s_main.setsockopt(zmq.LINGER, 0)
        s_main.bind(f"tcp://*:{self.port}")
        self.sockets["main"] = s_main
        logger.info(f"✓ ZMQ Head Camera (global_view) bound to tcp://*:{self.port}")

        # 2. Left wrist socket (Port 5557)
        if self.multi_port and self.left_wrist_port:
            s_lw = self.context.socket(zmq.PUB)
            s_lw.setsockopt(zmq.SNDHWM, 20)
            s_lw.setsockopt(zmq.LINGER, 0)
            s_lw.bind(f"tcp://*:{self.left_wrist_port}")
            self.sockets["left_wrist"] = s_lw
            logger.info(f"✓ ZMQ Left Wrist Camera bound to tcp://*:{self.left_wrist_port}")

        # 3. Right wrist socket (Port 5558)
        if self.multi_port and self.right_wrist_port:
            s_rw = self.context.socket(zmq.PUB)
            s_rw.setsockopt(zmq.SNDHWM, 20)
            s_rw.setsockopt(zmq.LINGER, 0)
            s_rw.bind(f"tcp://*:{self.right_wrist_port}")
            self.sockets["right_wrist"] = s_rw
            logger.info(f"✓ ZMQ Right Wrist Camera bound to tcp://*:{self.right_wrist_port}")

    def publish_frame(
        self,
        main_b64: str,
        left_b64: str | None = None,
        right_b64: str | None = None,
        ts: float | None = None,
    ):
        """Publishes camera frames across ZMQ ports formatted for ZMQCamera clients."""
        if ts is None:
            ts = time.time()

        if left_b64 is None:
            left_b64 = main_b64
        if right_b64 is None:
            right_b64 = main_b64

        # 1. Head / Main socket payload (port 5556)
        payload_main = {
            "timestamps": {"head_camera": ts, "global_view": ts, "left_wrist": ts, "right_wrist": ts},
            "images": {
                "head_camera": main_b64,
                "global_view": main_b64,
                "cam_high": main_b64,
                "left_wrist": left_b64,
                "right_wrist": right_b64,
            },
            "head_camera": main_b64,
            "global_view": main_b64,
        }
        try:
            self.sockets["main"].send_string(json.dumps(payload_main), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

        # 2. Left wrist socket payload (port 5557)
        if "left_wrist" in self.sockets:
            payload_left = {
                "timestamps": {"left_wrist": ts, "cam_left_wrist": ts},
                "images": {
                    "left_wrist": left_b64,
                    "cam_left_wrist": left_b64,
                },
                "left_wrist": left_b64,
            }
            try:
                self.sockets["left_wrist"].send_string(json.dumps(payload_left), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

        # 3. Right wrist socket payload (port 5558)
        if "right_wrist" in self.sockets:
            payload_right = {
                "timestamps": {"right_wrist": ts, "cam_right_wrist": ts},
                "images": {
                    "right_wrist": right_b64,
                    "cam_right_wrist": right_b64,
                },
                "right_wrist": right_b64,
            }
            try:
                self.sockets["right_wrist"].send_string(json.dumps(payload_right), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

    def _stream_dataset(self):
        """Streams native multi-camera dataset in 100% frame-level lockstep."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_name = self.dataset_root.name
        logger.info(f"Loading LeRobot dataset '{repo_name}' from: {self.dataset_root}")
        ds = LeRobotDataset(repo_name, root=str(self.dataset_root))

        total_episodes = ds.num_episodes
        ep_idx = min(self.episode_idx, total_episodes - 1)

        # Calculate episode frame range
        from_idx = sum(ds.meta.episodes[i]["length"] for i in range(ep_idx))
        ep_len = ds.meta.episodes[ep_idx]["length"]
        to_idx = from_idx + ep_len

        logger.info(f"Selected Episode #{ep_idx} (Frames {from_idx} -> {to_idx}, total {ep_len} frames)")

        # Find camera feature names
        high_key = "observation.images.cam_high"
        left_key = "observation.images.cam_left_wrist"
        right_key = "observation.images.cam_right_wrist"

        cycle = 0
        total_pub = 0
        t_start = time.perf_counter()

        while self.running:
            cycle += 1
            logger.info(f"--- [Cycle #{cycle}] Starting playback of Episode #{ep_idx} ---")
            cycle_pub = 0

            for f_idx in range(from_idx, to_idx):
                if not self.running:
                    break
                t0 = time.perf_counter()
                item = ds[f_idx]

                # Convert torch tensors (C, H, W in [0, 1]) to OpenCV BGR uint8
                t_h = item[high_key]
                t_l = item[left_key] if left_key in item else t_h
                t_r = item[right_key] if right_key in item else t_h

                bgr_h = cv2.cvtColor((t_h.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_l = cv2.cvtColor((t_l.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_r = cv2.cvtColor((t_r.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

                b64_h = encode_bgr_frame(bgr_h, quality=self.quality)
                b64_l = encode_bgr_frame(bgr_l, quality=self.quality)
                b64_r = encode_bgr_frame(bgr_r, quality=self.quality)

                self.publish_frame(b64_h, b64_l, b64_r)

                cycle_pub += 1
                total_pub += 1

                if cycle_pub % 60 == 0:
                    fps_real = total_pub / (time.perf_counter() - t_start)
                    logger.info(f"Cycle #{cycle} | 3-Cam Synced Frames: {cycle_pub}/{ep_len} | Real-Time FPS: {fps_real:.1f}")

                sleep_time = self.interval - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self.loop:
                logger.info("Dataset episode playback completed (loop=False). Exiting.")
                break

    def _stream_video_files(self):
        """Streams from single or 3 video files using PyAV for AV1/H264 support."""
        if not HAS_AV:
            raise RuntimeError("PyAV is required for decoding AV1/H264 video files. Please run: pip install av")

        logger.info(f"Opening main video via PyAV: {self.video_path}")
        cycle = 0
        total_pub = 0
        t_start = time.perf_counter()

        while self.running:
            cycle += 1
            logger.info(f"--- [Cycle #{cycle}] Starting video playback loop ---")
            cycle_pub = 0

            container_main = av.open(str(self.video_path))
            stream_main = container_main.decode(video=0)

            container_left = av.open(str(self.left_video_path)) if (self.left_video_path and self.left_video_path.exists()) else None
            stream_left = container_left.decode(video=0) if container_left else None

            container_right = av.open(str(self.right_video_path)) if (self.right_video_path and self.right_video_path.exists()) else None
            stream_right = container_right.decode(video=0) if container_right else None

            while self.running:
                t0 = time.perf_counter()
                try:
                    f_main = next(stream_main)
                    frame_main = f_main.to_ndarray(format="bgr24")
                except StopIteration:
                    break

                frame_l = None
                if stream_left:
                    try:
                        f_l = next(stream_left)
                        frame_l = f_l.to_ndarray(format="bgr24")
                    except StopIteration:
                        pass

                frame_r = None
                if stream_right:
                    try:
                        f_r = next(stream_right)
                        frame_r = f_r.to_ndarray(format="bgr24")
                    except StopIteration:
                        pass

                b64_main = encode_bgr_frame(frame_main, quality=self.quality)
                b64_left = encode_bgr_frame(frame_l, quality=self.quality) if frame_l is not None else b64_main
                b64_right = encode_bgr_frame(frame_r, quality=self.quality) if frame_r is not None else b64_main

                self.publish_frame(b64_main, b64_left, b64_right)

                cycle_pub += 1
                total_pub += 1

                if cycle_pub % 60 == 0:
                    fps_real = total_pub / (time.perf_counter() - t_start)
                    logger.info(f"Cycle #{cycle} | Frames: {cycle_pub} | Real-Time FPS: {fps_real:.1f}")

                sleep_time = self.interval - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            container_main.close()
            if container_left:
                container_left.close()
            if container_right:
                container_right.close()

            if not self.loop:
                break

    def run(self):
        """Starts streaming."""
        self._init_zmq()
        self.running = True

        try:
            if self.dataset_root is not None:
                self._stream_dataset()
            elif self.video_path is not None:
                self._stream_video_files()
            else:
                raise ValueError("Must provide either --dataset-root or --video-path!")
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """Cleans up ZMQ resources."""
        self.running = False
        for s in self.sockets.values():
            s.close()
        self.sockets.clear()
        if self.context:
            self.context.term()
            self.context = None
        logger.info("ZMQVideoStreamer stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Universal ZMQ Multi-Camera Video Streamer for LeRobot Rollout")
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=str,
        default=None,
        help="Path to LeRobot dataset directory (e.g. datasets/g1_pick_put_dex1_0923)",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index when streaming from dataset (default: 0)")
    parser.add_argument(
        "--video-path",
        "--video_path",
        dest="video_path",
        type=str,
        default=None,
        help="Path to main/head camera video file",
    )
    parser.add_argument(
        "--left-video-path",
        "--left_video_path",
        "--video-left",
        dest="left_video_path",
        type=str,
        default=None,
        help="Path to left wrist camera video file",
    )
    parser.add_argument(
        "--right-video-path",
        "--right_video_path",
        "--video-right",
        dest="right_video_path",
        type=str,
        default=None,
        help="Path to right wrist camera video file",
    )
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port for head camera (default: 5556)")
    parser.add_argument("--left-wrist-port", "--left_wrist_port", dest="left_wrist_port", type=int, default=5557, help="ZMQ PUB port for left wrist (default: 5557)")
    parser.add_argument("--right-wrist-port", "--right_wrist_port", dest="right_wrist_port", type=int, default=5558, help="ZMQ PUB port for right wrist (default: 5558)")
    parser.add_argument("--no-wrist-cameras", dest="multi_port", action="store_false", help="Only stream on main port 5556")
    parser.add_argument("--fps", type=float, default=30.0, help="Publish framerate in Hz (default: 30.0)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop playback indefinitely (default: True)")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop video")
    parser.add_argument("--quality", type=int, default=80, help="JPEG encoding quality (default: 80)")
    return parser.parse_args()


def main():
    args = parse_args()
    streamer = ZMQVideoStreamer(
        video_path=args.video_path,
        left_video_path=args.left_video_path,
        right_video_path=args.right_video_path,
        dataset_root=args.dataset_root,
        episode_idx=args.episode,
        port=args.port,
        left_wrist_port=args.left_wrist_port,
        right_wrist_port=args.right_wrist_port,
        multi_port=args.multi_port,
        fps=args.fps,
        loop=args.loop,
        quality=args.quality,
    )

    def handle_signal(sig, frame):
        logger.info("Received termination signal.")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    streamer.run()


if __name__ == "__main__":
    main()
