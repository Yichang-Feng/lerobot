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
High-Performance Universal ZMQ Video Streamer for LeRobot Rollout.
Supports AV1, H.264, VP9 using PyAV (with OpenCV fallback).
Supports precise time/frame offsets (--start_time, --end_time, --start_frame, --end_frame).
"""

import argparse
import base64
import json
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
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


def encode_bgr_frame(frame_bgr, quality: int = 80) -> str:
    """Encodes an OpenCV BGR numpy array into a Base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


class ZMQVideoStreamer:
    def __init__(
        self,
        video_path: str | Path,
        port: int = 5556,
        camera_names: list[str] | None = None,
        fps: float = 30.0,
        loop: bool = True,
        quality: int = 80,
        start_time: str | float | None = None,
        end_time: str | float | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ):
        self.video_path = Path(video_path).expanduser().resolve()
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.port = port
        self.camera_names = camera_names or ["head_camera", "global_view"]
        self.fps = fps
        self.interval = 1.0 / fps if fps > 0 else 0.0333
        self.loop = loop
        self.quality = quality

        self.start_sec = parse_time_str(start_time) or 0.0
        self.end_sec = parse_time_str(end_time)
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.running = False
        self.context = None
        self.socket = None

    def _init_zmq(self):
        """Initializes the ZMQ PUB socket."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"ZMQ PUB socket bound to tcp://*:{self.port}")

    def publish_frame(self, img_dict: dict[str, str], ts: float):
        """Publishes a single frame JSON packet over ZMQ."""
        payload = {
            "timestamps": {cam: ts for cam in self.camera_names},
            "images": img_dict,
        }
        for cam, b64_img in img_dict.items():
            payload[cam] = b64_img

        json_str = json.dumps(payload)
        try:
            self.socket.send_string(json_str, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _stream_with_pyav(self):
        """Streams video using PyAV with seeking and timestamp filtering."""
        logger.info("Using PyAV video decoding backend (libdav1d / ffmpeg)")
        cycle_count = 0
        total_published = 0
        stream_start_time = time.perf_counter()

        while self.running:
            cycle_count += 1
            start_desc = f"{self.start_sec:.1f}s" if self.start_sec > 0 else "0.0s"
            end_desc = f"{self.end_sec:.1f}s" if self.end_sec else "end"
            logger.info(f"--- Starting video cycle #{cycle_count} (Range: {start_desc} -> {end_desc}) ---")

            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                total_frames = stream.frames or "unknown"
                native_fps = float(stream.average_rate) if stream.average_rate else 30.0

                if cycle_count == 1:
                    logger.info(
                        f"Video opened: {self.video_path.name} | Codec: {stream.codec_context.name} | "
                        f"Resolution: {stream.width}x{stream.height} | Total Frames: {total_frames} | Native FPS: {native_fps:.1f}"
                    )

                # Seek to start time if requested
                if self.start_sec > 0:
                    seek_target = int(self.start_sec / stream.time_base)
                    container.seek(seek_target, stream=stream, backward=True)

                raw_frame_idx = 0
                published_in_cycle = 0

                for frame in container.decode(video=0):
                    if not self.running:
                        break

                    raw_frame_idx += 1
                    cur_time = float(frame.pts * stream.time_base) if frame.pts is not None else (raw_frame_idx / native_fps)

                    # Filter by start time
                    if cur_time < self.start_sec - 0.01:
                        continue

                    # Filter by start frame
                    if self.start_frame is not None and raw_frame_idx < self.start_frame:
                        continue

                    # Filter by end time
                    if self.end_sec is not None and cur_time > self.end_sec:
                        logger.info(f"Reached end time ({self.end_sec:.1f}s). Looping.")
                        break

                    # Filter by end frame
                    if self.end_frame is not None and raw_frame_idx > self.end_frame:
                        logger.info(f"Reached end frame ({self.end_frame}). Looping.")
                        break

                    t0 = time.perf_counter()
                    frame_bgr = frame.to_ndarray(format="bgr24")
                    encoded_jpg = encode_bgr_frame(frame_bgr, quality=self.quality)
                    img_dict = {cam: encoded_jpg for cam in self.camera_names}
                    ts = time.time()
                    self.publish_frame(img_dict, ts)

                    published_in_cycle += 1
                    total_published += 1

                    if published_in_cycle % 60 == 0:
                        elapsed_total = time.perf_counter() - stream_start_time
                        cur_fps = total_published / elapsed_total if elapsed_total > 0 else 0
                        logger.info(
                            f"Cycle #{cycle_count} | Video Time: {cur_time:.1f}s | "
                            f"Frames Pub: {published_in_cycle} | Avg FPS: {cur_fps:.1f}"
                        )

                    elapsed = time.perf_counter() - t0
                    sleep_time = self.interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if not self.loop:
                logger.info("Video stream finished (loop=False). Stopping.")
                break

    def run(self):
        """Main streaming entry point."""
        self._init_zmq()
        self.running = True
        logger.info(
            f"Streaming at {self.fps} Hz on port {self.port} "
            f"(camera_names={self.camera_names}, loop={self.loop}, start_time={self.start_sec}s)..."
        )

        try:
            if HAS_AV:
                self._stream_with_pyav()
            else:
                raise RuntimeError("PyAV is required for streaming. Please ensure 'av' is installed in the conda environment.")
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """Cleans up ZMQ resources."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
        logger.info("ZMQVideoStreamer stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Universal ZMQ Video Streamer for LeRobot Rollout")
    parser.add_argument(
        "--video_path",
        type=str,
        default="datasets/box_pick/videos/observation.images.global_view/chunk-000/file-002.mp4",
        help="Path to MP4 video file to stream",
    )
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (default: 5556)")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["head_camera", "global_view"],
        help="List of camera names/topics to publish",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Publish framerate in Hz (default: 30.0)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop video playback indefinitely")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop video")
    parser.add_argument("--quality", type=int, default=80, help="JPEG encoding quality (default: 80)")
    parser.add_argument(
        "--start_time",
        type=str,
        default="3:10",
        help="Start time offset in seconds or MM:SS (default: '3:10' / 190s)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default=None,
        help="End time offset in seconds or MM:SS (default: None, plays until end)",
    )
    parser.add_argument("--start_frame", type=int, default=None, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    return parser.parse_args()


def main():
    args = parse_args()
    streamer = ZMQVideoStreamer(
        video_path=args.video_path,
        port=args.port,
        camera_names=args.camera_names,
        fps=args.fps,
        loop=args.loop,
        quality=args.quality,
        start_time=args.start_time,
        end_time=args.end_time,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
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
