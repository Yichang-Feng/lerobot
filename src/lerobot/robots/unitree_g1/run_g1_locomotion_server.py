#!/usr/bin/env python3

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
G1 Locomotion and Simulation Server.

This server runs in Terminal 1 and hosts:
1. G1 Robot hardware interface (via DDS) OR MuJoCo simulation environment.
2. Background 50Hz GROOT Locomotion Controller (Balance + Walk) for waist and legs.
3. Upper-body default pose holding and action execution.
4. ZMQ state broadcaster (PUB on port 6001) for 29-DoF lowstate.
5. ZMQ action receiver (PULL on port 6002) for receiving 18-DoF actions from VLA client.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
import threading
import time

import numpy as np
import zmq

from lerobot.cameras.zmq import ZMQCameraConfig
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointIndex,
)
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("g1_locomotion_server")


def state_publisher_loop(
    robot: UnitreeG1,
    state_sock: zmq.Socket,
    fps: float,
    shutdown_event: threading.Event,
) -> None:
    """Publish 29-DoF joint state and IMU to ZMQ clients at constant rate."""
    period = 1.0 / fps
    logger.info("State publisher thread started at %.1fHz", fps)

    while not shutdown_event.is_set():
        t_start = time.time()

        with robot._lowstate_lock:
            lowstate = robot._lowstate

        if lowstate is not None:
            motors_dict = {}
            for i in range(NUM_MOTORS):
                motor_name = G1_29_JointIndex(i).name
                m = lowstate.motor_state[i]
                motors_dict[motor_name] = {
                    "q": float(m.q) if m.q is not None else 0.0,
                    "dq": float(m.dq) if m.dq is not None else 0.0,
                    "tau": float(m.tau_est) if m.tau_est is not None else 0.0,
                }

            imu = lowstate.imu_state
            imu_dict = {
                "quaternion": list(imu.quaternion) if imu.quaternion is not None else [1.0, 0.0, 0.0, 0.0],
                "gyroscope": list(imu.gyroscope) if imu.gyroscope is not None else [0.0, 0.0, 0.0],
                "accelerometer": list(imu.accelerometer) if imu.accelerometer is not None else [0.0, 0.0, 0.0],
                "rpy": list(imu.rpy) if imu.rpy is not None else [0.0, 0.0, 0.0],
            }

            msg = {
                "motors": motors_dict,
                "imu": imu_dict,
                "mode_machine": int(lowstate.mode_machine),
                "timestamp": time.time(),
            }

            payload = json.dumps(msg).encode("utf-8")
            with contextlib.suppress(zmq.Again):
                state_sock.send(payload, zmq.NOBLOCK)

        elapsed = time.time() - t_start
        sleep_time = max(0.0, period - elapsed)
        time.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree G1 Locomotion & Simulation Server (Terminal 1)")
    parser.add_argument("--sim", action="store_true", default=True, help="Run in MuJoCo simulation mode (default)")
    parser.add_argument("--real", action="store_true", help="Run on real physical robot")
    parser.add_argument("--controller", type=str, default="GrootLocomotionController", help="Controller name")
    parser.add_argument("--robot-ip", type=str, default="192.168.123.164", help="G1 Robot IP (for real robot)")
    parser.add_argument("--state-port", type=int, default=6001, help="ZMQ PUB port for 29-DoF lowstate")
    parser.add_argument("--action-port", type=int, default=6002, help="ZMQ PULL port for 18-DoF actions")
    parser.add_argument("--camera-port", type=int, default=0, help="Camera port (default: 5556 for sim, 5555 for real)")
    parser.add_argument("--robot-scene", type=str, default="assets/scene_29dof.xml", help="MuJoCo scene XML")
    parser.add_argument("--locomotion-mode", type=str, default="stand", help="Initial locomotion mode: stand or walk")
    parser.add_argument("--zero-locomotion-cmd", action="store_true", default=False, help="Force zero velocity to WBC")
    parser.add_argument("--fps", type=float, default=50.0, help="State publisher rate in Hz (default: 50)")

    args = parser.parse_args()

    # Handle real vs sim flags
    is_simulation = not args.real
    cam_port = args.camera_port if args.camera_port > 0 else (5556 if is_simulation else 5555)

    print("=" * 80)
    print(" [Unitree G1 Locomotion & Simulation Server]")
    print(f" 运行模式     : {'MuJoCo 物理仿真' if is_simulation else f'物理实机 ({args.robot_ip})'}")
    print(f" 平衡控制器   : {args.controller} (50Hz)")
    print(f" 状态广播端口 : tcp://0.0.0.0:{args.state_port} (ZMQ PUB, {args.fps}Hz)")
    print(f" 动作接收端口 : tcp://0.0.0.0:{args.action_port} (ZMQ PULL)")
    print(f" 相机推流端口 : {cam_port} (ZMQ PUB)")
    print("=" * 80)

    # 1. Build UnitreeG1 configuration
    cam_address = "localhost" if is_simulation else args.robot_ip
    cameras_cfg = {
        "global_view": ZMQCameraConfig(
            server_address=cam_address,
            port=cam_port,
            camera_name="head_camera",
            width=640,
            height=480,
            fps=30,
            warmup_s=5,
        )
    }

    robot_config = UnitreeG1Config(
        is_simulation=is_simulation,
        robot_ip=args.robot_ip,
        controller=args.controller,
        locomotion_mode=args.locomotion_mode,
        zero_locomotion_cmd=args.zero_locomotion_cmd,
        robot_scene=args.robot_scene,
        cameras=cameras_cfg,
    )

    logger.info("Initializing UnitreeG1 instance...")
    robot = UnitreeG1(robot_config)
    robot.connect()
    logger.info("Robot connected. Initializing ZeroMQ communication channels...")

    # 2. Setup ZeroMQ sockets
    ctx = zmq.Context.instance()

    state_sock = ctx.socket(zmq.PUB)
    state_sock.setsockopt(zmq.CONFLATE, 1)
    state_sock.bind(f"tcp://0.0.0.0:{args.state_port}")

    action_sock = ctx.socket(zmq.PULL)
    action_sock.setsockopt(zmq.CONFLATE, 1)
    action_sock.bind(f"tcp://0.0.0.0:{args.action_port}")

    shutdown_event = threading.Event()

    def handle_signal(sig, frame):
        logger.info("Received exit signal (%s). Shutting down...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 3. Start state broadcast thread
    state_thread = threading.Thread(
        target=state_publisher_loop,
        args=(robot, state_sock, args.fps, shutdown_event),
        daemon=True,
    )
    state_thread.start()

    logger.info(
        "Locomotion Server is LIVE! Robot balance controller running. Waiting for VLA client on port %d...",
        args.action_port,
    )

    # 4. Main action dispatch loop
    last_action_time = 0.0
    vla_connected = False

    try:
        while not shutdown_event.is_set():
            # Non-blocking check for incoming actions
            if action_sock.poll(timeout=20):  # 20ms check
                try:
                    payload = action_sock.recv(zmq.NOBLOCK)
                    data = json.loads(payload.decode("utf-8"))
                    cmd = data.get("cmd", "action")

                    if cmd == "action":
                        action = data.get("action", {})
                        robot.send_action(action)
                        last_action_time = time.time()
                        if not vla_connected:
                            logger.info(">>> [VLA Client CONNECTED] Streaming 18-DoF actions to robot! <<<")
                            vla_connected = True

                    elif cmd == "reset":
                        logger.info(">>> [RESET REQUESTED] Returning arms smoothly to default position. <<<")
                        # Zero locomotion input
                        robot.send_action({k: 0.0 for k in REMOTE_AXES})
                        robot.reset()
                        vla_connected = False

                    elif cmd == "stop":
                        logger.info(">>> [STOP REQUESTED] VLA client disconnected. Halting locomotion velocity. <<<")
                        robot.send_action({k: 0.0 for k in REMOTE_AXES})
                        vla_connected = False

                except zmq.ContextTerminated:
                    break
                except Exception as e:
                    logger.debug("Error processing action packet: %s", e)

            # Watchdog timeout: if VLA stopped sending packets for > 1.0s, stop walking
            if vla_connected and (time.time() - last_action_time > 1.0):
                logger.warning(">>> [WATCHDOG] VLA packet timeout (>1.0s). Halting velocity and holding arms. <<<")
                robot.send_action({k: 0.0 for k in REMOTE_AXES})
                vla_connected = False

    finally:
        logger.info("Cleaning up Locomotion Server...")
        shutdown_event.set()
        state_thread.join(timeout=1.0)
        state_sock.close(linger=0)
        action_sock.close(linger=0)
        ctx.term()
        robot.disconnect()
        logger.info("Locomotion Server shutdown complete.")


if __name__ == "__main__":
    main()
