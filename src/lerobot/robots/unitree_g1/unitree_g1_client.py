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

from __future__ import annotations

import json
import logging
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import zmq

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.zmq import ZMQCameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)

logger = logging.getLogger(__name__)


def _default_unitree_g1_client_cameras(server_address: str = "localhost", port: int = 5556) -> dict[str, CameraConfig]:
    return {
        "global_view": ZMQCameraConfig(
            server_address=server_address,
            port=port,
            camera_name="head_camera",
            width=1280,
            height=720,
            fps=30,
            warmup_s=5,
        )
    }


@RobotConfig.register_subclass("unitree_g1_client")
@dataclass
class UnitreeG1ClientConfig(RobotConfig):
    """Configuration for decoupled Unitree G1 robot client.

    Connects to an external G1 locomotion server (e.g. run_g1_locomotion_server.py)
    via ZeroMQ.
    """

    # Target host IP where the locomotion/state server is running
    robot_ip: str = "localhost"

    # Destination host IP where actions should be sent (defaults to robot_ip if empty)
    action_ip: str = ""

    # Source host IP where camera stream is running (defaults to robot_ip if empty)
    camera_ip: str = "localhost"

    # ZMQ port for receiving 29-DoF lowstate (PUB on server, SUB here)
    state_port: int = 6001

    # ZMQ port for sending 18-DoF actions (PULL on server, PUSH here)
    action_port: int = 6002

    # Camera streaming port
    camera_port: int = 5556
    camera_name: str = "head_camera"

    # Cameras dict (defaults to ZMQ camera connected to camera_ip:camera_port)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Control loop timestep for the client
    control_dt: float = 1.0 / 25.0

    # Default joint positions (for reference / reset)
    default_positions: list[float] = field(default_factory=lambda: [0.0] * NUM_MOTORS)

    # Timeout in seconds before issuing warning (or raising if wait_until_ready=False)
    connect_timeout: float = 60.0

    # If True, keep waiting and listening for locomotion server instead of exiting/crashing,
    # preserving pre-loaded policy in GPU memory.
    wait_until_ready: bool = True

    # If True, mock the 29-DoF joint lowstate with default zero values instead of waiting for locomotion server.
    # Enables camera-only rollout for pure visual inference testing without robot motors.
    mock_state: bool = False

    def __post_init__(self) -> None:
        if not self.action_ip:
            self.action_ip = self.robot_ip
        if not self.camera_ip:
            self.camera_ip = self.robot_ip

        if not self.cameras:
            self.cameras = _default_unitree_g1_client_cameras(
                server_address=self.camera_ip,
                port=self.camera_port,
            )
        elif "global_view" in self.cameras:
            # Sync server address and port if user specified them on CLI
            cam = self.cameras["global_view"]
            if hasattr(cam, "server_address") and (not cam.server_address or cam.server_address == "localhost"):
                cam.server_address = self.camera_ip
            if hasattr(cam, "port") and self.camera_port != 5556:
                cam.port = self.camera_port


class UnitreeG1Client(Robot):
    """Decoupled client for Unitree G1.

    This client does NOT run MuJoCo or the 50Hz GROOT locomotion controller locally.
    Instead, it communicates over ZeroMQ with a remote or separate server process
    running `run_g1_locomotion_server.py`.

    - Subscribes to 29-DoF joint state + IMU on `tcp://{robot_ip}:{state_port}`
    - Subscribes to camera feed on `tcp://{robot_ip}:{camera_port}`
    - Pushes 18-DoF actions (14-DoF arms + 4-DoF remote axes) to `tcp://{robot_ip}:{action_port}`
    """

    config_class = UnitreeG1ClientConfig
    name = "unitree_g1_client"

    def __init__(self, config: UnitreeG1ClientConfig):
        super().__init__(config)
        self.config = config
        self.control_dt = config.control_dt

        self._cameras = make_cameras_from_configs(config.cameras)

        self._ctx: zmq.Context | None = None
        self._state_sub: zmq.Socket | None = None
        self._action_push: zmq.Socket | None = None

        self._latest_state: dict[str, Any] | None = None
        self._state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._state_thread: threading.Thread | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Motor joint positions for all 29 joints + camera RGB features."""
        proprio_ft = {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}
        cameras_ft: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                cameras_ft[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                cameras_ft[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return {**proprio_ft, **cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """14-DoF arm joint targets + 4-DoF remote joystick axes."""
        arm_features = {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}
        remote_features = dict.fromkeys(REMOTE_AXES, float)
        return {**arm_features, **remote_features}

    @property
    def cameras(self) -> dict:
        return self._cameras

    @property
    def is_connected(self) -> bool:
        with self._state_lock:
            return self._latest_state is not None and not self._shutdown_event.is_set()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _init_sockets(self) -> None:
        """Initialize or re-initialize ZMQ sockets for state subscriber and action pusher."""
        ctx = zmq.Context.instance()
        self._ctx = ctx

        if self._state_sub is not None:
            try:
                self._state_sub.close(linger=0)
            except Exception:
                pass
            self._state_sub = None

        self._state_sub = ctx.socket(zmq.SUB)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._state_sub.connect(f"tcp://{self.config.robot_ip}:{self.config.state_port}")

        if self._action_push is not None:
            try:
                self._action_push.close(linger=0)
            except Exception:
                pass
            self._action_push = None

        self._action_push = ctx.socket(zmq.PUSH)
        self._action_push.setsockopt(zmq.CONFLATE, 1)
        self._action_push.connect(f"tcp://{self.config.action_ip}:{self.config.action_port}")

    def _start_state_thread(self) -> None:
        """Start the background state receiving thread if not already running."""
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._state_thread = threading.Thread(target=self._state_reader_loop, daemon=True)
        self._state_thread.start()

    def _update_target_ip(self, new_ip: str) -> None:
        """Dynamically update target IP without restarting or reloading the model."""
        clean_ip = new_ip.strip()
        for prefix in ("--robot_ip=", "--robot.robot_ip=", "robot_ip=", "http://", "tcp://"):
            if clean_ip.startswith(prefix):
                clean_ip = clean_ip[len(prefix):]
        if ":" in clean_ip:
            clean_ip = clean_ip.split(":")[0]

        if not clean_ip:
            return

        old_ip = self.config.robot_ip
        self.config.robot_ip = clean_ip
        if self.config.action_ip in (old_ip, "localhost", "127.0.0.1", "") or not self.config.action_ip:
            self.config.action_ip = clean_ip

        if self.config.camera_ip in (old_ip, "localhost", "127.0.0.1", "") or not self.config.camera_ip:
            self.config.camera_ip = clean_ip
            for cam in self._cameras.values():
                if hasattr(cam, "server_address"):
                    cam.server_address = clean_ip
                if hasattr(cam, "disconnect"):
                    try:
                        cam.disconnect()
                    except Exception:
                        pass

        # Re-initialize sockets with new IP
        self._init_sockets()
        self._try_connect_cameras()
        logger.info(
            "🔄 [UnitreeG1Client] 目标 IP 已热切换为: %s (State: %d, Action: %d, Camera: %d)",
            clean_ip,
            self.config.state_port,
            self.config.action_port,
            self.config.camera_port,
        )

    def _check_stdin(self, timeout: float = 0.0) -> str | None:
        """Check for user input on stdin without blocking."""
        try:
            if sys.stdin:
                rlist, _, _ = select.select([sys.stdin], [], [], timeout)
                if rlist:
                    return sys.stdin.readline()
        except Exception:
            pass
        return None

    def _all_cameras_have_frames(self) -> bool:
        """Check if all configured cameras have received at least one valid frame."""
        for cam in self._cameras.values():
            if hasattr(cam, "latest_frame"):
                frame_lock = getattr(cam, "frame_lock", None)
                if frame_lock:
                    with frame_lock:
                        if cam.latest_frame is None:
                            return False
                elif cam.latest_frame is None:
                    return False
            elif not getattr(cam, "is_connected", False):
                return False
        return True

    def _try_connect_cameras(self) -> bool:
        """Attempt to connect any disconnected cameras or resurrect dead camera threads."""
        for cam_name, cam in self._cameras.items():
            thread_dead = hasattr(cam, "thread") and (cam.thread is None or not cam.thread.is_alive())
            not_connected = not getattr(cam, "is_connected", False) or thread_dead
            if not_connected:
                try:
                    if hasattr(cam, "disconnect"):
                        try:
                            cam.disconnect()
                        except Exception:
                            pass
                    if hasattr(cam, "connect"):
                        try:
                            cam.connect(warmup=False)
                        except TypeError:
                            cam.connect()
                    logger.info(f"🔄 [UnitreeG1Client] 尝试连通机载相机 [{cam_name}] (tcp://{self.config.camera_ip}:{self.config.camera_port})...")
                except Exception as e:
                    logger.debug("Camera connect attempt for %s: %s", cam_name, e)

        return self._all_cameras_have_frames()

    def _state_reader_loop(self) -> None:
        """Background thread to continuously pull the newest lowstate from the server."""
        while not self._shutdown_event.is_set():
            sock = self._state_sub
            if sock is None:
                time.sleep(0.05)
                continue
            try:
                if sock.poll(timeout=100):
                    payload = sock.recv(zmq.NOBLOCK)
                    msg = json.loads(payload.decode("utf-8"))
                    with self._state_lock:
                        self._latest_state = msg
            except zmq.ContextTerminated:
                break
            except (zmq.ZMQError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger.debug("Exception in state_reader_loop: %s", e)
                time.sleep(0.05)

    def connect(self, calibrate: bool = True) -> None:
        self._init_sockets()
        self._start_state_thread()
        self._try_connect_cameras()

        if self.config.mock_state:
            mock_motors = {
                G1_29_JointIndex(i).name: {"q": float(self.config.default_positions[i]), "dq": 0.0, "tau": 0.0}
                for i in range(NUM_MOTORS)
            }
            with self._state_lock:
                self._latest_state = {
                    "motors": mock_motors,
                    "imu": {
                        "gyroscope": [0.0, 0.0, 0.0],
                        "accelerometer": [0.0, 0.0, 9.81],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "rpy": [0.0, 0.0, 0.0],
                    },
                    "timestamp": time.time(),
                }
            logger.info("ℹ️  [UnitreeG1Client] 纯视觉 (Camera-Only) 模式已启用：关节使用虚拟零位，仅监听机载相机。")

        logger.info("=" * 80)
        logger.info(" [VLA 策略模型已加载至 GPU 显存]")
        logger.info("-" * 80)
        logger.info("当前网络接入监听配置:")
        if self.config.mock_state:
            logger.info("  • 机器人状态 (State):  [虚拟零位 / Mock] (纯视觉测试，不依赖 6001 端口)")
            logger.info("  • 动作下发   (Action): [已旁路 / Bypassed] (不下发到物理电机)")
        else:
            logger.info("  • 机器人状态 (State):  tcp://%s:%d", self.config.robot_ip, self.config.state_port)
            logger.info("  • 动作下发   (Action): tcp://%s:%d", self.config.action_ip, self.config.action_port)
        logger.info("  • 机载视觉   (Camera): tcp://%s:%d", self.config.camera_ip, self.config.camera_port)
        logger.info("-" * 80)
        logger.info("⏳ 正在等待机器人与视觉服务接入 (检测到数据流将自动接入并启动推理)...")
        logger.info("💡 提示: 模型常驻显存，连接未就绪也不会中断退出！可在机器人端随时启动服务端。")
        logger.info("   [交互指令] 终端直接输入新 IP + 回车可热切换目标地址；输入 'q' + 回车退出")
        logger.info("=" * 80)

        start_time = time.time()
        last_log_time = start_time
        last_cam_retry_time = start_time
        warned_timeout = False

        while True:
            # 1. Check if state packet arrived
            with self._state_lock:
                has_state = self._latest_state is not None

            # 2. Check and periodically retry cameras
            now = time.time()
            if now - last_cam_retry_time >= 2.0:
                has_camera = self._try_connect_cameras()
                last_cam_retry_time = now
            else:
                has_camera = self._all_cameras_have_frames()

            # If state is ready and cameras have frames, connection complete!
            if has_state and has_camera:
                break

            # 3. Check for non-blocking stdin input (hot-switching IP or quit)
            user_line = self._check_stdin(timeout=0.05)
            if user_line is not None:
                cmd = user_line.strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    logger.info("[UnitreeG1Client] 用户请求退出程序。")
                    raise KeyboardInterrupt("User requested exit.")
                elif cmd:
                    if cmd.startswith("--camera_ip=") or cmd.startswith("camera_ip="):
                        val = cmd.split("=", 1)[1].strip()
                        if ":" in val:
                            new_cam_ip, new_port_str = val.split(":", 1)
                            self.config.camera_ip = new_cam_ip
                            self.config.camera_port = int(new_port_str)
                        else:
                            self.config.camera_ip = val
                        for cam in self._cameras.values():
                            if hasattr(cam, "server_address"):
                                cam.server_address = self.config.camera_ip
                            if hasattr(cam, "port"):
                                cam.port = self.config.camera_port
                            if hasattr(cam, "disconnect"):
                                try:
                                    cam.disconnect()
                                except Exception:
                                    pass
                        logger.info("🔄 [UnitreeG1Client] 相机目标已切换为: tcp://%s:%d", self.config.camera_ip, self.config.camera_port)
                    elif cmd.startswith("--camera_port=") or cmd.startswith("camera_port="):
                        new_port = int(cmd.split("=", 1)[1].strip())
                        self.config.camera_port = new_port
                        for cam in self._cameras.values():
                            if hasattr(cam, "port"):
                                cam.port = new_port
                                try:
                                    cam.disconnect()
                                except Exception:
                                    pass
                        logger.info("🔄 [UnitreeG1Client] 相机端口已切换为: %d", new_port)
                    else:
                        self._update_target_ip(cmd)
                    start_time = time.time()
                    last_log_time = start_time
                    warned_timeout = False
                else:
                    # User pressed Enter, print rich immediate diagnostic
                    cam_details = []
                    for name, cam in self._cameras.items():
                        c_alive = getattr(cam, "thread", None) is not None and cam.thread.is_alive()
                        c_frame = getattr(cam, "latest_frame", None) is not None
                        cam_details.append(f"[{name}] tcp://{self.config.camera_ip}:{self.config.camera_port} (线程:{'运行' if c_alive else '未起'}, 首帧:{'已收到' if c_frame else '未收到'})")
                    logger.info(
                        "🔍 [实时检测] 状态源(6001): %s (tcp://%s:%d) | 视觉源: %s",
                        "✅ 正常" if has_state else "❌ 等待数据",
                        self.config.robot_ip,
                        self.config.state_port,
                        ", ".join(cam_details),
                    )

            # 4. Periodic progress logging every 5s
            now = time.time()
            elapsed = now - start_time
            if now - last_log_time >= 5.0:
                last_log_time = now
                state_str = "✅ 收到数据" if has_state else "⏳ 等待数据包..."
                cam_str = "✅ 图像就绪" if has_camera else "⏳ 等待图像帧..."
                logger.info(
                    "⏳ [等待接入] 状态: %s (tcp://%s:%d) | 视觉: %s (tcp://%s:%d) | 已等待 %.0fs (回车查状态 / 输入新IP切换 / q退出)",
                    state_str,
                    self.config.robot_ip,
                    self.config.state_port,
                    cam_str,
                    self.config.camera_ip,
                    self.config.camera_port,
                    elapsed,
                )

            # 5. Handle timeout warning or strict error
            if elapsed > self.config.connect_timeout and not warned_timeout:
                warned_timeout = True
                if not self.config.wait_until_ready:
                    raise TimeoutError(
                        f"Timed out ({int(self.config.connect_timeout)}s) waiting for state from locomotion server at "
                        f"tcp://{self.config.robot_ip}:{self.config.state_port}."
                    )
                logger.warning("-" * 80)
                logger.warning(
                    " 已等待超过 %.0f 秒，仍未连通机器人！(状态: %s, 视觉: %s)",
                    self.config.connect_timeout,
                    "已就绪" if has_state else "未收到包",
                    "已就绪" if has_camera else "未收到帧",
                )
                logger.warning("💡 模型依然保存在显存中！请在目标机器人启动 run_g1_locomotion_server.py。")
                logger.warning("   程序将持续保持后台等待并自动接入；或在此直接输入正确 IP 并按回车。")
                logger.warning("-" * 80)

            time.sleep(0.05)

        logger.info("=" * 80)
        logger.info("✅ [UnitreeG1Client] 机器人与视觉服务接入成功！")
        logger.info("  • 机器人状态: 已连接 (tcp://%s:%d)", self.config.robot_ip, self.config.state_port)
        logger.info("  • 动作下发:   已连接 (tcp://%s:%d)", self.config.action_ip, self.config.action_port)
        logger.info("  • 机载视觉:   已连接 (tcp://%s:%d)", self.config.camera_ip, self.config.camera_port)
        logger.info("开始执行 VLA 策略推理...")
        logger.info("=" * 80)

    def disconnect(self) -> None:
        logger.info("Disconnecting UnitreeG1Client...")
        self._shutdown_event.set()

        # Send stop command to server so server knows client exited
        if self._action_push is not None:
            try:
                stop_msg = json.dumps({"cmd": "stop", "timestamp": time.time()}).encode("utf-8")
                self._action_push.send(stop_msg, flags=zmq.NOBLOCK)
            except Exception:
                pass

        # Disconnect cameras
        for cam in self._cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting camera: %s", e)

        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None

        if self._state_sub is not None:
            self._state_sub.close(linger=0)
            self._state_sub = None

        if self._action_push is not None:
            self._action_push.close(linger=0)
            self._action_push = None

        logger.info("[UnitreeG1Client] Disconnected.")

    def get_observation(self) -> RobotObservation:
        with self._state_lock:
            state = self._latest_state

        if state is None:
            return {}

        obs = {}

        # 1. Parse motor positions, velocities, torques
        motors = state.get("motors", {})
        for motor in G1_29_JointIndex:
            name = motor.name
            m_data = motors.get(name, {})
            obs[f"{name}.q"] = float(m_data.get("q", 0.0))
            obs[f"{name}.dq"] = float(m_data.get("dq", 0.0))
            obs[f"{name}.tau"] = float(m_data.get("tau", 0.0))

        # 2. Parse IMU state
        imu = state.get("imu", {})
        gyro = imu.get("gyroscope", [0.0, 0.0, 0.0])
        accel = imu.get("accelerometer", [0.0, 0.0, 0.0])
        quat = imu.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        rpy = imu.get("rpy", [0.0, 0.0, 0.0])

        obs["imu.gyro.x"] = gyro[0]
        obs["imu.gyro.y"] = gyro[1]
        obs["imu.gyro.z"] = gyro[2]

        obs["imu.accel.x"] = accel[0]
        obs["imu.accel.y"] = accel[1]
        obs["imu.accel.z"] = accel[2]

        obs["imu.quat.w"] = quat[0]
        obs["imu.quat.x"] = quat[1]
        obs["imu.quat.y"] = quat[2]
        obs["imu.quat.z"] = quat[3]

        obs["imu.rpy.roll"] = rpy[0]
        obs["imu.rpy.pitch"] = rpy[1]
        obs["imu.rpy.yaw"] = rpy[2]

        # 3. Read camera image frames
        for cam_name, cam in self._cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Forward action dict to locomotion server via ZMQ PUSH."""
        if self._action_push is None or self.config.mock_state:
            return action

        # Clean action dictionary with only serializable floats
        serializable_action = {}
        for k, v in action.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                serializable_action[k] = float(v)

        msg = {
            "cmd": "action",
            "action": serializable_action,
            "timestamp": time.time(),
        }
        try:
            payload = json.dumps(msg).encode("utf-8")
            self._action_push.send(payload, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.debug("Failed to send action via ZMQ: %s", e)

        return action

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:
        """Request the locomotion server to reset arms to default position."""
        if self._action_push is not None:
            logger.info("[UnitreeG1Client] Sending RESET command to locomotion server...")
            try:
                # Zero out any remote locomotion commands first
                zero_remote = {k: 0.0 for k in REMOTE_AXES}
                self.send_action(zero_remote)

                reset_msg = json.dumps({"cmd": "reset", "timestamp": time.time()}).encode("utf-8")
                self._action_push.send(reset_msg, flags=zmq.NOBLOCK)
                time.sleep(0.5)  # brief pause to allow server to start homing
            except Exception as e:
                logger.warning("Failed to send reset command: %s", e)
