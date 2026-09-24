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

import ast
import json
import logging
import math
import re
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
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


class RobotMode(str, Enum):
    """Operation mode of the robot controller / gamepad."""
    NAV = "nav"          # 导航模式 (Navigation)
    GAMEPAD = "gamepad"  # 手柄控制模式 (Manual / Gamepad)
    VLA = "vla"          # VLA 控制模式 (VLA control)
    UNKNOWN = "unknown"


def _parse_robot_mode(raw_data: Any) -> RobotMode:
    """Parse raw bytes, string, multipart list, or dict received on mode_port into a RobotMode enum."""
    try:
        # Handle multipart ZMQ messages (list or tuple of frames)
        if isinstance(raw_data, (list, tuple)):
            for part in reversed(raw_data):
                parsed = _parse_robot_mode(part)
                if parsed != RobotMode.UNKNOWN:
                    return parsed
            # If individual parts didn't match, join them and parse
            joined = b" ".join(part if isinstance(part, bytes) else str(part).encode() for part in raw_data)
            return _parse_robot_mode(joined)

        if isinstance(raw_data, bytes):
            s = raw_data.decode("utf-8", errors="ignore").strip().strip("\x00\r\n\t ")
        else:
            s = str(raw_data).strip().strip("\x00\r\n\t ")
        if not s:
            return RobotMode.UNKNOWN

        # Check if it contains JSON payload
        json_obj = None
        if (s.startswith("{") and s.endswith("}")) or ("{" in s and "}" in s):
            try:
                first_brace = s.find("{")
                last_brace = s.rfind("}")
                json_candidate = s[first_brace : last_brace + 1]
                json_obj = json.loads(json_candidate)
            except Exception:
                json_obj = None

        extracted_values: list[str] = []
        if isinstance(json_obj, dict):
            # Prioritize candidate keys
            for key in (
                "mode", "robot_mode", "control_mode", "state", "status",
                "robot_state", "action", "action_mode", "current_mode",
                "val", "data", "msg", "type"
            ):
                if key in json_obj:
                    v = json_obj[key]
                    if isinstance(v, (str, int, float)):
                        extracted_values.append(str(v).lower())
                    elif isinstance(v, dict):
                        for sub_k in ("mode", "state", "name", "val"):
                            if sub_k in v:
                                extracted_values.append(str(v[sub_k]).lower())

            # Also collect all primitive values in the dict
            for v in json_obj.values():
                if isinstance(v, (str, int, float)):
                    extracted_values.append(str(v).lower())

        # Also append the raw string lowered
        s_lower = s.lower()
        extracted_values.append(s_lower)

        # Check for VLA
        vla_keywords = ("vla", "vla_control", "vla_mode", "vla_teleop", "autonomous")
        for val in extracted_values:
            if val == "2" or any(k in val for k in vla_keywords):
                return RobotMode.VLA

        # Check for NAV
        nav_keywords = ("nav", "navigation", "daohang", "dao_hang")
        for val in extracted_values:
            if val == "0" or any(k in val for k in nav_keywords):
                return RobotMode.NAV

        # Check for GAMEPAD / MANUAL / LOCO
        gamepad_keywords = ("gamepad", "manual", "shoubing", "shou_bing", "joystick", "loco", "teleop")
        for val in extracted_values:
            if val == "1" or any(k in val for k in gamepad_keywords):
                return RobotMode.GAMEPAD

    except Exception as e:
        logger.debug("Error parsing robot mode: %s", e)
    return RobotMode.UNKNOWN


def _parse_gripper_data(raw_data: Any) -> tuple[float, float] | None:
    """Parse raw bytes, string, multipart list, or dict received on gripper_port into (left, right).
    
    Supports:
      - {"data": {"left": ..., "right": ...}}
      - "data {'left': ..., 'right': ...}" or "data{\"left\": ..., \"right\": ...}"
      - Multipart: [b"data", b'{"left": ..., "right": ...}']
      - {"left": ..., "right": ...}
      - {"data": [left, right]}
    """
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

        def _extract_val(v: Any) -> float:
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

        def extract_from_dict(d: Any) -> tuple[float, float] | None:
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

        # Regex fallback for left and right values
        m_left = re.search(r"left[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        m_right = re.search(r"right[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        if m_left or m_right:
            l = float(m_left.group(1)) if m_left else 5.0
            r = float(m_right.group(1)) if m_right else 5.0
            return l, r

    except Exception as e:
        logger.debug("Error parsing gripper data: %s", e)
    return None


def _default_unitree_g1_client_cameras(
    server_address: str = "localhost",
    port: int = 5556,
    enable_wrist_cameras: bool = False,
    left_wrist_port: int = 5557,
    right_wrist_port: int = 5558,
    left_wrist_ip: str = "",
    right_wrist_ip: str = "",
) -> dict[str, CameraConfig]:
    cams: dict[str, CameraConfig] = {
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
    if enable_wrist_cameras:
        cams["left_wrist"] = ZMQCameraConfig(
            server_address=left_wrist_ip or server_address,
            port=left_wrist_port,
            camera_name="left_wrist",
            width=640,
            height=480,
            fps=30,
            warmup_s=5,
        )
        cams["right_wrist"] = ZMQCameraConfig(
            server_address=right_wrist_ip or server_address,
            port=right_wrist_port,
            camera_name="right_wrist",
            width=640,
            height=480,
            fps=30,
            warmup_s=5,
        )
    return cams


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
    camera_ip: str = ""

    # ZMQ port for receiving 29-DoF lowstate (PUB on server, SUB here)
    state_port: int = 6001

    # ZMQ port for sending 18-DoF actions (PULL on server, PUSH here)
    action_port: int = 6002

    # ZMQ port for receiving robot mode stream (PUB on server, SUB here; default 6000: nav, gamepad, vla)
    mode_port: int = 6000

    # Source host IP where mode stream is published (defaults to robot_ip if empty)
    mode_ip: str = ""

    # ZMQ port for receiving gripper state stream (PUB on server, SUB here; default 6004: {left, right})
    gripper_port: int = 6004

    # Source host IP where gripper state is published (defaults to robot_ip if empty)
    gripper_ip: str = ""

    # If True, enable extra gripper receiver and include gripper in observation & action features
    enable_gripper: bool = False

    # If True, expose only 14 arm joints (+ optional 2 grippers) matching 16-DoF models (e.g. Dex-1)
    arm_only: bool = False

    # If True, smoothly interpolate from current physical joint angles to VLA action stream upon engagement
    enable_smooth_engagement: bool = True

    # Duration in seconds for smooth engagement interpolation
    engagement_duration: float = 1.2

    # Camera streaming port
    camera_port: int = 5556
    camera_name: str = "head_camera"

    # Wrist camera ports (optional, if using multi-camera streaming)
    left_wrist_port: int = 5557
    right_wrist_port: int = 5558
    left_wrist_ip: str = ""
    right_wrist_ip: str = ""
    enable_wrist_cameras: bool = False

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
        if not self.mode_ip:
            self.mode_ip = self.robot_ip
        if not self.gripper_ip:
            self.gripper_ip = self.robot_ip
        if not self.left_wrist_ip:
            self.left_wrist_ip = self.camera_ip
        if not self.right_wrist_ip:
            self.right_wrist_ip = self.camera_ip

        if not self.cameras:
            self.cameras = _default_unitree_g1_client_cameras(
                server_address=self.camera_ip,
                port=self.camera_port,
                enable_wrist_cameras=self.enable_wrist_cameras,
                left_wrist_port=self.left_wrist_port,
                right_wrist_port=self.right_wrist_port,
                left_wrist_ip=self.left_wrist_ip,
                right_wrist_ip=self.right_wrist_ip,
            )
        else:
            if "global_view" in self.cameras:
                # Sync server address and port if user specified them on CLI
                cam = self.cameras["global_view"]
                if hasattr(cam, "server_address") and (not cam.server_address or cam.server_address == "localhost"):
                    cam.server_address = self.camera_ip
                if hasattr(cam, "port") and self.camera_port != 5556:
                    cam.port = self.camera_port
            if "left_wrist" in self.cameras:
                cam = self.cameras["left_wrist"]
                if hasattr(cam, "server_address") and (not cam.server_address or cam.server_address == "localhost"):
                    cam.server_address = self.left_wrist_ip
                if hasattr(cam, "port") and self.left_wrist_port != 5557:
                    cam.port = self.left_wrist_port
            if "right_wrist" in self.cameras:
                cam = self.cameras["right_wrist"]
                if hasattr(cam, "server_address") and (not cam.server_address or cam.server_address == "localhost"):
                    cam.server_address = self.right_wrist_ip
                if hasattr(cam, "port") and self.right_wrist_port != 5558:
                    cam.port = self.right_wrist_port


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

        # Robot mode tracking (port 6003: NAV, GAMEPAD, VLA)
        self._mode_sub: zmq.Socket | None = None
        self._current_mode: RobotMode = RobotMode.UNKNOWN
        self._mode_lock = threading.Lock()
        self._mode_thread: threading.Thread | None = None
        self._mode_packet_count: int = 0
        self._last_mode_time: float = 0.0
        self._last_mode_raw: str = ""
        self._first_mode_packet_received: bool = False

        # Gripper state tracking (extra port, e.g. 6004: data{left, right})
        self._gripper_sub: zmq.Socket | None = None
        self._latest_gripper: dict[str, float] = {"left": 1.0, "right": 1.0}
        self._gripper_lock = threading.Lock()

        self._gripper_thread: threading.Thread | None = None
        self._first_gripper_packet_received: bool = False

        # Engagement smoothing (S-curve blending from physical joint angles to VLA stream)
        self._engagement_smoothing_active: bool = False
        self._engagement_step: int = 0
        self._engagement_total_steps: int = 0
        self._engagement_start_pos: dict[str, float] = {}
        self._engagement_lock = threading.Lock()
        self._has_sent_action_since_idle: bool = False
        self._first_action_logged: bool = False

        # Camera frame cache to ensure observations remain complete even during transient packet drops
        self._last_camera_frames: dict[str, Any] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Motor joint positions for all 29 joints (or 14 arms in arm_only mode) + camera RGB features (+ optional gripper)."""
        if self.config.arm_only:
            proprio_ft = {f"{G1_29_JointArmIndex(motor).name}": float for motor in G1_29_JointArmIndex}
            if self.config.enable_gripper:
                proprio_ft["kLeftGripper"] = float
                proprio_ft["kRightGripper"] = float
        else:
            proprio_ft = {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}
            if self.config.enable_gripper:
                proprio_ft["gripper.left"] = float
                proprio_ft["gripper.right"] = float
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
        """14-DoF arm joint targets + optional 2-DoF grippers + optional 4-DoF remote joystick axes."""
        if self.config.arm_only:
            arm_features = {f"{G1_29_JointArmIndex(motor).name}": float for motor in G1_29_JointArmIndex}
            if self.config.enable_gripper:
                gripper_features = {
                    "kLeftGripper": float,
                    "kRightGripper": float,
                }
                return {**arm_features, **gripper_features}
            return arm_features

        arm_features = {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}
        remote_features = dict.fromkeys(REMOTE_AXES, float)
        if self.config.enable_gripper:
            gripper_features = {
                "gripper.right": float,
                "gripper.left": float,
            }
            return {**arm_features, **gripper_features, **remote_features}
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

    @property
    def current_mode(self) -> RobotMode:
        with self._mode_lock:
            return self._current_mode

    @property
    def is_vla_mode(self) -> bool:
        return self.current_mode == RobotMode.VLA

    @property
    def mode_packet_count(self) -> int:
        with self._mode_lock:
            return self._mode_packet_count

    @property
    def last_mode_time(self) -> float:
        with self._mode_lock:
            return self._last_mode_time

    @property
    def mode_port(self) -> int:
        return self.config.mode_port

    @property
    def last_mode_raw(self) -> str:
        with self._mode_lock:
            return self._last_mode_raw

    def mode_status_summary(self) -> str:
        """Produce a formatted human-readable diagnostic report of the 6003 mode stream."""
        mode_ip = self.config.mode_ip or self.config.robot_ip
        count = self.mode_packet_count
        last_t = self.last_mode_time
        raw = self.last_mode_raw
        mode = self.current_mode
        elapsed = (time.time() - last_t) if last_t > 0 else -1.0
        mode_desc = {
            RobotMode.NAV: "导航模式 (NAV)",
            RobotMode.GAMEPAD: "手柄控制模式 (GAMEPAD)",
            RobotMode.VLA: "VLA 控制模式 (VLA)",
            RobotMode.UNKNOWN: "未知/尚未收到 (UNKNOWN)",
        }.get(mode, str(mode))

        lines = [
            f"  • 监听地址: tcp://{mode_ip}:{self.config.mode_port}",
            f"  • 接收统计: 共接收 {count} 个数据包" + (f" (最近接收: {elapsed:.2f} 秒前)" if elapsed >= 0 else " (尚未接收到任何数据)"),
            f"  • 最新原始数据: {raw!r}" if raw else "  • 最新原始数据: (暂无)",
            f"  • 当前解析模式: {mode_desc}",
            f"  • VLA 触发状态: {'★ 激活中 (已接入 VLA)' if mode == RobotMode.VLA else '○ 等待中 / 非 VLA'}",
        ]
        return "\n".join(lines)

    def set_mock_mode(self, mode: RobotMode | str) -> None:
        """Helper to set or simulate robot mode for testing."""
        if isinstance(mode, str):
            mode = _parse_robot_mode(mode)
        with self._mode_lock:
            self._current_mode = mode

    def _extract_arm_joint_positions(self) -> dict[str, float]:
        """Extract current arm joint angles and grippers from latest state without touching cameras."""
        with self._state_lock:
            state = self._latest_state

        start_pos: dict[str, float] = {}
        data = state.get("data", state) if isinstance(state, dict) else {}
        motors_dict = data.get("motors") or (state.get("motors") if isinstance(state, dict) else None)
        motor_list = data.get("motor_state") or (state.get("motor_state") if isinstance(state, dict) else None)

        if motors_dict or motor_list:
            for motor in G1_29_JointArmIndex:
                name = motor.name
                idx = motor.value
                q_val = None
                if motors_dict and name in motors_dict:
                    m_data = motors_dict[name]
                    if isinstance(m_data, dict):
                        q_val = float(m_data.get("q", 0.0))
                    elif isinstance(m_data, (int, float)):
                        q_val = float(m_data)
                elif motor_list and idx < len(motor_list):
                    m_data = motor_list[idx]
                    if isinstance(m_data, dict):
                        q_val = float(m_data.get("q", 0.0))
                    elif isinstance(m_data, (int, float)):
                        q_val = float(m_data)

                if q_val is None:
                    q_val = float(self.config.default_positions[motor.value])
                start_pos[f"{name}.q"] = q_val
                start_pos[name] = q_val
        else:
            for motor in G1_29_JointArmIndex:
                q_val = float(self.config.default_positions[motor.value])
                start_pos[f"{motor.name}.q"] = q_val
                start_pos[motor.name] = q_val

        if self.config.enable_gripper:
            with self._gripper_lock:
                g = dict(self._latest_gripper)
            r_val = float(g.get("right", 1.0))
            l_val = float(g.get("left", 1.0))
            start_pos["gripper.right"] = r_val
            start_pos["gripper.left"] = l_val
            start_pos["kRightGripper"] = r_val
            start_pos["kLeftGripper"] = l_val

        return start_pos

    def trigger_engagement_smoothing(self) -> None:
        """Trigger S-curve smooth interpolation from current joint angles to VLA action stream."""
        if not self.config.enable_smooth_engagement:
            return

        start_pos = self._extract_arm_joint_positions()
        total_steps = max(int(self.config.engagement_duration / self.control_dt), 1)
        with self._engagement_lock:
            self._engagement_start_pos = start_pos
            self._engagement_step = 0
            self._engagement_total_steps = total_steps
            self._engagement_smoothing_active = True

        logger.info(
            "✨ [UnitreeG1Client] 触发平滑介入插值 (Engagement Smoothing: %d 步 / %.2fs)...",
            total_steps,
            self.config.engagement_duration,
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _init_sockets(self) -> None:
        """Initialize or re-initialize ZMQ sockets for state subscriber, action pusher, and mode subscriber."""
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

        if self._mode_sub is not None:
            try:
                self._mode_sub.close(linger=0)
            except Exception:
                pass
            self._mode_sub = None

        if self.config.mode_port > 0:
            self._mode_sub = ctx.socket(zmq.SUB)
            self._mode_sub.setsockopt(zmq.RCVHWM, 10)
            self._mode_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            mode_ip = self.config.mode_ip or self.config.robot_ip
            self._mode_sub.connect(f"tcp://{mode_ip}:{self.config.mode_port}")

        if self._gripper_sub is not None:
            try:
                self._gripper_sub.close(linger=0)
            except Exception:
                pass
            self._gripper_sub = None

        if self.config.enable_gripper and self.config.gripper_port > 0:
            self._gripper_sub = ctx.socket(zmq.SUB)
            self._gripper_sub.setsockopt(zmq.CONFLATE, 1)
            self._gripper_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            gripper_ip = self.config.gripper_ip or self.config.robot_ip
            self._gripper_sub.connect(f"tcp://{gripper_ip}:{self.config.gripper_port}")

    def _start_gripper_thread(self) -> None:
        """Start the background gripper receiving thread if enabled."""
        if self._gripper_thread is not None and self._gripper_thread.is_alive():
            return
        if not self.config.enable_gripper or self.config.gripper_port <= 0:
            return
        self._shutdown_event.clear()
        self._gripper_thread = threading.Thread(target=self._gripper_reader_loop, daemon=True)
        self._gripper_thread.start()

    def _gripper_reader_loop(self) -> None:
        """Background thread to continuously receive latest data{left, right} gripper state from gripper_port."""
        gripper_ip = self.config.gripper_ip or self.config.robot_ip
        while not self._shutdown_event.is_set():
            sock = self._gripper_sub
            if sock is None:
                time.sleep(0.05)
                continue
            try:
                if sock.poll(timeout=50):
                    newest_parts: list[bytes] | None = None
                    while True:
                        try:
                            parts = sock.recv_multipart(zmq.NOBLOCK)
                            newest_parts = parts
                        except zmq.Again:
                            break

                    if newest_parts is None:
                        continue

                    parsed = _parse_gripper_data(newest_parts)
                    if parsed is not None:
                        l_val, r_val = parsed
                        if not self._first_gripper_packet_received:
                            self._first_gripper_packet_received = True
                            raw_preview = " | ".join(p.decode("utf-8", errors="ignore").strip().strip("\x00\r\n\t ") for p in newest_parts)
                            sys.stdout.write(
                                f"\n📡 [{self.config.gripper_port} 端口] 首次接收到夹爪数据包! 来源: tcp://{gripper_ip}:{self.config.gripper_port}\n"
                                f"   ├─ 原始数据: {raw_preview!r}\n"
                                f"   └─ 解析夹爪: left={l_val:.2f}, right={r_val:.2f}\n"
                            )
                            sys.stdout.flush()

                        with self._gripper_lock:
                            self._latest_gripper = {
                                "left": float(np.clip(l_val, 0.0, 5.0)),
                                "right": float(np.clip(r_val, 0.0, 5.0)),
                            }
            except zmq.Again:
                pass
            except zmq.ContextTerminated:
                break
            except (zmq.ZMQError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger.debug("Exception in gripper_reader_loop: %s", e)
                time.sleep(0.05)

    def _start_state_thread(self) -> None:
        """Start the background state receiving thread if not already running."""
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._state_thread = threading.Thread(target=self._state_reader_loop, daemon=True)
        self._state_thread.start()

    def _start_mode_thread(self) -> None:
        """Start the background mode receiving thread (port 6003) if not already running."""
        if self._mode_thread is not None and self._mode_thread.is_alive():
            return
        if self.config.mode_port <= 0:
            return
        self._shutdown_event.clear()
        self._mode_thread = threading.Thread(target=self._mode_reader_loop, daemon=True)
        self._mode_thread.start()

    def _mode_reader_loop(self) -> None:
        """Background thread to continuously pull the newest robot mode from mode_port (6003)."""
        mode_ip = self.config.mode_ip or self.config.robot_ip
        while not self._shutdown_event.is_set():
            sock = self._mode_sub
            if sock is None:
                time.sleep(0.05)
                continue
            try:
                if sock.poll(timeout=50):
                    newest_parts: list[bytes] | None = None
                    drain_count = 0
                    while True:
                        try:
                            parts = sock.recv_multipart(zmq.NOBLOCK)
                            newest_parts = parts
                            drain_count += 1
                        except zmq.Again:
                            break

                    if newest_parts is None:
                        continue

                    # Record statistics & raw string representation
                    raw_preview = " | ".join(p.decode("utf-8", errors="ignore").strip().strip("\x00\r\n\t ") for p in newest_parts)
                    now = time.time()
                    with self._mode_lock:
                        self._mode_packet_count += drain_count
                        self._last_mode_time = now
                        self._last_mode_raw = raw_preview

                    new_mode = _parse_robot_mode(newest_parts)

                    if not self._first_mode_packet_received:
                        self._first_mode_packet_received = True
                        sys.stdout.write(
                            f"\n📡 [{self.config.mode_port} 端口] 首次接收到模式数据包! 来源: tcp://{mode_ip}:{self.config.mode_port}\n"
                            f"   ├─ 原始数据: {raw_preview!r}\n"
                            f"   └─ 解析模式: {new_mode.value}\n"
                        )
                        sys.stdout.flush()

                    if new_mode != RobotMode.UNKNOWN:
                        with self._mode_lock:
                            changed = (new_mode != self._current_mode)
                            old_mode = self._current_mode
                            self._current_mode = new_mode
                        if changed:
                            mode_desc = {
                                RobotMode.NAV: "导航模式 (NAV)",
                                RobotMode.GAMEPAD: "手柄控制模式 (GAMEPAD)",
                                RobotMode.VLA: "VLA 控制模式 (VLA)",
                            }.get(new_mode, str(new_mode))
                            sys.stdout.write(
                                f"\n📡 [{self.config.mode_port} 端口] 机器人模式切换: {old_mode.value} -> {mode_desc} (原始数据: {raw_preview!r})\n"
                            )
                            sys.stdout.flush()
                            logger.info("📡 [%d 端口] 机器人模式变更为: %s", self.config.mode_port, mode_desc)
                            if new_mode == RobotMode.VLA:
                                self.trigger_engagement_smoothing()
                            elif new_mode in (RobotMode.NAV, RobotMode.GAMEPAD):
                                self._has_sent_action_since_idle = False
                                with self._engagement_lock:
                                    self._engagement_smoothing_active = False
            except zmq.ContextTerminated:
                break
            except (zmq.ZMQError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger.debug("Exception in mode_reader_loop: %s", e)
                time.sleep(0.05)

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
                cam_addr = getattr(cam, "server_address", self.config.camera_ip)
                cam_port = getattr(cam, "port", self.config.camera_port)
                try:
                    print(f"🔄 [UnitreeG1Client] 尝试连通机载相机 [{cam_name}] (tcp://{cam_addr}:{cam_port})...", flush=True)
                    if hasattr(cam, "disconnect"):
                        try:
                            cam.disconnect()
                        except Exception:
                            pass
                    if hasattr(cam, "connect"):
                        try:
                            print(f"DEBUG: Calling cam.connect(warmup=False) for {cam_name}", flush=True)
                            cam.connect(warmup=False)
                            print(f"DEBUG: Returned from cam.connect for {cam_name}", flush=True)
                        except TypeError:
                            cam.connect()
                except Exception as e:
                    logger.warning(f"⚠️ [UnitreeG1Client] 相机 [{cam_name}] 连接失败 (tcp://{cam_addr}:{cam_port}): {e}")

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
        print("=" * 80, flush=True)
        print(" [VLA 策略模型已加载至 GPU 显存]", flush=True)
        print("-" * 80, flush=True)
        print("当前网络接入监听配置:", flush=True)
        if self.config.mock_state:
            logger.info("  • 机器人状态 (State):  [虚拟零位 / Mock] (纯视觉测试，不依赖 6001 端口)")
            logger.info("  • 动作下发   (Action): [已旁路 / Bypassed] (不下发到物理电机)")
        else:
            logger.info("  • 机器人状态 (State):  tcp://%s:%d", self.config.robot_ip, self.config.state_port)
            logger.info("  • 动作下发   (Action): tcp://%s:%d", self.config.action_ip, self.config.action_port)
        for cname, ccfg in self.config.cameras.items():
            c_addr = getattr(ccfg, "server_address", self.config.camera_ip)
            c_port = getattr(ccfg, "port", self.config.camera_port)
            logger.info("  • 机载视觉   (%s): tcp://%s:%d", cname, c_addr, c_port)
        if self.config.mode_port > 0:
            logger.info("  • 模式监听   (Mode):   tcp://%s:%d (监听手柄模式: 导航/手柄/VLA)", self.config.robot_ip, self.config.mode_port)
        if self.config.enable_gripper:
            logger.info("  • 夹爪状态   (Gripper): tcp://%s:%d (监听 {left, right} 夹爪实测角度)", self.config.gripper_ip, self.config.gripper_port)
        logger.info("-" * 80)
        logger.info("⏳ 正在等待机器人与视觉服务接入 (检测到数据流将自动接入并启动推理)...")
        logger.info("💡 提示: 模型常驻显存，连接未就绪也不会中断退出！可在机器人端随时启动服务端。")
        logger.info("   [交互指令] 终端直接输入新 IP + 回车可热切换目标地址；输入 'q' + 回车退出")
        logger.info("=" * 80)

        self._init_sockets()
        self._start_state_thread()
        self._start_mode_thread()
        self._start_gripper_thread()
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
            if self.config.enable_gripper:
                with self._gripper_lock:
                    self._latest_gripper = {"left": 5.0, "right": 5.0}
            logger.info("ℹ️  [UnitreeG1Client] 纯视觉 (Camera-Only) 模式已启用：关节使用虚拟零位，仅监听机载相机。")

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
            
            print(f"DEBUG LOOP: has_state={has_state}, has_camera={has_camera}", flush=True)

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
                        c_addr = getattr(cam, "server_address", self.config.camera_ip)
                        c_port = getattr(cam, "port", self.config.camera_port)
                        c_alive = getattr(cam, "thread", None) is not None and cam.thread.is_alive()
                        c_frame = getattr(cam, "latest_frame", None) is not None
                        cam_details.append(f"[{name}] tcp://{c_addr}:{c_port} (线程:{'运行' if c_alive else '未起'}, 首帧:{'已收到' if c_frame else '未收到'})")
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
                cam_status_list = []
                for name, cam in self._cameras.items():
                    c_port = getattr(cam, "port", self.config.camera_port)
                    c_ok = getattr(cam, "latest_frame", None) is not None
                    cam_status_list.append(f"{name}(:{c_port}):{'✅' if c_ok else '⏳等待'}")
                cam_str = ", ".join(cam_status_list)
                logger.info(
                    "⏳ [等待接入] 状态: %s (tcp://%s:%d) | 视觉: [%s] | 已等待 %.0fs (回车查状态 / 输入新IP切换 / q退出)",
                    state_str,
                    self.config.robot_ip,
                    self.config.state_port,
                    cam_str,
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
        if self.config.enable_gripper:
            logger.info("  • 夹爪状态:   已连接 (tcp://%s:%d)", self.config.gripper_ip, self.config.gripper_port)
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

        if self._mode_thread is not None:
            self._mode_thread.join(timeout=1.0)
            self._mode_thread = None

        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
            self._gripper_thread = None

        if self._state_sub is not None:
            self._state_sub.close(linger=0)
            self._state_sub = None

        if self._mode_sub is not None:
            try:
                self._mode_sub.close(linger=0)
            except Exception:
                pass
            self._mode_sub = None

        if self._gripper_sub is not None:
            try:
                self._gripper_sub.close(linger=0)
            except Exception:
                pass
            self._gripper_sub = None

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
        data = state.get("data", state) if isinstance(state, dict) else {}
        motors_dict = data.get("motors") or (state.get("motors") if isinstance(state, dict) else None)
        motor_list = data.get("motor_state") or (state.get("motor_state") if isinstance(state, dict) else None)

        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            m_data = {}
            if motors_dict and name in motors_dict:
                m_data = motors_dict[name]
            elif motor_list and idx < len(motor_list):
                m_data = motor_list[idx]

            q_val = float(m_data.get("q", 0.0))
            obs[f"{name}.q"] = q_val
            obs[name] = q_val
            obs[f"{name}.dq"] = float(m_data.get("dq", 0.0))
            obs[f"{name}.tau"] = float(m_data.get("tau", m_data.get("tau_est", 0.0)))

        # 2. Parse IMU state
        imu = data.get("imu_state") or data.get("imu") or (state.get("imu", {}) if isinstance(state, dict) else {})
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
            try:
                if getattr(cam, "is_connected", True):
                    if getattr(cam, "use_rgb", True):
                        frame = cam.read_latest()
                        obs[cam_name] = frame
                        self._last_camera_frames[cam_name] = frame
                    if getattr(cam, "use_depth", False):
                        depth_frame = cam.read_latest_depth()
                        obs[f"{cam_name}_depth"] = depth_frame
                        self._last_camera_frames[f"{cam_name}_depth"] = depth_frame
            except Exception as e:
                # If reading latest frame fails or times out momentarily, reuse last valid frame to prevent dropping observation
                if cam_name in self._last_camera_frames:
                    obs[cam_name] = self._last_camera_frames[cam_name]
                    logger.debug("Camera %s read failed (%s); reusing previous valid frame.", cam_name, e)
                else:
                    logger.debug("Failed to read camera %s: %s", cam_name, e)
                if getattr(cam, "use_depth", False) and f"{cam_name}_depth" in self._last_camera_frames:
                    obs[f"{cam_name}_depth"] = self._last_camera_frames[f"{cam_name}_depth"]

        # 4. Parse Gripper state (if enabled)
        if self.config.enable_gripper:
            with self._gripper_lock:
                g = dict(self._latest_gripper)
            l_val = float(g.get("left", 1.0))
            r_val = float(g.get("right", 1.0))
            obs["gripper.left"] = l_val
            obs["gripper.right"] = r_val
            obs["kLeftGripper"] = l_val
            obs["kRightGripper"] = r_val
            obs["observation.left_gripper"] = l_val
            obs["observation.right_gripper"] = r_val

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Forward action dict to locomotion server via ZMQ PUSH, applying engagement smoothing if active."""
        blended_action = dict(action)
        with self._engagement_lock:
            if self._engagement_smoothing_active:
                total = self._engagement_total_steps
                step = self._engagement_step
                tau = min(step / max(total, 1), 1.0)
                # Cosine S-curve: 0.0 at step=0, 1.0 at step=total
                alpha = 0.5 * (1.0 - math.cos(math.pi * tau))

                # Blend arm and gripper joints: (1 - alpha) * q_start + alpha * q_vla
                for k, start_q in self._engagement_start_pos.items():
                    target_k = k
                    if target_k not in blended_action:
                        if target_k.endswith(".q") and target_k[:-2] in blended_action:
                            target_k = target_k[:-2]
                        elif f"{target_k}.q" in blended_action:
                            target_k = f"{target_k}.q"
                    if target_k in blended_action:
                        target_q = float(blended_action[target_k])
                        blended_action[target_k] = (1.0 - alpha) * start_q + alpha * target_q

                # Damp remote locomotion axes during engagement to avoid sudden twitching
                for axis in REMOTE_AXES:
                    if axis in blended_action:
                        blended_action[axis] = alpha * float(blended_action[axis])

                self._engagement_step += 1
                if self._engagement_step >= total:
                    self._engagement_smoothing_active = False
                    logger.info("✨ [UnitreeG1Client] 平滑介入插值已完成，现全权由 VLA 动作流接管。")

        if self._action_push is None or self.config.mock_state:
            return blended_action

        # Clean action dictionary with only serializable floats, packaging gripper into action["gripper"]
        serializable_action = {}
        for k, v in blended_action.items():
            if k in ("gripper.right", "gripper.left", "kRightGripper", "kLeftGripper"):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                val = float(v)
                serializable_action[k] = val
                if not k.endswith(".q"):
                    serializable_action[f"{k}.q"] = val

        # Ensure chassis remote control axes are present (default 0.0)
        for axis in REMOTE_AXES:
            if axis in blended_action:
                serializable_action[axis] = float(blended_action[axis])
            elif axis not in serializable_action:
                serializable_action[axis] = 0.0

        gripper_obj = None
        has_gripper_keys = (
            "gripper.right" in blended_action
            or "gripper.left" in blended_action
            or "kRightGripper" in blended_action
            or "kLeftGripper" in blended_action
        )
        if self.config.enable_gripper or has_gripper_keys:
            r_val = float(blended_action.get("kRightGripper", blended_action.get("gripper.right", 1.0)))
            l_val = float(blended_action.get("kLeftGripper", blended_action.get("gripper.left", 1.0)))
            r_val_clipped = float(np.clip(r_val, 0.0, 5.0))
            l_val_clipped = float(np.clip(l_val, 0.0, 5.0))

            gripper_obj = {
                "right": {"q": r_val_clipped},
                "left":  {"q": l_val_clipped},
            }
            serializable_action["gripper"] = gripper_obj

        msg = {
            "cmd": "action",
            "action": serializable_action,
            "timestamp": time.time(),
        }

        if not self._first_action_logged:
            self._first_action_logged = True
            if gripper_obj is not None:
                logger.info("📡 [动作下发首包] 成功打包夹爪信息 -> action['gripper']: %s (开合范围: 0.0~5.0)", json.dumps(gripper_obj))
            else:
                logger.info("📡 [动作下发首包] 未包含夹爪 (enable_gripper=%s, blended_keys=%s)", self.config.enable_gripper, list(blended_action.keys())[:5])

        try:
            payload = json.dumps(msg).encode("utf-8")
            self._action_push.send(payload, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.debug("Failed to send action via ZMQ: %s", e)

        return blended_action

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:
        """Request the locomotion server to reset arms to default position."""
        self._has_sent_action_since_idle = False
        self._first_action_logged = False
        with self._engagement_lock:
            self._engagement_smoothing_active = False
            self._engagement_step = 0
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
