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

"""Thread-safe diagnostic logger for LeRobot + Unitree G1 + SONIC WBC + PI0.5.

Records:
1. PI0.5 model outputs (18-D: 14 arm joint positions + 4 remote joystick axes)
2. Robot state / proprioception (29 joint positions, velocities, IMU quaternion, gyroscope, gravity dir)
3. SONIC WBC inputs & inference (64-D motion token, 994-D decoder input, 29-D residual action)
4. Final published DDS lowcmd (29-D target angles, gains Kp/Kd, velocities dq, feedforward torques)
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .g1_utils import (
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    get_gravity_orientation,
)

DEFAULT_DUMP_PATH = os.environ.get(
    "SONIC_DUMP_FILE",
    "/home/yichangfeng/lerobot/outputs/sonic_io_dump.txt",
)


class SonicIODumper:
    """Thread-safe dumper writing comprehensive step-by-step diagnostic logs."""

    _instance: SonicIODumper | None = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls, file_path: str | Path | None = None) -> SonicIODumper:
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(file_path or DEFAULT_DUMP_PATH)
            return cls._instance

    def __init__(self, file_path: str | Path = DEFAULT_DUMP_PATH):
        self.file_path = str(file_path)
        self._lock = threading.Lock()
        self._step_count = 0
        self._init_file()

    def _init_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("        LeRobot + Unitree G1 + SONIC WBC + PI0.5 Diagnostic Execution Log\n")
                f.write(f"        Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
                f.write(f"        Dump File : {self.file_path}\n")
                f.write("=" * 100 + "\n\n")
                f.write("Structure of Logged Data:\n")
                f.write("  [1] PI0.5 Policy 18-D Action: 14 Arm Joint Positions (.q) + 4 Remote Joystick Axes\n")
                f.write("  [2] Robot LowState Proprioception: 29 Joint Angles (q), Velocities (dq), IMU Quat/Gyro\n")
                f.write("  [3] SONIC Controller: 64-D Latent Token, 994-D Decoder Input, 29-D Residual Action\n")
                f.write("  [4] Published LowCmd: 29-D Target Joint Positions, PD Gains (Kp, Kd), Target dq\n")
                f.write("=" * 100 + "\n\n")
        except Exception as e:
            print(f"[SonicIODumper] Failed to initialize log file: {e}")

    def log_pi05_action(self, action_dict: dict, step_idx: int | None = None) -> None:
        """Log raw action dict from PI0.5 strategy."""
        with self._lock:
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            step = step_idx if step_idx is not None else self._step_count
            lines = [
                f"[Step {step:05d} | {now_str}] === [1] PI0.5 MODEL OUTPUT (18-D Action) ===",
            ]

            # Remote axes
            remote_lx = action_dict.get("remote.lx", 0.0)
            remote_ly = action_dict.get("remote.ly", 0.0)
            remote_rx = action_dict.get("remote.rx", 0.0)
            remote_ry = action_dict.get("remote.ry", 0.0)
            lines.append(
                f"  > Remote Velocity Axes : lx={remote_lx:+.4f}, ly={remote_ly:+.4f}, "
                f"rx={remote_rx:+.4f}, ry={remote_ry:+.4f}"
            )

            # Left Arm
            l_arm_keys = [
                "kLeftShoulderPitch.q",
                "kLeftShoulderRoll.q",
                "kLeftShoulderYaw.q",
                "kLeftElbow.q",
                "kLeftWristRoll.q",
                "kLeftWristPitch.q",
                "kLeftWristYaw.q",
            ]
            l_arm_vals = [action_dict.get(k, action_dict.get(k.lower(), 0.0)) for k in l_arm_keys]
            lines.append(
                f"  > Left Arm (7D)        : {np.array2string(np.array(l_arm_vals, dtype=np.float32), precision=4, suppress_small=True)}"
            )

            # Right Arm
            r_arm_keys = [
                "kRightShoulderPitch.q",
                "kRightShoulderRoll.q",
                "kRightShoulderYaw.q",
                "kRightElbow.q",
                "kRightWristRoll.q",
                "kRightWristPitch.q",
                "kRightWristYaw.q",
            ]
            r_arm_vals = [action_dict.get(k, action_dict.get(k.lower(), 0.0)) for k in r_arm_keys]
            lines.append(
                f"  > Right Arm (7D)       : {np.array2string(np.array(r_arm_vals, dtype=np.float32), precision=4, suppress_small=True)}"
            )

            lines.append("")
            self._write_lines(lines)

    def log_sonic_step(
        self,
        input_action: dict,
        lowstate,
        cmd_vel: np.ndarray,
        height_cmd: float,
        last_token: np.ndarray | None,
        decoder_input_obs: np.ndarray,
        raw_action_mj: np.ndarray,
        final_targets: dict,
    ) -> None:
        """Log one 50Hz control tick of SonicWholeBodyController."""
        with self._lock:
            self._step_count += 1
            step = self._step_count
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            lines = [
                f"[Step {step:05d} | {now_str}] === [2] ROBOT PROPRIOCEPTION & SONIC CONTROL STEP ===",
            ]

            # 1. LowState Proprioception
            if lowstate is not None:
                q = np.array([lowstate.motor_state[m.value].q for m in G1_29_JointIndex], dtype=np.float32)
                dq = np.array([lowstate.motor_state[m.value].dq for m in G1_29_JointIndex], dtype=np.float32)
                quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)
                gyro = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
                grav = get_gravity_orientation(quat)

                lines.append(
                    f"  > IMU Quaternion (w,x,y,z) : [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
                )
                lines.append(
                    f"  > IMU Gyroscope (wx,wy,wz) : [{gyro[0]:+.4f}, {gyro[1]:+.4f}, {gyro[2]:+.4f}] rad/s"
                )
                lines.append(
                    f"  > Gravity Vector (gx,gy,gz) : [{grav[0]:+.4f}, {grav[1]:+.4f}, {grav[2]:+.4f}]"
                )
                lines.append(
                    f"  > Left Leg q   [0:6]        : {np.array2string(q[0:6], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Right Leg q  [6:12]       : {np.array2string(q[6:12], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Waist q      [12:15]      : {np.array2string(q[12:15], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Left Arm q   [15:22]      : {np.array2string(q[15:22], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Right Arm q  [22:29]      : {np.array2string(q[22:29], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Legs dq Mean / Max        : mean={np.mean(np.abs(dq[0:12])):.4f}, max={np.max(np.abs(dq[0:12])):.4f} rad/s"
                )

            # 2. SONIC WBC Commands & Token
            lines.append("  --- SONIC Planner & Token ---")
            lines.append(
                f"  > Commanded Vel (vx,vy,vyaw) : [{cmd_vel[0]:+.4f}, {cmd_vel[1]:+.4f}, {cmd_vel[2]:+.4f}] | Height: {height_cmd:.3f}m"
            )
            if last_token is not None:
                token_arr = np.asarray(last_token, dtype=np.float32).flatten()
                lines.append(
                    f"  > Motion Token (64-D)        : norm={np.linalg.norm(token_arr):.4f}, "
                    f"min={np.min(token_arr):+.4f}, max={np.max(token_arr):+.4f}"
                )
                lines.append(
                    f"    Token [0:8] slice          : {np.array2string(token_arr[0:8], precision=4, suppress_small=True)}"
                )

            # 3. SONIC Decoder Input/Output
            lines.append("  --- SONIC Decoder I/O ---")
            if decoder_input_obs is not None:
                dec_in = np.asarray(decoder_input_obs, dtype=np.float32).flatten()
                lines.append(
                    f"  > 994-D Decoder Input Norm   : {np.linalg.norm(dec_in):.4f} (finite={bool(np.all(np.isfinite(dec_in)))})"
                )

            if raw_action_mj is not None:
                raw_act = np.asarray(raw_action_mj, dtype=np.float32).flatten()
                lines.append(
                    f"  > Raw ONNX Residual (IsaacLab) : norm={np.linalg.norm(raw_act):.4f}, "
                    f"min={np.min(raw_act):+.4f}, max={np.max(raw_act):+.4f}"
                )

            # 4. Final Published Targets
            if final_targets:
                target_q = np.array([final_targets.get(f"{m.name}.q", 0.0) for m in G1_29_JointIndex], dtype=np.float32)
                lines.append("  --- Final 29-Joint Published Targets ---")
                lines.append(
                    f"  > Target Left Leg  [0:6]     : {np.array2string(target_q[0:6], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Target Right Leg [6:12]    : {np.array2string(target_q[6:12], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Target Waist     [12:15]   : {np.array2string(target_q[12:15], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Target Left Arm  [15:22]   : {np.array2string(target_q[15:22], precision=4, suppress_small=True)}"
                )
                lines.append(
                    f"  > Target Right Arm [22:29]   : {np.array2string(target_q[22:29], precision=4, suppress_small=True)}"
                )

            lines.append("-" * 100 + "\n")
            self._write_lines(lines)

    def log_groot_step(
        self,
        cmd: np.ndarray,
        selected_policy_name: str,
        groot_action: np.ndarray,
        target_dict: dict,
        lowstate=None,
    ) -> None:
        """Log one control tick of GrootLocomotionController."""
        with self._lock:
            self._step_count += 1
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            lines = [
                f"[Step {self._step_count:05d} | {now_str}] === [2] GR00T LOCOMOTION CONTROL STEP ===",
            ]

            lines.append(
                f"  > Commanded Vel (vx,vy,vyaw) : [{cmd[0]:+.4f}, {cmd[1]:+.4f}, {cmd[2]:+.4f}] | "
                f"Active Policy: {selected_policy_name.upper()}"
            )

            if lowstate is not None and hasattr(lowstate, "imu_state"):
                quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)
                gyro = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
                grav = get_gravity_orientation(quat)
                lines.append(f"  > IMU Quaternion (w,x,y,z) : [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                lines.append(f"  > Gravity Vector (gx,gy,gz) : [{grav[0]:+.4f}, {grav[1]:+.4f}, {grav[2]:+.4f}]")
                lines.append(f"  > IMU Gyroscope (wx,wy,wz) : [{gyro[0]:+.4f}, {gyro[1]:+.4f}, {gyro[2]:+.4f}] rad/s")

            if groot_action is not None:
                act = np.asarray(groot_action, dtype=np.float32).flatten()
                lines.append(
                    f"  > GR00T Policy Action (15D) : norm={np.linalg.norm(act):.4f}, "
                    f"min={np.min(act):+.4f}, max={np.max(act):+.4f}"
                )

            if target_dict:
                target_q = np.array([target_dict.get(f"{G1_29_JointIndex(i).name}.q", 0.0) for i in range(15)], dtype=np.float32)
                lines.append("  --- Published Lower Body & Waist Targets ---")
                lines.append(f"  > Target Left Leg  [0:6]     : {np.array2string(target_q[0:6], precision=4, suppress_small=True)}")
                lines.append(f"  > Target Right Leg [6:12]    : {np.array2string(target_q[6:12], precision=4, suppress_small=True)}")
                lines.append(f"  > Target Waist     [12:15]   : {np.array2string(target_q[12:15], precision=4, suppress_small=True)}")

            lines.append("-" * 100 + "\n")
            self._write_lines(lines)

    def _write_lines(self, lines: list[str]) -> None:
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"[SonicIODumper] Write error: {e}")
