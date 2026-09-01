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

"""SONIC decoder whole-body controller for the Unitree G1 (token-only).

Pure-Python/ONNX re-implementation of the *decode* half of NVIDIA's SONIC deploy stack.
The encoder is intentionally absent: a token-output VLA (e.g. ``nepyope/sonic_walk``)
supplies the 64-D latent ``motion_token`` directly each tick, and the SONIC **decoder**
maps ``token + recent proprioception history`` to a residual action that is scaled and
added onto the standing pose (``default_angles``) to produce 50 Hz joint-position targets
for the robot's PD controller.

Index spaces: joints exist in two orderings — **IsaacLab** (policy/training order) and
**MuJoCo** (deploy order). ``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB`` (in g1_utils)
convert between them. Quaternions are scalar-first ``(w, x, y, z)``.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    NUM_MOTORS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    get_gravity_orientation,
)
from ..sonic_io_dumper import SonicIODumper
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size
HISTORY_LEN = 10  # proprioception frames the decoder conditions on

# Latent-token feature-key prefixes: action carries the token, obs echoes it back.
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - feature-key prefix, not a secret
TOKEN_STATE_PREFIX = "motion_token_state"  # nosec B105 - feature-key prefix, not a secret

# SONIC decoder checkpoint. Deploy constants (kp/kd, default_angles, action_scale,
# neutral_token) are baked into the ONNX metadata; see upload_sonic_decoder.py.
DEFAULT_SONIC_REPO_ID = "lerobot/sonic_decoder"
# token + HISTORY_LEN frames of (angular velocity, joint pos, joint vel, last action) + gravity
DECODER_INPUT_DIM = TOKEN_DIM + HISTORY_LEN * (3 + 3 * NUM_MOTORS) + HISTORY_LEN * 3  # 994

# Decoder filename mapping: the full decoder (default) or NVIDIA's distilled low-latency one.
POLICY_FILES = {
    "default": "model_decoder.onnx",
    "low_latency": "low_latency/model_decoder.onnx",
}

# Local candidate paths in ~/GR00T-WholeBodyControl
LOCAL_SONIC_BASE = Path("/home/yichangfeng/GR00T-WholeBodyControl/gear_sonic_deploy")
LOCAL_DECODER_FILES = {
    "default": LOCAL_SONIC_BASE / "policy/sonic_v1_1/model_decoder.onnx",
    "low_latency": LOCAL_SONIC_BASE / "policy/low_latency/model_decoder.onnx",
}
LOCAL_PLANNER_FILE = LOCAL_SONIC_BASE / "planner/target_vel/V2/planner_sonic.onnx"
LOCAL_ENCODER_FILES = {
    "default": LOCAL_SONIC_BASE / "policy/sonic_v1_1/model_encoder.onnx",
    "low_latency": LOCAL_SONIC_BASE / "policy/low_latency/model_encoder.onnx",
}

# Default deploy constants for Unitree G1 (29-DoF) matching SONIC policy_parameters.hpp
DEFAULT_KP = np.array([
    99.1005, 99.1005, 40.1793, 99.1005, 28.5010, 28.5010,  # left leg
    99.1005, 99.1005, 40.1793, 99.1005, 28.5010, 28.5010,  # right leg
    40.1793, 28.5010, 28.5010,                               # waist
    14.2505, 14.2505, 14.2505, 14.2505, 14.2505, 16.7783, 16.7783,  # left arm
    14.2505, 14.2505, 14.2505, 14.2505, 14.2505, 16.7783, 16.7783,  # right arm
], dtype=np.float32)

DEFAULT_KD = np.array([
    6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # left leg
    6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,  # right leg
    2.5579, 1.8144, 1.8144,                           # waist
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # left arm
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,  # right arm
], dtype=np.float32)

DEFAULT_STANDING_ANGLES = np.zeros(NUM_MOTORS, dtype=np.float32)
# Legs (trained SONIC standing angles from policy_parameters.hpp)
DEFAULT_STANDING_ANGLES[[0, 6]] = -0.312  # Hip pitch
DEFAULT_STANDING_ANGLES[[3, 9]] = 0.669   # Knee
DEFAULT_STANDING_ANGLES[[4, 10]] = -0.363 # Ankle pitch
# Arms (standard forward-reaching teleop/manipulation pose)
DEFAULT_STANDING_ANGLES[15] = 0.5   # L Shoulder Pitch
DEFAULT_STANDING_ANGLES[17] = 0.2   # L Shoulder Yaw
DEFAULT_STANDING_ANGLES[18] = 0.3   # L Elbow
DEFAULT_STANDING_ANGLES[22] = 0.5   # R Shoulder Pitch
DEFAULT_STANDING_ANGLES[24] = -0.2  # R Shoulder Yaw
DEFAULT_STANDING_ANGLES[25] = 0.3   # R Elbow

DEFAULT_ACTION_SCALE = np.array([
    0.350654, 0.350654, 0.547545, 0.350654, 0.438581, 0.438581,  # left leg
    0.350654, 0.350654, 0.547545, 0.350654, 0.438581, 0.438581,  # right leg
    0.547545, 0.438581, 0.438581,                                 # waist
    0.438581, 0.438581, 0.438581, 0.438581, 0.438581, 0.074501, 0.074501,  # left arm
    0.438581, 0.438581, 0.438581, 0.438581, 0.438581, 0.074501, 0.074501,  # right arm
], dtype=np.float32)
DEFAULT_NEUTRAL_TOKEN = np.array(
    [
        -0.0625,  0.0000, -0.0625, -0.1250, -0.1875, -0.0625,  0.1875,
         0.2500,  0.1875, -0.1250,  0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625,  0.0000, -0.0625, -0.1250, -0.1875,  0.0000,
        -0.2500,  0.0000, -0.2500, -0.0625,  0.0625,  0.1250, -0.1250,
         0.2500,  0.1875,  0.2500, -0.1250,  0.1250,  0.1875, -0.0625,
         0.0000, -0.1875, -0.1875,  0.2500,  0.0000,  0.0000, -0.1250,
         0.0625,  0.0000, -0.0625, -0.0625,  0.1875, -0.0625,  0.0000,
         0.0625,  0.1250,  0.0625,  0.1250,  0.0625,  0.1250,  0.0000,
         0.1250,  0.1875,  0.0000,  0.0000,  0.0625,  0.0625,  0.1875,
         0.0625,
    ],
    dtype=np.float32,
)


def load_planner(planner_path: str | Path | None = None) -> ort.InferenceSession | None:
    """Load the neural locomotion planner (planner_sonic.onnx) if available."""
    path = None
    if planner_path and os.path.isfile(planner_path):
        path = str(planner_path)
    elif LOCAL_PLANNER_FILE.is_file():
        path = str(LOCAL_PLANNER_FILE)
    if path is not None:
        try:
            planner = ort.InferenceSession(path)
            logger.info(f"Loaded SONIC planner: {path}")
            return planner
        except Exception as e:
            logger.warning(f"Could not load planner ONNX ({e})")
    return None


def load_policy(
    repo_id: str = DEFAULT_SONIC_REPO_ID,
    policy_type: str = "default",
    local_path: str | Path | None = None,
) -> tuple[ort.InferenceSession, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the SONIC decoder and its deploy constants.

    Checks local filesystem paths first (e.g. ~/GR00T-WholeBodyControl), falling back
    to downloading from the Hugging Face Hub.

    Args:
        repo_id: Hugging Face Hub repo ID
        policy_type: Either "default" (full decoder) or "low_latency" (distilled)
        local_path: Optional explicit path to an ONNX model file.

    Returns:
        (decoder, kp, kd, default_angles, action_scale, neutral_token) tuple.
    """
    if policy_type not in POLICY_FILES:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {list(POLICY_FILES.keys())}")

    decoder_path = None
    if local_path and os.path.isfile(local_path):
        decoder_path = str(local_path)
        logger.info(f"Loading SONIC decoder from explicit local path: {decoder_path}")
    elif policy_type in LOCAL_DECODER_FILES and LOCAL_DECODER_FILES[policy_type].is_file():
        decoder_path = str(LOCAL_DECODER_FILES[policy_type])
        logger.info(f"Loading SONIC decoder from local GR00T-WholeBodyControl: {decoder_path}")
    else:
        filename = POLICY_FILES[policy_type]
        logger.info(f"Loading {policy_type.upper()} SONIC decoder from Hub: {repo_id}/{filename}")
        decoder_path = hf_hub_download(repo_id=repo_id, filename=filename)

    decoder = ort.InferenceSession(decoder_path)
    logger.info(f"Decoder loaded: {decoder.get_inputs()[0].shape} → {decoder.get_outputs()[0].shape}")

    # Extract deploy constants from ONNX metadata (with robust default fallbacks)
    kp = DEFAULT_KP.copy()
    kd = DEFAULT_KD.copy()
    default_angles = DEFAULT_STANDING_ANGLES.copy()
    action_scale = DEFAULT_ACTION_SCALE.copy()
    neutral_token = DEFAULT_NEUTRAL_TOKEN.copy()

    try:
        model = onnx.load(decoder_path, load_external_data=False)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        if "kp" in metadata:
            kp = np.array(json.loads(metadata["kp"]), dtype=np.float32)
        if "kd" in metadata:
            kd = np.array(json.loads(metadata["kd"]), dtype=np.float32)
        if "default_angles" in metadata:
            default_angles = np.array(json.loads(metadata["default_angles"]), dtype=np.float32)
        if "action_scale" in metadata:
            action_scale = np.array(json.loads(metadata["action_scale"]), dtype=np.float32)
        if "neutral_token" in metadata:
            neutral_token = np.array(json.loads(metadata["neutral_token"]), dtype=np.float32)
        logger.info(f"Loaded SONIC deploy constants ({len(kp)} joints)")
    except Exception as e:
        logger.warning(f"Could not read metadata from ONNX ({e}); using default G1 deploy constants.")

    return decoder, kp, kd, default_angles, action_scale, neutral_token


class SonicWholeBodyController(RobotController):
    """Full-body SONIC controller for UnitreeG1's background controller thread.

    Supports multiple input modalities:
    1. Latent Token Mode (64-D continuous motion token: ``motion_token.{i}.pos``)
    2. Velocity / Loco-Manipulation Mode: Accepts 18-D action (14 arm joints + 4 remote
       joystick velocity axes) matching standard G1 manipulation policies and datasets.
    """

    control_dt = CONTROL_DT

    def __init__(self, policy_type: str = "default", mode: str = "auto"):
        """
        Args:
            policy_type: "default" or "low_latency"
            mode: "auto", "token" (64-D latent token), or "velocity" (18-D arm+remote action)
        """
        self.mode = mode
        self.decoder, self.kp, self.kd, self.default_angles, self.action_scale, self.neutral_token = (
            load_policy(policy_type=policy_type)
        )
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.default_angles_mj = self.default_angles[MUJOCO_TO_ISAACLAB]

        if mode == "velocity":
            # In velocity mode, action and observation spaces fall back to standard 18-D action
            # and 29-D joint state so that UnitreeG1 aligns with policies like box_move_blue.
            self.action_ft = None
            self.observation_ft = None
        else:
            # 64-D latent-token action and proprio space for Token policies.
            self.action_ft = {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
            self.observation_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

        self.planner = load_planner()
        self.dumper = SonicIODumper.get_instance()
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # vx, vy, vyaw
        self.height_cmd = 0.74
        self.arm_targets = {}

        self.reset()
        logger.info(f"SonicWholeBodyController initialized (mode={mode}, planner={self.planner is not None})")

    def reset(self) -> None:
        """Reset internal state for a new episode: held token and proprioception history."""
        self.last_action_mj = np.zeros(NUM_MOTORS, np.float32)
        self.h_q_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_dq_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_ang = [np.zeros(3, np.float32) for _ in range(HISTORY_LEN)]
        self.h_act_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_quat = [np.array([1, 0, 0, 0], np.float32) for _ in range(HISTORY_LEN)]
        self.context_qpos = [np.zeros(36, dtype=np.float32) for _ in range(4)]
        self._last_token = None
        self.cmd_vel[:] = 0.0
        self.height_cmd = 0.74
        # Default arm pose: forward-reaching manipulation posture (shoulder pitch 0.5, yaw ±0.2, elbow 0.3)
        self.arm_targets = {
            G1_29_JointIndex.kLeftShoulderPitch.value: 0.5,
            G1_29_JointIndex.kLeftShoulderRoll.value: 0.0,
            G1_29_JointIndex.kLeftShoulderYaw.value: 0.2,
            G1_29_JointIndex.kLeftElbow.value: 0.3,
            G1_29_JointIndex.kLeftWristRoll.value: 0.0,
            G1_29_JointIndex.kLeftWristPitch.value: 0.0,
            G1_29_JointIndex.kLeftWristYaw.value: 0.0,
            G1_29_JointIndex.kRightShoulderPitch.value: 0.5,
            G1_29_JointIndex.kRightShoulderRoll.value: 0.0,
            G1_29_JointIndex.kRightShoulderYaw.value: -0.2,
            G1_29_JointIndex.kRightElbow.value: 0.3,
            G1_29_JointIndex.kRightWristRoll.value: 0.0,
            G1_29_JointIndex.kRightWristPitch.value: 0.0,
            G1_29_JointIndex.kRightWristYaw.value: 0.0,
        }

    def observation_state(self) -> dict[str, float]:
        """Echo the last decoded token as ``observation.state`` when in token mode."""
        if self.observation_ft is None:
            return {}
        token = self._last_token if self._last_token is not None else self.neutral_token.copy()
        return {f"{TOKEN_STATE_PREFIX}.{i}.pos": float(v) for i, v in enumerate(token)}

    def run_step(self, action: dict, lowstate) -> dict:
        """Decode one control tick into absolute joint-position targets.

        Args:
            action: Action dict. May contain 64-D ``motion_token.{i}.pos`` or 18-D
                arm joint targets and ``remote.lx/ly/rx/ry`` joystick velocity axes.
            lowstate: Unitree lowstate carrying joint positions/velocities and IMU state.

        Returns:
            Absolute joint targets keyed ``<joint>.q`` for all 29 joints.
        """
        if lowstate is None:
            return {}

        # 1. Parse Input Modalities
        token_keys = [f"{TOKEN_ACTION_PREFIX}.{i}.pos" for i in range(TOKEN_DIM)]
        has_token = action and all(k in action for k in token_keys)

        if has_token:
            # Mode A: 64-D Latent Token Input
            self._last_token = np.fromiter(
                (float(action[k]) for k in token_keys), dtype=np.float32, count=TOKEN_DIM
            )
        else:
            zero_cmd = (
                os.environ.get("UNITREE_G1_ZERO_VELOCITY", "0").lower() in ("1", "true")
                or os.environ.get("ZERO_LOCOMOTION_CMD", "0").lower() in ("1", "true")
                or os.environ.get("STAND_ONLY", "0").lower() in ("1", "true")
            )
            if zero_cmd:
                self.cmd_vel[:] = 0.0
            else:
                lx, ly, rx, _ry = (float(action.get(k, 0.0)) for k in REMOTE_AXES)
                self.cmd_vel[0] = ly   # Forward / Backward
                self.cmd_vel[1] = -lx  # Left / Right
                self.cmd_vel[2] = -rx  # Yaw rate

            # Adjust height via buttons if present
            buttons = [int(action.get(k, 0)) for k in REMOTE_BUTTONS]
            if len(buttons) > 0 and buttons[0]:
                self.height_cmd = float(np.clip(self.height_cmd + 0.001, 0.50, 1.00))
            if len(buttons) > 4 and buttons[4]:
                self.height_cmd = float(np.clip(self.height_cmd - 0.001, 0.50, 1.00))

            # Store arm joint targets (case-insensitive for dataset compatibility)
            action_lower = {k.lower(): v for k, v in action.items()} if action else {}
            for arm_joint in G1_29_JointArmIndex:
                q_key = f"{arm_joint.name}.q"
                if q_key in action:
                    self.arm_targets[arm_joint.value] = float(action[q_key])
                elif q_key.lower() in action_lower:
                    self.arm_targets[arm_joint.value] = float(action_lower[q_key.lower()])

            # Token selection: stable neutral standing pose or planner
            speed = float(np.linalg.norm(self.cmd_vel[:2]))
            if speed < 0.05 and abs(self.cmd_vel[2]) < 0.05:
                self._last_token = self.neutral_token.copy()
            else:
                if self.planner is not None:
                    try:
                        input_name = self.planner.get_inputs()[0].name
                        in_shape = self.planner.get_inputs()[0].shape
                        if in_shape and in_shape[-1] == 4:
                            cmd_in = np.array([*self.cmd_vel, self.height_cmd], dtype=np.float32).reshape(1, -1)
                        else:
                            cmd_in = self.cmd_vel.reshape(1, -1)
                        self._last_token = self.planner.run(None, {input_name: cmd_in})[0].squeeze().astype(np.float32)
                    except Exception as e:
                        logger.debug(f"Planner inference failed ({e}), using neutral token")
                        self._last_token = self.neutral_token.copy()
                else:
                    self._last_token = self.neutral_token.copy()

        # 2. Read proprioception from lowstate (IsaacLab joint order)
        q = np.array([lowstate.motor_state[m.value].q for m in G1_29_JointIndex], np.float32)
        dq = np.array([lowstate.motor_state[m.value].dq for m in G1_29_JointIndex], np.float32)
        quat = np.array(lowstate.imu_state.quaternion, np.float32)  # (w, x, y, z)
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        ang = np.array(lowstate.imu_state.gyroscope, np.float32)

        # 3. Update 10-frame history
        self.h_q_mj = [q[MUJOCO_TO_ISAACLAB] - self.default_angles_mj] + self.h_q_mj[:-1]
        self.h_dq_mj = [dq[MUJOCO_TO_ISAACLAB]] + self.h_dq_mj[:-1]
        self.h_ang = [ang] + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat = [quat] + self.h_quat[:-1]

        # 4. Assemble the 994-D decoder input
        obs = np.zeros(DECODER_INPUT_DIM, np.float32)
        obs[:TOKEN_DIM] = self._last_token
        off = TOKEN_DIM
        for hist, sz in (
            (self.h_ang, 3),
            (self.h_q_mj, NUM_MOTORS),
            (self.h_dq_mj, NUM_MOTORS),
            (self.h_act_mj, NUM_MOTORS),
        ):
            for frame in reversed(hist):
                obs[off : off + sz] = frame
                off += sz
        for hquat in reversed(self.h_quat):
            obs[off : off + 3] = get_gravity_orientation(hquat)
            off += 3

        # 5. Decode -> residual action (MuJoCo order) added onto default angles
        action_mj = (
            self.decoder.run(None, {self.decoder_input: obs.reshape(1, -1)})[0].squeeze().astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = self.default_angles + action_mj[ISAACLAB_TO_MUJOCO] * self.action_scale

        target_dict = {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}

        # In Mode B (18-D action space), merge the 14 arm joint targets directly from policy/teleop
        if not has_token:
            for arm_joint in G1_29_JointArmIndex:
                if arm_joint.value in self.arm_targets:
                    target_dict[f"{arm_joint.name}.q"] = float(self.arm_targets[arm_joint.value])

        # Log complete diagnostic step
        try:
            self.dumper.log_sonic_step(
                input_action=action,
                lowstate=lowstate,
                cmd_vel=self.cmd_vel,
                height_cmd=self.height_cmd,
                last_token=self._last_token,
                decoder_input_obs=obs,
                raw_action_mj=action_mj,
                final_targets=target_dict,
            )
        except Exception:
            pass

        return target_dict


class SonicLocoManipulationController(SonicWholeBodyController):
    """SONIC Whole-Body Controller for 18-D Loco-Manipulation (14 Arm + 4 Remote Velocity)."""

    def __init__(self, policy_type: str = "default"):
        super().__init__(policy_type=policy_type, mode="velocity")


