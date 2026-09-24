#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from enum import IntEnum

import numpy as np

# ruff: noqa: N801, N815

NUM_MOTORS = 29

# Joint-order permutation between IsaacLab and Mujoco convention
ISAACLAB_TO_MUJOCO = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int32,
)
MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO).astype(np.int32)

REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
REMOTE_BUTTONS = tuple(f"remote.button.{i}" for i in range(16))
REMOTE_KEYS = REMOTE_AXES + REMOTE_BUTTONS


def default_remote_input() -> dict[str, float]:
    """Return a zeroed-out remote input dict (axes + buttons)."""
    return dict.fromkeys(REMOTE_KEYS, 0.0)


def get_gravity_orientation(quaternion: list[float] | np.ndarray) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


# Dex1 Gripper Constants
DEX1_LEFT_FINGER_ACTUATORS = ("left_dex1_finger_joint_1", "left_dex1_finger_joint_2")
DEX1_RIGHT_FINGER_ACTUATORS = ("right_dex1_finger_joint_1", "right_dex1_finger_joint_2")
DEX1_ALL_FINGER_ACTUATORS = DEX1_LEFT_FINGER_ACTUATORS + DEX1_RIGHT_FINGER_ACTUATORS
NUM_DEX1_FINGER_MOTORS = len(DEX1_ALL_FINGER_ACTUATORS)


def map_gripper_cmd_to_pos(
    val: float,
    min_val: float = 0.0,
    max_val: float = 5.0,
    min_pos: float = -0.02,
    max_pos: float = 0.0245,
) -> float:
    """Map client gripper command [0.0 ~ 5.0] (0=closed, 5=open) to MuJoCo joint position [-0.02 ~ 0.0245]."""
    val_clipped = float(np.clip(val, min_val, max_val))
    norm = (val_clipped - min_val) / (max_val - min_val) if max_val != min_val else 0.0
    return min_pos + norm * (max_pos - min_pos)


def map_pos_to_gripper_val(
    pos: float,
    min_pos: float = -0.02,
    max_pos: float = 0.0245,
    min_val: float = 0.0,
    max_val: float = 5.0,
) -> float:
    """Map MuJoCo joint position [-0.02 ~ 0.0245] to client gripper command [0.0 ~ 5.0]."""
    pos_clipped = float(np.clip(pos, min_pos, max_pos))
    norm = (pos_clipped - min_pos) / (max_pos - min_pos) if max_pos != min_pos else 0.0
    return min_val + norm * (max_val - min_val)

