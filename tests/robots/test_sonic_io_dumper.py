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

from types import SimpleNamespace
from pathlib import Path
import numpy as np

from lerobot.robots.unitree_g1.g1_utils import NUM_MOTORS
from lerobot.robots.unitree_g1.sonic_io_dumper import SonicIODumper


def test_sonic_io_dumper_logging(tmp_path: Path):
    dump_file = tmp_path / "test_dump.txt"
    dumper = SonicIODumper(file_path=dump_file)

    # 1. Log PI0.5 action
    action = {
        "remote.lx": 0.1,
        "remote.ly": 0.2,
        "remote.rx": -0.05,
        "remote.ry": 0.0,
        "kLeftShoulderPitch.q": 0.5,
        "kLeftShoulderRoll.q": 0.0,
        "kLeftShoulderYaw.q": 0.2,
        "kLeftElbow.q": 0.3,
        "kLeftWristRoll.q": 0.0,
        "kLeftWristPitch.q": 0.0,
        "kLeftWristYaw.q": 0.0,
        "kRightShoulderPitch.q": 0.5,
        "kRightShoulderRoll.q": 0.0,
        "kRightShoulderYaw.q": -0.2,
        "kRightElbow.q": 0.3,
        "kRightWristRoll.q": 0.0,
        "kRightWristPitch.q": 0.0,
        "kRightWristYaw.q": 0.0,
    }
    dumper.log_pi05_action(action, step_idx=1)

    # 2. Log SONIC step
    lowstate = SimpleNamespace(
        motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(NUM_MOTORS)],
        imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0, 0.0, 0.0]),
    )
    dumper.log_sonic_step(
        input_action=action,
        lowstate=lowstate,
        cmd_vel=np.array([0.2, -0.1, 0.05], dtype=np.float32),
        height_cmd=0.74,
        last_token=np.zeros(64, dtype=np.float32),
        decoder_input_obs=np.zeros(994, dtype=np.float32),
        raw_action_mj=np.zeros(29, dtype=np.float32),
        final_targets={"kLeftHipPitch.q": -0.1, "kRightHipPitch.q": -0.1},
    )

    assert dump_file.exists()
    content = dump_file.read_text()
    assert "PI0.5 MODEL OUTPUT" in content
    assert "ROBOT PROPRIOCEPTION & SONIC CONTROL STEP" in content
    assert "Remote Velocity Axes" in content
    assert "Target Left Leg" in content
