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

"""Unit tests for Unitree G1 feature extraction and PI0.5 processor integration in rollout."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import make_default_processors
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts


def test_unitree_g1_rollout_feature_aggregation():
    """Verify that Unitree G1 with SonicLocoManipulationController correctly produces
    29-D observation.state and 18-D action features for policy consumption."""

    # 1. Mock lowstate message and SDK to create UnitreeG1 in isolation
    lowstate_msg = MagicMock()
    lowstate_msg.motor_state = [MagicMock(q=i * 0.05, dq=0.0, tau_est=0.0) for i in range(35)]
    lowstate_msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    lowstate_msg.imu_state.gyroscope = [0.0, 0.0, 0.0]
    lowstate_msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    lowstate_msg.imu_state.rpy = [0.0, 0.0, 0.0]
    lowstate_msg.wireless_remote = b"\x00" * 40
    lowstate_msg.mode_machine = 0

    mock_cam = MagicMock(is_connected=True)
    mock_cam.use_rgb = True
    mock_cam.use_depth = False
    mock_cam.read_latest.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    module = "lerobot.robots.unitree_g1.unitree_g1"
    with patch(f"{module}.require_package", MagicMock()), \
         patch(f"{module}.make_cameras_from_configs", lambda cfgs: {"global_view": mock_cam}), \
         patch(f"{module}.load_policy", return_value=(
             MagicMock(get_inputs=lambda: [SimpleNamespace(name="obs")]),
             np.zeros(NUM_MOTORS, np.float32),
             np.zeros(NUM_MOTORS, np.float32),
             np.zeros(NUM_MOTORS, np.float32),
             np.zeros(NUM_MOTORS, np.float32),
             np.zeros(64, np.float32),
         )):
        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        cfg = UnitreeG1Config(
            is_simulation=True,
            gravity_compensation=False,
            controller="SonicLocoManipulationController",
            cameras={"global_view": SimpleNamespace(width=640, height=480, fps=30, use_rgb=True, use_depth=False)},
        )
        robot = UnitreeG1(cfg)
        robot._lowstate = SimpleNamespace(
            motor_state=[SimpleNamespace(q=float(i) * 0.1, dq=0.0, tau_est=0.0) for i in range(NUM_MOTORS)],
            imu_state=SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0],
                gyroscope=[0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 9.81],
                rpy=[0.0, 0.0, 0.0],
            ),
            wireless_remote=None,
            mode_machine=0,
        )

        all_obs_features = robot.observation_features
        action_features = robot.action_features

        # Check observation features: 29 joints + 1 camera
        assert len(all_obs_features) == 30
        assert "global_view" in all_obs_features
        for joint in G1_29_JointIndex:
            assert f"{joint.name}.q" in all_obs_features

        # Check action features: 14 arm joints + 4 remote axes
        assert len(action_features) == 18
        for arm_joint in G1_29_JointArmIndex:
            assert f"{arm_joint.name}.q" in action_features
        for remote_axis in REMOTE_AXES:
            assert remote_axis in action_features

        # 2. Extract hardware features using updated context.py logic
        observation_features_hw = {
            k: v
            for k, v in all_obs_features.items()
            if isinstance(v, tuple) or v is float or (isinstance(v, PolicyFeature) and v.type != FeatureType.VISUAL)
        }
        action_features_hw = {
            k: v
            for k, v in action_features.items()
            if v is float or (isinstance(v, PolicyFeature) and v.type != FeatureType.VISUAL)
        }

        assert len(observation_features_hw) == 30
        assert len(action_features_hw) == 18

        # 3. Aggregate dataset features
        teleop_proc, robot_act_proc, robot_obs_proc = make_default_processors()
        action_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        )
        observation_dataset_features = aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=True,
        )
        dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)

        assert "observation.state" in dataset_features
        assert dataset_features["observation.state"]["shape"] == (29,)
        assert len(dataset_features["observation.state"]["names"]) == 29

        assert "observation.images.global_view" in dataset_features
        assert dataset_features["observation.images.global_view"]["shape"] == (480, 640, 3)

        assert ACTION in dataset_features
        assert dataset_features[ACTION]["shape"] == (18,)
        assert len(dataset_features[ACTION]["names"]) == 18

        # 4. Build dataset frame from raw robot observation
        raw_obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset_features, raw_obs, prefix=OBS_STR)

        assert "observation.state" in obs_frame
        assert isinstance(obs_frame["observation.state"], np.ndarray)
        assert obs_frame["observation.state"].shape == (29,)
        assert obs_frame["observation.state"][0] == pytest.approx(0.0)
        assert obs_frame["observation.state"][1] == pytest.approx(0.1)

        assert "observation.images.global_view" in obs_frame
        assert obs_frame["observation.images.global_view"].shape == (480, 640, 3)

        # 5. Prepare observation for inference
        inference_obs = prepare_observation_for_inference(
            obs_frame,
            device=torch.device("cpu"),
            task="move blue box",
            robot_type="unitree_g1",
        )

        assert inference_obs["observation.state"].shape == (1, 29)
        assert inference_obs["observation.images.global_view"].shape == (1, 3, 480, 640)
        assert inference_obs["task"] == "move blue box"

        # 6. Pass through PI0.5 State Tokenizer Step
        step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=32, task_key="task")
        transition = EnvTransition(
            observation=inference_obs,
            action=None,
            reward=None,
            done=None,
            truncated=None,
            info=None,
            complementary_data={"task": ["move blue box"]},
        )
        processed_transition = step(transition)

        assert TransitionKey.COMPLEMENTARY_DATA in processed_transition
        prompt = processed_transition[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
        assert prompt.startswith("Task: move blue box, State:")
        assert prompt.endswith(";\nAction: ")
