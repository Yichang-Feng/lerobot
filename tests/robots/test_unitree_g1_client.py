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

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)
from lerobot.robots.unitree_g1.unitree_g1_client import (
    UnitreeG1Client,
    UnitreeG1ClientConfig,
)
from lerobot.robots.utils import make_robot_from_config


def test_unitree_g1_client_registration():
    """Verify that UnitreeG1ClientConfig is properly registered in RobotConfig ChoiceRegistry."""
    cfg_cls = RobotConfig.get_choice_class("unitree_g1_client")
    assert cfg_cls is UnitreeG1ClientConfig

    config = UnitreeG1ClientConfig(robot_ip="127.0.0.1", state_port=6001, action_port=6002, camera_port=5556)
    assert config.type == "unitree_g1_client"
    assert config.connect_timeout == 60.0
    assert "global_view" in config.cameras
    assert config.cameras["global_view"].port == 5556
    assert config.cameras["global_view"].server_address == "127.0.0.1"

    custom_config = UnitreeG1ClientConfig(connect_timeout=30.0)
    assert custom_config.connect_timeout == 30.0


def test_unitree_g1_client_features():
    """Verify that UnitreeG1Client exposes the exact 29-DoF observation and 18-DoF action features."""
    config = UnitreeG1ClientConfig()
    client = UnitreeG1Client(config)

    # Observation features: 29 joints + camera
    obs_ft = client.observation_features
    for motor in G1_29_JointIndex:
        assert f"{motor.name}.q" in obs_ft
        assert obs_ft[f"{motor.name}.q"] is float

    assert "global_view" in obs_ft
    assert obs_ft["global_view"] == (480, 640, 3)

    # Action features: 14 arms + 4 remote axes
    act_ft = client.action_features
    assert len(act_ft) == 18

    for motor in G1_29_JointArmIndex:
        assert f"{motor.name}.q" in act_ft
        assert act_ft[f"{motor.name}.q"] is float

    for axis in REMOTE_AXES:
        assert axis in act_ft
        assert act_ft[axis] is float


def test_unitree_g1_client_get_observation_and_send_action():
    """Verify observation parsing and action sending over mocked ZMQ sockets."""
    config = UnitreeG1ClientConfig()
    client = UnitreeG1Client(config)

    # Mock internal state and camera
    mock_cam = MagicMock(is_connected=True)
    fake_img = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read_latest.return_value = fake_img
    client._cameras = {"global_view": mock_cam}

    # Populate mock lowstate
    mock_motors = {
        G1_29_JointIndex(i).name: {"q": float(i) * 0.1, "dq": 0.05, "tau": 1.2}
        for i in range(NUM_MOTORS)
    }
    client._latest_state = {
        "motors": mock_motors,
        "imu": {
            "gyroscope": [0.1, -0.2, 0.3],
            "accelerometer": [0.0, 0.0, 9.81],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "rpy": [0.01, -0.02, 0.03],
        },
        "timestamp": 123456.789,
    }

    obs = client.get_observation()
    assert obs["global_view"] is fake_img
    assert obs["kLeftHipPitch.q"] == 0.0
    assert obs["kLeftHipRoll.q"] == 0.1 or math.isclose(obs["kLeftHipRoll.q"], 0.1, abs_tol=1e-4)
    assert obs["imu.gyro.x"] == 0.1 or math.isclose(obs["imu.gyro.x"], 0.1, abs_tol=1e-4)
    assert obs["imu.accel.z"] == 9.81 or math.isclose(obs["imu.accel.z"], 9.81, abs_tol=1e-4)

    # Test send_action
    mock_push = MagicMock()
    client._action_push = mock_push

    action = {"kLeftShoulderPitch.q": 0.5, "remote.lx": 0.1, "remote.ly": 0.2}
    result = client.send_action(action)
    assert result == action

    mock_push.send.assert_called_once()
    payload = mock_push.send.call_args[0][0]
    data = json.loads(payload.decode("utf-8"))
    assert data["cmd"] == "action"
    assert data["action"]["kLeftShoulderPitch.q"] == 0.5
    assert data["action"]["remote.lx"] == 0.1


def test_make_robot_from_config_creates_client():
    """Verify that make_robot_from_config instantiates UnitreeG1Client directly."""
    config = UnitreeG1ClientConfig()
    assert config.wait_until_ready is True
    robot = make_robot_from_config(config)
    assert isinstance(robot, UnitreeG1Client)
    assert robot.name == "unitree_g1_client"


def test_unitree_g1_client_hot_switch_ip():
    """Verify that _update_target_ip correctly changes target IPs on the fly."""
    config = UnitreeG1ClientConfig(robot_ip="127.0.0.1")
    client = UnitreeG1Client(config)
    assert client.config.robot_ip == "127.0.0.1"
    assert client.config.action_ip == "127.0.0.1"

    with patch.object(client, "_init_sockets") as mock_init:
        client._update_target_ip("192.168.123.165")
        assert client.config.robot_ip == "192.168.123.165"
        assert client.config.action_ip == "192.168.123.165"
        mock_init.assert_called_once()


def test_unitree_g1_client_mock_state():
    """Verify that mock_state populates virtual zero state and bypasses action sending."""
    config = UnitreeG1ClientConfig(mock_state=True)
    client = UnitreeG1Client(config)

    with patch.object(client, "_init_sockets"), \
         patch.object(client, "_start_state_thread"), \
         patch.object(client, "_try_connect_cameras", return_value=True):
        client.connect()
        assert client.is_connected is True
        obs = client.get_observation()
        assert "kLeftHipPitch.q" in obs
        assert obs["kLeftHipPitch.q"] == 0.0

        # Action send is bypassed
        action = {"kLeftShoulderPitch.q": 0.5}
        ret = client.send_action(action)
        assert ret == action
