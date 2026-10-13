#!/usr/bin/env python3
"""
Unit test for WholeSessionRecorder and replay_recorded_session.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from lerobot.rollout.whole_session_recorder import WholeSessionRecorder


class DummyCamera:
    def __init__(self, color=(100, 150, 200), label="cam"):
        self.color = color
        self.label = label
        self.frame_idx = 0

    def read_latest(self):
        self.frame_idx += 1
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = self.color
        cv2.putText(
            img,
            f"{self.label} #{self.frame_idx}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return img


class DummyRobot:
    def __init__(self):
        self._cameras = {
            "global_view": DummyCamera((40, 40, 180), "Global"),
            "left_wrist": DummyCamera((40, 180, 40), "LeftWrist"),
            "right_wrist": DummyCamera((180, 40, 40), "RightWrist"),
        }
        self._latest_state = {
            "q": np.linspace(-1.0, 1.0, 29, dtype=np.float32),
            "dq": np.zeros(29, dtype=np.float32),
            "tau": np.zeros(29, dtype=np.float32),
            "left_gripper": 5.0,
            "right_gripper": 3.4,
            "imu": [0.01, -0.02, 0.05],
        }


class DummyRobotWrapper:
    def __init__(self):
        self.inner = DummyRobot()


def test_whole_session_recorder():
    test_dir = Path("outputs/test_recordings/test_session_1")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    robot_wrapper = DummyRobotWrapper()
    controller = MagicMock()
    controller.running = True
    controller.action = np.ones(29, dtype=np.float32) * 0.1
    controller.task = "pick up the water bottle from the table"

    recorder = WholeSessionRecorder(
        robot_wrapper=robot_wrapper,
        controller=controller,
        output_dir=str(test_dir),
        fps=30.0,
    )

    print("\n--- 1. Starting recorder ---")
    recorder.start()
    time.sleep(0.5)

    print("--- 2. Recording mode transition ---")
    recorder.record_mode_transition("NAV", "VLA", raw_state_preview="state=vla")
    time.sleep(0.3)

    print("--- 3. Recording JEV decision ---")
    dummy_jev_result = MagicMock()
    dummy_jev_result.is_success = True
    dummy_jev_result.choice = "water_bottle_present"
    dummy_jev_result.confidence = 0.985
    dummy_jev_result.probabilities = {"water_bottle_present": 0.985, "no_water_bottle": 0.015}
    dummy_jev_result.cost_ms = 142.0
    dummy_jev_result.error = None

    recorder.record_jev_decision(
        table=1,
        phase=4,
        description="Table 1 water bottle presence polling",
        result=dummy_jev_result,
        questions={"test": "question"},
        context_text="test context",
    )
    time.sleep(0.5)

    print("--- 4. Stopping recorder ---")
    recorder.stop()
    print("Recorder stopped.")

    # Verification of files
    assert (test_dir / "session_meta.json").exists(), "session_meta.json missing"
    assert (test_dir / "telemetry.csv").exists(), "telemetry.csv missing"
    assert (test_dir / "joint_states.npy").exists(), "joint_states.npy missing"
    assert (test_dir / "camera_global_view.mp4").exists(), "camera_global_view.mp4 missing"
    assert (test_dir / "camera_left_wrist.mp4").exists(), "camera_left_wrist.mp4 missing"
    assert (test_dir / "camera_right_wrist.mp4").exists(), "camera_right_wrist.mp4 missing"
    assert (test_dir / "mode_transitions.jsonl").exists(), "mode_transitions.jsonl missing"
    assert (test_dir / "jev_decisions.jsonl").exists(), "jev_decisions.jsonl missing"

    with open(test_dir / "session_meta.json") as f:
        meta = json.load(f)
        assert meta["total_frames"] > 10, f"Expected >10 frames, got {meta['total_frames']}"
        assert meta["mode_transitions_count"] == 1
        assert meta["jev_decisions_count"] == 1

    joint_arr = np.load(test_dir / "joint_states.npy")
    assert joint_arr.ndim == 2 and joint_arr.shape[1] == 29, f"Invalid joint shape: {joint_arr.shape}"

    print(f"✅ Recording verified! Total recorded frames: {meta['total_frames']}")

    print("\n--- 5. Testing replay inspector ---")
    from dataset_tools.replay_recorded_session import SessionInspector, export_replay_mosaic

    inspector = SessionInspector(test_dir)
    inspector.print_summary()

    mosaic_output = test_dir / "test_mosaic.mp4"
    export_replay_mosaic(inspector, mosaic_output, target_fps=30.0)

    assert mosaic_output.exists() and mosaic_output.stat().st_size > 1000, "Mosaic output missing or empty"
    print(f"✅ Mosaic video export verified ({mosaic_output.stat().st_size} bytes)")


if __name__ == "__main__":
    test_whole_session_recorder()
