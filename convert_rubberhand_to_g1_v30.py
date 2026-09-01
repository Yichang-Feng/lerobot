#!/usr/bin/env python3
"""
Convert SonicStar rubberhand_pickbox dataset (LeRobot v2.1 format, 43-D state/action)
to LeRobot v3.0 format (29-D state, 18-D action, observation.images.global_view)
for fine-tuning and training with model/unitree_box_move.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure lerobot source is in PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# G1 29-DoF Joint Names (aligned with unitree_box_move / box_pick)
STATE_NAMES = [
    # 12 leg joints
    "kLeftHipPitch.q",
    "kLeftHipRoll.q",
    "kLeftHipYaw.q",
    "kLeftKnee.q",
    "kLeftAnklePitch.q",
    "kLeftAnkleRoll.q",
    "kRightHipPitch.q",
    "kRightHipRoll.q",
    "kRightHipYaw.q",
    "kRightKnee.q",
    "kRightAnklePitch.q",
    "kRightAnkleRoll.q",
    # 3 waist joints
    "kWaistYaw.q",
    "kWaistRoll.q",
    "kWaistPitch.q",
    # 7 left arm joints
    "kLeftShoulderPitch.q",
    "kLeftShoulderRoll.q",
    "kLeftShoulderYaw.q",
    "kLeftElbow.q",
    "kLeftWristRoll.q",
    "kLeftWristPitch.q",
    "kLeftWristyaw.q",
    # 7 right arm joints
    "kRightShoulderPitch.q",
    "kRightShoulderRoll.q",
    "kRightShoulderYaw.q",
    "kRightElbow.q",
    "kRightWristRoll.q",
    "kRightWristPitch.q",
    "kRightWristYaw.q",
]

# 18-DoF Action Names (14 arm joints + 4 base teleop velocities)
ACTION_NAMES = [
    # 7 left arm target angles
    "kLeftShoulderPitch.q",
    "kLeftShoulderRoll.q",
    "kLeftShoulderYaw.q",
    "kLeftElbow.q",
    "kLeftWristRoll.q",
    "kLeftWristPitch.q",
    "kLeftWristyaw.q",
    # 7 right arm target angles
    "kRightShoulderPitch.q",
    "kRightShoulderRoll.q",
    "kRightShoulderYaw.q",
    "kRightElbow.q",
    "kRightWristRoll.q",
    "kRightWristPitch.q",
    "kRightWristYaw.q",
    # 4 teleop velocity commands
    "remote.lx",
    "remote.ly",
    "remote.rx",
    "remote.ry",
]

FEATURES = {
    "observation.images.global_view": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (29,),
        "names": STATE_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (18,),
        "names": ACTION_NAMES,
    },
}


def convert_dataset(
    src_dir: Path,
    dst_dir: Path,
    repo_id: str = "unitree_g1/rubberhand_pickbox_g1",
    fps: int = 30,
    task_desc: str = "pick the rubber hand box",
    overwrite: bool = True,
):
    src_dir = Path(src_dir).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    if dst_dir.exists():
        if overwrite:
            print(f"Removing existing destination directory: {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"Destination directory already exists: {dst_dir}")

    parquet_files = sorted((src_dir / "data/chunk-000").glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {src_dir / 'data/chunk-000'}")

    print(f"Source dataset: {src_dir}")
    print(f"Destination dataset: {dst_dir}")
    print(f"Found {len(parquet_files)} episodes to convert at {fps} FPS...")

    # Create destination LeRobot v3.0 dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="unitree_g1",
        features=FEATURES,
        root=dst_dir,
        use_videos=True,
    )

    for ep_idx, pq_path in enumerate(tqdm(parquet_files, desc="Converting episodes")):
        df = pq.read_table(pq_path).to_pandas()

        # Locate corresponding video file
        vid_path = src_dir / f"videos/chunk-000/observation.images.ego_view/episode_{ep_idx:06d}.mp4"
        if not vid_path.exists():
            print(f"Warning: Video file not found: {vid_path}, skipping episode {ep_idx}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        num_frames = min(len(df), len(frames))
        if num_frames == 0:
            print(f"Warning: Episode {ep_idx} has 0 frames, skipping.")
            continue

        for i in range(num_frames):
            raw_state = np.array(df["observation.state"].iloc[i], dtype=np.float32)
            raw_action = np.array(df["action.wbc"].iloc[i], dtype=np.float32)

            # 1. State Mapping (43-D -> 29-D):
            # [0:15] legs (12) + waist (3)
            # [15:22] left arm (7)
            # [22:29] left fingers (7) -> discarded
            # [29:36] right arm (7)
            # [36:43] right fingers (7) -> discarded
            state_29 = np.concatenate([raw_state[0:22], raw_state[29:36]])

            # 2. Action Mapping (43-D -> 18-D):
            # [15:22] left arm target angles (7)
            # [29:36] right arm target angles (7)
            # 4-D remote velocity commands [remote.lx, remote.ly, remote.rx, remote.ry]
            remote_vel = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            action_18 = np.concatenate([raw_action[15:22], raw_action[29:36], remote_vel])

            frame_data = {
                "observation.images.global_view": frames[i],
                "observation.state": state_29,
                "action": action_18,
                "task": task_desc,
            }
            dataset.add_frame(frame_data)

        dataset.save_episode()

    dataset.finalize()
    print("\n" + "=" * 60)
    print("Conversion successfully completed!")
    print(f"Saved to: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SonicStar dataset to LeRobot v3.0 G1 dataset")
    parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path.home() / "SonicStar/wbc/outputs/rubberhand_pickbox"),
        help="Source directory containing v2.1 dataset",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/rubberhand_pickbox_g1"),
        help="Destination directory for converted v3.0 dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="unitree_g1/rubberhand_pickbox_g1",
        help="Repo ID for metadata",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate (default: 30)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick the rubber hand box",
        help="Task description string",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite destination directory if it exists",
    )

    args = parser.parse_args()
    convert_dataset(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        task_desc=args.task,
        overwrite=not args.no_overwrite,
    )
