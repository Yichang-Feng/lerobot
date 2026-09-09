#!/usr/bin/env python3
"""
replay_g1_dataset.py

在 MuJoCo 仿真中直观回放 Unitree G1 采集/转化后的数据集动作（如 g1_box_pick_turn_v30）。
支持：
1. 播放实测真实状态 (--mode state)：查看电机编码器实际物理关节角度 (q_meas)
2. 播放期望目标动作 (--mode action)：查看下发的期望目标角度 (q_des)，直观验证抱箱时的手臂内收与夹紧效果
3. 底盘偏航旋转同步：根据 remote.rx 偏航角速度实时积分推算基座朝向，真实还原“抱箱转身”全过程
4. 交互式 3D 视窗 (GUI) 与离线高清 MP4 视频导出 (--save-video)
5. 左右分屏对比 (--side-by-side)：左侧播放采集时的真实机载视觉，右侧播放 3D 仿真机器人姿态

使用示例：
    # 1. 交互式 3D 窗口播放 Episode 0（实测状态）
    python replay_g1_dataset.py --dataset datasets/g1_box_pick_turn_v30 --episode 0 --mode state

    # 2. 交互式 3D 窗口播放 Episode 0（期望动作，观察更夹紧的手臂姿态）
    python replay_g1_dataset.py --dataset datasets/g1_box_pick_turn_v30 --episode 0 --mode action

    # 3. 离线导出左右分屏 MP4 视频（左边相机视角，右边 3D 仿真）
    python replay_g1_dataset.py --dataset datasets/g1_box_pick_turn_v30 --episode 0 --save-video replay_ep0.mp4 --side-by-side
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_XML_PATH = "/home/yichangfeng/SonicStar/wbc/gear_sonic/data/robots/g1/g1_29dof.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Replay G1 dataset motions in MuJoCo simulation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/g1_box_pick_turn_v30",
        help="Path to dataset directory (e.g. datasets/g1_box_pick_turn_v30)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["state", "action"],
        default="state",
        help="Playback mode: 'state' for measured actual positions, 'action' for commanded target positions (shows clamping)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate (default: 30.0)",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=DEFAULT_XML_PATH,
        help="Path to G1 29-DoF MuJoCo XML model",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Output path to save playback as MP4 video (runs in headless mode)",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="If set, concatenates original camera video (left) with 3D simulation render (right)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop playback continuously in GUI viewer",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to play/render (useful for quick test)",
    )
    return parser.parse_args()


def load_episode_data(dataset_path: Path, episode_idx: int):
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    df_target = None
    for pf in parquet_files:
        df = pq.read_table(pf).to_pandas()
        if "episode_index" in df.columns and episode_idx in df["episode_index"].values:
            df_target = df[df["episode_index"] == episode_idx].reset_index(drop=True)
            break

    if df_target is None:
        raise ValueError(f"Episode {episode_idx} not found across parquet files in {dataset_path}")

    return df_target


def load_episode_video_frames(dataset_path: Path, episode_idx: int, num_frames: int) -> list:
    """Attempt to load camera frames from dataset for side-by-side visualization."""
    # Check v2.1 structure: videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
    v21_path = dataset_path / f"videos/chunk-{episode_idx // 1000:03d}/observation.images.ego_view/episode_{episode_idx:06d}.mp4"
    if v21_path.exists():
        cap = cv2.VideoCapture(str(v21_path))
        frames = []
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    # Try LeRobotDataset API if available
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(dataset_path.name, root=dataset_path)
        ep_info = ds.meta.episodes[episode_idx]
        start_idx = ep_info["dataset_from_index"]
        end_idx = min(ep_info["dataset_to_index"], start_idx + num_frames)
        ep_frames = []
        for i in range(start_idx, end_idx):
            img = ds[i]["observation.images.global_view"]
            if isinstance(img, torch.Tensor):
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8) if img.dtype != torch.uint8 else img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)
            ep_frames.append(img_np)
        return ep_frames
    except Exception as e:
        print(f"[Warning] Failed to load camera frames via LeRobotDataset: {e}")
        pass

    return []


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle (rad) to quaternion [w, x, y, z]."""
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def main():
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve()

    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        sys.exit(1)

    print(f"[Replay] Dataset: {dataset_path}")
    print(f"[Replay] Episode: {args.episode}")
    print(f"[Replay] Playback Mode: {args.mode.upper()}")
    print(f"[Replay] Loading MuJoCo model: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    df = load_episode_data(dataset_path, args.episode)
    total_frames = len(df)
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)
    print(f"[Replay] Loaded Episode {args.episode} with {total_frames} frames.")

    has_state = "observation.state" in df.columns
    has_action = "action" in df.columns

    states = np.vstack(df["observation.state"].values) if has_state else None
    actions = np.vstack(df["action"].values) if has_action else None

    # Precompute base yaw trajectory from remote.rx
    dt = 1.0 / args.fps
    base_yaws = np.zeros(total_frames, dtype=np.float64)
    if actions is not None and actions.shape[1] >= 17:
        # remote.rx = -yaw_rate -> delta_yaw = -remote.rx * dt
        rx = actions[:total_frames, 16]
        cum_dyaw = np.cumsum(-rx * dt)
        base_yaws = cum_dyaw

    # Side-by-side camera frames if requested
    camera_frames = []
    if args.side_by_side and args.save_video:
        print("[Replay] Loading camera frames for side-by-side visualization...")
        camera_frames = load_episode_video_frames(dataset_path, args.episode, total_frames)
        print(f"[Replay] Loaded {len(camera_frames)} camera frames.")

    # -------------------------------------------------------------
    # Execution Mode: Save to Video (Headless)
    # -------------------------------------------------------------
    if args.save_video:
        out_path = Path(args.save_video).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Replay] Rendering offline video to: {out_path} ...")

        render_h, render_w = 480, 640
        renderer = mujoco.Renderer(model, render_h, render_w)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

        out_w = render_w * 2 if (args.side_by_side and camera_frames) else render_w
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, render_h))

        for i in range(total_frames):
            # Base pose
            data.qpos[0] = 0.0
            data.qpos[1] = 0.0
            data.qpos[2] = 0.793
            quat = yaw_to_quat(base_yaws[i])
            data.qpos[3:7] = quat

            # Joints
            if args.mode == "state" and states is not None:
                data.qpos[7:36] = states[i]
            elif args.mode == "action" and actions is not None:
                # Legs & waist from state (if available) or default
                if states is not None:
                    data.qpos[7:22] = states[i, 0:15]
                # Arm joints from action: [0:7] left arm, [7:14] right arm
                data.qpos[22:29] = actions[i, 0:7]
                data.qpos[29:36] = actions[i, 7:14]

            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            sim_img = renderer.render()
            sim_bgr = cv2.cvtColor(sim_img, cv2.COLOR_RGB2BGR)

            # Draw HUD info
            mode_text = f"Mode: {args.mode.upper()} | Frame: {i:04d}/{total_frames} | Yaw: {np.degrees(base_yaws[i]):+.1f} deg"
            cv2.putText(sim_bgr, mode_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if args.mode == "action" and actions is not None:
                rx_val = actions[i, 16]
                cv2.putText(sim_bgr, f"Target Action (Remote RX: {rx_val:+.2f})", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if args.side_by_side and camera_frames:
                cam_img = camera_frames[i] if i < len(camera_frames) else np.zeros((render_h, render_w, 3), dtype=np.uint8)
                cam_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                cam_bgr = cv2.resize(cam_bgr, (render_w, render_h))
                cv2.putText(cam_bgr, "Original Camera (Global View)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                combined = np.hstack([cam_bgr, sim_bgr])
                writer.write(combined)
            else:
                writer.write(sim_bgr)

            if i % 60 == 0 or i == total_frames - 1:
                print(f"  Rendering [{i:04d}/{total_frames:04d}] ({i/total_frames*100:.1f}%)", end="\r", flush=True)

        writer.release()
        print(f"\n[Success] Video successfully saved to: {out_path}")
        return

    # -------------------------------------------------------------
    # Execution Mode: Interactive GUI Viewer
    # -------------------------------------------------------------
    print("\n==============================================================")
    print("                 MuJoCo Dataset Replayer                     ")
    print("==============================================================")
    print(f"  Playing: Episode {args.episode} ({args.mode.upper()} mode)")
    print("  Controls in viewer:")
    print("    - Space: Pause / Resume playback")
    print("    - Right Mouse: Rotate camera, Left Mouse: Pan")
    print("    - Mouse Scroll: Zoom in/out")
    print("    - Esc: Exit viewer")
    print("==============================================================\n")

    if not hasattr(mujoco, "viewer"):
        print("[Error] mujoco.viewer is not available in your Python environment.")
        print("Tip: Please use --save-video <output.mp4> to export an MP4 video instead!")
        return

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                frame_i = 0
                while frame_i < total_frames and viewer.is_running():
                    t0 = time.time()

                    # Set base pose with yaw
                    data.qpos[0] = 0.0
                    data.qpos[1] = 0.0
                    data.qpos[2] = 0.793
                    quat = yaw_to_quat(base_yaws[frame_i])
                    data.qpos[3:7] = quat

                    # Set joint positions according to mode
                    if args.mode == "state" and states is not None:
                        data.qpos[7:36] = states[frame_i]
                    elif args.mode == "action" and actions is not None:
                        if states is not None:
                            data.qpos[7:22] = states[frame_i, 0:15]
                        data.qpos[22:29] = actions[frame_i, 0:7]
                        data.qpos[29:36] = actions[frame_i, 7:14]

                    mujoco.mj_forward(model, data)
                    viewer.sync()

                    ts = frame_i * dt
                    rx_str = f" | remote.rx={actions[frame_i, 16]:+.2f}" if actions is not None else ""
                    print(f"Episode {args.episode} [{frame_i:04d}/{total_frames:04d}] (t={ts:5.2f}s, Yaw={np.degrees(base_yaws[frame_i]):+5.1f} deg){rx_str}", end="\r", flush=True)

                    frame_i += 1

                    elapsed = time.time() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                print(f"\nFinished Episode {args.episode}.")
                if not args.loop:
                    break
                time.sleep(1.0)
    except Exception as e:
        print(f"\n[Note] GUI Viewer encountered: {e}")
        print("Tip: If you are running over remote SSH without X11, please use --save-video to export an MP4 video instead!")


if __name__ == "__main__":
    main()
