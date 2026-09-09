#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""
Offline Evaluation Script for Trained VLA Policy Checkpoints.

Features:
1. Automatically scans all checkpoints in outputs/train/ (e.g. pi05_box_pick_turn_aligned, pi05_box_pick_turn_final).
2. Supports step 5000 (even if training_state was truncated, pretrained_model is intact).
3. Evaluates on held-out validation episodes from the specified dataset.
4. Computes:
   - Val Loss: Flow Matching validation loss
   - Arm MAE / MSE: 14-DoF arm joint angle error (in radians)
   - Locomotion MAE: 4-DoF remote joystick command error (remote.lx/ly/rx/ry)
   - Action Cosine Similarity: Overall trajectory direction agreement
5. Outputs a ranked leaderboard in terminal and exports results to JSON & Markdown.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_offline")


def find_candidate_checkpoints(train_dir: Path, run_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Discover all available checkpoints across training runs."""
    candidates = []
    if not train_dir.exists():
        logger.error(f"Train directory does not exist: {train_dir}")
        return candidates

    for run_dir in sorted(train_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_names and run_dir.name not in run_names:
            continue

        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.exists():
            continue

        for ckpt_dir in sorted(ckpt_root.iterdir()):
            if ckpt_dir.name == "last" or not ckpt_dir.is_dir():
                continue

            model_dir = ckpt_dir / "pretrained_model"
            if not model_dir.exists():
                model_dir = ckpt_dir

            # Verify presence of model.safetensors and config.json
            if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
                candidates.append({
                    "run_name": run_dir.name,
                    "step": ckpt_dir.name,
                    "ckpt_path": model_dir,
                    "display_name": f"{run_dir.name}/{ckpt_dir.name}",
                })

    return candidates


def evaluate_single_checkpoint(
    ckpt_info: dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
    task_prompt: str,
    max_batches: int = -1,
) -> dict[str, float]:
    """Evaluate one checkpoint on the validation dataloader."""
    ckpt_path = ckpt_info["ckpt_path"]
    logger.info(f">>> Evaluating: {ckpt_info['display_name']} ...")

    # 1. Load config & enforce compile_model=False to avoid compilation overhead
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt_path))
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False

    # 2. Instantiate policy
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(str(ckpt_path), config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    # 3. Instantiate preprocessor & postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    total_loss = 0.0
    arm_maes = []
    arm_mses = []
    loco_maes = []
    cos_sims = []
    num_samples = 0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches > 0 and batch_count >= max_batches:
                break

            # Override/insert language task string if needed
            batch["task"] = [task_prompt] * len(batch[list(batch.keys())[0]])

            # Preprocess inputs
            processed_batch = preprocessor(batch)

            # Metric 1: Flow Matching Validation Loss
            try:
                loss_val, _ = policy.forward(processed_batch)
                total_loss += loss_val.item() * len(batch["action"])
            except Exception as e:
                logger.debug(f"Loss forward failed: {e}")

            # Metric 2: Predict action chunk and compare
            try:
                # Predict action chunk
                pred_action_norm = policy.predict_action_chunk(processed_batch)
                
                # Unnormalize ground truth and predicted actions to physical space
                gt_action_norm = processed_batch[ACTION]
                
                pred_action_phys = postprocessor(pred_action_norm)
                gt_action_phys = postprocessor(gt_action_norm)

                # Ensure dimensions match for first execution step
                if pred_action_phys.ndim == 3:
                    pred_step0 = pred_action_phys[:, 0, :]
                else:
                    pred_step0 = pred_action_phys

                if gt_action_phys.ndim == 3:
                    gt_step0 = gt_action_phys[:, 0, :]
                else:
                    gt_step0 = gt_action_phys

                # Slice arms (first 14) and locomotion axes (last 4)
                # G1 Action space: 14 Arm Joints + 4 Remote Axes
                arm_pred = pred_step0[:, :14].cpu().numpy()
                arm_gt = gt_step0[:, :14].cpu().numpy()

                loco_pred = pred_step0[:, 14:18].cpu().numpy()
                loco_gt = gt_step0[:, 14:18].cpu().numpy()

                # Arm MAE & MSE (in radians)
                arm_maes.append(np.abs(arm_pred - arm_gt).mean())
                arm_mses.append(np.mean((arm_pred - arm_gt) ** 2))

                # Locomotion MAE (remote joystick command scale)
                loco_maes.append(np.abs(loco_pred - loco_gt).mean())

                # Cosine similarity across full action vector
                pred_flat = pred_step0.reshape(len(pred_step0), -1)
                gt_flat = gt_step0.reshape(len(gt_step0), -1)
                cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean().item()
                cos_sims.append(cos_sim)

            except Exception as e:
                logger.warning(f"Action prediction failed on batch: {e}")

            num_samples += len(batch["action"])
            batch_count += 1

    # Cleanup GPU memory immediately
    del policy, preprocessor, postprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    mean_val_loss = total_loss / max(1, num_samples)
    mean_arm_mae = float(np.mean(arm_maes)) if arm_maes else float("nan")
    mean_arm_mse = float(np.mean(arm_mses)) if arm_mses else float("nan")
    mean_loco_mae = float(np.mean(loco_maes)) if loco_maes else float("nan")
    mean_cos_sim = float(np.mean(cos_sims)) if cos_sims else float("nan")

    logger.info(
        f"  ✓ {ckpt_info['display_name']} | Val Loss: {mean_val_loss:.4f} | "
        f"Arm MAE: {mean_arm_mae:.4f} rad ({np.degrees(mean_arm_mae):.2f}°) | "
        f"Loco MAE: {mean_loco_mae:.4f} | Cos Sim: {mean_cos_sim:.4f}"
    )

    return {
        "val_loss": round(mean_val_loss, 5),
        "arm_mae_rad": round(mean_arm_mae, 5),
        "arm_mae_deg": round(float(np.degrees(mean_arm_mae)), 2),
        "arm_mse_rad": round(mean_arm_mse, 5),
        "loco_mae": round(mean_loco_mae, 5),
        "action_cos_sim": round(mean_cos_sim, 4),
        "num_samples_evaluated": num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline Evaluation Benchmark for G1 PI0.5 Checkpoints")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train"),
        help="Root directory containing training runs (default: outputs/train)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated run directory names to evaluate (e.g. pi05_box_pick_turn_aligned,pi05_box_pick_turn_final)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated step numbers to filter (e.g. 003000,004000,005000). Default: all available",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="g1_box_pick_turn_v30_aligned",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/g1_box_pick_turn_v30_aligned"),
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--val-episodes",
        type=str,
        default="last_10%",
        help="Validation episodes: 'last_10%', 'last_5', or comma-separated list of episode IDs (e.g. 58,59,60,61,62)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60,
        help="Maximum validation frames to evaluate per checkpoint for speed (default: 60, use -1 for all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the box, turn right, and place it on the table",
        help="Task description string for the policy prompt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("eval_results.json"),
        help="Path to export evaluation results JSON (default: eval_results.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("eval_results.md"),
        help="Path to export summary leaderboard Markdown (default: eval_results.md)",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print(" [G1 PI0.5 策略模型离线综合评测工具]")
    print(f" 训练产物根目录 : {args.train_dir}")
    print(f" 评测验证数据集 : {args.dataset_root}")
    print(f" 计算运行设备   : {device}")
    print(f" 评估样本数量   : {args.max_samples if args.max_samples > 0 else '全部'}")
    print("=" * 80)

    # 1. Discover Checkpoints
    run_filter = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
    step_filter = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None

    all_ckpts = find_candidate_checkpoints(args.train_dir, run_filter)
    if step_filter:
        all_ckpts = [c for c in all_ckpts if c["step"] in step_filter]

    if not all_ckpts:
        logger.error("No valid checkpoints found! Check --train-dir or filter arguments.")
        sys.exit(1)

    logger.info(f"Discovered {len(all_ckpts)} checkpoint(s) to evaluate:")
    for c in all_ckpts:
        logger.info(f"  - {c['display_name']} -> {c['ckpt_path']}")

    # 2. Setup Dataset & Validation Split
    if not args.dataset_root.exists():
        logger.error(f"Dataset directory not found: {args.dataset_root}")
        sys.exit(1)

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    total_episodes = ds_meta.total_episodes

    # Resolve validation episode indices
    if args.val_episodes == "last_10%":
        val_count = max(1, int(total_episodes * 0.10))
        val_ep_indices = list(range(total_episodes - val_count, total_episodes))
    elif args.val_episodes.startswith("last_"):
        n = int(args.val_episodes.replace("last_", ""))
        val_ep_indices = list(range(max(0, total_episodes - n), total_episodes))
    else:
        val_ep_indices = [int(x.strip()) for x in args.val_episodes.split(",") if x.strip()]

    logger.info(f"Using {len(val_ep_indices)} validation episodes: {val_ep_indices} (total: {total_episodes})")

    # Pick first checkpoint's config to resolve delta timestamps
    sample_cfg = PreTrainedConfig.from_pretrained(str(all_ckpts[0]["ckpt_path"]))
    delta_timestamps = resolve_delta_timestamps(sample_cfg, ds_meta)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=val_ep_indices,
        delta_timestamps=delta_timestamps,
    )
    logger.info(f"Validation dataset loaded: {len(dataset)} frames total across episodes {val_ep_indices}")

    # Subsample frames if max_samples is specified
    if 0 < args.max_samples < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.max_samples, dtype=int).tolist()
        eval_subset = Subset(dataset, indices)
    else:
        eval_subset = dataset

    dataloader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Run Benchmark on all Checkpoints
    results = []
    for ckpt_info in all_ckpts:
        metrics = evaluate_single_checkpoint(
            ckpt_info=ckpt_info,
            dataloader=dataloader,
            device=device,
            task_prompt=args.task,
        )
        results.append({
            "run_name": ckpt_info["run_name"],
            "step": ckpt_info["step"],
            "model_name": ckpt_info["display_name"],
            **metrics,
        })

    # 4. Rank Models (Primary: Arm MAE in rad, Secondary: Val Loss)
    ranked_results = sorted(
        results,
        key=lambda x: (x["arm_mae_rad"] if not np.isnan(x["arm_mae_rad"]) else 999.0, x["val_loss"]),
    )

    # 5. Print Formatted Leaderboard
    print("\n" + "=" * 92)
    print("                      【G1 PI0.5 策略模型离线评测排行榜】")
    print("=" * 92)
    header = (
        f"{'排名':<4} | {'模型名称 (Run/Step)':<38} | {'Val Loss':<9} | "
        f"{'手臂 MAE (°)':<12} | {'遥控 MAE':<9} | {'动作相似度':<10}"
    )
    print(header)
    print("-" * 92)

    for rank, res in enumerate(ranked_results, start=1):
        deg_str = f"{res['arm_mae_deg']:.2f}°"
        print(
            f" #{rank:<3} | {res['model_name']:<38} | {res['val_loss']:<9.4f} | "
            f"{deg_str:<12} | {res['loco_mae']:<9.4f} | {res['action_cos_sim']:<10.4f}"
        )
    print("=" * 92)

    best_model = ranked_results[0]
    print(f"\n🏆 [推荐最优模型]: {best_model['model_name']}")
    print(f"   - 手臂关节平均误差: {best_model['arm_mae_deg']:.2f}° ({best_model['arm_mae_rad']:.4f} rad)")
    print(f"   - 动作方向相似度:   {best_model['action_cos_sim']:.4f}")
    print(f"   - 验证集 Loss:     {best_model['val_loss']:.4f}\n")

    # 6. Save JSON & Markdown
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=4, ensure_ascii=False)
    logger.info(f"Detailed JSON results exported to: {args.output_json.resolve()}")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# G1 PI0.5 策略模型离线评测排行榜\n\n")
        f.write(f"- **评测路径**: `{Path.cwd()}`\n")
        f.write(f"- **验证数据集**: `{args.dataset_root}`\n")
        f.write(f"- **验证 Episode 列表**: `{val_ep_indices}`\n\n")
        f.write("| 排名 | 模型名称 | Val Loss | 手臂 MAE (rad) | 手臂 MAE (°) | 遥控 MAE | 动作相似度 |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for rank, res in enumerate(ranked_results, start=1):
            f.write(
                f"| **#{rank}** | `{res['model_name']}` | {res['val_loss']:.4f} | "
                f"{res['arm_mae_rad']:.4f} | {res['arm_mae_deg']:.2f}° | "
                f"{res['loco_mae']:.4f} | {res['action_cos_sim']:.4f} |\n"
            )
        f.write(f"\n> **推荐优先闭环部署/仿真测试的模型**: `{best_model['model_name']}`\n")
    logger.info(f"Markdown leaderboard exported to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
