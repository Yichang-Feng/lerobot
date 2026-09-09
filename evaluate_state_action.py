#!/usr/bin/env python3
"""
evaluate_state_action.py

评估并对比 LeRobot 数据集中 State（实测物理关节状态）与 Action（期望目标控制动作）之间的差距。
生成三张专业分析图表：
  1. eval_1_jitter_comparison.png:       Action 抖动 vs State 抖动程度对比（时域波形与关节均方差）
  2. eval_2_position_discrepancy.png:     Action 与 State 之间的位置偏差（时序跟踪轨迹与抱箱夹持力矩/末尾投降姿态分析）
  3. eval_3_cross_dataset_comparison.png: 自身数据集 vs 开源数据集（unitree_box_move_blue_full）的横向指标对比（抖动比率与末尾姿态超调）

使用方法：
    python evaluate_state_action.py
    # 或指定数据集与 Episode：
    python evaluate_state_action.py \
        --dataset-ours datasets/g1_box_pick_turn_v30 --episode-ours 0 \
        --dataset-open datasets/unitree_box_move_blue_full --episode-open 20 \
        --output-dir evaluation_plots
"""

import argparse
import os
from pathlib import Path

# Set Matplotlib headless backend and temporary config dir before importing pyplot
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ARM_JOINT_NAMES = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow",
    "L_WristRoll", "L_WristPitch", "L_WristYaw",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow",
    "R_WristRoll", "R_WristPitch", "R_WristYaw",
]

ARM_SHORT_NAMES = [
    "L_ShP", "L_ShR", "L_ShY", "L_Elb", "L_WrR", "L_WrP", "L_WrY",
    "R_ShP", "R_ShR", "R_ShY", "R_Elb", "R_WrR", "R_WrP", "R_WrY",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate State vs Action discrepancy and compare datasets")
    parser.add_argument(
        "--dataset-ours",
        type=str,
        default="datasets/g1_box_pick_turn_v30",
        help="Path to current/own dataset directory",
    )
    parser.add_argument(
        "--episode-ours",
        type=int,
        default=0,
        help="Episode index for current dataset (default: 0)",
    )
    parser.add_argument(
        "--dataset-open",
        type=str,
        default="datasets/unitree_box_move_blue_full",
        help="Path to open-source dataset directory",
    )
    parser.add_argument(
        "--episode-open",
        type=int,
        default=20,
        help="Episode index for open-source dataset (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_plots",
        help="Directory to save the 3 generated figures",
    )
    return parser.parse_args()


def load_episode_data(dataset_path: Path, episode_idx: int) -> pd.DataFrame:
    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")

    target_df = None
    for pf in parquet_files:
        temp = pq.read_table(pf).to_pandas()
        if "episode_index" in temp.columns and episode_idx in temp["episode_index"].values:
            target_df = temp[temp["episode_index"] == episode_idx].reset_index(drop=True)
            break

    if target_df is None:
        raise ValueError(f"Episode {episode_idx} not found in {dataset_path}")

    return target_df


def extract_arm_arrays(df: pd.DataFrame):
    """Extract arm action (14-D) and arm state (14-D) arrays from dataframe."""
    st_full = np.vstack(df["observation.state"].values)
    act_full = np.vstack(df["action"].values)

    # In 29-D state: [0:12] legs, [12:15] waist, [15:22] left arm, [22:29] right arm
    arm_state = st_full[:, 15:29]
    # In 18-D action: [0:7] left arm, [7:14] right arm, [14:18] remote
    arm_action = act_full[:, 0:14]

    timestamps = (
        df["timestamp"].values
        if "timestamp" in df.columns
        else np.arange(len(df)) / 30.0
    )

    return arm_state, arm_action, timestamps


def compute_jitter_metrics(arm_data: np.ndarray, dt: float = 1.0 / 30.0):
    """
    Compute jitter as second-order finite difference (discrete acceleration / jerk proxy):
    jitter_t = |q_{t+1} - 2q_t + q_{t-1}|.
    """
    diff2 = np.diff(np.diff(arm_data, axis=0), axis=0)
    per_joint_mean = np.mean(np.abs(diff2), axis=0)
    overall_norm = np.linalg.norm(diff2, axis=1)
    return diff2, per_joint_mean, overall_norm


# ==============================================================================
# Plot 1: Action Jitter vs State Jitter Comparison
# ==============================================================================
def plot_figure_1_jitter(
    arm_st_ours, arm_act_ours, t_ours, output_path: Path, ep_idx: int
):
    print(f"Generating Plot 1: Jitter Comparison -> {output_path} ...")

    diff2_st, mean_jit_st, norm_jit_st = compute_jitter_metrics(arm_st_ours)
    diff2_act, mean_jit_act, norm_jit_act = compute_jitter_metrics(arm_act_ours)

    t_diff = t_ours[1:-1]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # 1. Bar Chart: Per-joint jitter comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(14)
    width = 0.38

    b1 = ax1.bar(
        x - width / 2,
        mean_jit_st * 1000,
        width,
        label="State (Measured Physical Position)",
        color="#2b5c8f",
        alpha=0.9,
    )
    b2 = ax1.bar(
        x + width / 2,
        mean_jit_act * 1000,
        width,
        label="Action (Commanded Target Position)",
        color="#d95f02",
        alpha=0.9,
    )

    ax1.set_title(
        f"Figure 1A: Arm Joint Jitter Comparison (Episode {ep_idx})\n"
        r"Jitter Metric = Mean Second Difference $\mathbb{E}[|q_{t+1} - 2q_t + q_{t-1}|]$ ($10^{-3}$ rad)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARM_SHORT_NAMES, rotation=30, ha="right", fontsize=11)
    ax1.set_ylabel("Jitter Magnitude ($10^{-3}$ rad)", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=12, loc="upper right")

    # Annotate ratio on top of bars
    for i in range(14):
        ratio = mean_jit_act[i] / max(mean_jit_st[i], 1e-6)
        h = mean_jit_act[i] * 1000
        ax1.text(
            x[i] + width / 2,
            h + 0.3,
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#b33000",
            fontweight="bold",
        )

    # 2. Time-series: Total Instantaneous Jitter Norm
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        t_diff,
        norm_jit_st,
        label="State Jitter Norm",
        color="#2b5c8f",
        linewidth=1.2,
        alpha=0.85,
    )
    ax2.plot(
        t_diff,
        norm_jit_act,
        label="Action Jitter Norm",
        color="#d95f02",
        linewidth=1.0,
        alpha=0.75,
    )
    ax2.set_title(
        "Figure 1B: Total Arm Jitter Across Time",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel(r"$\|\Delta^2 q\|_2$ (rad)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    # 3. Time-series zoom on a key joint (Left Shoulder Roll)
    ax3 = fig.add_subplot(gs[1, 1])
    # Show zoom window (e.g. 5 seconds in holding phase)
    mid_t = len(t_ours) // 2
    win = slice(max(0, mid_t - 75), min(len(t_ours), mid_t + 75))
    ax3.plot(
        t_ours[win],
        np.degrees(arm_act_ours[win, 1]),
        label="Action (Commanded)",
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
    )
    ax3.plot(
        t_ours[win],
        np.degrees(arm_st_ours[win, 1]),
        label="State (Measured)",
        color="#2b5c8f",
        linewidth=2.0,
    )
    ax3.set_title(
        "Figure 1C: Zoom-in Waveform (Left Shoulder Roll, 5s window)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Joint Angle (deg)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Done] Saved: {output_path}")


# ==============================================================================
# Plot 2: Position Discrepancy & Clamping / Posture Analysis
# ==============================================================================
def plot_figure_2_discrepancy(
    arm_st_ours, arm_act_ours, t_ours, output_path: Path, ep_idx: int
):
    print(f"Generating Plot 2: Position Discrepancy -> {output_path} ...")

    diff_deg = np.degrees(arm_act_ours - arm_st_ours)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Subplot 1: Clamping Joints Tracking & Error (Left Shoulder Roll & Elbow)
    ax1 = axes[0]
    ax1.plot(
        t_ours,
        np.degrees(arm_act_ours[:, 1]),
        label="Action Left Shoulder Roll (Command)",
        color="#d95f02",
        linestyle="--",
        linewidth=1.5,
    )
    ax1.plot(
        t_ours,
        np.degrees(arm_st_ours[:, 1]),
        label="State Left Shoulder Roll (Physical Surface)",
        color="#2b5c8f",
        linewidth=2.0,
    )
    ax1.plot(
        t_ours,
        np.degrees(arm_act_ours[:, 8]),
        label="Action Right Shoulder Roll",
        color="#e7298a",
        linestyle="--",
        linewidth=1.5,
    )
    ax1.plot(
        t_ours,
        np.degrees(arm_st_ours[:, 8]),
        label="State Right Shoulder Roll",
        color="#7570b3",
        linewidth=2.0,
    )
    ax1.set_title(
        f"Figure 2A: Shoulder Roll Position Tracking (Clamping Force Generation, Episode {ep_idx})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("Angle (deg)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", ncol=2, fontsize=10)

    # Subplot 2: Discrepancy Error Curves across Key Joints
    ax2 = axes[1]
    ax2.plot(
        t_ours,
        diff_deg[:, 0],
        label=r"$\Delta$ Left Shoulder Pitch (Lifting/Holding Force)",
        color="#1b9e77",
        linewidth=1.5,
    )
    ax2.plot(
        t_ours,
        diff_deg[:, 1],
        label=r"$\Delta$ Left Shoulder Roll (Inward Squeeze)",
        color="#d95f02",
        linewidth=1.8,
    )
    ax2.plot(
        t_ours,
        diff_deg[:, 3],
        label=r"$\Delta$ Left Elbow (Forearm Hugging Force)",
        color="#7570b3",
        linewidth=1.5,
    )
    ax2.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax2.set_title(
        r"Figure 2B: Position Discrepancy Error $\Delta q = q_{\mathrm{action}} - q_{\mathrm{state}}$ (deg)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_ylabel("Error (deg)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10)

    # Highlight box holding zone vs release zone
    # Find release phase (last 15% of episode)
    end_t = t_ours[-1]
    ax2.axvspan(
        end_t - 3.0,
        end_t,
        color="red",
        alpha=0.15,
        label="Post-Placement Phase (Surrender Pose)",
    )
    ax2.legend(loc="lower left", fontsize=10)

    # Subplot 3: Arm Outward Spread / Elevation at End of Episode (Surrender Pose)
    ax3 = axes[2]
    # Shoulder Roll difference: positive for L, negative for R indicates outward spread!
    ax3.plot(
        t_ours,
        np.degrees(arm_act_ours[:, 1]),
        label="Action L_ShoulderRoll (>0 = Abducted / Spread Out)",
        color="#d95f02",
        linewidth=1.8,
    )
    ax3.plot(
        t_ours,
        np.degrees(arm_act_ours[:, 8]),
        label="Action R_ShoulderRoll (<0 = Abducted / Spread Out)",
        color="#7570b3",
        linewidth=1.8,
    )
    ax3.plot(
        t_ours,
        np.degrees(arm_act_ours[:, 0]),
        label="Action L_ShoulderPitch (<0 = Raised to Chest)",
        color="#1b9e77",
        linestyle="-.",
        linewidth=1.5,
    )
    ax3.axvspan(end_t - 3.0, end_t, color="red", alpha=0.15)
    ax3.text(
        end_t - 2.8,
        35,
        "Arms Open & Raised\n(Surrender Pose Zone)",
        color="darkred",
        fontweight="bold",
        fontsize=11,
    )
    ax3.set_title(
        "Figure 2C: Arm Abduction & Elevation Trajectory (Analysis of Surrender Pose at End)",
        fontsize=13,
        fontweight="bold",
    )
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Angle (deg)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", ncol=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Done] Saved: {output_path}")


# ==============================================================================
# Plot 3: Cross-Dataset Comparison (Ours vs Open-source Baseline)
# ==============================================================================
def plot_figure_3_cross_comparison(
    arm_st_ours,
    arm_act_ours,
    arm_st_open,
    arm_act_open,
    output_path: Path,
    ep_ours_idx: int,
    ep_open_idx: int,
):
    print(f"Generating Plot 3: Cross-Dataset Comparison -> {output_path} ...")

    # Jitter calculations
    _, mean_jit_st_ours, _ = compute_jitter_metrics(arm_st_ours)
    _, mean_jit_act_ours, _ = compute_jitter_metrics(arm_act_ours)
    _, mean_jit_st_open, _ = compute_jitter_metrics(arm_st_open)
    _, mean_jit_act_open, _ = compute_jitter_metrics(arm_act_open)

    ratio_ours = mean_jit_act_ours / np.maximum(mean_jit_st_ours, 1e-6)
    ratio_open = mean_jit_act_open / np.maximum(mean_jit_st_open, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Action/State Jitter Ratio Bar Chart across 14 arm joints
    ax1 = axes[0, 0]
    x = np.arange(14)
    width = 0.38
    ax1.bar(
        x - width / 2,
        ratio_ours,
        width,
        label=f"Our Dataset (g1_box_pick_turn, Ep {ep_ours_idx})",
        color="#d95f02",
        alpha=0.9,
    )
    ax1.bar(
        x + width / 2,
        ratio_open,
        width,
        label=f"Open Baseline (unitree_box_move_blue, Ep {ep_open_idx})",
        color="#2b5c8f",
        alpha=0.9,
    )
    ax1.axhline(
        1.0, color="gray", linestyle="--", linewidth=1.2, label="Ratio = 1.0 (No Added Jitter)"
    )
    ax1.set_title(
        "Figure 3A: Action-to-State Jitter Ratio Comparison (Jitter Multiplier)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARM_SHORT_NAMES, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Jitter Ratio (Action / State)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # 2. Overall Jitter Summary (Box / Bar metric)
    ax2 = axes[0, 1]
    datasets = ["Our Dataset\n(g1_box_pick_turn)", "Open Baseline\n(box_move_blue)"]
    overall_ratio_ours = np.mean(mean_jit_act_ours) / np.mean(mean_jit_st_ours)
    overall_ratio_open = np.mean(mean_jit_act_open) / np.mean(mean_jit_st_open)
    bars = ax2.bar(
        datasets,
        [overall_ratio_ours, overall_ratio_open],
        color=["#d95f02", "#2b5c8f"],
        width=0.45,
        alpha=0.9,
    )
    ax2.set_title(
        "Figure 3B: Mean Action Jitter Inflation Factor",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Overall Jitter Ratio (Action / State)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(bars, [overall_ratio_ours, overall_ratio_open]):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    # 3. End-of-Episode Arm Rest Angles (Surrender Pose Check)
    ax3 = axes[1, 0]
    # Take last 30 frames
    last_act_ours_deg = np.degrees(np.mean(arm_act_ours[-30:], axis=0))
    last_act_open_deg = np.degrees(np.mean(arm_act_open[-30:], axis=0))

    key_indices = [0, 1, 3, 7, 8, 10]
    key_labels = [
        "L_ShPitch\n(Raise)",
        "L_ShRoll\n(Abduct)",
        "L_Elbow\n(Bend)",
        "R_ShPitch\n(Raise)",
        "R_ShRoll\n(Abduct)",
        "R_Elbow\n(Bend)",
    ]
    x_k = np.arange(len(key_indices))
    ax3.bar(
        x_k - width / 2,
        last_act_ours_deg[key_indices],
        width,
        label=f"Our Dataset (Ep {ep_ours_idx})",
        color="#d95f02",
        alpha=0.9,
    )
    ax3.bar(
        x_k + width / 2,
        last_act_open_deg[key_indices],
        width,
        label=f"Open Baseline (Ep {ep_open_idx})",
        color="#2b5c8f",
        alpha=0.9,
    )
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_title(
        "Figure 3C: End-of-Episode Rest Pose Comparison\n(Exaggerated Surrender Pose Check)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xticks(x_k)
    ax3.set_xticklabels(key_labels, fontsize=10)
    ax3.set_ylabel("Angle (deg)", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9, loc="lower right")

    # 4. Position Tracking Error (q_act - q_st) Distribution
    ax4 = axes[1, 1]
    err_ours = (arm_act_ours - arm_st_ours).flatten()
    err_open = (arm_act_open - arm_st_open).flatten()

    ax4.hist(
        np.degrees(err_ours),
        bins=50,
        density=True,
        alpha=0.6,
        color="#d95f02",
        label=f"Our Dataset (Std={np.std(np.degrees(err_ours)):.1f}°)",
    )
    ax4.hist(
        np.degrees(err_open),
        bins=50,
        density=True,
        alpha=0.6,
        color="#2b5c8f",
        label=f"Open Baseline (Std={np.std(np.degrees(err_open)):.1f}°)",
    )
    ax4.set_title(
        r"Figure 3D: Error Distribution $\Delta q = q_{\mathrm{act}} - q_{\mathrm{st}}$ (All Arm Joints)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Discrepancy (deg)", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Done] Saved: {output_path}")


def main():
    args = parse_args()
    ds_ours_path = Path(args.dataset_ours).expanduser().resolve()
    ds_open_path = Path(args.dataset_open).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      State vs Action Discrepancy & Cross-Dataset Evaluator     ")
    print("=" * 70)
    print(f"Dataset 1 (Ours): {ds_ours_path} (Episode {args.episode_ours})")
    print(f"Dataset 2 (Open): {ds_open_path} (Episode {args.episode_open})")
    print(f"Output Directory: {out_dir}\n")

    # Load datasets
    df_ours = load_episode_data(ds_ours_path, args.episode_ours)
    df_open = load_episode_data(ds_open_path, args.episode_open)

    arm_st_ours, arm_act_ours, t_ours = extract_arm_arrays(df_ours)
    arm_st_open, arm_act_open, t_open = extract_arm_arrays(df_open)

    # 1. Generate Figure 1
    p1 = out_dir / "eval_1_jitter_comparison.png"
    plot_figure_1_jitter(arm_st_ours, arm_act_ours, t_ours, p1, args.episode_ours)

    # 2. Generate Figure 2
    p2 = out_dir / "eval_2_position_discrepancy.png"
    plot_figure_2_discrepancy(arm_st_ours, arm_act_ours, t_ours, p2, args.episode_ours)

    # 3. Generate Figure 3
    p3 = out_dir / "eval_3_cross_dataset_comparison.png"
    plot_figure_3_cross_comparison(
        arm_st_ours,
        arm_act_ours,
        arm_st_open,
        arm_act_open,
        p3,
        args.episode_ours,
        args.episode_open,
    )

    # Output Numerical Summary
    _, j_st_ours, _ = compute_jitter_metrics(arm_st_ours)
    _, j_act_ours, _ = compute_jitter_metrics(arm_act_ours)
    _, j_st_open, _ = compute_jitter_metrics(arm_st_open)
    _, j_act_open, _ = compute_jitter_metrics(arm_act_open)

    print("\n" + "=" * 70)
    print("                      EVALUATION SUMMARY REPORT                         ")
    print("=" * 70)
    print(f"1. Action Jitter Multiplier (Action / State):")
    print(f"   - Our Dataset (Ep {args.episode_ours:2d})  : {np.mean(j_act_ours)/np.mean(j_st_ours):.2f}x (Act: {np.mean(j_act_ours)*1000:.2f} vs St: {np.mean(j_st_ours)*1000:.2f} mrad)")
    print(f"   - Open Baseline (Ep {args.episode_open:2d}): {np.mean(j_act_open)/np.mean(j_st_open):.2f}x (Act: {np.mean(j_act_open)*1000:.2f} vs St: {np.mean(j_st_open)*1000:.2f} mrad)")
    print()
    print(f"2. End-of-Episode Rest Posture (Last 30 frames avg):")
    last_ours = np.degrees(np.mean(arm_act_ours[-30:], axis=0))
    last_open = np.degrees(np.mean(arm_act_open[-30:], axis=0))
    print(f"   - Our Dataset  : L_Pitch={last_ours[0]:+5.1f}°, L_Roll={last_ours[1]:+5.1f}°, R_Roll={last_ours[8]:+5.1f}° (Surrender Pose)")
    print(f"   - Open Baseline: L_Pitch={last_open[0]:+5.1f}°, L_Roll={last_open[1]:+5.1f}°, R_Roll={last_open[8]:+5.1f}° (Relaxed / Forward)")
    print("=" * 70)
    print(f"All 3 evaluation figures saved successfully to: {out_dir}")
    print(f"  - 1. {p1}")
    print(f"  - 2. {p2}")
    print(f"  - 3. {p3}")
    print("=" * 70)


if __name__ == "__main__":
    main()
