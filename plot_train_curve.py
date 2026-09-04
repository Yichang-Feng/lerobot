#!/usr/bin/env python3
"""
Plot training loss curve from LeRobot training logs.
Usage:
    python plot_train_curve.py [log_file]
    If no log_file is provided, it reads from stdin or default train.log.
"""
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(lines):
    steps = []
    losses = []
    lrs = []
    
    # Matches: step:100 ... loss:0.123 ... lr:2.5e-05
    step_pattern = re.compile(r"step:([0-9kK\.]+)")
    loss_pattern = re.compile(r"loss:([0-9\.]+)")
    lr_pattern = re.compile(r"lr:([0-9\.eE\-\+]+)")
    
    for line in lines:
        if "loss:" in line and "step:" in line:
            m_step = step_pattern.search(line)
            m_loss = loss_pattern.search(line)
            m_lr = lr_pattern.search(line)
            if m_step and m_loss:
                s_val = m_step.group(1).lower()
                if "k" in s_val:
                    step_val = int(float(s_val.replace("k", "")) * 1000)
                else:
                    step_val = int(s_val)
                loss_val = float(m_loss.group(1))
                steps.append(step_val)
                losses.append(loss_val)
                if m_lr:
                    lrs.append(float(m_lr.group(1)))
    return steps, losses, lrs

def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
        if not log_path.exists():
            print(f"Error: Log file '{log_path}' not found.")
            sys.exit(1)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    elif not sys.stdin.isatty():
        lines = sys.stdin.readlines()
    elif Path("train.log").exists():
        with open("train.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        print("Usage: python plot_train_curve.py <log_file>")
        print("Or pipe log text: python plot_train_curve.py < train.log")
        sys.exit(1)

    steps, losses, lrs = parse_log(lines)
    if not steps:
        print("No training steps/losses found in the provided log.")
        sys.exit(1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(steps, losses, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Loss", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if lrs and len(lrs) == len(steps):
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, "g--", alpha=0.6, label="Learning Rate")
        ax2.set_ylabel("Learning Rate", color="g", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="g")

    plt.title("LeRobot Pi0.5 Fine-tuning Loss Curve", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_png = "training_loss_curve.png"
    plt.savefig(output_png, dpi=200)
    print(f"Loss curve successfully saved to: {output_png} ({len(steps)} data points plotted)")

if __name__ == "__main__":
    main()
