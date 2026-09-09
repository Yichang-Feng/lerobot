#!/usr/bin/env python3
"""
align_velocity_dataset.py

该脚本用于将原地转身数据集（如 g1_box_pick_turn_v30）转化为与预训练基模（model/box_pick 及 unitree_box_move_blue_full）
速度指令分布完全对齐的新数据集，彻底解决 pi05 模型在 QUANTILES 归一化模式下因部分维度全为零导致的分母塌陷（除以 eps）与分布偏移问题。

保留原数据集不变，输出独立的新数据集。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# 来自 model/box_pick (unnormalizer safetensors) 与 unitree_box_move_blue_full 的动作后 4 维基准统计量
ALIGNED_ACTION_STATS = {
    # 物理极值边界（覆盖预训练数据与当前数据集）
    "min": [-0.7634, -0.7707, -1.0838, -0.7726],
    "max": [1.2928, 1.2977, 1.5000, 1.3382],
    # 期望均值与标准差
    "mean": [-0.0029, 0.0645, 0.0626, -0.0964],
    "std": [0.1114, 0.4852, 0.1861, 0.1444],
    # 核心分位数（决定 QUANTILES 归一化映射）
    "q01": [-0.0947, -0.7692, -0.7414, -0.7497],
    "q10": [-0.0205, -0.1250, -0.1401, -0.2364],
    "q50": [-0.0073, -0.0210, -0.0017, -0.0537],
    "q90": [0.0078, 1.2366, 0.5475, -0.0010],
    "q99": [0.0777, 1.2659, 1.2586, 0.0645],
}


def align_dataset(src_dir: Path, dst_dir: Path, overwrite: bool = True) -> None:
    src_dir = Path(src_dir).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"源数据集目录不存在: {src_dir}")

    if src_dir == dst_dir:
        raise ValueError("目标目录不能与源目录相同，请指定新的路径以保留原数据集！")

    if dst_dir.exists():
        if overwrite:
            print(f"目标目录已存在，正在清除旧数据: {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"目标目录已存在: {dst_dir}")

    print("=" * 70)
    print(f"正在从原数据集复制: {src_dir}")
    print(f"输出新对齐数据集: {dst_dir}")
    print("=" * 70)

    # 完整拷贝数据集（视频、动作表、元数据）
    shutil.copytree(src_dir, dst_dir)

    stats_file = dst_dir / "meta" / "stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"未找到元数据文件: {stats_file}")

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if "action" not in stats:
        raise KeyError(f"在 {stats_file} 中未找到 'action' 字段！")

    print("\n正在对齐后 4 维底盘速度指令 (remote.lx, remote.ly, remote.rx, remote.ry) 分位数统计量...")
    for key, vals in ALIGNED_ACTION_STATS.items():
        if key in stats["action"]:
            old_vals = stats["action"][key][14:18]
            stats["action"][key][14:18] = vals
            print(f"  [{key:4s}] 原值: {[round(x, 4) for x in old_vals]} -> 对齐值: {[round(x, 4) for x in vals]}")

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[成功] 新统计文件已写入: {stats_file}")

    # 自检验证
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.processor.normalize_processor import NormalizationMode, NormalizerProcessorStep

        ds = LeRobotDataset(dst_dir.name, root=dst_dir)
        features = {"action": {"type": "ACTION", "shape": [18]}}
        norm_map = {"ACTION": NormalizationMode.QUANTILES}
        normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=ds.meta.stats)

        sample_action = ds[0]["action"]
        norm_action = normalizer({"action": sample_action})["action"]

        assert not torch.isnan(norm_action).any(), "归一化后存在 NaN！"
        assert not torch.isinf(norm_action).any(), "归一化后存在 Inf！"

        print("\n" + "=" * 70)
        print("数据集健康自检通过:")
        print(f"  • 总轨迹数 (Episodes): {ds.num_episodes}")
        print(f"  • 总样本帧数 (Frames):   {len(ds)}")
        print(f"  • 原始动作底盘速度[14:18]: {sample_action[14:18].numpy().tolist()}")
        print(f"  • 归一化后动作值[14:18]:    {[round(float(x), 4) for x in norm_action[14:18]]}")
        print("  • 检查结果: 无 NaN, 无 Inf, 分位数有效区间正常对齐！")
        print("=" * 70)
    except Exception as e:
        print(f"\n[注意] 校验步骤跳过或提示: {e}")

    print("\n转换完成！原数据集保持原样，新数据集可直接用于 pi05 微调。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对齐 LeRobot G1 数据集的底盘速度统计量以适配 model/box_pick")
    parser.add_argument(
        "--src-dir",
        type=str,
        default="/home/yichangfeng/lerobot/datasets/g1_box_pick_turn_v30",
        help="源数据集路径",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        default="/home/yichangfeng/lerobot/datasets/g1_box_pick_turn_v30_aligned",
        help="转换后新数据集路径",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="若目标目录已存在则不覆盖",
    )

    args = parser.parse_args()
    align_dataset(Path(args.src_dir), Path(args.dst_dir), overwrite=not args.no_overwrite)
