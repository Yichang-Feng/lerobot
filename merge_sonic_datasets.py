#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SonicStar / WBC 多批次数据集自动化合并工具 (merge_sonic_datasets.py)

功能：
1. 支持将多个独立采集并清洗后的 Session 数据集文件夹（带时间戳）按时序一键合并为一个统一的大数据集；
2. 自动过滤备份目录（如 *_backup_* 和 *_staged_*），避免误合并历史备份；
3. 严格重编排全局 Episode 序号 (episode_000000 ~ episode_{N-1})，无缝平移 Parquet 与 MP4 文件；
4. 全局累加自增 index 字段与 episode_index，确保下游 DataLoader 与转码脚本 100% 兼容；
5. 自动整合并重写 info.json, episodes.jsonl, episodes_stats.jsonl 元数据；
6. 目标目录若已存在，自动执行时间戳全量安全备份后再进行覆盖替换；
7. 最终进行全量 1:1 帧率与帧数严密校验 (Diff = 0)。

使用示例：
    # 场景 1: 合并所有带时间戳的批次目录到主数据集目录
    /home/yichangfeng/miniforge3/envs/lerobot/bin/python ~/lerobot/merge_sonic_datasets.py \
        --src-dirs ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_* \
        --output-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn

    # 场景 2: 预览合并计划（干跑模式，不修改磁盘）
    /home/yichangfeng/miniforge3/envs/lerobot/bin/python ~/lerobot/merge_sonic_datasets.py \
        --src-dirs ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn_* \
        --output-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn \
        --dry-run

    # 场景 3: 显式指定多个目录按顺序合并
    /home/yichangfeng/miniforge3/envs/lerobot/bin/python ~/lerobot/merge_sonic_datasets.py \
        --src-dirs ~/SonicStar/wbc/outputs/session_1 ~/SonicStar/wbc/outputs/session_2 \
        --output-dir ~/SonicStar/wbc/outputs/g1_rubberhand_pick_turn
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def compute_column_stats(df: pa.Table) -> Dict[str, dict]:
    """计算单个 Episode 的全量列统计数据"""
    pandas_df = df.to_pandas()
    stats = {}
    for col in pandas_df.columns:
        series = pandas_df[col]
        first_val = series.iloc[0]
        if hasattr(first_val, "__len__") and not isinstance(first_val, str):
            val_array = np.vstack(series.values)
        else:
            val_array = np.array(series.values)[:, None]

        stats[col] = {
            "min": np.min(val_array, axis=0).tolist(),
            "max": np.max(val_array, axis=0).tolist(),
            "mean": np.mean(val_array, axis=0).tolist(),
            "std": np.std(val_array, axis=0).tolist(),
            "count": [len(val_array)],
        }
    return stats


def resolve_source_dirs(raw_inputs: List[str], output_dir: Path) -> List[Path]:
    """解析并过滤输入的源数据集目录列表"""
    matched_dirs = set()
    explicit_paths = set()
    for item in raw_inputs:
        expanded = str(Path(item).expanduser())
        p_raw = Path(expanded)
        if p_raw.exists() and p_raw.is_dir():
            explicit_paths.add(p_raw.resolve())

        hits = glob.glob(expanded)
        if not hits:
            if p_raw.exists() and p_raw.is_dir():
                matched_dirs.add(p_raw.resolve())
        else:
            for hit in hits:
                p = Path(hit).resolve()
                if p.is_dir():
                    matched_dirs.add(p)

    resolved = []
    output_resolved = output_dir.resolve()

    for p in sorted(list(matched_dirs)):
        name = p.name
        # 自动过滤备份目录与暂存目录
        if any(keyword in name for keyword in ["_backup_", "_staged_", "_staging_", "_trial_backup"]):
            continue
        # 如果是通配符匹配出的 output_dir 本身，且用户并未显式将其作为独立参数传入，则自动跳过避免重复
        if p == output_resolved and output_resolved not in explicit_paths:
            continue
        # 校验是否包含合法的 data 与 meta 结构
        if (p / "data/chunk-000").exists() and (p / "meta/info.json").exists():
            resolved.append(p)
        else:
            print(f"[!] 提示: 跳过非有效 Sonic 数据集目录: {p}")

    return resolved


def inspect_dataset(ds_dir: Path) -> Tuple[List[int], Dict[int, int]]:
    """检查数据集中的连续 Episode 及帧数"""
    data_dir = ds_dir / "data/chunk-000"
    vid_dir = ds_dir / "videos/chunk-000/observation.images.ego_view"

    pqs = sorted(data_dir.glob("episode_*.parquet"))
    valid_eps = []
    ep_lengths = {}

    for p in pqs:
        try:
            ep_idx = int(p.stem.split("_")[1])
        except Exception:
            continue
        vid_file = vid_dir / f"episode_{ep_idx:06d}.mp4"
        if not vid_file.exists():
            continue
        # 读取表格长度
        try:
            meta = pq.read_metadata(p)
            n_rows = meta.num_rows
        except Exception:
            df = pq.read_table(p)
            n_rows = len(df)
        valid_eps.append(ep_idx)
        ep_lengths[ep_idx] = n_rows

    valid_eps.sort()
    return valid_eps, ep_lengths


def main():
    parser = argparse.ArgumentParser(description="SonicStar / WBC 多批次数据集自动化合并工具")
    parser.add_argument(
        "--src-dirs",
        type=str,
        nargs="+",
        required=True,
        help="待合并的源数据集目录列表或通配符匹配 (如 --src-dirs outputs/session_* outputs/other_dir)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="合并后生成的目标主数据集路径 (如 outputs/g1_rubberhand_pick_turn)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="采集帧率 (默认 50.0 Hz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演练模式：仅显示合并计划与汇总，不修改任何磁盘文件",
    )
    args = parser.parse_args()

    dst_dir = Path(args.output_dir).expanduser().resolve()
    src_dirs = resolve_source_dirs(args.src_dirs, dst_dir)

    if not src_dirs:
        print("[!] 错误: 未发现任何符合条件的源数据集目录！请检查 --src-dirs 参数。")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("             SonicStar 多批次数据集整合规划 (Dataset Merge Plan)")
    print("=" * 78)
    print(f" 目标主目录 : {dst_dir}")
    print(f" 参与合并的批次目录 ({len(src_dirs)} 个):")

    total_episodes_planned = 0
    total_frames_planned = 0
    batch_plan = []

    for idx, s_dir in enumerate(src_dirs):
        eps, ep_lens = inspect_dataset(s_dir)
        n_eps = len(eps)
        n_frames = sum(ep_lens.values())
        batch_plan.append((s_dir, eps, ep_lens))
        print(f"   [{idx + 1}] {s_dir.name:<36} -> {n_eps:3d} 条 Episode, 共 {n_frames:6d} 帧 ({n_frames/args.fps:6.1f}s)")
        total_episodes_planned += n_eps
        total_frames_planned += n_frames

    print("-" * 78)
    print(f" 整合后总规模 : {total_episodes_planned} 条 Episode (新编号: episode_000000 ~ episode_{total_episodes_planned - 1:06d})")
    print(f" 整合后总帧数 : {total_frames_planned} 帧 (总录制时长: {total_frames_planned / args.fps:.2f} 秒)")
    print("=" * 78)

    if args.dry_run:
        print("\n[INFO] 当前处于 --dry-run 演练模式，未修改磁盘。确认无误后去掉 --dry-run 参数执行即可。")
        return

    # 1. 安全备份目标目录（若已存在）
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if dst_dir.exists():
        backup_path = dst_dir.parent / f"{dst_dir.name}_backup_{timestamp_str}"
        print(f"\n[1/5] 检测到目标目录已存在，正在创建安全备份:\n      -> {backup_path}")
        shutil.copytree(dst_dir, backup_path)
        print("[+] 目标目录备份完成！随时可通过备份还原。")

    # 2. 准备暂存隔离区 (Staging Area)
    staged_dir = dst_dir.parent / f"{dst_dir.name}_merge_staging_{timestamp_str}"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)

    staged_data = staged_dir / "data/chunk-000"
    staged_vid = staged_dir / "videos/chunk-000/observation.images.ego_view"
    staged_meta = staged_dir / "meta"
    staged_data.mkdir(parents=True, exist_ok=True)
    staged_vid.mkdir(parents=True, exist_ok=True)
    staged_meta.mkdir(parents=True, exist_ok=True)

    # 3. 按批次顺序合并并连续重排
    print("\n[2/5] 正在重构全局时序流并平移数据...")
    episodes_jsonl_entries = []
    episodes_stats_entries = []
    current_global_index = 0
    next_new_idx = 0

    # 选取第一个合法的 info.json / modality.json 作为模板
    template_info = None
    template_modality = None
    template_tasks = None

    for s_idx, (s_dir, eps, ep_lens) in enumerate(batch_plan):
        print(f"\n  >> 正在处理批次 [{s_idx + 1}/{len(batch_plan)}]: {s_dir.name} (含 {len(eps)} 条轨迹)")
        s_data = s_dir / "data/chunk-000"
        s_vid = s_dir / "videos/chunk-000/observation.images.ego_view"
        s_meta = s_dir / "meta"

        if template_info is None and (s_meta / "info.json").exists():
            template_info = json.loads((s_meta / "info.json").read_text())
        if template_modality is None and (s_meta / "modality.json").exists():
            template_modality = s_meta / "modality.json"
        if template_tasks is None and (s_meta / "tasks.jsonl").exists():
            template_tasks = s_meta / "tasks.jsonl"

        # 读取该批次的 stats 备用
        batch_stats = {}
        if (s_meta / "episodes_stats.jsonl").exists():
            for line in (s_meta / "episodes_stats.jsonl").read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    batch_stats[item["episode_index"]] = item["stats"]
                except Exception:
                    pass

        # 读取该批次的 tasks 描述备用
        batch_tasks = {}
        if (s_meta / "episodes.jsonl").exists():
            for line in (s_meta / "episodes.jsonl").read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    batch_tasks[item["episode_index"]] = item.get(
                        "tasks", ["pick up the box, turn right, and place it on the table"]
                    )
                except Exception:
                    pass

        for old_idx in eps:
            new_idx = next_new_idx
            old_pq = s_data / f"episode_{old_idx:06d}.parquet"
            old_vid = s_vid / f"episode_{old_idx:06d}.mp4"

            # 复制并平移视频
            dst_vid = staged_vid / f"episode_{new_idx:06d}.mp4"
            shutil.copy2(old_vid, dst_vid)

            # 更新 Parquet 表格全局索引与序号
            df = pq.read_table(old_pq).to_pandas()
            n_frames = len(df)
            df["episode_index"] = new_idx
            df["index"] = current_global_index + np.arange(n_frames, dtype=np.int64)
            current_global_index += n_frames

            dst_pq = staged_data / f"episode_{new_idx:06d}.parquet"
            out_table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(out_table, dst_pq)

            # episodes.jsonl 条目
            ep_task = batch_tasks.get(
                old_idx, ["pick up the box, turn right, and place it on the table"]
            )
            episodes_jsonl_entries.append({
                "episode_index": new_idx,
                "tasks": ep_task,
                "length": n_frames,
            })

            # episodes_stats.jsonl 条目
            if old_idx in batch_stats:
                episodes_stats_entries.append({
                    "episode_index": new_idx,
                    "stats": batch_stats[old_idx],
                })
            else:
                ep_stats = compute_column_stats(out_table)
                episodes_stats_entries.append({"episode_index": new_idx, "stats": ep_stats})

            next_new_idx += 1

    # 4. 生成新主元数据 (info.json, episodes.jsonl, stats)
    print("\n[3/5] 正在重新生成一致性元数据 (info.json, episodes.jsonl, stats)...")
    if template_info is None:
        print("[!] 错误: 未能在任何源目录中读取到 info.json 模板！")
        sys.exit(1)

    template_info["total_episodes"] = next_new_idx
    template_info["total_frames"] = current_global_index
    template_info["total_videos"] = next_new_idx
    template_info["splits"] = {"train": f"0:{next_new_idx}"}
    (staged_meta / "info.json").write_text(json.dumps(template_info, indent=4))

    with open(staged_meta / "episodes.jsonl", "w") as f:
        for entry in episodes_jsonl_entries:
            f.write(json.dumps(entry) + "\n")

    with open(staged_meta / "episodes_stats.jsonl", "w") as f:
        for entry in episodes_stats_entries:
            f.write(json.dumps(entry) + "\n")

    if template_modality and template_modality.exists():
        shutil.copy2(template_modality, staged_meta / "modality.json")
    if template_tasks and template_tasks.exists():
        shutil.copy2(template_tasks, staged_meta / "tasks.jsonl")

    # 5. 原子替换到目标主目录
    print("\n[4/5] 正在发布合并后的正式数据集...")
    if dst_dir.exists():
        for item in dst_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for item in staged_dir.iterdir():
        shutil.move(str(item), str(dst_dir / item.name))
    shutil.rmtree(staged_dir)

    # 6. 最终严密对齐校验
    print("\n[5/5] 执行合并后音视频全量帧数严密校验...")
    final_pqs = sorted((dst_dir / "data/chunk-000").glob("*.parquet"))
    final_vids = sorted((dst_dir / "videos/chunk-000/observation.images.ego_view").glob("*.mp4"))

    all_pass = True
    print("-" * 75)
    print(f" 新序号 | 帧数 | 时长(秒) | 动作表/视频差值 | 状态")
    print("-" * 75)
    for i in range(len(final_pqs)):
        df_chk = pq.read_table(final_pqs[i]).to_pandas()
        in_c = av.open(str(final_vids[i]))
        v_count = in_c.streams.video[0].frames
        if v_count == 0:
            v_count = sum(1 for _ in in_c.decode(video=0))
        in_c.close()

        diff = len(df_chk) - v_count
        status = "对齐" if diff == 0 else "异常!"
        if diff != 0:
            all_pass = False
        print(f" Ep {i:02d}  | {len(df_chk):4d} | {len(df_chk)/args.fps:7.2f}s | Diff = {diff:2d}      | {status}")
    print("-" * 75)

    if all_pass:
        print("\n" + "=" * 78)
        print(f" 恭喜！成功合并 {len(src_dirs)} 个批次，共 {len(final_pqs)} 条轨迹全部校验通过！")
        print(f" 主数据集路径 : {dst_dir}")
        print(f" 总帧数       : {current_global_index} (总时长: {current_global_index / args.fps:.2f} 秒)")
        print(f" 下次启动采集器将自动顺延为: Episode {len(final_pqs)}")
        print("=" * 78)
    else:
        print("[!] 警告: 发现部分视频与 Parquet 帧数存在偏差，请检查备份！")


if __name__ == "__main__":
    main()
