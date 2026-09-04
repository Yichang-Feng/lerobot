#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SonicStar / GR00T 采集数据通用清洗与整理工具 (sanitize_sonic_dataset.py)

功能：
1. 自动备份原始数据，杜绝误操作导致数据丢失
2. 支持剔除失败/废弃的 Episode（自动同步删除 parquet 表格与 mp4 视频）
3. 支持精确时间裁剪（截取某 Episode 的指定时间段，音画同步 0 帧差）
4. 自动全量连续重编号（重置 episode_000000 ~ episode_{N-1}，消除索引断层）
5. 自动校准全套元数据（info.json, episodes.jsonl, episodes_stats.jsonl）
6. 整理后可无缝继续在终端 4 追加录制后续 Episode，或直接转码为 LeRobot v3.0 进行微调

使用示例：
    # 场景 1: 删除失败的第 5 和第 14 条数据
    python sanitize_sonic_dataset.py --delete-episodes 5 14

    # 场景 2: 将第 2 条数据从第 13.0 秒起步（裁剪掉前 13 秒），并删除第 5 条
    python sanitize_sonic_dataset.py --trim-start 2:13.0 --delete-episodes 5

    # 场景 3: 仅检查并连续重排现有所有数据（不删除、不裁剪）
    python sanitize_sonic_dataset.py

    # 场景 4: 预览清洗计划（干跑模式，不修改任何文件）
    python sanitize_sonic_dataset.py --trim-start 2:13.0 --delete-episodes 5 14 --dry-run
"""

import argparse
import datetime
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


def parse_trim_arg(arg_str: Optional[str]) -> Dict[int, float]:
    """解析裁剪参数，格式为 'ep_idx:seconds'，支持逗号分隔多个，如 '2:13.0,3:5.5'"""
    trims = {}
    if not arg_str:
        return trims
    for item in arg_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"裁剪参数格式错误: '{item}'，应为 'episode_index:seconds' (如 2:13.0)")
        ep_idx = int(parts[0])
        sec = float(parts[1])
        trims[ep_idx] = sec
    return trims


def compute_column_stats(df: pa.Table) -> Dict[str, dict]:
    """计算单个 Episode 的全量列统计数据，用于更新 episodes_stats.jsonl"""
    pandas_df = df.to_pandas()
    stats = {}
    for col in pandas_df.columns:
        series = pandas_df[col]
        first_val = series.iloc[0]
        if hasattr(first_val, "__len__") and not isinstance(first_val, str):
            val_array = np.vstack(series.values)
        else:
            # 标量列增加维度 (N, 1)，确保其 min/max/mean/std 为 1D list 而非标量 float
            val_array = np.array(series.values)[:, None]

        stats[col] = {
            "min": np.min(val_array, axis=0).tolist(),
            "max": np.max(val_array, axis=0).tolist(),
            "mean": np.mean(val_array, axis=0).tolist(),
            "std": np.std(val_array, axis=0).tolist(),
            "count": [len(val_array)],
        }
    return stats


def trim_and_save_video(
    src_vid_path: Path,
    dst_vid_path: Path,
    start_frame: int,
    end_frame: Optional[int],
    fps: float = 50.0,
    width: int = 640,
    height: int = 480,
) -> int:
    """使用 PyAV 进行精确到帧的视频切片与重编码"""
    in_container = av.open(str(src_vid_path))
    frames = [f.to_ndarray(format="rgb24") for f in in_container.decode(video=0)]
    in_container.close()

    total_in = len(frames)
    end_idx = end_frame if end_frame is not None else total_in
    sliced_frames = frames[start_frame:end_idx]

    dst_vid_path.parent.mkdir(parents=True, exist_ok=True)
    out_container = av.open(str(dst_vid_path), mode="w")
    stream = out_container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame_rgb in sliced_frames:
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(av_frame):
            out_container.mux(packet)
    for packet in stream.encode():
        out_container.mux(packet)
    out_container.close()

    return len(sliced_frames)


def main():
    parser = argparse.ArgumentParser(description="SonicStar / WBC 采集数据清洗与重编号工具")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/yichangfeng/SonicStar/wbc/outputs/g1_rubberhand_pick_turn",
        help="待整理的原始数据集根路径",
    )
    parser.add_argument(
        "--delete-episodes",
        type=int,
        nargs="*",
        default=[],
        help="要删除的 Episode 编号列表 (例如: --delete-episodes 5 14)",
    )
    parser.add_argument(
        "--trim-start",
        type=str,
        default="",
        help="裁剪 Episode 起始时间 (格式: 'ep_idx:秒数', 例如: --trim-start 2:13.0)",
    )
    parser.add_argument(
        "--trim-end",
        type=str,
        default="",
        help="裁剪 Episode 结束时间 (格式: 'ep_idx:秒数', 例如: --trim-end 3:20.0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="原始采集帧率 (默认 50.0 Hz)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="",
        help="自定义备份目录 (留空则自动以时间戳命名备份)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演练模式：仅打印清洗计划，不实际修改任何磁盘文件",
    )
    args = parser.parse_args()

    raw_dir = Path(args.dataset_dir).expanduser().resolve()
    if not raw_dir.exists():
        print(f"[!] 错误: 数据集目录不存在: {raw_dir}")
        sys.exit(1)

    data_dir = raw_dir / "data/chunk-000"
    vid_dir = raw_dir / "videos/chunk-000/observation.images.ego_view"
    meta_dir = raw_dir / "meta"

    if not data_dir.exists() or not meta_dir.exists():
        print(f"[!] 错误: 目标目录不符合 SonicStar 数据格式 (缺少 data 或 meta 目录): {raw_dir}")
        sys.exit(1)

    # 扫描现有的 parquet 与 mp4 文件
    pqs = sorted(data_dir.glob("episode_*.parquet"))
    vids = sorted(vid_dir.glob("episode_*.mp4")) if vid_dir.exists() else []

    pq_indices = set()
    for p in pqs:
        try:
            pq_indices.add(int(p.stem.split("_")[1]))
        except Exception:
            continue

    vid_indices = set()
    for v in vids:
        try:
            vid_indices.add(int(v.stem.split("_")[1]))
        except Exception:
            continue

    all_indices = sorted(list(pq_indices | vid_indices))
    if not all_indices:
        print(f"[!] 错误: 未发现任何 episode 数据在 {raw_dir}")
        sys.exit(1)

    # 自动识别单边缺失的 Episode (例如手动在 video 目录删除了某些视频)
    missing_video_eps = sorted(list(pq_indices - vid_indices))
    missing_pq_eps = sorted(list(vid_indices - pq_indices))
    inconsistent_eps = set(missing_video_eps) | set(missing_pq_eps)

    trim_starts = parse_trim_arg(args.trim_start)
    trim_ends = parse_trim_arg(args.trim_end)
    delete_set = set(args.delete_episodes) | inconsistent_eps

    # 规划保留的 Episode
    kept_indices = [idx for idx in all_indices if idx not in delete_set]

    print("\n" + "=" * 75)
    print("           SonicStar 数据集清洗与重编排规划 (Sanitize Plan)")
    print("=" * 75)
    print(f" 数据集路径 : {raw_dir}")
    print(f" 现有样本   : {all_indices} (共 {len(all_indices)} 条)")
    if missing_video_eps:
        print(f" 缺失视频   : {missing_video_eps} (视频已手动删除，自动同步剔除并重排)")
    if missing_pq_eps:
        print(f" 缺失动作表 : {missing_pq_eps} (动作表缺失，自动同步剔除并重排)")
    print(f" 计划删除   : {sorted(list(delete_set)) if delete_set else '无'}")
    print(f" 起始裁剪   : {trim_starts if trim_starts else '无'}")
    print(f" 结束裁剪   : {trim_ends if trim_ends else '无'}")
    print(f" 清洗后保留 : {kept_indices} (共 {len(kept_indices)} 条，将重编号为 0 ~ {len(kept_indices)-1})")
    print("=" * 75)

    if args.dry_run:
        print("\n[INFO] 当前处于 --dry-run 预览模式，未做任何修改。确认无误后去掉 --dry-run 执行即可。")
        return

    # 1. 自动执行全量安全备份
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        Path(args.backup_dir).expanduser().resolve()
        if args.backup_dir
        else raw_dir.parent / f"{raw_dir.name}_backup_{timestamp_str}"
    )
    print(f"\n[1/5] 正在创建完整安全备份至:\n      -> {backup_path}")
    if backup_path.exists():
        shutil.rmtree(backup_path)
    shutil.copytree(raw_dir, backup_path)
    print("[+] 备份完成！无论任何意外，数据均可通过备份秒级还原。")

    # 2. 准备暂存隔离区 (Staging Area)
    staged_dir = raw_dir.parent / f"{raw_dir.name}_staged_{timestamp_str}"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)

    staged_data = staged_dir / "data/chunk-000"
    staged_vid = staged_dir / "videos/chunk-000/observation.images.ego_view"
    staged_meta = staged_dir / "meta"
    staged_data.mkdir(parents=True, exist_ok=True)
    staged_vid.mkdir(parents=True, exist_ok=True)
    staged_meta.mkdir(parents=True, exist_ok=True)

    # 读取旧统计信息备用
    old_stats_by_idx = {}
    episodes_stats_file = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_file.exists():
        for line in episodes_stats_file.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            old_stats_by_idx[item["episode_index"]] = item["stats"]

    # 3. 逐条处理、裁剪、重编排
    print("\n[2/5] 正在重构数据流并执行时间裁剪...")
    episodes_jsonl_entries = []
    episodes_stats_entries = []
    current_global_index = 0

    next_new_idx = 0
    for old_idx in kept_indices:
        old_pq = data_dir / f"episode_{old_idx:06d}.parquet"
        old_vid = vid_dir / f"episode_{old_idx:06d}.mp4"

        if not old_pq.exists():
            print(f"[!] 警告: 找不到 Parquet 文件: {old_pq}，跳过该条。")
            continue
        if not old_vid.exists():
            print(f"[!] 警告: 找不到视频文件: {old_vid}，跳过该条。")
            continue

        new_idx = next_new_idx
        next_new_idx += 1

        df = pq.read_table(old_pq).to_pandas()
        orig_len = len(df)

        start_sec = trim_starts.get(old_idx, 0.0)
        end_sec = trim_ends.get(old_idx, None)

        start_frame = int(round(start_sec * args.fps))
        end_frame = int(round(end_sec * args.fps)) if end_sec is not None else orig_len

        # 边界保护
        start_frame = max(0, min(start_frame, orig_len - 1))
        end_frame = max(start_frame + 1, min(end_frame, orig_len))

        is_trimmed = (start_frame > 0) or (end_frame < orig_len)

        if is_trimmed:
            print(
                f"  [*] Episode {old_idx:02d} -> 新序号 {new_idx:02d}: "
                f"时间裁剪 [{start_sec:.2f}s ~ {end_sec if end_sec else orig_len/args.fps:.2f}s] "
                f"(帧数 {orig_len} -> {end_frame - start_frame})"
            )
            df = df.iloc[start_frame:end_frame].copy()
            # 重置时间戳与帧号从 0.0 开始
            df["timestamp"] = np.arange(len(df), dtype=np.float64) / args.fps
            df["frame_index"] = np.arange(len(df), dtype=np.int64)

            # 裁剪视频
            dst_vid = staged_vid / f"episode_{new_idx:06d}.mp4"
            actual_vid_frames = trim_and_save_video(
                old_vid, dst_vid, start_frame, end_frame, fps=args.fps
            )
            assert actual_vid_frames == len(
                df
            ), f"裁剪后帧数不一致: Parquet {len(df)} vs 视频 {actual_vid_frames}"
        else:
            print(f"  [+] Episode {old_idx:02d} -> 新序号 {new_idx:02d} (保持原长 {orig_len} 帧)")
            dst_vid = staged_vid / f"episode_{new_idx:06d}.mp4"
            shutil.copy2(old_vid, dst_vid)

        # 更新全局索引和新 Episode 编号
        n_frames = len(df)
        df["episode_index"] = new_idx
        df["index"] = current_global_index + np.arange(n_frames, dtype=np.int64)
        current_global_index += n_frames

        # 写入新 Parquet
        dst_pq = staged_data / f"episode_{new_idx:06d}.parquet"
        out_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_table, dst_pq)

        # 构造 episodes.jsonl 记录
        episodes_jsonl_entries.append({
            "episode_index": new_idx,
            "tasks": ["pick up the box, turn right, and place it on the table"],
            "length": n_frames,
        })

        # 构造 episodes_stats.jsonl 记录
        if is_trimmed or old_idx not in old_stats_by_idx:
            ep_stats = compute_column_stats(out_table)
            episodes_stats_entries.append({"episode_index": new_idx, "stats": ep_stats})
        else:
            episodes_stats_entries.append({"episode_index": new_idx, "stats": old_stats_by_idx[old_idx]})

    # 4. 生成新元数据
    print("\n[3/5] 正在重新生成一致性元数据 (info.json, episodes.jsonl, stats)...")
    info = json.loads((meta_dir / "info.json").read_text())
    info["total_episodes"] = len(episodes_jsonl_entries)
    info["total_frames"] = current_global_index
    info["total_videos"] = len(episodes_jsonl_entries)
    info["splits"] = {"train": f"0:{len(episodes_jsonl_entries)}"}
    (staged_meta / "info.json").write_text(json.dumps(info, indent=4))

    with open(staged_meta / "episodes.jsonl", "w") as f:
        for entry in episodes_jsonl_entries:
            f.write(json.dumps(entry) + "\n")

    with open(staged_meta / "episodes_stats.jsonl", "w") as f:
        for entry in episodes_stats_entries:
            f.write(json.dumps(entry) + "\n")

    if (meta_dir / "modality.json").exists():
        shutil.copy2(meta_dir / "modality.json", staged_meta / "modality.json")
    if (meta_dir / "tasks.jsonl").exists():
        shutil.copy2(meta_dir / "tasks.jsonl", staged_meta / "tasks.jsonl")

    # 5. 原子替换目标目录
    print("\n[4/5] 正在应用整理后的正式数据...")
    for item in raw_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in staged_dir.iterdir():
        shutil.move(str(item), str(raw_dir / item.name))
    shutil.rmtree(staged_dir)

    # 6. 最终严密自检
    print("\n[5/5] 执行最终音视频帧数严密自检...")
    final_pqs = sorted((raw_dir / "data/chunk-000").glob("*.parquet"))
    final_vids = sorted((raw_dir / "videos/chunk-000/observation.images.ego_view").glob("*.mp4"))

    all_pass = True
    print("-" * 75)
    print(f" 新序号 | 帧数 | 时长(秒) | 动作表/视频差值 | 状态")
    print("-" * 75)
    for i in range(len(final_pqs)):
        df_chk = pq.read_table(final_pqs[i]).to_pandas()
        in_c = av.open(str(final_vids[i]))
        v_count = in_c.streams.video[0].frames
        if v_count == 0:  # 部分容器未存帧数字段时手动解算
            v_count = sum(1 for _ in in_c.decode(video=0))
        in_c.close()

        diff = len(df_chk) - v_count
        status = "对齐" if diff == 0 else "异常!"
        if diff != 0:
            all_pass = False
        print(f" Ep {i:02d}  | {len(df_chk):4d} | {len(df_chk)/args.fps:7.2f}s | Diff = {diff:2d}      | {status}")
    print("-" * 75)

    if all_pass:
        print("\n" + "=" * 75)
        print(f" 全部 {len(final_pqs)} 条轨迹校验通过！总帧数: {current_global_index} (时长: {current_global_index/args.fps:.2f} 秒)")
        print(f" 下次启动采集器将自动顺延为: Episode {len(final_pqs)}")
        print("=" * 75)
    else:
        print("[!] 警告: 发现部分视频与 Parquet 帧数存在偏差，请检查备份！")


if __name__ == "__main__":
    main()
