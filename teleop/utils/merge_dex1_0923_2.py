#!/usr/bin/env python3
"""
Merge g1_pick_put_dex1_0923_2 into g1_pick_put_dex1, sort/reindex, and update prompts.
"""

import json
import shutil
import sys
from pathlib import Path


TASK_PROMPT = {
    "goal": "pick up the water bottle from the table, place it into the blue box, then take it out and place it back on the table.",
    "desc": "从桌子上夹起水瓶后放到蓝色盒子里面，然后再拿出来放到桌子上",
    "steps": "step1: clamp and pick up the water bottle from the table; step2: place it into the blue box; step3: take it out of the blue box; step4: place it back on the table.",
}


def update_episode_prompt(data_json_path: Path, prompt_dict: dict) -> int:
    """Update text field in data.json and return the frame count."""
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["text"] = prompt_dict

    with open(data_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return len(data.get("data", []))


def main():
    base_data_dir = Path("/home/yichangfeng/xr_teleoperate/teleop/utils/data")
    dest_dir = base_data_dir / "g1_pick_put_dex1"
    src_dir = base_data_dir / "g1_pick_put_dex1_0923_2"

    print("=" * 80)
    print("STEP 1: VALIDATE DIRECTORIES")
    print("=" * 80)
    assert dest_dir.exists(), f"Destination directory {dest_dir} does not exist!"
    assert src_dir.exists(), f"Source directory {src_dir} does not exist!"

    # Existing episodes in g1_pick_put_dex1
    existing_eps = sorted([
        d for d in dest_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_") and (d / "data.json").exists()
    ], key=lambda x: int(x.name.split("_")[-1]))

    print(f"Existing episodes in {dest_dir.name}: {len(existing_eps)}")
    assert len(existing_eps) == 62, f"Expected 62 episodes in {dest_dir.name}, found {len(existing_eps)}"
    assert existing_eps[0].name == "episode_0000"
    assert existing_eps[-1].name == "episode_0061"

    # New episodes in g1_pick_put_dex1_0923_2
    new_eps = sorted([
        d for d in src_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_") and (d / "data.json").exists()
    ], key=lambda x: int(x.name.split("_")[-1]))

    print(f"New episodes in {src_dir.name}: {len(new_eps)}")
    assert len(new_eps) == 23, f"Expected 23 episodes in {src_dir.name}, found {len(new_eps)}"
    print(f"Source episode sequence: {[d.name for d in new_eps]}")

    print("\n" + "=" * 80)
    print("STEP 2: COPY AND RE-INDEX NEW EPISODES INTO DESTINATION")
    print("=" * 80)
    start_index = len(existing_eps)  # 62

    for i, ep_src in enumerate(new_eps):
        target_idx = start_index + i
        target_name = f"episode_{target_idx:04d}"
        target_path = dest_dir / target_name

        if target_path.exists():
            print(f"Target {target_name} already exists, checking contents...")
            assert (target_path / "data.json").exists(), f"{target_name} exists but missing data.json"
        else:
            print(f"[{i+1}/{len(new_eps)}] Copying {ep_src.name} -> {target_name} ...")
            shutil.copytree(str(ep_src), str(target_path))

        # Update prompt in target
        update_episode_prompt(target_path / "data.json", TASK_PROMPT)

        # Also update in src for consistency
        update_episode_prompt(ep_src / "data.json", TASK_PROMPT)

    print("\n" + "=" * 80)
    print("STEP 3: VERIFY ALL EPISODES IN DESTINATION")
    print("=" * 80)
    all_merged_eps = sorted([
        d for d in dest_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_") and (d / "data.json").exists()
    ], key=lambda x: int(x.name.split("_")[-1]))

    print(f"Total episodes in merged directory: {len(all_merged_eps)}")
    assert len(all_merged_eps) == 85, f"Expected 85 episodes, got {len(all_merged_eps)}"
    assert all_merged_eps[0].name == "episode_0000"
    assert all_merged_eps[-1].name == "episode_0084"

    total_frames = 0
    for idx, ep_path in enumerate(all_merged_eps):
        assert ep_path.name == f"episode_{idx:04d}", f"Discontinuous episode name: {ep_path.name} vs episode_{idx:04d}"
        with open(ep_path / "data.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        frames = len(d.get("data", []))
        total_frames += frames
        assert d["text"]["goal"] == TASK_PROMPT["goal"]

    print("\n" + "=" * 80)
    print("MERGE & PROMPT UPDATE COMPLETED SUCCESSFULLY!")
    print(f"  Target directory: {dest_dir}")
    print(f"  Total episodes:   {len(all_merged_eps)} (episode_0000 - episode_0084)")
    print(f"  Total frames:     {total_frames}")
    print("  Prompt updated:")
    print(f"    - Goal:  {TASK_PROMPT['goal']}")
    print(f"    - Desc:  {TASK_PROMPT['desc']}")
    print(f"    - Steps: {TASK_PROMPT['steps']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
