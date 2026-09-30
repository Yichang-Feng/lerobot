#!/usr/bin/env python3
"""
run_vla_water_bottle_transfer.py

跨桌水瓶转运任务 — VLA + JEV 部署启动器 (Unitree G1 Dex-1)
=============================================================

任务流程:
  Phase 0  桌 A 前 | VLA 控制上肢执行夹取水瓶
           检测到夹紧并抬起后自动冻结姿态与夹爪快照
  Phase 1  上肢平滑回默认位置 (2.5s 余弦 S 曲线，夹爪全程锁定夹紧)
  Phase 2  [等待导航] 机器人由导航/底盘移动至桌 B
           上肢固定在默认位置，夹爪全程锁定夹紧
           ★ 导航到达后，输入 'd' 回车 (或 /navdone)
  Phase 3  上肢从默认位置平滑恢复到抓取时的姿态 (2.5s，夹爪保持夹紧)
  Phase 4  桌 B 前 | JEV 视觉判断放置目标 (盒子 or 桌面)
           VLA 接管上肢执行精确放置

快捷指令:
  's' (回车) : 启动 Phase 0 推理 (策略加载完毕后手动触发)
  'd' (回车) : 导航到达桌 B 信号，触发 Phase 3 姿态恢复与 Phase 4 放置
  'r' (回车) : 复位 (/reset)，清空转运状态并回位
  'q' (回车) : 安全停止并退出 (/stop)
  (直接回车) : 即时查看当前转运阶段与夹爪角度状态

使用示例:
  python run_vla_water_bottle_transfer.py \\
      --robot_ip 0.0.0.0 \\
      --policy.path outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model \\
      --valen_evaluator true \\
      --valen_ip 10.8.8.98
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── 离线环境与动态库配置 ────────────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
lerobot_lib = "/home/yichangfeng/miniforge3/envs/lerobot/lib"
if os.path.isdir(lerobot_lib):
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lerobot_lib not in cur_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{lerobot_lib}:{cur_ld}" if cur_ld else lerobot_lib

if "PALIGEMMA_TOKENIZER_PATH" not in os.environ:
    cand = os.path.expanduser("~/lerobot/paligemma_tokenizer")
    if os.path.isdir(cand):
        os.environ["PALIGEMMA_TOKENIZER_PATH"] = cand

DEFAULT_POLICY_PATH = (
    "outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="跨桌水瓶转运任务 VLA+JEV 部署启动器 (Unitree G1 Dex-1)"
    )
    parser.add_argument("--robot_ip", default="192.168.123.164", help="G1 机器人 IP (0.0.0.0 自动转为 127.0.0.1)")
    parser.add_argument("--action_ip", default="", help="动作下发目标 IP (留空则同 robot_ip)")
    parser.add_argument("--camera_ip", default="", help="相机流接收 IP (留空则同 robot_ip)")
    parser.add_argument("--camera_port", type=int, default=5556, help="全局主视角相机端口 (默认 5556)")
    parser.add_argument("--left_wrist_port", type=int, default=5557, help="左腕相机端口 (默认 5557)")
    parser.add_argument("--right_wrist_port", type=int, default=5558, help="右腕相机端口 (默认 5558)")
    parser.add_argument("--state_port", type=int, default=6001, help="状态流端口 (默认 6001)")
    parser.add_argument("--action_port", type=int, default=6002, help="动作流端口 (默认 6002)")
    parser.add_argument("--arm_only", type=str, default="auto", choices=["auto", "true", "false"],
                        help="是否开启 16-DoF 纯双臂+双夹爪模式 (auto: 自动检测)")
    parser.add_argument("--policy.path", default=DEFAULT_POLICY_PATH, help="策略模型路径")
    parser.add_argument("--queue_threshold", type=int, default=35, help="RTC 队列门限")
    parser.add_argument("--interpolation_multiplier", type=int, default=3, help="动作插值倍率")
    parser.add_argument("--fps", type=int, default=30, help="控制帧率 (30Hz)")
    parser.add_argument("--duration", type=int, default=1000, help="最大单次推理秒数")
    parser.add_argument("--display_data", default="true", help="是否开启 Rerun 实时监控 (默认 true)")
    parser.add_argument("--valen_evaluator", default="false", help="是否启用 Valen JEV 决策评估")
    parser.add_argument("--valen_ip", default="10.8.8.98", help="Valen JEV 服务 IP")
    parser.add_argument("--valen_port", type=int, default=5559, help="Valen JEV 服务端口")
    parser.add_argument("--transfer_grasp_thresh", type=float, default=3.5, help="夹爪闭合阈值 (rad)")
    parser.add_argument("--transfer_lift_pitch", type=float, default=-0.15, help="右肩俯仰阈值 (rad)，判定已抬起")
    parser.add_argument("--transfer_grasp_frames", type=int, default=8, help="连续确认帧数 (×0.05s)")
    parser.add_argument("--transfer_restore_duration", type=float, default=2.5, help="姿态恢复插值时长 (s)")
    parser.add_argument("--transfer_gripper_closed", type=float, default=3.4, help="备用夹爪角度 (rad)")
    parser.add_argument("--transfer_place_box_task",
                        default="pick up the water bottle and place it into the blue box")
    parser.add_argument("--transfer_place_table_task",
                        default="pick up the water bottle and place it on the table")
    parser.add_argument("--subtask_homing_duration", type=float, default=2.5, help="回默认位置插值时长 (s)")
    parser.add_argument("--transfer_hud", action="store_true", default=False, help="是否打印 1.5s 周期性 HUD (默认关闭，保持终端整洁)")

    args, unknown_args = parser.parse_known_args()

    # 将 0.0.0.0 / localhost 规范化为 127.0.0.1
    robot_ip = "127.0.0.1" if args.robot_ip in ("0.0.0.0", "localhost") else args.robot_ip
    action_ip = args.action_ip if args.action_ip else robot_ip
    camera_ip = args.camera_ip if args.camera_ip else robot_ip
    if action_ip in ("0.0.0.0", "localhost"):
        action_ip = "127.0.0.1"
    if camera_ip in ("0.0.0.0", "localhost"):
        camera_ip = "127.0.0.1"

    policy_path_str = getattr(args, "policy.path")
    policy_path = Path(policy_path_str).expanduser()
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path

    if not policy_path.is_dir():
        print("\n" + "=" * 76)
        print(f"❌ 错误: 指定的本地模型路径不存在: '{policy_path_str}'")
        print("💡 提示: outputs/train/ 下可用的本地策略模型目录包括:")
        for cand in sorted(Path("outputs/train").glob("*/merged_model*")):
            print(f"   - {cand}")
        print("=" * 76 + "\n")
        sys.exit(1)

    lerobot_rollout = "/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"
    if not os.path.exists(lerobot_rollout):
        lerobot_rollout = sys.executable

    # 自动判断是否开启 16-DoF 纯双臂+双夹爪模式 (如 Dex-1，无下肢/无底盘摇杆轴)
    arm_only_flag = False
    if args.arm_only == "true":
        arm_only_flag = True
    elif args.arm_only == "auto":
        cfg_file = policy_path / "config.json"
        if cfg_file.is_file():
            try:
                with open(cfg_file) as f:
                    content = f.read()
                    if '"shape": [16]' in content.replace(" ", "") or "kLeftGripper" in content:
                        arm_only_flag = True
            except Exception:
                pass

    valen_enabled = args.valen_evaluator.lower() in ("true", "1", "yes")

    base_cmd = [
        lerobot_rollout,
        "--strategy.type=base",
        "--inference.type=rtc",
        f"--inference.queue_threshold={args.queue_threshold}",
        f"--interpolation_multiplier={args.interpolation_multiplier}",
        f"--policy.path={policy_path.resolve()}",
        "--policy.device=cuda",
        '--rename_map={"observation.images.global_view":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}',
        "--robot.type=unitree_g1_client",
        f"--robot.robot_ip={robot_ip}",
        f"--robot.action_ip={action_ip}",
        f"--robot.camera_ip={camera_ip}",
        f"--robot.camera_port={args.camera_port}",
        "--robot.enable_wrist_cameras=true",
        f"--robot.left_wrist_port={args.left_wrist_port}",
        f"--robot.right_wrist_port={args.right_wrist_port}",
        f"--robot.state_port={args.state_port}",
        f"--robot.action_port={args.action_port}",
        "--robot.enable_gripper=true",
        "--robot.connect_timeout=60",
        "--robot.wait_until_ready=true",
        f"--duration={args.duration}",
        f"--fps={args.fps}",
        f"--display_data={args.display_data}",
        "--transfer_mode=true",
        f"--transfer_grasp_thresh={args.transfer_grasp_thresh}",
        f"--transfer_lift_pitch={args.transfer_lift_pitch}",
        f"--transfer_grasp_frames={args.transfer_grasp_frames}",
        f"--transfer_restore_duration={args.transfer_restore_duration}",
        f"--transfer_gripper_closed={args.transfer_gripper_closed}",
        f"--subtask_homing_duration={args.subtask_homing_duration}",
        f"--transfer_place_box_task={args.transfer_place_box_task}",
        f"--transfer_place_table_task={args.transfer_place_table_task}",
        f"--transfer_hud={'true' if args.transfer_hud else 'false'}",
        f"--valen_evaluator={'true' if valen_enabled else 'false'}",
        f"--valen_ip={args.valen_ip}",
        f"--valen_port={args.valen_port}",
        "--interactive=true",
        "--auto_mode=false",
        "--subtasks=false",
        "--task=pick up the water bottle from the table",
    ]
    if arm_only_flag:
        base_cmd.append("--robot.arm_only=true")
    base_cmd += unknown_args

    print("=" * 76)
    print(" 🚀 [终端 2: 跨桌水瓶转运 (Unitree G1 Dex-1: VLA + JEV)]")
    print(f" 模型路径 : {policy_path.resolve()}")
    print(f" 机器人IP : {robot_ip} (状态: {args.state_port}, 动作: {args.action_port})")
    print(f" 视觉机位 : 全局主摄({args.camera_port}) | 左腕({args.left_wrist_port}) | 右腕({args.right_wrist_port})")
    if valen_enabled:
        print(f" 决策大脑 : Valen JEV ({args.valen_ip}:{args.valen_port})")
    print(" 操作说明 :")
    print("   1. 正在载入 VLA 模型到 GPU 显存 (约需 15~30 秒)...")
    print("   2. 提示就绪后，按 's' 回车 开始 Phase 0 桌A夹水瓶推理；")
    print("   3. 夹起水瓶后机器人自动保持夹持并平滑回默认位置；")
    print("   4. 导航走至桌B后，按 'd' 回车 恢复姿态并由 JEV 决策放置；")
    print("   5. 随时可按 'r' 回车 复位，'q' 回车 退出。")
    print("=" * 76 + "\n")
    sys.stdout.flush()

    try:
        os.execvp(lerobot_rollout, base_cmd)
    except Exception as exc:
        print(f"[Launcher] os.execvp 失败 ({exc})，使用 subprocess 启动…")
        sys.exit(subprocess.run(base_cmd).returncode)


if __name__ == "__main__":
    main()
