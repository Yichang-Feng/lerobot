#!/usr/bin/env python3
"""
PICO VR ZeroMQ Teleoperation Client (NO ROS 2 DEPENDENCY).

Controls:
- VR Headset + Handheld Controllers (6D Poses) -> G1 Upper Body (14-DoF Arms IK)
- Left Joystick (x, y) -> G1 Forward/Backward (vx) & Strafe (vy)
- Right Joystick (x)   -> G1 Yaw Turn (yaw_rate)
- Hand Triggers/Grips  -> Grippers / Hand actions
- Button A             -> Start / Save Data Collection Episode
- Button B             -> Discard Episode
- Left Menu + Right Trigger -> Toggle Upper Body Teleop Active
- Left Menu + Left Trigger  -> Toggle Lower Body Walk / Stand

Streams 18-DoF actions to G1 Locomotion Server via ZMQ on port 6002.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq

# Ensure GR00T-WholeBodyControl and its submodules are in python path
repo_root = Path("/home/yichangfeng/GR00T-WholeBodyControl")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "decoupled_wbc") not in sys.path:
    sys.path.insert(0, str(repo_root / "decoupled_wbc"))

ARM_MOTOR_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def main():
    parser = argparse.ArgumentParser(description="PICO VR Teleoperation Client via ZeroMQ")
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="Target server IP (default: 127.0.0.1)")
    parser.add_argument("--action-port", type=int, default=6002, help="ZMQ action port on locomotion server (default: 6002)")
    parser.add_argument("--fps", type=float, default=50.0, help="Teleoperation publish rate in Hz (default: 50)")
    parser.add_argument("--high-elbow", action="store_true", help="Use high elbow pose configuration")
    parser.add_argument("--deadzone", type=float, default=0.1, help="Joystick deadzone")
    parser.add_argument("--max-lin-vel", type=float, default=0.4, help="Max linear speed (m/s)")
    parser.add_argument("--max-ang-vel", type=float, default=0.6, help="Max angular yaw speed (rad/s)")
    args = parser.parse_args()

    print("=" * 75)
    print(" [PICO VR -> G1 ZeroMQ Teleop Client]")
    print(f" 目标运控服务器 : tcp://{args.server_ip}:{args.action_port}")
    print(f" 遥控更新频率   : {args.fps} Hz")
    print(f" 通信协议       : ZeroMQ PUSH (纯 Python，无 ROS2 依赖)")
    print("=" * 75)

    # 1. Initialize ZMQ socket
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{args.server_ip}:{args.action_port}")

    # 2. Initialize Teleop Policy (Bypassing ROS2 keyboard dispatcher)
    print("\n[1/3] 正在加载 G1 运动学模型与上肢解耦 IK 求解器...")
    from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
        instantiate_g1_hand_ik_solver,
    )
    from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK

    robot_model = instantiate_g1_robot_model(
        waist_location="lower_body",
        high_elbow_pose=args.high_elbow,
    )
    left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()

    retargeting_ik = TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_ik_solver,
        right_hand_ik_solver=right_hand_ik_solver,
        enable_visualization=False,
        body_active_joint_groups=["upper_body"],  # 仅解算双臂 14 自由度
    )

    print("[2/3] 正在初始化 PICO VR 驱动 (XRoboToolkit)...")
    teleop_policy = TeleopPolicy(
        robot_model=robot_model,
        retargeting_ik=retargeting_ik,
        body_control_device="pico",
        hand_control_device="pico",
        enable_real_device=True,
        activate_keyboard_listener=False,  # 关键：彻底禁用 ROS2 键盘监听模块
    )

    print("[3/3] VR 遥控已就绪！")
    print("-" * 75)
    print("【手柄操作指引】:")
    print("  • 双手手柄空间移动  -> 控制机器人双臂 (14 关节跟随)")
    print("  • 左手柄摇杆 前后/左右 -> 控制机器人底盘 前进/后退/横移")
    print("  • 右手柄摇杆 左右推    -> 控制机器人底盘 原地旋转转身")
    print("  • Menu + 右扳机       -> 切换上肢遥控激活状态 (首次需激活)")
    print("  • Menu + 左扳机       -> 切换下肢步态允许行走")
    print("  • A 键               -> 触发录制开始 / 保存")
    print("  • B 键               -> 触发录制废弃")
    print("-" * 75)

    period = 1.0 / args.fps
    step_count = 0
    last_print = time.time()

    try:
        while True:
            t_start = time.time()

            # Read action from PICO VR policy
            raw_action = teleop_policy.get_action()

            # 1. Extract 14-DoF target arm positions
            target_upper = raw_action.get("target_upper_body_pose", None)
            arm_dict = {}
            if target_upper is not None and len(target_upper) >= 14:
                for idx, name in enumerate(ARM_MOTOR_NAMES):
                    arm_dict[f"{name}.q"] = float(target_upper[idx])

            # 2. Extract navigation command [lin_vel_x, lin_vel_y, ang_vel_z]
            nav_cmd = raw_action.get("navigate_cmd", [0.0, 0.0, 0.0])
            vx = float(nav_cmd[0])
            vy = float(nav_cmd[1])
            vyaw = float(nav_cmd[2])

            # Map to G1 controller remote axes
            remote_dict = {
                "remote.lx": -vy,
                "remote.ly": vx,
                "remote.rx": -vyaw,
                "remote.ry": 0.0,
            }

            # 3. Assemble unified 18-DoF action packet
            combined_action = {**arm_dict, **remote_dict}

            # Include button triggers for data exporter
            payload = {
                "cmd": "action",
                "action": combined_action,
                "toggle_data_collection": bool(raw_action.get("toggle_data_collection", False)),
                "toggle_data_abort": bool(raw_action.get("toggle_data_abort", False)),
                "is_active": bool(teleop_policy.is_active),
                "timestamp": t_start,
            }

            sock.send_json(payload)
            step_count += 1

            # Print status every 1 second
            now = time.time()
            if now - last_print >= 1.0:
                active_str = "ACTIVE (控制中)" if teleop_policy.is_active else "IDLE (请按 Menu+右扳机 激活)"
                print(
                    f"\r[VR Teleop] 状态: {active_str} | 速度: vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, yaw={vyaw:+.2f} rad/s | 帧率: {step_count}Hz",
                    end="",
                    flush=True,
                )
                step_count = 0
                last_print = now

            elapsed = time.time() - t_start
            sleep_time = max(0.0, period - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n>>> 用户中断，正在停止 VR 遥控并重置底盘速度... <<<")
        # Send zero velocity before exiting
        zero_action = {
            "remote.lx": 0.0,
            "remote.ly": 0.0,
            "remote.rx": 0.0,
            "remote.ry": 0.0,
        }
        sock.send_json({"cmd": "action", "action": zero_action})
    finally:
        teleop_policy.close()
        sock.close()
        ctx.term()
        print(">>> VR Teleop 客户端已安全退出。 <<<")


if __name__ == "__main__":
    main()
