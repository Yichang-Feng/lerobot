#!/usr/bin/env python3
"""
run_vla_dex1_2subtasks.py

Unitree G1 Dex-1 (3Cam + 16DoF) 抓水瓶与蓝盒放置 2-Subtask 部署调度器。
专用于模型: outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model_15000

子任务 Prompt 定义:
  Phase 1 (Subtask 0): "pick up the water bottle from the table and place it into the blue box"
  Phase 2 (Subtask 1): "take the water bottle out of the blue box and place it back on the table"

核心功能:
  1. 异步模型加载监听: 等待 Pi0.5 模型权重加载到 GPU 并完成 ZMQ 网络握手后自动发送 /start；
  2. 交互式流转切换:
     - 手动模式 (Manual):
       * 按 [Space] / [Enter] / [n]: 一键流转到下一阶段 (Phase 1 -> Phase 2)；
       * 按 [1]: 立即注入 Phase 1 Prompt (桌面夹水瓶放入盒子)；
       * 按 [2]: 立即注入 Phase 2 Prompt (从盒子拿水瓶放回桌上)；
       * 按 [r]: 发送复位命令 (/r)；
       * 按 [q]: 发送安全停止退出 (/stop)；
     - 自动模式 (Auto): 根据右夹爪抓取与抬起动作及时间门限自动触发流转；
  3. 实时终端 HUD 仪表盘: 显示当前阶段、耗时、右夹爪开闭弧度及右肩俯仰/下摆状态。
"""

import argparse
import json
import logging
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "PALIGEMMA_TOKENIZER_PATH" not in os.environ:
    cand = os.path.expanduser("~/lerobot/paligemma_tokenizer")
    if os.path.isdir(cand):
        os.environ["PALIGEMMA_TOKENIZER_PATH"] = cand

SUBTASKS = [
    "pick up the water bottle from the table and place it into the blue box",
    "take the water bottle out of the blue box and place it back on the table",
]

DEFAULT_POLICY_PATH = "outputs/train/pi05_lora_5090_g1_pick_put_dex1_subtasks/merged_model_15000"

logger = logging.getLogger("Dex1SubtasksSupervisor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def getch_timeout(timeout: float = 0.1) -> Optional[str]:
    """非阻塞读取单个键盘按键"""
    if not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class LiveDex1Monitor:
    """监听 G1 ZMQ 状态流 (端口 6001)，提取右臂关节弧度及 Dex-1 夹爪开合状态"""

    def __init__(self, robot_ip: str, state_port: int = 6001):
        self.robot_ip = robot_ip
        self.state_port = state_port
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(f"tcp://{robot_ip}:{state_port}")

        self.running = True
        self.lock = threading.Lock()
        self.latest_state = None
        self.last_recv_time = 0.0
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            try:
                if self.sock.poll(timeout=100):
                    payload = self.sock.recv(zmq.NOBLOCK)
                    msg = json.loads(payload.decode("utf-8"))
                    with self.lock:
                        self.latest_state = msg
                        self.last_recv_time = time.time()
            except Exception:
                time.sleep(0.05)

    def get_metrics(self) -> dict:
        with self.lock:
            state = self.latest_state
            is_active = (time.time() - self.last_recv_time) < 1.0

        if not state or not is_active:
            return {
                "connected": False,
                "r_pitch": 0.0,
                "r_roll": 0.0,
                "r_gripper": 5.4,
            }

        motors = state.get("motors", {})

        def _get_q(candidates: list[str], default_val: float = 0.0) -> float:
            for k in candidates:
                if k in motors:
                    val = motors[k]
                    if isinstance(val, dict):
                        return float(val.get("q", default_val))
                    elif isinstance(val, (int, float)):
                        return float(val)
            return default_val

        r_pitch = _get_q(["right_shoulder_pitch_joint", "kRightShoulderPitch", "right_shoulder_pitch"])
        r_roll = _get_q(["right_shoulder_roll_joint", "kRightShoulderRoll", "right_shoulder_roll"])
        r_gripper = _get_q(["right_gripper", "kRightGripper", "right_ee"], default_val=5.4)

        return {
            "connected": True,
            "r_pitch": r_pitch,
            "r_roll": r_roll,
            "r_gripper": r_gripper,
        }

    def close(self):
        self.running = False
        try:
            self.sock.close(linger=0)
            self.ctx.term()
        except Exception:
            pass


class Dex1SubtaskSupervisor:
    """Dex-1 双子任务调度器"""

    def __init__(
        self,
        mode: str,
        rollout_cmd: list[str],
        robot_ip: str,
        state_port: int = 6001,
        auto_switch_time: float = 12.0,
        record: bool = False,
        diagnostics_dir: str = "",
    ):
        self.mode = mode.lower()
        self.rollout_cmd = rollout_cmd
        self.robot_ip = robot_ip
        self.state_port = state_port
        self.auto_switch_time = auto_switch_time
        self.record = record
        self.diagnostics_dir = diagnostics_dir

        self.current_phase = 0
        self.phase_start_time = 0.0
        self.running = True

        self.ready_event = threading.Event()
        self.running_event = threading.Event()
        self.monitor = LiveDex1Monitor(robot_ip, state_port)

    def send_cmd(self, proc: subprocess.Popen, text: str):
        if proc.poll() is not None:
            return
        cmd = text.strip() + "\n"
        try:
            proc.stdin.write(cmd.encode("utf-8"))
            proc.stdin.flush()
        except Exception as e:
            logger.debug("写入 stdin 失败: %s", e)

    def switch_to_phase(self, proc: subprocess.Popen, new_phase: int, reason: str = ""):
        if new_phase < 0 or new_phase >= len(SUBTASKS):
            return
        self.current_phase = new_phase
        self.phase_start_time = time.time()
        task_name = SUBTASKS[new_phase]

        print("\n" + "=" * 82)
        print(f" 🚀 [子任务切换] 切换至 Phase [{new_phase + 1}/2]: \"{task_name}\"")
        if reason:
            print(f"    触发原因: {reason}")
        print("=" * 82 + "\n")

        self.send_cmd(proc, f"/subtask {task_name}")

    def _stream_stdout(self, proc: subprocess.Popen):
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()

                if (
                    "Interactive rollout session" in text
                    or "the robot will NOT move until you type /start" in text
                    or "type /help for commands" in text
                ):
                    self.ready_event.set()

                if "Rollout running" in text or "Starting rollout" in text:
                    self.running_event.set()
        except Exception:
            pass

    def run_manual_supervisor(self, proc: subprocess.Popen):
        print("\n" + "=" * 82)
        print("🎮 [手动子任务流转模式已就绪]")
        print("   快捷按键指引:")
        print("     [Space] / [Enter] / 'n' : 立即流转至下一个子任务阶段 (Phase 1 -> Phase 2)")
        print("     '1'                     : 切换到 Phase 1 (\"pick up ... into blue box\")")
        print("     '2'                     : 切换到 Phase 2 (\"take out ... onto table\")")
        print("     'r'                     : 发送复位指令 (/r)")
        print("     'q'                     : 安全停止退出 (/stop)")
        print("=" * 82)
        print(f"👉 当前就绪阶段 [1/2]: \"{SUBTASKS[0]}\" (放入盒子后按 Space 切入下一阶段)\n")

        self.phase_start_time = time.time()
        last_status_print = 0.0

        while self.running and proc.poll() is None:
            time.sleep(0.04)
            now = time.time()
            elapsed = now - self.phase_start_time

            # 终端 HUD 状态行刷新 (每 0.5 秒一次)
            if now - last_status_print >= 0.5:
                last_status_print = now
                metrics = self.monitor.get_metrics()
                conn_str = "🟢 ZMQ连接正常" if metrics["connected"] else "🔴 等待ZMQ(6001)"
                grip_val = metrics["r_gripper"]
                grip_status = "闭合抓取" if grip_val < 4.0 else ("全开" if grip_val > 5.0 else "中间态")
                phase_title = "Phase 1:桌→盒" if self.current_phase == 0 else "Phase 2:盒→桌"
                hud_msg = (
                    f"[{conn_str}] {phase_title} | 耗时: {elapsed:4.1f}s | "
                    f"右夹爪: {grip_val:4.2f} rad ({grip_status}) | 右肩俯仰: {metrics['r_pitch']:+.2f}"
                )
                print(f"   [HUD] {hud_msg}", end="\r", flush=True)

            key = getch_timeout(timeout=0.04)
            if key is None:
                continue

            if key in (" ", "\n", "\r", "n", "N"):
                if self.current_phase == 0:
                    self.switch_to_phase(proc, 1, reason="用户按 Space/Enter 切换到 Phase 2 (盒到桌)")
                else:
                    self.switch_to_phase(proc, 0, reason="用户按 Space/Enter 循环切换回 Phase 1 (桌到盒)")
            elif key == "1":
                self.switch_to_phase(proc, 0, reason="用户按 1 切换到 Phase 1")
            elif key == "2":
                self.switch_to_phase(proc, 1, reason="用户按 2 切换到 Phase 2")
            elif key in ("r", "R"):
                print("\n[User Request] 发送复位命令 /r ...")
                self.send_cmd(proc, "/r")
                self.switch_to_phase(proc, 0, reason="用户重置任务")
            elif key in ("q", "Q"):
                print("\n[User Request] 发送退出命令 /stop ...")
                self.send_cmd(proc, "/stop")
                break

    def run_auto_supervisor(self, proc: subprocess.Popen):
        print("\n" + "=" * 82)
        print("🤖 [自动化子任务流转模式已就绪]")
        print("   当前设定: Phase 1 执行至放入盒子后自动流转至 Phase 2，支持随时按键盘干预！")
        print("   快捷按键: [Space] 强制提前流转 | '1'/'2' 手动跳段 | 'r' 复位 | 'q' 退出")
        print("=" * 82 + "\n")

        self.phase_start_time = time.time()
        has_grasped = False

        while self.running and proc.poll() is None:
            time.sleep(0.08)
            now = time.time()
            elapsed = now - self.phase_start_time
            metrics = self.monitor.get_metrics()

            key = getch_timeout(timeout=0.02)
            if key in (" ", "\n", "\r", "n", "N"):
                self.switch_to_phase(proc, 1 if self.current_phase == 0 else 0, reason="用户按键手动干预流转")
                continue
            elif key == "1":
                self.switch_to_phase(proc, 0, reason="用户跳转至 Phase 1")
                continue
            elif key == "2":
                self.switch_to_phase(proc, 1, reason="用户跳转至 Phase 2")
                continue
            elif key in ("r", "R"):
                self.send_cmd(proc, "/r")
                self.switch_to_phase(proc, 0, reason="用户重置")
                continue
            elif key in ("q", "Q"):
                self.send_cmd(proc, "/stop")
                break

            # 自动流转判断: Phase 0 中，水瓶被抓取后并在盒中松开 (>5.0)，或超过预设时长
            if self.current_phase == 0:
                if metrics["r_gripper"] < 4.0:
                    has_grasped = True
                if (has_grasped and metrics["r_gripper"] > 5.0) or (elapsed >= self.auto_switch_time):
                    reason = "检测到水瓶在盒中松开释放" if (has_grasped and metrics["r_gripper"] > 5.0) else f"达到阶段超时门限 ({self.auto_switch_time}s)"
                    self.switch_to_phase(proc, 1, reason=reason)

    def start(self):
        initial_task = SUBTASKS[0]
        cmd = list(self.rollout_cmd)
        cmd.append(f"--task={initial_task}")
        cmd.append("--interactive=true")

        print("=" * 82)
        print(" [Dex1 Subtask Supervisor] 启动底层 LeRobot Rollout 推理客户端...")
        print(f" 目标策略路径: {getattr(args, 'policy.path', DEFAULT_POLICY_PATH)}")
        print(" 等待 Pi0.5 大模型权重加载到 GPU 显存并连接 G1 (约需 10~20 秒)...")
        print("=" * 82)

        sub_env = os.environ.copy()
        sub_env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=sub_env,
            bufsize=1,
        )

        stdout_thread = threading.Thread(target=self._stream_stdout, args=(proc,), daemon=True)
        stdout_thread.start()

        ready = self.ready_event.wait(timeout=180.0)
        if not ready or proc.poll() is not None:
            print("\n❌ 错误: 等待 Rollout 客户端就绪超时或进程异常退出！")
            return

        print("\n" + "=" * 82)
        print("✅ [Supervisor] 检测到策略已加载就绪！正在自动激活控制循环 /start ...")
        print("=" * 82 + "\n")

        time.sleep(0.5)
        self.send_cmd(proc, "/start")
        self.running_event.wait(timeout=5.0)

        try:
            if self.mode == "auto":
                self.run_auto_supervisor(proc)
            else:
                self.run_manual_supervisor(proc)
        except KeyboardInterrupt:
            print("\n[Supervisor] 捕获 Ctrl+C，正在安全关闭...")
            self.send_cmd(proc, "/stop")
        finally:
            self.monitor.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()


def main():
    global args
    parser = argparse.ArgumentParser(description="Unitree G1 Dex-1 2-Subtask Deployment Supervisor.")
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual", help="子任务流转模式 (manual: 单键手动, auto: 自动+按键)")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="G1 机器人或底层服务 IP")
    parser.add_argument("--action_ip", default="", help="动作指令发送 IP")
    parser.add_argument("--camera_ip", default="", help="机载相机推流 IP")
    parser.add_argument("--camera_port", type=int, default=55555, help="头部主相机端口 (默认 55555)")
    parser.add_argument("--left_wrist_port", type=int, default=55556, help="左腕相机端口 (默认 55556)")
    parser.add_argument("--right_wrist_port", type=int, default=55557, help="右腕相机端口 (默认 55557)")
    parser.add_argument("--state_port", type=int, default=6001, help="状态端口 (默认 6001)")
    parser.add_argument("--action_port", type=int, default=6002, help="动作端口 (默认 6002)")
    parser.add_argument("--policy.path", default=DEFAULT_POLICY_PATH, help="融合后的策略模型路径")
    parser.add_argument("--queue_threshold", type=int, default=35, help="RTC 队列打断门限")
    parser.add_argument("--interpolation_multiplier", type=int, default=3, help="动作插值倍率")
    parser.add_argument("--fps", type=int, default=30, help="控制帧率 (30Hz)")
    parser.add_argument("--duration", type=int, default=1000, help="单次测试最大总秒数")
    parser.add_argument("--display_data", default="false", help="是否打开图像监控窗口")

    args, unknown_args = parser.parse_known_args()

    action_ip = args.action_ip if args.action_ip else args.robot_ip
    camera_ip = args.camera_ip if args.camera_ip else args.robot_ip

    lerobot_rollout = "/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"
    if not os.path.exists(lerobot_rollout):
        lerobot_rollout = sys.executable

    base_cmd = [
        lerobot_rollout,
        "--strategy.type=base",
        "--inference.type=rtc",
        f"--inference.queue_threshold={args.queue_threshold}",
        f"--interpolation_multiplier={args.interpolation_multiplier}",
        f"--policy.path={getattr(args, 'policy.path')}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--rename_map.observation.images.cam_high=observation.images.base_0_rgb",
        "--rename_map.observation.images.cam_left_wrist=observation.images.left_wrist_0_rgb",
        "--rename_map.observation.images.cam_right_wrist=observation.images.right_wrist_0_rgb",
        "--robot.type=unitree_g1_client",
        f"--robot.robot_ip={args.robot_ip}",
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
        "--subtasks=false",
    ]
    base_cmd += unknown_args

    supervisor = Dex1SubtaskSupervisor(
        mode=args.mode,
        rollout_cmd=base_cmd,
        robot_ip=args.robot_ip,
        state_port=args.state_port,
    )
    supervisor.start()


if __name__ == "__main__":
    main()
