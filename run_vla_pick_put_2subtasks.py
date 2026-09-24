#!/usr/bin/env python3
"""
run_vla_pick_put_2subtasks.py

Unitree G1 VLA 2-Subtask (抱箱 & 前伸放置) 协同部署调度器
专用于模型: outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000

两阶段子任务定义:
  Phase 1 (Subtask 0): "clamp and lift the box"
  Phase 2 (Subtask 1): "reach forward and place the box in the blue area"

核心功能:
  1. 异步模型载入自适应: 严密监听底层 lerobot-rollout 输出，等待大模型权重载入 GPU 完成、网络连接就绪后自动发送 /start；
  2. 真实时钟与本体姿态感知 (ZMQ 6001):
     - 自动检测双手内夹抱箱动作 (ShoulderRoll 夹紧姿态)；
     - 维持抱持达标时间 (默认 1.8s) 或达到安全时序门限 (默认 5.0s) 时，外部自动切换提示词至 Phase 2；
     - 提示词强行注入突破实机 RTC 在胸前的停滞微晃死锁；
  3. 双模式协同:
     - 自动模式 (Auto): 姿态门限 + 抱稳计时 + 超时保底全自动流转；
     - 手动模式 (Manual): 单键一键切词 (Space/Enter/n 立即切换到前伸, 1/2 选段, r 复位, q 退出)；
  4. 实时终端 HUD 仪表盘: 显示当前阶段、倒计时、双肩俯仰/侧摆/手肘关节角度与连接状态。
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

# 离线环境与 Tokenizer 配置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "PALIGEMMA_TOKENIZER_PATH" not in os.environ:
    cand = os.path.expanduser("~/lerobot/paligemma_tokenizer")
    if os.path.isdir(cand):
        os.environ["PALIGEMMA_TOKENIZER_PATH"] = cand

SUBTASKS = [
    "clamp and lift the box",
    "reach forward and place the box in the blue area",
]

DEFAULT_POLICY_PATH = "outputs/train/pi05_lora_g1_pick_put_subtasks/merged_model_10000"

logger = logging.getLogger("PickPut2Subtasks")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def getch_timeout(timeout: float = 0.1) -> Optional[str]:
    """非阻塞读取单字符键盘按键"""
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


class LiveStateMonitor:
    """监听 G1 ZMQ 状态流 (端口 6001)，提取双臂 14 自由度关键关节弧度"""

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
                "l_pitch": 0.0,
                "r_pitch": 0.0,
                "l_roll": 0.0,
                "r_roll": 0.0,
                "l_elbow": 0.0,
                "r_elbow": 0.0,
            }

        motors = state.get("motors", {})

        def _get_q(candidates: list[str]) -> float:
            for k in candidates:
                if k in motors:
                    val = motors[k]
                    if isinstance(val, dict):
                        return float(val.get("q", 0.0))
                    elif isinstance(val, (int, float)):
                        return float(val)
            return 0.0

        l_pitch = _get_q(["left_shoulder_pitch_joint", "kLeftShoulderPitch", "left_shoulder_pitch"])
        r_pitch = _get_q(["right_shoulder_pitch_joint", "kRightShoulderPitch", "right_shoulder_pitch"])
        l_roll = _get_q(["left_shoulder_roll_joint", "kLeftShoulderRoll", "left_shoulder_roll"])
        r_roll = _get_q(["right_shoulder_roll_joint", "kRightShoulderRoll", "right_shoulder_roll"])
        l_elbow = _get_q(["left_elbow_joint", "kLeftElbow", "left_elbow"])
        r_elbow = _get_q(["right_elbow_joint", "kRightElbow", "right_elbow"])

        return {
            "connected": True,
            "l_pitch": l_pitch,
            "r_pitch": r_pitch,
            "l_roll": l_roll,
            "r_roll": r_roll,
            "l_elbow": l_elbow,
            "r_elbow": r_elbow,
        }

    def close(self):
        self.running = False
        try:
            self.sock.close(linger=0)
            self.ctx.term()
        except Exception:
            pass


class PickPutSubtaskSupervisor:
    """2-Stage 子任务部署监管调度器"""

    def __init__(
        self,
        mode: str,
        rollout_cmd: list[str],
        robot_ip: str,
        state_port: int = 6001,
        hold_time: float = 1.8,
        max_clamp_time: float = 5.0,
        reach_time: float = 8.5,
        auto_reset: bool = False,
        record: bool = False,
        diagnostics_dir: str = "",
    ):
        self.mode = mode.lower()
        self.rollout_cmd = rollout_cmd
        self.robot_ip = robot_ip
        self.state_port = state_port
        self.hold_time = hold_time
        self.max_clamp_time = max_clamp_time
        self.reach_time = reach_time
        self.auto_reset = auto_reset
        self.record = record
        self.diagnostics_dir = diagnostics_dir

        self.current_phase = 0
        self.phase_start_time = 0.0
        self.clamp_sustained_seconds = 0.0
        self.running = True

        self.ready_event = threading.Event()
        self.running_event = threading.Event()
        self.monitor = LiveStateMonitor(robot_ip, state_port)

    def send_cmd(self, proc: subprocess.Popen, text: str):
        """向底层 lerobot-rollout 发送交互控制命令"""
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
        self.clamp_sustained_seconds = 0.0
        task_name = SUBTASKS[new_phase]

        print("\n" + "=" * 82)
        print(f" 🚀 [阶段流转] 切换至 Phase [{new_phase + 1}/2]: \"{task_name}\"")
        if reason:
            print(f"    触发原因: {reason}")
        print("=" * 82 + "\n")

        # 记录流转切词时间戳到诊断目录
        if self.record and self.diagnostics_dir:
            try:
                switch_file = Path(self.diagnostics_dir) / "prompt_switches.json"
                switches = []
                if switch_file.exists():
                    try:
                        with open(switch_file, "r", encoding="utf-8") as f:
                            switches = json.load(f)
                    except Exception:
                        pass
                switches.append({
                    "timestamp": time.time(),
                    "phase": new_phase,
                    "task": task_name,
                    "reason": reason,
                })
                with open(switch_file, "w", encoding="utf-8") as f:
                    json.dump(switches, f, indent=2, ensure_ascii=False)
            except Exception as e_sw:
                logger.debug("写入 prompt_switches.json 失败: %s", e_sw)

        self.send_cmd(proc, f"/subtask {task_name}")

    def _stream_stdout(self, proc: subprocess.Popen):
        """实时输出底层日志并监听就绪信号"""
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()

                # 识别交互就绪 Banner
                if (
                    "Interactive rollout session" in text
                    or "the robot will NOT move until you type /start" in text
                    or "type /help for commands" in text
                ):
                    self.ready_event.set()

                # 识别运行开始
                if "Rollout running" in text or "Starting rollout" in text:
                    self.running_event.set()
        except Exception:
            pass

    def run_auto_supervisor(self, proc: subprocess.Popen):
        """模式 1: 自动化子任务流转 (抱稳计时 + 姿态触发 + 超时保底)"""
        print("\n" + "=" * 82)
        print("🤖 [自动化 2-Subtask 流转模式已激活]")
        print("   流转控制策略:")
        print(f"     Phase 1 (抱箱夹紧): 检测到双臂内夹且保持稳态 >= {self.hold_time:.1f}s，或总时间 >= {self.max_clamp_time:.1f}s")
        print(f"     Phase 2 (前伸放下): 强行注入 'reach forward...' 驱动双臂前伸放下，持续 {self.reach_time:.1f}s")
        print("   快捷按键:")
        print("     [Space] / [Enter] / 'n': 强制立即进入 Phase 2 (前伸放下)")
        print("     '1' / '2'              : 跳转至指定阶段")
        print("     'r'                    : 发送复位命令 (/r)")
        print("     'q'                    : 安全退出停止 (/stop)")
        print("=" * 82 + "\n")

        self.phase_start_time = time.time()
        last_status_print = 0.0

        while self.running and proc.poll() is None:
            time.sleep(0.08)
            now = time.time()
            elapsed = now - self.phase_start_time
            metrics = self.monitor.get_metrics()

            # 键盘按键响应
            key = getch_timeout(timeout=0.02)
            if key in (" ", "\n", "\r", "n", "N"):
                if self.current_phase == 0:
                    self.switch_to_phase(proc, 1, reason="用户按键手动触发提前进入前伸阶段")
                    continue
            elif key in ("r", "R"):
                print("\n[User Request] 发送复位命令 /r ...")
                self.send_cmd(proc, "/r")
                self.switch_to_phase(proc, 0, reason="用户重置任务")
                continue
            elif key in ("q", "Q"):
                print("\n[User Request] 发送退出命令 /stop ...")
                self.send_cmd(proc, "/stop")
                break
            elif key == "1":
                self.switch_to_phase(proc, 0, reason="用户跳转至 Phase 1")
                continue
            elif key == "2":
                self.switch_to_phase(proc, 1, reason="用户跳转至 Phase 2")
                continue

            # 终端 HUD 状态行刷新 (约 0.5 秒一次)
            if now - last_status_print >= 0.5:
                last_status_print = now
                conn_str = "🟢 ZMQ连接正常" if metrics["connected"] else "🔴 等待ZMQ状态(6001)"
                l_roll = metrics["l_roll"]
                r_roll = metrics["r_roll"]
                l_pitch = metrics["l_pitch"]
                r_pitch = metrics["r_pitch"]

                if self.current_phase == 0:
                    hud_msg = (
                        f"[{conn_str}] Phase 1 (抱箱) | 耗时: {elapsed:4.1f}s | "
                        f"夹持内倾: L={l_roll:+.2f}, R={r_roll:+.2f} | 抱稳进度: {self.clamp_sustained_seconds:3.1f}s/{self.hold_time:.1f}s"
                    )
                else:
                    hud_msg = (
                        f"[{conn_str}] Phase 2 (前伸) | 耗时: {elapsed:4.1f}s/{self.reach_time:.1f}s | "
                        f"俯仰: L={l_pitch:+.2f}, R={r_pitch:+.2f} rad"
                    )
                print(f"   [HUD] {hud_msg}", end="\r", flush=True)

            if not metrics["connected"]:
                continue

            # -----------------------------------------------------------------
            # Phase 0 -> Phase 1 流转判断
            # -----------------------------------------------------------------
            if self.current_phase == 0:
                # 判定夹取姿态: 左肩向内摆 (l_roll < -0.08) 且 右肩向内摆 (r_roll > 0.08)
                # 或双臂俯仰已抬离桌面 (l_pitch < -0.05)
                is_clamping = (metrics["l_roll"] <= -0.08 and metrics["r_roll"] >= 0.08) or (metrics["l_pitch"] <= -0.08)

                if is_clamping:
                    self.clamp_sustained_seconds += 0.08
                else:
                    self.clamp_sustained_seconds = max(0.0, self.clamp_sustained_seconds - 0.04)

                # 满足稳定抱持条件，或达到阶段最大保护超时
                if self.clamp_sustained_seconds >= self.hold_time or elapsed >= self.max_clamp_time:
                    trigger_reason = (
                        f"双手夹紧抱持稳定持续 {self.clamp_sustained_seconds:.1f}s (L_Roll={metrics['l_roll']:.2f}, R_Roll={metrics['r_roll']:.2f})"
                        if self.clamp_sustained_seconds >= self.hold_time
                        else f"Phase 1 达到防死锁保底时间上限 {elapsed:.1f}s"
                    )
                    self.switch_to_phase(proc, 1, reason=trigger_reason)

            # -----------------------------------------------------------------
            # Phase 1 -> 结束/复位判断
            # -----------------------------------------------------------------
            elif self.current_phase == 1:
                if elapsed >= self.reach_time:
                    print("\n" + "=" * 82)
                    print(f"🎉 [任务圆满完成] 前伸与放置阶段已执行完毕 (持续 {elapsed:.1f}s)！")
                    if self.auto_reset:
                        print("   正在自动发送复位命令 (/r)...")
                        self.send_cmd(proc, "/r")
                        time.sleep(2.0)
                    else:
                        print("   按 [r] 复位重新开始，或按 [q] 退出。")
                    print("=" * 82 + "\n")
                    if self.auto_reset:
                        break

    def run_manual_supervisor(self, proc: subprocess.Popen):
        """模式 2: 单键手动流转模式"""
        print("\n" + "=" * 82)
        print("🎮 [手动子任务流转模式已激活]")
        print("   快捷操作指引:")
        print("     [Space] / [Enter] / 'n' : 立即流转至下一个阶段 (Phase 1 -> Phase 2)")
        print("     '1'                     : 切换到 Phase 1 (\"clamp and lift the box\")")
        print("     '2'                     : 切换到 Phase 2 (\"reach forward and place the box in the blue area\")")
        print("     'r'                     : 发送复位命令 (/r)")
        print("     'q'                     : 安全停止退出 (/stop)")
        print("=" * 82)
        print(f"👉 当前就绪阶段 [1/2]: \"{SUBTASKS[0]}\" (夹起后按 Space 触发前伸)\n")

        self.phase_start_time = time.time()

        while self.running and proc.poll() is None:
            key = getch_timeout(timeout=0.1)
            if key is None:
                continue

            if key in (" ", "\n", "\r", "n", "N"):
                if self.current_phase == 0:
                    self.switch_to_phase(proc, 1, reason="用户按键触发进入前伸阶段")
                else:
                    print("⚠️ 当前已处于前伸放置阶段 (Phase 2)！按 'r' 复位，或按 'q' 退出。")
            elif key == "1":
                self.switch_to_phase(proc, 0, reason="用户跳转至 Phase 1")
            elif key == "2":
                self.switch_to_phase(proc, 1, reason="用户跳转至 Phase 2")
            elif key in ("r", "R"):
                print("\n[User Request] 发送复位命令 /r ...")
                self.send_cmd(proc, "/r")
                self.switch_to_phase(proc, 0, reason="用户重置任务")
            elif key in ("q", "Q"):
                print("\n[User Request] 发送退出命令 /stop ...")
                self.send_cmd(proc, "/stop")
                break

    def start(self):
        """启动底层 Rollout 并托管子任务流转"""
        initial_task = SUBTASKS[0]
        cmd = list(self.rollout_cmd)
        cmd.append(f"--task={initial_task}")
        cmd.append("--interactive=true")

        print("=" * 82)
        print(" [Subtask Supervisor] 启动底层 LeRobot Rollout 推理客户端...")
        print(f" 目标策略路径: {getattr(args, 'policy.path', DEFAULT_POLICY_PATH)}")
        if self.record and self.diagnostics_dir:
            print(f" 🔴 [诊断录制已开启] 50步Chunk预测与切片执行数据将保存至:")
            print(f"    {Path(self.diagnostics_dir).resolve()}")
        print(" 等待 VLA 大模型权重加载到 GPU 显存并连接机器人 (预计需 20~30 秒)...")
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
            if self.record and self.diagnostics_dir:
                print("\n" + "=" * 82)
                print("📊 [诊断录制已完成] 本次实机部署数据已成功保存！")
                print(f"   数据目录: {Path(self.diagnostics_dir).resolve()}")
                print(f"   一键分析: python dataset_tools/analyze_recorded_chunks.py {self.diagnostics_dir}")
                print("=" * 82 + "\n")


def main():
    global args
    parser = argparse.ArgumentParser(description="Unitree G1 2-Subtask Pick & Put Deployment Supervisor.")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto", help="子任务流转模式 (auto: 自动, manual: 手动按键)")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="G1 机器人或底层服务 IP")
    parser.add_argument("--action_ip", default="", help="动作指令发送 IP (默认等于 robot_ip)")
    parser.add_argument("--camera_ip", default="", help="机载相机推流 IP (默认等于 robot_ip)")
    parser.add_argument("--camera_port", type=int, default=5556, help="相机推流端口 (默认 5556)")
    parser.add_argument("--state_port", type=int, default=6001, help="29-DoF 关节状态端口 (默认 6001)")
    parser.add_argument("--action_port", type=int, default=6002, help="18-DoF 动作流接收端口 (默认 6002)")
    parser.add_argument("--policy.path", default=DEFAULT_POLICY_PATH, help="2-Subtask 训练好的模型路径")
    parser.add_argument("--queue_threshold", type=int, default=35, help="RTC 队列打断门限")
    parser.add_argument("--interpolation_multiplier", type=int, default=3, help="动作插值倍率")
    parser.add_argument("--fps", type=int, default=30, help="控制帧率 (30Hz)")
    parser.add_argument("--duration", type=int, default=1000, help="单次测试最大总秒数")
    parser.add_argument("--display_data", default="false", help="是否打开图像监控窗口")
    parser.add_argument("--hold_time", type=float, default=1.8, help="抱箱稳定维持时长门限 (秒)，触发进入前伸")
    parser.add_argument("--max_clamp_time", type=float, default=5.0, help="Phase 1 最大抱持超时保护门限 (秒)")
    parser.add_argument("--reach_time", type=float, default=8.5, help="Phase 2 前伸与放下持续时间 (秒)")
    parser.add_argument("--auto_reset", action="store_true", help="任务完成后是否自动复位 /r")
    parser.add_argument("--record", "--record_chunks", "--record_diagnostics", dest="record", action="store_true", help="记录大模型每次输出的50步Chunk和RTC切片执行数据")
    parser.add_argument("--diagnostics_dir", default="", help="诊断数据保存目录 (默认自动按时间戳在 outputs/diagnostics/ 生成)")

    args, unknown_args = parser.parse_known_args()

    action_ip = args.action_ip if args.action_ip else args.robot_ip
    camera_ip = args.camera_ip if args.camera_ip else args.robot_ip

    lerobot_rollout = "/home/yichangfeng/miniforge3/envs/lerobot/bin/lerobot-rollout"
    if not os.path.exists(lerobot_rollout):
        lerobot_rollout = sys.executable

    diag_dir = args.diagnostics_dir
    is_recording = args.record or bool(diag_dir)
    if is_recording and not diag_dir:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_dir = f"outputs/diagnostics/session_{ts}"

    base_cmd = [
        lerobot_rollout,
        "--strategy.type=base",
        "--inference.type=rtc",
        f"--inference.queue_threshold={args.queue_threshold}",
        f"--interpolation_multiplier={args.interpolation_multiplier}",
        f"--policy.path={getattr(args, 'policy.path')}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--robot.type=unitree_g1_client",
        f"--robot.robot_ip={args.robot_ip}",
        f"--robot.action_ip={action_ip}",
        f"--robot.camera_ip={camera_ip}",
        f"--robot.camera_port={args.camera_port}",
        f"--robot.state_port={args.state_port}",
        f"--robot.action_port={args.action_port}",
        "--robot.connect_timeout=60",
        "--robot.wait_until_ready=true",
        f"--duration={args.duration}",
        f"--fps={args.fps}",
        f"--display_data={args.display_data}",
        "--subtasks=false",
    ]
    if is_recording:
        Path(diag_dir).mkdir(parents=True, exist_ok=True)
        base_cmd.append("--record=true")
        base_cmd.append(f"--diagnostics_dir={diag_dir}")

    base_cmd += unknown_args

    supervisor = PickPutSubtaskSupervisor(
        mode=args.mode,
        rollout_cmd=base_cmd,
        robot_ip=args.robot_ip,
        state_port=args.state_port,
        hold_time=args.hold_time,
        max_clamp_time=args.max_clamp_time,
        reach_time=args.reach_time,
        auto_reset=args.auto_reset,
        record=is_recording,
        diagnostics_dir=diag_dir,
    )
    supervisor.start()


if __name__ == "__main__":
    main()
