# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive rollout session: chat-style stdin commands for ``lerobot-rollout``.

Enabled with ``--interactive=true``, this module lets the operator drive a rollout from the terminal
(``/help`` lists the commands) while hardware and policy stay connected and warm.  It adds only the
CLI front-end — stdin reading, command parsing, terminal output, and log muting.  Real shutdown
signals (SIGINT/SIGTERM) propagate through the session's :class:`LinkedEvent` parent, so Ctrl-C
behaves exactly as in non-interactive runs.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import threading
import time
from typing import IO, TYPE_CHECKING
import numpy as np

from lerobot.utils.stdin_input import StdinCommandListener
from lerobot.utils.utils import log_say

from .controller import AskResult, RolloutController, RolloutEvent
from .inference import QueryAnswer, QueryKind

if TYPE_CHECKING:
    from .context import RolloutContext
    from .strategies import RolloutStrategy

logger = logging.getLogger(__name__)

_BANNER_RULE = "─" * 60


@contextlib.contextmanager
def _mute_system_output() -> Iterator[None]:
    """Suppress log records below ERROR and Python warnings, process-wide.

    Routine system logs would contend with the chat prompt.  ``logging.disable`` gates records
    before handler dispatch, so non-propagating library loggers and loggers created mid-session are
    covered too (as are file handlers); ERROR and above still get through, so failures stay visible.
    """
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        # catch_warnings also restores the mutation counter and showwarning, unlike a filters snapshot.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous_disable)


@dataclass(frozen=True)
class InteractiveCommand:
    """A parsed ``/name args`` line from the interactive prompt."""

    name: str
    args: str = ""


def _format_task(task: str) -> str:
    """Render a task string for the operator, naming the empty case explicitly."""
    return repr(task) if task else "(none — set one with /subtask <text>)"


def _strip_quotes(text: str) -> str:
    """Drop one layer of matching surrounding quotes from a command argument."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


DEFAULT_SUBTASKS: list[str] = [
    "clamp and lift the box",
    "hold the box and turn right",
    "place the box on the table and release",
]

COMMAND_ALIASES: dict[str, str] = {
    "r": "reset",
    "reset": "reset",
    "s": "start",
    "start": "start",
    "q": "stop",
    "stop": "stop",
    "h": "help",
    "help": "help",
    "n": "next",
    "next": "next",
    "m": "mode",
    "mode": "mode",
    "1": "phase1",
    "2": "phase2",
    "3": "phase3",
    "phase1": "phase1",
    "phase2": "phase2",
    "phase3": "phase3",
}


def parse_command(line: str) -> InteractiveCommand | None:
    """Parse an input line into an :class:`InteractiveCommand`.

    Commands are ``/name`` (or single-character shortcuts ``/s``, ``/r``, ``/q``, ``/h``,
    and bare ``s``, ``r``, ``q``, ``h``) optionally followed by free-text arguments.
    Returns ``None`` for lines that are not commands.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("/"):
        parts = line[1:].split(maxsplit=1)
        if not parts or not parts[0]:
            return None
        head = parts[0]
        args = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Check single-character shortcuts without slash
        parts = line.split(maxsplit=1)
        if parts[0].lower() in COMMAND_ALIASES:
            head = parts[0]
            args = parts[1].strip() if len(parts) > 1 else ""
        else:
            return None

    name = head.lower()
    name = COMMAND_ALIASES.get(name, name)
    return InteractiveCommand(name=name, args=args)


class InteractiveSession:
    """Drive a rollout from chat-style stdin commands.

    A thin terminal front-end over :class:`RolloutController`, exposed as :attr:`controller` for
    tests and embedders: the stdin listener parses lines into commands that call the controller's
    thread-safe methods, and controller events are rendered back as terminal output.

    Commands are last-write-wins: ``/reset`` and ``/stop`` cancel a pending ``/start``.  EOF on the
    command stream stops the session (nothing is left to command the robot with), so piped scripts
    must keep stdin open for the intended duration, e.g.
    ``(printf '/start\\n'; sleep 60; printf '/stop\\n') | lerobot-rollout ... --interactive=true``.
    """

    def __init__(
        self,
        strategy: RolloutStrategy,
        ctx: RolloutContext,
        input_stream: IO[str] | None = None,
    ) -> None:
        self.ctx = ctx
        self.controller = RolloutController(strategy, ctx, on_event=self._on_event)
        self._runtime = ctx.runtime
        self._play_sounds = ctx.runtime.cfg.play_sounds
        self._listener = StdinCommandListener(self._handle_line, on_eof=self._handle_eof, stream=input_stream)

        # Auto mode (port 6003 supervision: NAV/GAMEPAD/VLA mode transitions)
        self._auto_mode = getattr(ctx.runtime.cfg, "auto_mode", False)
        self._is_resetting = False
        self._is_starting = False
        self._auto_supervisor_thread: threading.Thread | None = None
        self._auto_supervisor_stop_event = threading.Event()

        # Subtask automatic sequencing support
        raw_subtasks = getattr(ctx.runtime.cfg, "subtasks", None)
        policy_path = str(getattr(getattr(ctx.runtime.cfg, "policy", None), "path", ""))

        if isinstance(raw_subtasks, str) and raw_subtasks.lower() in ("false", "0", "none", "off", "no"):
            self._subtasks_enabled = False
            self._subtask_list = []
        elif isinstance(raw_subtasks, str) and ("," in raw_subtasks or len(raw_subtasks.strip()) > 0):
            self._subtask_list = [s.strip() for s in raw_subtasks.split(",") if s.strip()]
            self._subtasks_enabled = len(self._subtask_list) > 0
        elif isinstance(raw_subtasks, (list, tuple)):
            self._subtask_list = list(raw_subtasks)
            self._subtasks_enabled = bool(raw_subtasks)
        elif raw_subtasks is True:
            self._subtasks_enabled = True
            self._subtask_list = list(DEFAULT_SUBTASKS)
        else:
            # Auto-detect only if this is specifically the legacy 3-stage turn-right model
            if "turn" in policy_path.lower() and "subtask" in policy_path.lower():
                self._subtasks_enabled = True
                self._subtask_list = list(DEFAULT_SUBTASKS)
            else:
                self._subtasks_enabled = False
                self._subtask_list = []

        self._current_phase = 0
        self._phase_start_time = 0.0
        self._phase1_start_yaw = 0.0
        self._lift_sustained_seconds = 0.0
        self._subtask_tracker_thread: threading.Thread | None = None
        self._subtask_tracker_running = False
        self._subtask_stop_event = threading.Event()

        if self._subtasks_enabled and self._subtask_list:
            self.controller._initial_task = self._subtask_list[0]
            self.controller.set_task(self._subtask_list[0])

        # name -> (handler, argument hint, help line); /help and the banner render from this table.
        self._commands: dict[str, tuple[Callable[[InteractiveCommand], None], str, str]] = {
            "start": (self._cmd_start, "", "start (or restart) the policy control loop (shortcut: /s, s)"),
            "subtask": (self._cmd_subtask, " <text>", "set the instruction the policy follows"),
            "vqa": (self._cmd_vqa, " <text>", "ask the policy a question about what it sees"),
            "autosteer": (
                self._cmd_autosteer,
                " <goal>|off",
                "let the policy pick its own subtasks toward a high-level goal",
            ),
            "reset": (self._cmd_reset, "", "stop movement, return to initial position, reset VLA context (shortcut: /r, r)"),
            "mode": (self._cmd_mode, "", "show 6003 robot mode stream status & diagnostics (shortcut: /mode, m)"),
            "stop": (self._cmd_stop, "", "end the session and shut down (shortcut: /q, q)"),
            "help": (self._cmd_help, "", "show this help (shortcut: /h, h)"),
        }
        if self._subtasks_enabled:
            self._commands["next"] = (self._cmd_next_subtask, "", "advance to next subtask phase (shortcut: /n, n)")
            self._commands["phase1"] = (lambda cmd: self._cmd_jump_phase(0), "", "jump to phase 1: clamp and lift the box (shortcut: /1, 1)")
            self._commands["phase2"] = (lambda cmd: self._cmd_jump_phase(1), "", "jump to phase 2: hold and turn right (shortcut: /2, 2)")
            self._commands["phase3"] = (lambda cmd: self._cmd_jump_phase(2), "", "jump to phase 3: place on table and release (shortcut: /3, 3)")

    @property
    def robot_wrapper(self):
        """Retrieve the robot wrapper instance reliably across sub-context structures."""
        hw = getattr(self.ctx, "hardware", None)
        return getattr(hw, "robot_wrapper", None) if hw is not None else getattr(self.ctx, "robot_wrapper", None)

    @contextlib.contextmanager
    def _route_cadence_reports(self) -> Iterator[None]:
        """Send the control loop's cadence summaries to the chat stream, not the muted log.

        Boundary-only output, printed on the serve thread; scoped like :func:`_mute_system_output`.
        """
        previous = self._runtime.cadence_report
        self._runtime.cadence_report = self._print
        try:
            yield
        finally:
            self._runtime.cadence_report = previous

    def run(self) -> None:
        """Run the session until ``/stop``, EOF, engine failure, or a shutdown signal."""
        try:
            with _mute_system_output(), self._route_cadence_reports():
                self._print(self._render_banner())
                self._listener.start()
                if self._auto_mode:
                    self._start_auto_supervisor()
                try:
                    self.controller.serve()
                finally:
                    self._listener.stop()
        finally:
            self._stop_auto_supervisor()
            self._stop_subtask_tracker()
            # Outside the muting context, so the announcement and teardown logs are visible again.
            log_say("Interactive session ended", self._play_sounds)

    # ------------------------------------------------------------------
    # Controller events (fired on the serve thread) -> terminal output
    # ------------------------------------------------------------------

    def _on_event(self, event: RolloutEvent, payload: QueryAnswer | None = None) -> None:
        if event is RolloutEvent.QUERY_ANSWERED and payload is not None:
            self._report_answer(payload)
        elif event is RolloutEvent.SEGMENT_STARTED:
            self._is_starting = False
            log_say("Starting rollout", self._play_sounds)
            if self._subtasks_enabled:
                self.controller.set_task(self._subtask_list[self._current_phase])
                self._print(
                    f"Rollout running — [3-Subtask 模式] 当前阶段 [{self._current_phase + 1}/3]: "
                    f"\"{self._subtask_list[self._current_phase]}\".\n"
                    "快捷指令: 'n' 跳下一阶段, '1'/'2'/'3' 选段, 'r' 复位, 'q' 退出。"
                )
                self._start_subtask_tracker()
            else:
                self._print(
                    f"Rollout running — task {_format_task(self.controller.task)}. "
                    "/subtask <text> to change it, /reset to return to initial position, /stop to shut down."
                )
        elif event is RolloutEvent.SEGMENT_ENDED:
            self._is_starting = False
            self._stop_subtask_tracker()
            self._print(
                "Rollout run ended on its own (duration reached). Robot is holding position — "
                "/start to run again, /reset to return to initial position, /stop to shut down."
            )
        elif event is RolloutEvent.RESET_STARTED:
            self._is_starting = False
            self._is_resetting = True
            self._stop_subtask_tracker()
            log_say("Resetting robot to initial position", self._play_sounds)
            self._print("Resetting — returning the robot to its initial position...")
        elif event is RolloutEvent.RESET_DONE:
            self._is_starting = False
            self._is_resetting = False
            self._print("Robot reset — holding at initial position. /start (or s) to run.")
        elif event is RolloutEvent.RESET_SKIPPED:
            self._is_starting = False
            self._is_resetting = False
            self._print("Robot paused — no initial position captured, holding current pose. /start to run.")
        elif event is RolloutEvent.RESET_FAILED:
            self._is_starting = False
            self._is_resetting = False
            self._print(
                "Reset FAILED — the return move errored, so the robot may NOT be at its "
                "initial position. Check the robot before /start."
            )
        elif event is RolloutEvent.ENGINE_FAILED:
            self._is_starting = False
            self._stop_subtask_tracker()
            self._report_failure("Inference engine failed — shutting down.")
        elif event is RolloutEvent.STRATEGY_FAILED:
            self._is_starting = False
            self._stop_subtask_tracker()
            self._report_failure("Rollout strategy failed (robot or recording error) — shutting down.")
        elif event is RolloutEvent.STOPPED:
            self._is_starting = False
            self._stop_subtask_tracker()

    def _report_answer(self, answer: QueryAnswer) -> None:
        """Render a resolved text query (an operator question or an autosteer turn)."""
        if answer.kind is QueryKind.NEXT_SUBTASK:
            if answer.ok:
                # The engine has already applied it via set_task; just announce.
                self._print(f"Autosteer subtask: {answer.answer!r}")
            else:
                self._print(
                    f"Autosteer stopped — could not plan the next subtask for {answer.question!r}: "
                    f"{answer.error}"
                )
        elif answer.ok:
            self._print(f"Q: {answer.question}\nA: {answer.answer}")
        else:
            self._print(f"Could not answer {answer.question!r} — {answer.error}")

    def _report_failure(self, headline: str) -> None:
        """Surface a fatal engine/strategy error despite the muted console logging."""
        self._print(headline)
        failure_traceback = self.controller.failure_traceback
        if failure_traceback:
            self._print(failure_traceback)
        else:
            self._print("Re-run without --interactive=true to see the error output.")

    # ------------------------------------------------------------------
    # Auto Mode Supervisor (Port 6003: NAV / GAMEPAD / VLA transitions)
    # ------------------------------------------------------------------

    def _start_auto_supervisor(self) -> None:
        if self._auto_supervisor_thread is not None and self._auto_supervisor_thread.is_alive():
            return
        self._auto_supervisor_stop_event.clear()
        self._auto_supervisor_thread = threading.Thread(
            target=self._auto_supervisor_loop, daemon=True, name="AutoModeSupervisor"
        )
        self._auto_supervisor_thread.start()

    def _stop_auto_supervisor(self) -> None:
        self._auto_supervisor_stop_event.set()
        if self._auto_supervisor_thread is not None:
            self._auto_supervisor_thread.join(timeout=1.0)
            self._auto_supervisor_thread = None

    def _auto_supervisor_loop(self) -> None:
        """Supervises robot mode on port 6003 for automatic start and reset.

        Mode lifecycle:
        - Handheld controller in NAV or GAMEPAD: Wait idly (inference paused).
        - Handheld controller enters VLA: Automatically invoke controller.start() and trigger smooth engagement.
        - Handheld controller leaves VLA: Wait 0.2s debounce; if still non-VLA, invoke controller.reset().
        - Resetting completes: Wait for subsequent VLA entry to re-trigger start with smooth engagement.
        """
        robot = self.robot_wrapper
        debounce_start_time: float | None = None

        while not self._auto_supervisor_stop_event.is_set():
            if self.controller.stopped:
                break

            current_mode = getattr(robot, "current_mode", None)
            is_vla = bool(getattr(robot, "is_vla_mode", False))
            mode_port = getattr(robot, "mode_port", 6000)

            # Case 1: Currently IDLE (not running, not resetting, and not starting) -> Watch for VLA entry
            if not self.controller.running and not self._is_resetting and not self._is_starting:
                debounce_start_time = None
                if is_vla:
                    self._is_starting = True
                    mode_val = current_mode.value if hasattr(current_mode, "value") else str(current_mode)
                    self._print(
                        f"\n🤖 [自动模式] 监听到 {mode_port} 端口切换为 VLA 模式 ({mode_val})！\n"
                        "   ├─ 自动启动策略推理 (Controller.start)\n"
                        "   └─ 激活关节平滑过渡 (S-curve Smoothing) 消除初始突变..."
                    )
                    if hasattr(robot, "trigger_engagement_smoothing"):
                        robot.trigger_engagement_smoothing()
                    if not self.controller.start():
                        self._is_starting = False

            # Case 2: Currently RUNNING -> Watch for non-VLA exit with 0.2s debounce
            elif self.controller.running and not self._is_resetting:
                # ONLY trigger reset when mode is explicitly recognized as NAV or GAMEPAD!
                # NEVER reset when mode is UNKNOWN (e.g. no packets, or manual start without mode stream)
                from lerobot.robots.unitree_g1.unitree_g1_client import RobotMode
                if current_mode in (RobotMode.NAV, RobotMode.GAMEPAD):
                    if debounce_start_time is None:
                        debounce_start_time = time.time()
                    elif time.time() - debounce_start_time >= 0.2:
                        mode_val = current_mode.value if hasattr(current_mode, "value") else str(current_mode)
                        self._print(
                            f"\n🤖 [自动模式] 监听到 {mode_port} 端口退出 VLA 模式 (切换为手柄/导航模式: {mode_val})，持续超过 0.2s！\n"
                            "   └─ 自动触发复位 (Reset) 归位并清空 RTC 历史..."
                        )
                        self._cmd_reset(InteractiveCommand(name="reset"))
                        debounce_start_time = None
                else:
                    debounce_start_time = None

            time.sleep(0.02)

    # ------------------------------------------------------------------
    # 3-Subtask Kinematics & Automated Progression
    # ------------------------------------------------------------------

    def _get_robot_metrics(self) -> dict:
        """Extract arm shoulder pitch and IMU yaw from robot state."""
        try:
            robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
            if robot is not None and hasattr(robot, "_latest_state") and hasattr(robot, "_state_lock"):
                with robot._state_lock:
                    state = robot._latest_state
                if state is not None:
                    motors = state.get("motors", {})
                    l_pitch = float(motors.get("left_shoulder_pitch_joint", {}).get("q", 0.0))
                    r_pitch = float(motors.get("right_shoulder_pitch_joint", {}).get("q", 0.0))
                    l_roll = float(motors.get("left_shoulder_roll_joint", {}).get("q", 0.0))
                    r_roll = float(motors.get("right_shoulder_roll_joint", {}).get("q", 0.0))
                    imu = state.get("imu", {})
                    rpy = imu.get("rpy", [0.0, 0.0, 0.0])
                    yaw = float(rpy[2]) if len(rpy) >= 3 else 0.0
                    return {
                        "connected": True,
                        "l_pitch": l_pitch,
                        "r_pitch": r_pitch,
                        "l_roll": l_roll,
                        "r_roll": r_roll,
                        "yaw": yaw,
                    }
        except Exception as e:
            logger.debug("Error reading robot metrics: %s", e)
        return {
            "connected": False,
            "l_pitch": 0.0,
            "r_pitch": 0.0,
            "l_roll": 0.0,
            "r_roll": 0.0,
            "yaw": 0.0,
        }

    def _start_subtask_tracker(self) -> None:
        if not self._subtasks_enabled:
            return
        self._stop_subtask_tracker()
        self._subtask_stop_event.clear()
        self._subtask_tracker_running = True
        self._phase_start_time = time.time()
        self._lift_sustained_seconds = 0.0
        self._subtask_tracker_thread = threading.Thread(
            target=self._subtask_tracker_loop,
            name="SubtaskTrackerThread",
            daemon=True,
        )
        self._subtask_tracker_thread.start()

    def _stop_subtask_tracker(self) -> None:
        self._subtask_tracker_running = False
        self._subtask_stop_event.set()
        if self._subtask_tracker_thread is not None and self._subtask_tracker_thread.is_alive():
            self._subtask_tracker_thread.join(timeout=0.5)
        self._subtask_tracker_thread = None

    def _advance_phase(self, new_phase: int, reason: str = "") -> None:
        if not self._subtasks_enabled or new_phase < 0 or new_phase >= len(self._subtask_list):
            return
        self._current_phase = new_phase
        self._phase_start_time = time.time()
        self._lift_sustained_seconds = 0.0
        new_task = self._subtask_list[new_phase]

        if new_phase == 1:
            metrics = self._get_robot_metrics()
            self._phase1_start_yaw = metrics["yaw"]

        self.controller.set_task(new_task)
        msg = (
            f"\n" + "=" * 60 + "\n"
            f" [Subtasks 流转] 切换至 Subtask [{new_phase + 1}/3]: \"{new_task}\"\n"
        )
        if reason:
            msg += f"   原因: {reason}\n"
        msg += "=" * 60
        self._print(msg)

    def _subtask_tracker_loop(self) -> None:
        """Background thread monitoring kinematics for 3-stage subtask transitions."""
        logger.info("Subtask tracker thread started.")
        self._phase_start_time = time.time()
        self._phase1_start_yaw = 0.0
        self._lift_sustained_seconds = 0.0
        last_hud_time = 0.0

        while self._subtask_tracker_running and not self._subtask_stop_event.is_set():
            time.sleep(0.1)
            now = time.time()
            elapsed = now - self._phase_start_time
            metrics = self._get_robot_metrics()

            if not metrics["connected"]:
                continue

            # Status HUD every 2.0s
            if now - last_hud_time >= 2.0:
                last_hud_time = now
                if self._current_phase == 0:
                    self._print(
                        f"📊 [Subtask 1/3: 抱箱] 耗时: {elapsed:4.1f}s | "
                        f"双肩俯仰: L={metrics['l_pitch']:+.2f}, R={metrics['r_pitch']:+.2f} rad (目标 <= -0.35)"
                    )
                elif self._current_phase == 1:
                    curr_yaw = metrics["yaw"]
                    dyaw = (curr_yaw - self._phase1_start_yaw + np.pi) % (2.0 * np.pi) - np.pi
                    dyaw_deg = float(np.degrees(dyaw))
                    self._print(
                        f"📊 [Subtask 2/3: 右转] 耗时: {elapsed:4.1f}s | "
                        f"累积转角: {dyaw_deg:+5.1f}° (目标 <= -78°)"
                    )
                elif self._current_phase == 2:
                    self._print(
                        f"📊 [Subtask 3/3: 放箱] 耗时: {elapsed:4.1f}s / 8.5s"
                    )

            # Phase 0: "clamp and lift the box"
            if self._current_phase == 0:
                pitch_lifted = (metrics["l_pitch"] <= -0.35 and metrics["r_pitch"] <= -0.35)
                if pitch_lifted:
                    self._lift_sustained_seconds += 0.1
                else:
                    self._lift_sustained_seconds = max(0.0, self._lift_sustained_seconds - 0.05)

                if self._lift_sustained_seconds >= 1.0 or elapsed >= 12.0:
                    reason = (
                        f"双臂夹紧抬箱姿态稳定维持 {self._lift_sustained_seconds:.1f}s (L={metrics['l_pitch']:.2f}, R={metrics['r_pitch']:.2f})"
                        if self._lift_sustained_seconds >= 1.0
                        else f"抱箱抬升时间达到经验门限 {elapsed:.1f}s"
                    )
                    self._advance_phase(1, reason=reason)

            # Phase 1: "hold the box and turn right"
            elif self._current_phase == 1:
                curr_yaw = metrics["yaw"]
                dyaw = (curr_yaw - self._phase1_start_yaw + np.pi) % (2.0 * np.pi) - np.pi
                dyaw_deg = float(np.degrees(dyaw))
                turn_completed = (dyaw_deg <= -78.0 or abs(dyaw_deg) >= 78.0)
                if turn_completed or elapsed >= 6.5:
                    reason = (
                        f"底盘右转已到位 ({dyaw_deg:+.1f}°, 目标 -78°)"
                        if turn_completed
                        else f"踏步转弯时间达到经验门限 {elapsed:.1f}s (当前转角: {dyaw_deg:+.1f}°)"
                    )
                    self._advance_phase(2, reason=reason)

            # Phase 2: "place the box on the table and release"
            elif self._current_phase == 2:
                if elapsed >= 8.5:
                    self._print("\n" + "=" * 60)
                    self._print("🎉 [Subtasks] 3 阶段子任务已完整执行完毕！")
                    self._print("   机器人保持当前位置。输入 /r (或 r) 复位回到初始位置，/s 再次运行，/q 退出。")
                    self._print("=" * 60 + "\n")
                    self._current_phase = 3

    # ------------------------------------------------------------------
    # Command handlers (called from the listener thread)
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            # User pressed empty Enter: print a one-line quick status
            robot = self.robot_wrapper
            mode = getattr(robot, "current_mode", "N/A")
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            pkts = getattr(robot, "mode_packet_count", 0)
            mode_port = getattr(robot, "mode_port", 6000)
            ctrl_state = "RUNNING (运行中)" if self.controller.running else ("RESETTING (复位中)" if self._is_resetting else "IDLE (待命)")
            self._print(f"[当前状态] 模式端口({mode_port}): {mode_val} (收包: {pkts}) | 控制器: {ctrl_state} | 输入 'm' 查看详情, 's' 启动, 'r' 复位")
            return
        cmd = parse_command(stripped)
        if cmd is None:
            self._print("Input not recognized — commands: s (start), r (reset), m (mode), n (next), 1/2/3, q (stop), /help.")
            return
        entry = self._commands.get(cmd.name)
        if entry is None:
            self._print(f"Unknown command '/{cmd.name}'. Type /help for the list.")
            return
        handler = entry[0]
        handler(cmd)

    def _handle_eof(self) -> None:
        self._print("Input stream closed — stopping the session.")
        self.controller.stop()

    def _cmd_mode(self, cmd: InteractiveCommand) -> None:
        """Display live mode monitoring statistics and troubleshooting tips."""
        robot = self.robot_wrapper
        mode_port = getattr(robot, "mode_port", 6000)
        self._print("\n" + "=" * 65)
        self._print(f"📡 [{mode_port} 端口模式监听诊断报告]")
        if robot is not None and hasattr(robot, "mode_status_summary"):
            self._print(robot.mode_status_summary())
        else:
            self._print(f"  • 当前模式: {getattr(robot, 'current_mode', 'N/A')}")

        mode_packets = getattr(robot, "mode_packet_count", 0)
        if mode_packets == 0:
            self._print(f"\n💡 [排查提示 - 尚未收到任何 {mode_port} 数据包]:")
            self._print(f"   1. 请确认状态发送端 (手柄遥控/导航节点) 已启动并正在向 {mode_port} 端口发送 PUB 广播。")
            self._print(f"   2. 请确认发送端绑定的 IP 是 0.0.0.0 或 *，且本机与机器人 IP 之间网络互通。")
            self._print(f"   3. 可新开终端运行: python check_zmq_connection.py --mode-port={mode_port} 进行 2 秒快速排查。")
        self._print("=" * 65 + "\n")

    def _cmd_start(self, cmd: InteractiveCommand) -> None:
        robot = self.robot_wrapper
        if robot is not None and hasattr(robot, "trigger_engagement_smoothing"):
            robot.trigger_engagement_smoothing()
        if self.controller.start():
            return
        # start() also refuses while stopping or after a failure — don't mislabel an idle robot.
        if self.controller.running:
            self._print("Already running — /reset to pause first, or /stop to shut down.")
        else:
            self._print("Can't start — the session is stopping or has failed.")

    def _cmd_next_subtask(self, cmd: InteractiveCommand) -> None:
        if not self._subtasks_enabled:
            self._print("Subtasks mode is not enabled.")
            return
        if self._current_phase < len(self._subtask_list) - 1:
            self._advance_phase(self._current_phase + 1, reason="用户按键手动触发跳段")
        else:
            self._print(f"当前已是最后阶段: {self._subtask_list[-1]}")

    def _cmd_jump_phase(self, idx: int) -> None:
        if not self._subtasks_enabled:
            self._print("Subtasks mode is not enabled.")
            return
        if 0 <= idx < len(self._subtask_list):
            self._advance_phase(idx, reason=f"用户直接跳转至第 {idx + 1} 阶段")
        else:
            self._print(f"无效阶段索引: {idx + 1}")

    def _cmd_subtask(self, cmd: InteractiveCommand) -> None:
        # Strip quotes before the emptiness check, so /subtask "" reports the task instead of
        # silently applying the empty instruction.
        task = _strip_quotes(cmd.args)
        if not task:
            self._print(f"Current task: {_format_task(self.controller.task)}")
            return
        previous = self.controller.task
        steering = self.controller.autosteer_goal
        if steering is not None:
            self._print(f"Autosteer off (was {steering!r}) — setting the instruction by hand takes over.")
        if self.controller.set_task(task):
            self._print(
                f"Task: {_format_task(previous)} → {_format_task(task)} "
                "(applies from the next policy inference)"
            )
        elif task == self.controller.task:
            self._print(f"Task unchanged: {_format_task(task)}")
        else:
            # set_task also refuses while stopping; "unchanged" would imply it was applied.
            self._print("Can't change the task — the session is stopping.")

    def _cmd_vqa(self, cmd: InteractiveCommand) -> None:
        # Strip quotes first, so /vqa "" prints the usage hint instead of queueing an empty question.
        question = _strip_quotes(cmd.args)
        if not question:
            self._print("Usage: /vqa <question> — e.g. /vqa is the cube inside the box?")
            return
        result = self.controller.ask(question)
        if result is AskResult.QUEUED:
            self._print(f"Asked: {question!r} — answering from the next observation...")
        elif result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot answer questions.")
        elif result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to answer from.")
        elif result is AskResult.BUSY:
            # Could be a previous /vqa or an autosteer query — the channel does not say which.
            self._print("The policy is busy with another query — try again in a moment.")
        else:  # a future AskResult variant must not be mislabeled as busy
            logger.error("Unhandled AskResult %r for /vqa", result)
            self._print(f"Could not queue the question ({result.value}).")

    def _cmd_autosteer(self, cmd: InteractiveCommand) -> None:
        goal = _strip_quotes(cmd.args)
        if not goal:
            current = self.controller.autosteer_goal
            self._print(
                f"Autosteer on — goal {current!r}." if current else "Autosteer off. Usage: /autosteer <goal>"
            )
            return
        if goal.lower() == "off":
            stopped = self.controller.stop_autosteer()
            self._print(
                f"Autosteer off (was {stopped!r}). The last subtask stays in effect."
                if stopped
                else "Autosteer was not running."
            )
            return
        result = self.controller.autosteer(goal)
        if result is AskResult.UNSUPPORTED:
            self._print("This policy has no text head — it cannot plan subtasks.")
        elif result is AskResult.NOT_RUNNING:
            self._print("Not running — /start first so the policy has a live view to plan from.")
        elif result is AskResult.QUEUED:
            self._print(
                f"Autosteer on — goal {goal!r}. The policy picks its own subtasks; "
                "each one is announced here. Take over with /subtask <text> or /autosteer off."
            )
        else:  # a future AskResult variant must not be announced as success
            logger.error("Unhandled AskResult %r for /autosteer", result)
            self._print(f"Could not start autosteer ({result.value}).")

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        self._is_starting = False
        self._is_resetting = True
        self._stop_subtask_tracker()
        if self._subtasks_enabled:
            self._current_phase = 0
            self._lift_sustained_seconds = 0.0
            self._phase_start_time = 0.0
            self.controller._initial_task = self._subtask_list[0]
        self.controller.reset()
        if self.controller.stopped:
            self._print("Can't reset — the session has stopped.")
        else:
            if self._subtasks_enabled:
                self._print(
                    f"♻️ [Reset] 任务已重置为 [1/3]: {_format_task(self.controller.initial_task)}\n"
                    "   ├─ 已清空 RTC 异步动作队列与插值历史\n"
                    "   └─ ⚠️ 注意: 底盘物理偏航角未动！若机器人此前已转向，请将底盘/箱子调回视野正前方后再按 's' 启动。"
                )
            else:
                self._print(
                    f"♻️ [Reset] 任务已恢复为: {_format_task(self.controller.initial_task)}\n"
                    "   ├─ 已清空 RTC 异步动作队列与插值历史\n"
                    "   └─ ⚠️ 注意: 若机器人此前已移动或转向，请将机器人/目标物调回初始相对位置后再按 's' 启动。"
                )

    def _cmd_stop(self, cmd: InteractiveCommand) -> None:
        self.controller.stop()

    def _cmd_help(self, cmd: InteractiveCommand) -> None:
        self._print(self._render_help())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_help(self) -> str:
        usages = {name: f"/{name}{entry[1]}" for name, entry in self._commands.items()}
        width = max(len(usage) for usage in usages.values())
        lines = [f"  {usages[name]:<{width}}   {entry[2]}" for name, entry in self._commands.items()]
        return "Available commands:\n" + "\n".join(lines)

    def _render_banner(self) -> str:
        subtask_info = ""
        if self._subtasks_enabled:
            subtask_info = (
                "Subtasks Sequence: ON (3 阶段自动流转模式已激活)\n"
                "  [1] clamp and lift the box -> [2] hold the box and turn right -> [3] place the box on the table and release\n"
                "  (快捷键: 'n' 跳下一阶段, '1'/'2'/'3' 选阶段, 'r' 复位, 's' 启动, 'q' 退出)\n"
            )
        mode_info = ""
        robot = self.robot_wrapper
        mode_port = getattr(robot, "mode_port", getattr(getattr(self.ctx, "runtime", None), "cfg", None).mode_port if hasattr(getattr(self.ctx, "runtime", None), "cfg") else 6000)
        if self._auto_mode:
            mode_info = (
                "Deploy Mode: AUTO (★ 自动模式已激活)\n"
                f"  • 自动监听 {mode_port} 端口: 导航/手柄模式下待命，切入 VLA 模式时自动启动推理并平滑过渡\n"
                "  • 切出 VLA 模式后延迟 0.2s 自动执行复位 (Reset)，切回 VLA 再次自动接管\n"
                f"  • 随时输入 'm' (或 /mode) 查看 {mode_port} 端口实时收包与手柄模式状态\n"
                "  • 亦支持键盘指令: 's' (启动), 'r' (复位), 'q' (退出)\n"
            )
            session_lead = f"Interactive rollout session — waiting for VLA gamepad mode on port {mode_port} (or type /start).\n"
        else:
            mode_info = (
                "Deploy Mode: MANUAL (手动模式)\n"
                "  • 键盘输入 's' (或 /start) 启动推理，'r' 复位，'q' 退出\n"
                f"  • 随时输入 'm' (或 /mode) 查看 {mode_port} 端口实时收包与手柄模式状态\n"
            )
            session_lead = "Interactive rollout session — the robot will NOT move until you type /start (or /s).\n"

        return (
            f"{_BANNER_RULE}\n"
            f"{session_lead}"
            f"{mode_info}"
            f"{subtask_info}"
            f"Task: {_format_task(self.controller.initial_task)}\n"
            f"{self._render_help()}\n"
            "Routine system logs and warnings are muted during the session (errors and the "
            "cadence summary of each run still show).\n"
            f"{_BANNER_RULE}"
        )

    @staticmethod
    def _print(message: str) -> None:
        """User-facing chat output; logging stays on stderr, replies on stdout.

        One ``write`` call per message, newline included: ``print()``'s separate message/newline
        writes can interleave mid-line between the listener and serve threads.
        """
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
