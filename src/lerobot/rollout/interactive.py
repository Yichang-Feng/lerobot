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
    "d": "navdone",
    "nav": "navdone",
    "navdone": "navdone",
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
            if "dex1" in policy_path.lower():
                self._subtask_list = [
                    "pick up the water bottle from the table and place it into the blue box",
                    "take the water bottle out of the blue box and place it back on the table",
                ]
            else:
                self._subtask_list = list(DEFAULT_SUBTASKS)
        else:
            if "dex1" in policy_path.lower() and "subtask" in policy_path.lower():
                self._subtasks_enabled = True
                self._subtask_list = [
                    "pick up the water bottle from the table and place it into the blue box",
                    "take the water bottle out of the blue box and place it back on the table",
                ]
            elif "turn" in policy_path.lower() and "subtask" in policy_path.lower():
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

        # Transfer mode: cross-table water bottle transport
        # Enabled via --transfer_mode=true CLI flag
        raw_transfer = getattr(ctx.runtime.cfg, "transfer_mode", False)
        if isinstance(raw_transfer, str):
            self._transfer_mode = raw_transfer.lower() in ("true", "1", "yes")
        else:
            self._transfer_mode = bool(raw_transfer)

        raw_hold = getattr(ctx.runtime.cfg, "transfer_hold_after_grasp", False)
        if isinstance(raw_hold, str):
            self._transfer_hold_after_grasp = raw_hold.lower() in ("true", "1", "yes")
        else:
            self._transfer_hold_after_grasp = bool(raw_hold)

        raw_restore_b = getattr(ctx.runtime.cfg, "transfer_table_b_restore_pose", True)
        if isinstance(raw_restore_b, str):
            self._transfer_table_b_restore_pose = raw_restore_b.lower() in ("true", "1", "yes")
        else:
            self._transfer_table_b_restore_pose = bool(raw_restore_b)
        # Event set by /navdone command to signal navigation has completed
        self._nav_done_event = threading.Event()
        # Snapshot of robot joint state at the moment the bottle was confirmed grasped+lifted.
        # Used to restore arm pose at table B before VLA takeover (to prevent motion jump).
        self._frozen_joint_state: dict | None = None
        # Real-time gripper angle measured at the moment grasp success was confirmed.
        # Clamped throughout homing, navigation, and pose restore without VLA intervention.
        self._captured_gripper_val: float | None = None
        self._transfer_banner_printed: bool = False
        self._phase2_banner_printed: bool = False
        self._phase4_banner_printed: bool = False
        self._transfer_triggered: bool = False
        self._transfer_lock = threading.Lock()
        self._decided_placement_task: str | None = None
        self._jev_mode_evaluating = threading.Lock()
        self._last_observed_robot_mode: Any = getattr(self.robot_wrapper, "current_mode", None)
        self._vla_transfer_grasp_completed: bool = False
        self._vla_switched_out_after_grasp: bool = False
        self._vla_reentered_after_grasp: bool = False
        self._transfer_completed: bool = False
        self._transfer_supervisor_thread: threading.Thread | None = None
        self._transfer_supervisor_stop_event = threading.Event()
        self._transfer_polling_thread: threading.Thread | None = None
        self._transfer_polling_stop_event = threading.Event()
        self._transfer_polling_active: bool = False
        self._transfer_polling_lock = threading.Lock()
        if hasattr(self.robot_wrapper, "register_mode_change_callback"):
            self.robot_wrapper.register_mode_change_callback(self._on_robot_mode_changed)

        # Valen (Jev) Multi-Modal Decision Evaluator support
        raw_ve = getattr(ctx.runtime.cfg, "valen_evaluator", False)
        if isinstance(raw_ve, str):
            self._valen_enabled = raw_ve.lower() in ("true", "1", "yes")
        else:
            self._valen_enabled = bool(raw_ve)
        self._valen_client = None
        if self._valen_enabled:
            try:
                from lerobot.rollout.valen_evaluator import ValenClient
                valen_ip = getattr(ctx.runtime.cfg, "valen_ip", "10.8.8.98")
                valen_port = getattr(ctx.runtime.cfg, "valen_port", 5559)
                valen_endpoint = getattr(ctx.runtime.cfg, "valen_endpoint", None)
                self._valen_client = ValenClient(
                    host=valen_ip,
                    port=valen_port,
                    endpoint=valen_endpoint,
                    timeout_ms=500,
                )
                logger.info("Valen (Jev) Evaluator client initialized (%s)", self._valen_client.endpoint)
                self._print(f"🧠 [Valen JEV] 决策模型已连接: {self._valen_client.endpoint}")
            except Exception as e:
                logger.error("Failed to initialize ValenClient: %s", e)
                self._print(f"⚠️  [Valen JEV] 初始化连接失败: {e}")

        # Whole-session continuous recorder (headless, aligns 3 cameras, joints, mode, JEV)
        raw_rec = getattr(ctx.runtime.cfg, "record", False) or getattr(ctx.runtime.cfg, "record_diagnostics", False)
        if isinstance(raw_rec, str):
            self._record_enabled = raw_rec.lower() in ("true", "1", "yes")
        else:
            self._record_enabled = bool(raw_rec)
        self._recorder = None
        if self._record_enabled:
            try:
                from lerobot.rollout.whole_session_recorder import WholeSessionRecorder
                rec_dir = getattr(ctx.runtime.cfg, "record_dir", None) or getattr(ctx.runtime.cfg, "diagnostics_dir", None)
                self._recorder = WholeSessionRecorder(
                    output_dir=rec_dir,
                    fps=float(getattr(ctx.runtime.cfg, "fps", 30.0)),
                    robot_wrapper=self.robot_wrapper,
                    interactive_session=self,
                )
            except Exception as e:
                logger.error("Failed to initialize WholeSessionRecorder: %s", e)
                self._print(f"⚠️  [Recorder] 初始化录制器失败: {e}")

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
            for i, task_str in enumerate(self._subtask_list):
                idx = i
                self._commands[f"phase{i+1}"] = (
                    lambda cmd, p=idx: self._cmd_jump_phase(p),
                    "",
                    f"jump to phase {i+1}: {task_str} (shortcut: /{i+1}, {i+1})",
                )
        if self._transfer_mode:
            self._commands["navdone"] = (
                self._cmd_navdone,
                "",
                "signal navigation to table B done → restore arm & VLA place (shortcut: /d, d, /navdone)",
            )

    @property
    def robot_wrapper(self):
        """Retrieve the robot wrapper instance reliably across sub-context structures."""
        hw = getattr(self.ctx, "hardware", None)
        return getattr(hw, "robot_wrapper", None) if hw is not None else getattr(self.ctx, "robot_wrapper", None)

    @property
    def is_simulation(self) -> bool:
        """Check whether the current session is running against a simulation environment."""
        if getattr(self._runtime.cfg, "sim", None) is True or getattr(self._runtime.cfg, "is_simulation", None) is True:
            return True
        robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
        if robot is not None:
            if getattr(robot, "is_simulation", None) is True:
                return True
            cfg = getattr(robot, "config", None)
            if cfg is not None:
                if getattr(cfg, "is_simulation", None) is True:
                    return True
                robot_ip = getattr(cfg, "robot_ip", None)
                if isinstance(robot_ip, str):
                    if robot_ip.lower() and robot_ip.lower() not in ("localhost", "127.0.0.1"):
                        return False
                    if robot_ip.lower() in ("localhost", "127.0.0.1", ""):
                        return True
        import sys
        if "--real" in sys.argv:
            return False
        if "--sim" in sys.argv:
            return True
        return False

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
                if self._recorder is not None:
                    self._recorder.start()
                self._listener.start()
                if self._auto_mode:
                    self._start_auto_supervisor()
                elif self._transfer_mode:
                    self._start_transfer_supervisor()
                try:
                    self.controller.serve()
                finally:
                    self._listener.stop()
        finally:
            if self._recorder is not None:
                self._recorder.stop()
            self._stop_auto_supervisor()
            self._stop_transfer_polling()
            self._stop_transfer_supervisor()
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
            if self._transfer_mode:
                if self._current_phase == 4:
                    self._print(
                        f"Rollout running — [Transfer Phase 4] VLA 已接管放置任务: \"{self.controller.task}\""
                    )
                    self._start_transfer_phase4_tracker()
                elif not getattr(self, "_transfer_banner_printed", False):
                    self._transfer_banner_printed = True
                    self._current_phase = 0
                    self.controller.set_task(self.controller.task or "pick up the water bottle from the table")
                    hint_str = (
                        "检测到夹紧并抬起后自动保持当前抓取姿态；导航到达桌B后输入 'd' 回车继续。"
                        if getattr(self, "_transfer_hold_after_grasp", False) else
                        "检测到夹紧并抬起后自动夹持回默认位；导航到达桌B后输入 'd' 回车继续。"
                    )
                    self._print(
                        f"Rollout running — [跨桌水瓶转运] Phase 0: 夹取水瓶 (Task: \"{self.controller.task}\").\n"
                        f"提示: {hint_str}"
                    )
                    self._start_subtask_tracker()
                else:
                    self._start_subtask_tracker()
            elif self._subtasks_enabled:
                self.controller.set_task(self._subtask_list[self._current_phase])
                total_phases = len(self._subtask_list)
                shortcuts_str = ", ".join(f"'{i+1}'" for i in range(total_phases))
                self._print(
                    f"Rollout running — [{total_phases}-Subtask 模式] 当前阶段 [{self._current_phase + 1}/{total_phases}]: "
                    f"\"{self._subtask_list[self._current_phase]}\".\n"
                    f"快捷指令: 'n' 跳下一阶段, {shortcuts_str} 选段, 'r' 复位, 'q' 退出。"
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
    # Transfer Mode Supervisor (6000 Port Mode Switch + Dual-Table JEV Perception)
    # ------------------------------------------------------------------

    def _start_transfer_supervisor(self) -> None:
        if self._transfer_supervisor_thread is not None and self._transfer_supervisor_thread.is_alive():
            return
        self._transfer_supervisor_stop_event.clear()
        self._transfer_supervisor_thread = threading.Thread(
            target=self._transfer_supervisor_loop, daemon=True, name="TransferModeSupervisor"
        )
        self._transfer_supervisor_thread.start()

    def _stop_transfer_supervisor(self) -> None:
        self._transfer_supervisor_stop_event.set()
        if self._transfer_supervisor_thread is not None:
            self._transfer_supervisor_thread.join(timeout=1.0)
            self._transfer_supervisor_thread = None

    def _transfer_supervisor_loop(self) -> None:
        """Monitors robot mode on port 6000 and drives level-triggered JEV perception in transfer mode."""
        robot = self.robot_wrapper
        from lerobot.robots.unitree_g1.unitree_g1_client import RobotMode

        while not self._transfer_supervisor_stop_event.is_set():
            if self.controller.stopped:
                break

            if getattr(self, "_transfer_completed", False):
                time.sleep(0.08)
                continue

            current_mode = getattr(robot, "current_mode", None)

            # Edge trigger: Mode change notification & polling stop
            if current_mode != self._last_observed_robot_mode:
                old_m = self._last_observed_robot_mode
                self._last_observed_robot_mode = current_mode

                if old_m is not None:
                    if current_mode != RobotMode.VLA:
                        if self._transfer_polling_active:
                            self._stop_transfer_polling()
                        if getattr(self, "_vla_transfer_grasp_completed", False):
                            self._vla_switched_out_after_grasp = True
                        self._print(
                            f"\n📡 [6000 端口] 机器人模式切换: {old_m} -> {current_mode} (已退出 VLA)\n"
                            "   已暂停视觉感知与自动接管，机械臂保持安全静止。"
                        )
                    elif current_mode == RobotMode.VLA:
                        self._print(
                            "\n📡 [6000 端口] 机器人模式切换: -> VLA 控制模式 (VLA)！"
                        )
                        # 如果已经在桌A夹起水瓶且脱离过桌A (切出过VLA)，此时切入VLA代表到达桌B【再次切入VLA】
                        if (
                            getattr(self, "_vla_transfer_grasp_completed", False)
                            and getattr(self, "_vla_switched_out_after_grasp", False)
                        ):
                            self._vla_reentered_after_grasp = True
                            self._print(
                                "👉 [Transfer] 检测到到达桌B并再次切入 VLA 模式，开始向 JEV 传入图像进行目标识别..."
                            )
                            if self._current_phase == 2 and not self._nav_done_event.is_set():
                                self._start_transfer_polling(table=2)

            # Level trigger: If in VLA mode, not resetting, not starting, not currently polling
            if (
                current_mode == RobotMode.VLA
                and not self._is_resetting
                and not self._is_starting
                and not self._transfer_polling_active
            ):
                metrics = self._get_robot_metrics()
                r_gripper = float(metrics.get("r_gripper", 5.0))

                # Table 2: Phase 2 (navigating wait) or holding bottle with gripper closed
                is_table_2 = (self._current_phase == 2) or (
                    self._captured_gripper_val is not None and r_gripper <= 3.8
                )

                if is_table_2:
                    # 关键修改：夹起水瓶后绝对不能仅仅因为当前处于 VLA 模式就自动传图给 JEV！
                    # 必须等待机器人退出 VLA 后【再次切入 VLA】(self._vla_reentered_after_grasp == True)，才允许启动桌2感知！
                    if (
                        self._current_phase == 2
                        and not self._nav_done_event.is_set()
                        and getattr(self, "_vla_reentered_after_grasp", False)
                    ):
                        self._start_transfer_polling(table=2)
                else:
                    if self._current_phase == 0 and not self.controller.running:
                        self._start_transfer_polling(table=1)

            time.sleep(0.08)

    def _on_robot_mode_changed(self, old_mode: Any, new_mode: Any, raw_preview: str) -> None:
        """Callback invoked immediately when UnitreeG1Client receives a mode change on port 6000."""
        from lerobot.robots.unitree_g1.unitree_g1_client import RobotMode
        if getattr(self, "_recorder", None) is not None:
            self._recorder.record_mode_transition(
                old_mode=getattr(old_mode, "value", str(old_mode)),
                new_mode=getattr(new_mode, "value", str(new_mode)),
                raw_preview=raw_preview,
            )
        if not self._transfer_mode or getattr(self, "_transfer_completed", False):
            return
        if new_mode != RobotMode.VLA:
            if self._transfer_polling_active:
                self._stop_transfer_polling()
            if getattr(self, "_vla_transfer_grasp_completed", False):
                self._vla_switched_out_after_grasp = True
        elif new_mode == RobotMode.VLA and not self._is_resetting and not self._is_starting:
            metrics = self._get_robot_metrics()
            r_gripper = float(metrics.get("r_gripper", 5.0))
            is_table_2 = (self._current_phase == 2) or (
                self._captured_gripper_val is not None and r_gripper <= 3.8
            )
            if is_table_2:
                if (
                    self._current_phase == 2
                    and not self._nav_done_event.is_set()
                    and getattr(self, "_vla_switched_out_after_grasp", False)
                ):
                    self._vla_reentered_after_grasp = True
                    self._start_transfer_polling(table=2)
            else:
                if self._current_phase == 0 and not self.controller.running:
                    self._start_transfer_polling(table=1)

    def _start_transfer_polling(self, table: int) -> None:
        """Start background JEV perception polling loop for Table 1 (bottle) or Table 2 (box)."""
        if getattr(self, "_transfer_completed", False):
            logger.info("Transfer: ignoring polling request because transfer task is completed, awaiting 'r' reset.")
            return
        if table == 2 and not getattr(self, "_vla_reentered_after_grasp", False):
            logger.info("Transfer: ignoring Table 2 polling request because robot has not re-entered VLA mode after grasp.")
            return
        if not hasattr(self, "_transfer_polling_lock"):
            self._transfer_polling_lock = threading.Lock()
        if not hasattr(self, "_transfer_polling_stop_event"):
            self._transfer_polling_stop_event = threading.Event()
        if not hasattr(self, "_transfer_polling_active"):
            self._transfer_polling_active = False

        with self._transfer_polling_lock:
            if self._transfer_polling_active:
                return
            self._transfer_polling_active = True
            self._transfer_polling_stop_event.clear()
            self._transfer_polling_thread = threading.Thread(
                target=self._transfer_polling_worker,
                args=(table,),
                daemon=True,
                name=f"TransferJevPollingTable{table}",
            )
            self._transfer_polling_thread.start()

    def _stop_transfer_polling(self) -> None:
        """Stop active JEV perception polling loop."""
        if hasattr(self, "_transfer_polling_stop_event"):
            self._transfer_polling_stop_event.set()
        if hasattr(self, "_transfer_polling_lock"):
            with self._transfer_polling_lock:
                self._transfer_polling_active = False
        else:
            self._transfer_polling_active = False
        if getattr(self, "_transfer_polling_thread", None) is not None and self._transfer_polling_thread.is_alive():
            if threading.current_thread() != self._transfer_polling_thread:
                self._transfer_polling_thread.join(timeout=0.5)
        self._transfer_polling_thread = None

    def _get_live_camera_frames(self) -> tuple[Any, Any]:
        """Fetch fresh real-time camera frames [global_view, right_wrist] from robot.

        Directly queries the latest live camera buffers with age verification, guaranteeing
        that JEV visual perception always evaluates the current real-time scene rather than
        a stale startup or disconnected cache.
        """
        robot_wrapper = self.robot_wrapper
        robot = getattr(robot_wrapper, "inner", robot_wrapper)
        g_frame = None
        rw_frame = None

        # Method 1: Direct peek from live camera instances (fastest, zero motor/IMU overhead)
        cams = getattr(robot, "_cameras", {})
        if cams:
            for g_key in ("global_view", "base_0_rgb", "camera_0", "cam_high"):
                if g_key in cams and getattr(cams[g_key], "is_connected", True):
                    try:
                        g_frame = cams[g_key].read_latest(max_age_ms=1500)
                        break
                    except Exception:
                        pass
            for rw_key in ("right_wrist", "right_wrist_0_rgb", "camera_right_wrist", "cam_right_wrist"):
                if rw_key in cams and getattr(cams[rw_key], "is_connected", True):
                    try:
                        rw_frame = cams[rw_key].read_latest(max_age_ms=1500)
                        break
                    except Exception:
                        pass

        # Method 2: If direct camera peek didn't provide both, check robot_wrapper.get_observation()
        # Strictly verify freshness with timestamp check (<= 1.5s old)
        if g_frame is None or rw_frame is None:
            now = time.time()
            cam_ts = getattr(robot, "_last_camera_timestamps", {})
            try:
                obs = robot_wrapper.get_observation()
                if g_frame is None:
                    cand = (
                        obs.get("global_view")
                        or obs.get("base_0_rgb")
                        or obs.get("observation.images.global_view")
                        or obs.get("observation.images.base_0_rgb")
                    )
                    ts = max(cam_ts.get("global_view", 0.0), cam_ts.get("base_0_rgb", 0.0))
                    if cand is not None and (now - ts <= 1.5 or not cam_ts):
                        g_frame = cand
                if rw_frame is None:
                    cand = (
                        obs.get("right_wrist")
                        or obs.get("right_wrist_0_rgb")
                        or obs.get("observation.images.right_wrist")
                        or obs.get("observation.images.right_wrist_0_rgb")
                    )
                    ts = max(cam_ts.get("right_wrist", 0.0), cam_ts.get("right_wrist_0_rgb", 0.0))
                    if cand is not None and (now - ts <= 1.5 or not cam_ts):
                        rw_frame = cand
            except Exception as e:
                logger.debug("Failed to get observation for live cameras: %s", e)

        # Do NOT fall back to stale historical frames! If camera is offline, return None so JEV safely pauses.
        return g_frame, rw_frame

    def _transfer_polling_worker(self, table: int) -> None:
        """Worker thread for level-triggered JEV perception polling in transfer mode."""
        from lerobot.robots.unitree_g1.unitree_g1_client import RobotMode

        rule_65 = "=" * 65
        robot_wrapper = self.robot_wrapper
        robot = getattr(robot_wrapper, "inner", robot_wrapper)
        valen_client = getattr(self, "_valen_client", None)

        last_heartbeat_time = 0.0
        first_report = True
        box_confirm_count = 0

        logger.info("Transfer JEV perception polling started for Table %d", table)

        try:
            while not self._transfer_polling_stop_event.is_set():
                if self.controller.stopped or getattr(self, "_is_resetting", False):
                    break

                # 1. Mode check: must stay in VLA mode
                current_mode = getattr(robot_wrapper, "current_mode", getattr(robot, "current_mode", None))
                if current_mode != RobotMode.VLA:
                    logger.info("Transfer polling exiting: robot is no longer in VLA mode (%s)", current_mode)
                    break

                # 2. State validation
                if table == 1:
                    if self.controller.running or getattr(self, "_is_starting", False) or self._current_phase != 0:
                        break
                elif table == 2:
                    if self._current_phase != 2 or self._nav_done_event.is_set():
                        break

                now = time.time()

                # 3. Retrieve fresh real-time camera frames (always live)
                g_frame, rw_frame = self._get_live_camera_frames()

                if valen_client is None:
                    if first_report:
                        self._print(
                            f"\n{rule_65}\n"
                            f"⚠️  [桌{table} JEV 未连接] Valen 服务不可用，保持当前姿态待命！\n"
                            f"   👉 若需强制启动，请在终端手动输入 {'s' if table == 1 else 'd'} 回车。\n"
                            f"{rule_65}"
                        )
                        first_report = False
                    self._transfer_polling_stop_event.wait(1.0)
                    continue

                if g_frame is None or rw_frame is None:
                    if first_report:
                        self._print(
                            f"\n{rule_65}\n"
                            f"⚠️  [桌{table} 相机离线] 未获取到相机画面，保持当前姿态待命！\n"
                            f"   👉 若需强制启动，请在终端手动输入 {'s' if table == 1 else 'd'} 回车。\n"
                            f"{rule_65}"
                        )
                        first_report = False
                    self._transfer_polling_stop_event.wait(1.0)
                    continue

                if not first_report:
                    self._print(f"✅ [桌{table} 视觉已恢复] 成功获取到实时相机画面，继续执行 JEV 目标研判...")
                    first_report = True

                # 4. Perform JEV evaluation
                if table == 1:
                    ctx_t1 = (
                        "The input image is horizontally stitched from two camera views: "
                        "the left half is the third-person global overview showing Table 1; "
                        "the right half is the right-wrist camera close-up showing the gripper in front of Table 1. "
                        "The robot is standing in front of Table 1."
                    )
                    q_t1 = {
                        "task_phase": {
                            "type": "choice",
                            "instructions": "Look at the tabletop in front of the robot. What object is present on the table?",
                            "criteria": {
                                "water_bottle_present": "There is a water bottle standing or placed on the table surface.",
                                "no_water_bottle": "There is no water bottle on the table; only the bare table surface is present.",
                            },
                        }
                    }
                    v_res = valen_client.evaluate(
                        g_frame, rw_frame, phase=4, custom_questions=q_t1, custom_text=ctx_t1
                    )
                    if getattr(self, "_recorder", None) is not None:
                        self._recorder.record_jev_decision(
                            table=1,
                            phase=4,
                            description="Table 1 water bottle presence polling",
                            result=v_res,
                            questions=q_t1,
                            context_text=ctx_t1,
                        )
                    is_detected = False
                    prob = 0.0
                    if v_res.is_success:
                        prob = v_res.probabilities.get(v_res.choice, 0.0)
                        if v_res.choice in ("water_bottle_present", "bottle_present") and prob >= 0.50:
                            is_detected = True

                    if is_detected:
                        self._print(
                            f"\n{rule_65}\n"
                            f"🤖 [桌1 JEV 判定通过] 视觉确认桌面上存在水瓶！\n"
                            f"   ├─ 识别选项: {v_res.choice} (置信度 P={prob*100:.1f}%, 耗时: {v_res.cost_ms:.0f}ms)\n"
                            f"   └─ 自动切入 VLA 控制模式 (启动 Phase 0 水瓶夹取，和按 's' 效果一致)...\n"
                            f"{rule_65}"
                        )
                        if hasattr(robot_wrapper, "trigger_engagement_smoothing"):
                            robot_wrapper.trigger_engagement_smoothing()
                        self._is_starting = True
                        if not self.controller.start():
                            self._is_starting = False
                        break
                    else:
                        if first_report:
                            self._print(
                                f"\n{rule_65}\n"
                                f"⏳ [桌1 视觉感知中] 手柄处于 VLA 模式，桌面上暂未检测到水瓶。\n"
                                f"   ├─ 识别选项: {v_res.choice} (置信度 P={prob*100:.1f}%, 耗时: {v_res.cost_ms:.0f}ms)\n"
                                f"   └─ 机械臂保持静止待命，持续检测中... 请将水瓶摆放到桌面视野内。\n"
                                f"{rule_65}"
                            )
                            first_report = False
                            last_heartbeat_time = now
                        elif now - last_heartbeat_time >= 3.0:
                            last_heartbeat_time = now
                            self._print(
                                f"⏳ [桌1 持续感知中] 暂未检测到水瓶 ({v_res.choice}, P={prob*100:.1f}%)，保持静止等待摆放水瓶..."
                            )

                elif table == 2:
                    ctx_t2 = (
                        "The input image is horizontally stitched from two camera views: "
                        "the left half is the third-person global overview showing Table B; "
                        "the right half is the right-wrist camera close-up showing the robot gripper holding the water bottle. "
                        "The robot is standing in front of Table B to place the water bottle into a container."
                    )
                    q_t2 = {
                        "task_phase": {
                            "type": "choice",
                            "instructions": "Look at Table B in the camera views. What is present on the tabletop?",
                            "criteria": {
                                "place_into_box": "A box, storage container, cardboard box, or bin is present on Table B.",
                                "place_on_table": "Table B is flat and empty, with no box or container present.",
                            },
                        }
                    }
                    v_res = valen_client.evaluate(
                        g_frame, rw_frame, phase=2, custom_questions=q_t2, custom_text=ctx_t2
                    )
                    if getattr(self, "_recorder", None) is not None:
                        self._recorder.record_jev_decision(
                            table=2,
                            phase=2,
                            description="Table 2 placement target polling (box vs table)",
                            result=v_res,
                            questions=q_t2,
                            context_text=ctx_t2,
                        )
                    is_detected = False
                    p_box = 0.0
                    p_table = 0.0
                    if v_res.is_success:
                        p_box = v_res.probabilities.get("place_into_box", 0.0)
                        p_table = v_res.probabilities.get("place_on_table", 0.0)
                        # 倾向判定有盒子 (适应不同外观/材质/颜色的各类盒子):
                        # 1. 预测为 place_into_box / box_present (即使置信度不高)，或 P_box >= 0.35
                        # 2. 或当空桌面置信度不显著 (P_table < 0.70)，模型对不同盒子均倾向放行
                        is_box_inclined = (
                            (v_res.choice in ("place_into_box", "box_present"))
                            or (p_box >= 0.35)
                            or (p_table < 0.70)
                        )
                        if p_box >= 0.50:
                            # 明显倾向有盒，直接 1 帧快速放行
                            is_detected = True
                        elif is_box_inclined:
                            # 弱倾向或不同盒子边缘特征，连续 2 帧防抖确认放行
                            box_confirm_count += 1
                            if box_confirm_count >= 2:
                                is_detected = True
                        else:
                            box_confirm_count = 0

                    if is_detected:
                        self._print(
                            f"\n{rule_65}\n"
                            f"🤖 [桌2 JEV 判定通过] 视觉确认桌面上存在目标收纳盒 (倾向判定通过)！\n"
                            f"   ├─ 识别判断: {v_res.choice} (P_box={p_box*100:.1f}%, P_table={p_table*100:.1f}%, 耗时: {v_res.cost_ms:.0f}ms)\n"
                            f"   └─ 自动解除导航等待，开始进入 VLA 放置操作 (和按 'd' 效果一致)...\n"
                            f"{rule_65}"
                        )
                        self._cmd_navdone(InteractiveCommand(name="navdone"))
                        break
                    else:
                        # 仅当高度确认为空桌面时才显示空桌，其余显示识别与倾向感知状态
                        if p_table >= 0.70:
                            state_desc = f"桌面上未放置收纳盒 (空桌面 place_on_table, P={p_table*100:.1f}%)"
                        else:
                            state_desc = f"收纳盒倾向感知中 (P_box={p_box*100:.1f}%, P_table={p_table*100:.1f}%)"

                        if first_report:
                            self._print(
                                f"\n{rule_65}\n"
                                f"⏳ [桌2 视觉感知中] 手柄处于 VLA 模式，当前视觉感知状态:\n"
                                f"   ├─ 识别判断: {state_desc} (耗时: {v_res.cost_ms:.0f}ms)\n"
                                f"   └─ 机械臂持续锁紧夹爪保持姿态，静止等待盒子进入视野 (若需强制放置可按 'd')...\n"
                                f"{rule_65}"
                            )
                            first_report = False
                            last_heartbeat_time = now
                        elif now - last_heartbeat_time >= 3.0:
                            last_heartbeat_time = now
                            self._print(
                                f"⏳ [桌2 持续感知中] {state_desc}，机械臂保持静止待命中..."
                            )

                # Wait before next evaluation (interruptible)
                self._transfer_polling_stop_event.wait(0.6)

        except Exception as exc:
            logger.error("Error in transfer polling worker (table %d): %s", table, exc, exc_info=True)
        finally:
            with self._transfer_polling_lock:
                self._transfer_polling_active = False
            logger.info("Transfer JEV perception polling ended for Table %d", table)

    def _handle_vla_mode_switch_transfer(self) -> None:
        """Manual / external trigger for VLA transfer mode perception."""
        metrics = self._get_robot_metrics()
        r_gripper = float(metrics.get("r_gripper", 5.0))
        is_table_2 = (self._current_phase == 2) or (
            self._captured_gripper_val is not None and r_gripper <= 3.8
        )
        if is_table_2:
            if getattr(self, "_vla_reentered_after_grasp", False):
                self._start_transfer_polling(table=2)
        else:
            self._start_transfer_polling(table=1)

    # ------------------------------------------------------------------
    # 3-Subtask Kinematics & Automated Progression
    # ------------------------------------------------------------------

    def _get_robot_metrics(self) -> dict:
        """Extract arm shoulder pitch, IMU yaw, and right gripper angle from robot state."""
        r_gripper = getattr(self.robot_wrapper, "right_gripper_position", 5.0)

        def _extract_q(d: dict, *candidate_keys: str) -> float | None:
            for k in candidate_keys:
                if k in d:
                    entry = d[k]
                    if isinstance(entry, dict):
                        return float(entry.get("q", 0.0))
                    try:
                        return float(entry)
                    except (ValueError, TypeError):
                        pass
            return None

        try:
            robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
            state = None
            if robot is not None and hasattr(robot, "_latest_state") and hasattr(robot, "_state_lock"):
                with robot._state_lock:
                    state = robot._latest_state

            motors = state.get("motors", {}) if isinstance(state, dict) else {}

            # Read observation from robot wrapper as high-confidence fallback
            obs = {}
            if hasattr(self.robot_wrapper, "get_observation"):
                try:
                    obs = self.robot_wrapper.get_observation()
                except Exception:
                    pass

            r_pitch = _extract_q(motors, "kRightShoulderPitch", "right_shoulder_pitch", "right_shoulder_pitch_joint", "kRightShoulderPitch.q")
            if r_pitch is None:
                r_pitch = _extract_q(obs, "kRightShoulderPitch", "kRightShoulderPitch.q", "right_shoulder_pitch") or 0.0

            l_pitch = _extract_q(motors, "kLeftShoulderPitch", "left_shoulder_pitch", "left_shoulder_pitch_joint", "kLeftShoulderPitch.q")
            if l_pitch is None:
                l_pitch = _extract_q(obs, "kLeftShoulderPitch", "kLeftShoulderPitch.q", "left_shoulder_pitch") or 0.0

            r_roll = _extract_q(motors, "kRightShoulderRoll", "right_shoulder_roll", "right_shoulder_roll_joint", "kRightShoulderRoll.q")
            if r_roll is None:
                r_roll = _extract_q(obs, "kRightShoulderRoll", "kRightShoulderRoll.q", "right_shoulder_roll") or 0.0

            l_roll = _extract_q(motors, "kLeftShoulderRoll", "left_shoulder_roll", "left_shoulder_roll_joint", "kLeftShoulderRoll.q")
            if l_roll is None:
                l_roll = _extract_q(obs, "kLeftShoulderRoll", "kLeftShoulderRoll.q", "left_shoulder_roll") or 0.0

            # If gripper was not found from robot_wrapper property, check obs
            if r_gripper >= 4.95 and obs:
                obs_grip = _extract_q(obs, "kRightGripper", "right_gripper", "gripper.right")
                if obs_grip is not None and obs_grip < 4.95:
                    r_gripper = obs_grip

            imu = state.get("imu", {}) if isinstance(state, dict) else {}
            rpy = imu.get("rpy", [0.0, 0.0, 0.0])
            yaw = float(rpy[2]) if len(rpy) >= 3 else 0.0

            return {
                "connected": bool(state is not None or obs),
                "l_pitch": float(l_pitch),
                "r_pitch": float(r_pitch),
                "l_roll": float(l_roll),
                "r_roll": float(r_roll),
                "yaw": float(yaw),
                "r_gripper": float(r_gripper),
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
            "r_gripper": float(r_gripper),
        }

    def _start_subtask_tracker(self) -> None:
        if self._transfer_mode:
            pass
        elif not self._subtasks_enabled or len(self._subtask_list) not in (2, 3):
            return
        if self._subtask_tracker_thread is not None and self._subtask_tracker_thread.is_alive():
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
            self._subtask_tracker_thread.join(timeout=1.5)
        self._subtask_tracker_thread = None

    def _start_transfer_phase4_tracker(self) -> None:
        """Start the Phase 4 completion monitor thread to track bottle placement."""
        self._stop_subtask_tracker()
        self._subtask_stop_event.clear()
        self._subtask_tracker_running = True
        self._phase_start_time = time.time()
        self._subtask_tracker_thread = threading.Thread(
            target=self._subtask_tracker_loop_transfer_phase4,
            name="TransferPhase4TrackerThread",
            daemon=True,
        )
        self._subtask_tracker_thread.start()

    def _reset_transfer_state(self) -> None:
        """Reset transfer mode state flags so the session can cleanly start a new trial."""
        with self._transfer_lock:
            self._current_phase = 0
            self._transfer_triggered = False
            self._transfer_banner_printed = False
            self._phase2_banner_printed = False
            self._phase4_banner_printed = False
            self._frozen_joint_state = None
            self._captured_gripper_val = None
            self._decided_placement_task = None
            self._last_observed_robot_mode = None
            self._vla_transfer_grasp_completed = False
            self._vla_switched_out_after_grasp = False
            self._vla_reentered_after_grasp = False
            self._transfer_completed = False
            self._nav_done_event.clear()
            initial_task = getattr(self.controller, "_initial_task", None) or "pick up the water bottle from the table"
            self.controller.set_task(initial_task)

    def _on_transfer_round_completed(self) -> str | None:
        """Executed on serve thread once robot has smoothly returned to default stand pose."""
        self._transfer_completed = True
        self._stop_transfer_polling()
        rule = "-" * 65
        self._print(
            f"\n{rule}\n"
            "📍 [Transfer] 机器人已平稳停靠在身侧默认待命位置。\n"
            "   🔒 全套转运任务已圆满完成！视觉感知与自动控制已彻底休眠，机械臂保持静止不动。\n"
            "   👉 请在终端输入 'r' 回车 重置状态以准备下一轮实验；按 'q' 退出。\n"
            f"{rule}\n"
        )
        return None

    def _advance_phase(
        self,
        new_phase: int,
        reason: str = "",
        with_homing: bool = False,
        duration_s: float | None = None,
        post_homing_fn: Callable[[], str | None] | None = None,
    ) -> None:
        if not self._subtasks_enabled or new_phase < 0 or new_phase >= len(self._subtask_list):
            return
        self._current_phase = new_phase
        self._phase_start_time = time.time()
        self._lift_sustained_seconds = 0.0
        new_task = self._subtask_list[new_phase]

        if new_phase == 1 and len(self._subtask_list) == 3:
            metrics = self._get_robot_metrics()
            self._phase1_start_yaw = metrics["yaw"]

        msg = (
            f"\n" + "=" * 65 + "\n"
            f" [Subtasks 流转] 切换至 Subtask [{new_phase + 1}/{len(self._subtask_list)}]: \"{new_task}\"\n"
        )
        if reason:
            msg += f"   原因: {reason}\n"
        if with_homing:
            h_dur = duration_s if duration_s is not None else getattr(self._runtime.cfg, "subtask_homing_duration", 2.5)
            msg += (
                f"   优先平滑回位: 启动余弦 S 曲线归位插值 ({h_dur:.1f}s) 回到初始位置...\n"
                f"   自动接管模式: 归位到达后自动由 VLA 接管开始执行 Phase {new_phase + 1} 推理！\n"
            )
        else:
            msg += f"   模式: 直接动态切换 Prompt (保持机械臂当前姿态与环境状态，无缝流转)\n"
        msg += "=" * 65
        self._print(msg)

        if with_homing and hasattr(self.controller, "transition_with_homing"):
            h_dur = duration_s if duration_s is not None else getattr(self._runtime.cfg, "subtask_homing_duration", 2.0)
            if not isinstance(h_dur, (int, float)):
                h_dur = 2.0
            retract_first = getattr(self._runtime.cfg, "subtask_retract_first", True)
            if not isinstance(retract_first, bool):
                retract_first = True
            retract_dur = getattr(self._runtime.cfg, "subtask_retract_duration", 1.2)
            if not isinstance(retract_dur, (int, float)):
                retract_dur = 1.2
            self.controller.transition_with_homing(
                new_task,
                duration_s=h_dur,
                retract_first=retract_first,
                retract_duration_s=retract_dur,
                post_homing_fn=post_homing_fn,
            )
        else:
            self.controller.set_task(new_task)

    def _subtask_tracker_loop(self) -> None:
        """Background thread monitoring kinematics for subtask transitions."""
        if self._transfer_mode:
            self._subtask_tracker_loop_transfer()
        elif len(self._subtask_list) == 2:
            self._subtask_tracker_loop_2stage()
        else:
            self._subtask_tracker_loop_3stage()

    def _subtask_tracker_loop_2stage(self) -> None:
        """Dex-1 2-Subtask tracker: table to box, prioritized homing on release, then VLA takeover to table."""
        logger.info("Dex-1 2-Subtask tracker loop started.")
        self._phase_start_time = time.time()
        last_hud_time = 0.0
        has_grasped = False
        grasp_count = 0
        release_count = 0
        is_sim = self.is_simulation
        raw_auto_home = getattr(self._runtime.cfg, "subtask_auto_home", True)
        if isinstance(raw_auto_home, str):
            cfg_auto_home = raw_auto_home.lower() in ("true", "1", "yes")
        elif isinstance(raw_auto_home, bool):
            cfg_auto_home = raw_auto_home
        else:
            cfg_auto_home = True
        auto_home = cfg_auto_home
        homing_dur = getattr(self._runtime.cfg, "subtask_homing_duration", 2.5)
        if not isinstance(homing_dur, (int, float)):
            homing_dur = 2.5
        retract_first = getattr(self._runtime.cfg, "subtask_retract_first", True)
        if not isinstance(retract_first, bool):
            retract_first = True
        retract_dur = getattr(self._runtime.cfg, "subtask_retract_duration", 1.2)
        if not isinstance(retract_dur, (int, float)):
            retract_dur = 1.2
        raw_auto_advance = getattr(self._runtime.cfg, "subtask_auto_advance", False)
        if isinstance(raw_auto_advance, str):
            auto_advance = raw_auto_advance.lower() in ("true", "1", "yes")
        elif isinstance(raw_auto_advance, bool):
            auto_advance = raw_auto_advance
        else:
            auto_advance = False

        # Valen (Jev) decision evaluator tracking
        valen_client = getattr(self, "_valen_client", None)
        valen_last_time = 0.0
        valen_eval_interval = getattr(self._runtime.cfg, "valen_eval_interval_s", 0.4)
        if not isinstance(valen_eval_interval, (int, float)):
            valen_eval_interval = 0.4
        raw_vaa = getattr(self._runtime.cfg, "valen_auto_advance", True)
        if isinstance(raw_vaa, str):
            valen_auto_advance = raw_vaa.lower() in ("true", "1", "yes")
        elif isinstance(raw_vaa, bool):
            valen_auto_advance = raw_vaa
        else:
            valen_auto_advance = True
        valen_confirm_count = 0
        latest_valen_hud = ""

        def _evaluate_next_phase_at_home() -> str:
            """Executed at default position: inspect scene and decide whether next is Phase 1 or Phase 2."""
            time.sleep(0.4)
            g_frame, rw_frame = self._get_live_camera_frames()

            # 1. If Valen evaluator is active, evaluate scene from default position
            if valen_client is not None and g_frame is not None and rw_frame is not None:
                v_res = valen_client.evaluate(g_frame, rw_frame, phase=0)
                if getattr(self, "_recorder", None) is not None:
                    self._recorder.record_jev_decision(
                        table=1,
                        phase=0,
                        description="Single-table post-homing phase evaluation",
                        result=v_res,
                    )
                if v_res.is_success:
                    choice = v_res.choice
                    prob = v_res.probabilities.get(choice, 0.0)
                    box_prob = v_res.probabilities.get("bottle_in_box_completed", 0.0)
                    if valen_auto_advance and (choice == "bottle_in_box_completed" or box_prob >= 0.50):
                        self._current_phase = 1
                        next_task = self._subtask_list[1]
                        self._print(
                            f"\n" + "=" * 65 + "\n"
                            f" [默认位置状态判别] Valen Jev 判定水瓶已在蓝盒中！\n"
                            f"   判定结果: {choice} (置信度 P={box_prob:.2f}, {v_res.cost_ms:.0f}ms)\n"
                            f"   决策下发: 进入 Subtask [2/2]: \"{next_task}\"\n"
                            f"   接管模式: VLA 自动接管开始执行从盒中取出水瓶 (无缝流转，全程无 Reset)\n"
                            f"=" * 65 + "\n"
                        )
                        self._phase_start_time = time.time()
                        return next_task
                    else:
                        self._current_phase = 0
                        next_task = self._subtask_list[0]
                        self._print(
                            f"\n" + "=" * 65 + "\n"
                            f" [默认位置状态判别] Valen Jev 判定水瓶未入盒 / 仍在桌上 / 抓取失败！\n"
                            f"   判定结果: {choice} (P={prob:.2f}, {v_res.cost_ms:.0f}ms)\n"
                            f"   决策下发: 重新执行 Subtask [1/2]: \"{next_task}\"\n"
                            f"   接管模式: VLA 自动接管重新夹取水瓶放入蓝盒 (无缝重试，全程无 Reset)\n"
                            f"=" * 65 + "\n"
                        )
                        self._phase_start_time = time.time()
                        return next_task

            # 2. If Valen is not active or evaluation failed:
            if auto_advance:
                self._current_phase = 1
                next_task = self._subtask_list[1]
                self._print(
                    f"\n" + "=" * 65 + "\n"
                    f" [默认位置状态确认] 机器人已到达默认位置，流转至 Subtask [2/2]: \"{next_task}\"\n"
                    f"   接管模式: VLA 自动接管开始执行 Phase 2 推理 (全程无 Reset)\n"
                    f"=" * 65 + "\n"
                )
            else:
                self._current_phase = 0
                next_task = self._subtask_list[0]
                self._print(
                    f"\n" + "=" * 65 + "\n"
                    f" [默认位置状态保持] 机器人已到达默认位置，保持 Subtask [1/2]: \"{next_task}\"\n"
                    f"   手动切换提示: 准备好执行下一阶段时，随时按 'n' 或 '2' 手动切换至 Subtask 2\n"
                    f"=" * 65 + "\n"
                )
            self._phase_start_time = time.time()
            return next_task

        while self._subtask_tracker_running and not self._subtask_stop_event.is_set():
            time.sleep(0.05)
            # Skip evaluation while inference is paused or transitioning
            if not getattr(self.controller, "running", False):
                continue

            now = time.time()
            elapsed = now - self._phase_start_time
            metrics = self._get_robot_metrics()

            if not metrics.get("connected", False):
                continue

            r_gripper = float(metrics.get("r_gripper", 5.0))

            # Query Valen evaluator if enabled
            if valen_client is not None and now - valen_last_time >= valen_eval_interval:
                valen_last_time = now
                g_frame, rw_frame = self._get_live_camera_frames()

                if g_frame is not None and rw_frame is not None:
                    v_res = valen_client.evaluate(g_frame, rw_frame, phase=self._current_phase)
                    if getattr(self, "_recorder", None) is not None:
                        self._recorder.record_jev_decision(
                            table=1,
                            phase=self._current_phase,
                            description=f"Single-table loop completion check (phase={self._current_phase})",
                            result=v_res,
                        )
                    if v_res.is_success:
                        prob_val = v_res.probabilities.get(v_res.choice, 0.0)
                        latest_valen_hud = f" | Jev: {v_res.choice} (P={prob_val:.2f}, {v_res.cost_ms:.0f}ms)"

                        # Double verification with Valen + Gripper
                        if self._current_phase == 0:
                            is_target = (
                                (v_res.choice == "bottle_in_box_completed" and prob_val >= 0.55)
                                or v_res.probabilities.get("bottle_in_box_completed", 0.0) >= 0.65
                            )
                            # Only confirm completion if the bottle was previously grasped and is now released in box
                            if is_target and has_grasped and r_gripper >= 4.7:
                                valen_confirm_count += 1
                                if valen_confirm_count >= 2:
                                    reason = (
                                        f"Valen Jev+夹爪双校验确认瓶已入盒 "
                                        f"({v_res.choice}, P={v_res.probabilities.get('bottle_in_box_completed', 0.0):.2f}) 且已松爪"
                                    )
                                    valen_confirm_count = 0
                                    has_grasped = False
                                    grasp_count = 0
                                    release_count = 0
                                    self._phase_start_time = time.time()
                                    if auto_home:
                                        msg = (
                                            f"\n" + "=" * 65 + "\n"
                                            f" [Subtask 1/2 动作完成] {reason}\n"
                                            f"   平滑回位中: 先平收右臂 ({retract_dur:.1f}s) 避开盒子，再余弦 S 曲线归位 ({homing_dur:.1f}s) 回到默认位置...\n"
                                            f"   归位后规划: 到达默认位置后重新观察视野，判断接下来进入 Phase 1 还是 Phase 2 (全程无 Reset)\n"
                                            f"=" * 65
                                        )
                                        self._print(msg)
                                        if hasattr(self.controller, "transition_with_homing"):
                                            self.controller.transition_with_homing(
                                                None,
                                                duration_s=homing_dur,
                                                retract_first=retract_first,
                                                retract_duration_s=retract_dur,
                                                resume=True,
                                                post_homing_fn=_evaluate_next_phase_at_home,
                                            )
                                    else:
                                        if valen_auto_advance or auto_advance:
                                            self._advance_phase(1, reason=reason, with_homing=False)
                            else:
                                valen_confirm_count = max(0, valen_confirm_count - 1)

                        elif self._current_phase == 1:
                            is_target = (
                                (v_res.choice == "bottle_on_table_completed" and prob_val >= 0.55)
                                or v_res.probabilities.get("bottle_on_table_completed", 0.0) >= 0.65
                            )
                            if is_target and has_grasped and r_gripper >= 4.7:
                                valen_confirm_count += 1
                                if valen_confirm_count >= 2:
                                    self._print("\n" + "=" * 65)
                                    self._print("🎉 [Subtasks] 2 阶段抓放水瓶子任务已全部执行完毕！")
                                    self._print(
                                        f"   Valen Jev+夹爪双校验确认瓶已放回桌面 ({v_res.choice})，平滑返回默认位置待命...\n"
                                        f"   (保持当前大模型上下文与仿真场景，未自动Reset；如需重置模型上下文可手动按 'r' 或 /reset)"
                                    )
                                    self._print("=" * 65 + "\n")
                                    self._current_phase = 2
                                    valen_confirm_count = 0
                                    has_grasped = False
                                    grasp_count = 0
                                    release_count = 0
                                    if auto_home and hasattr(self.controller, "transition_with_homing"):
                                        self.controller.transition_with_homing(
                                            None,
                                            duration_s=homing_dur,
                                            retract_first=retract_first,
                                            retract_duration_s=retract_dur,
                                            resume=False,
                                        )
                            else:
                                valen_confirm_count = max(0, valen_confirm_count - 1)

            # Status HUD every 1.5s
            if now - last_hud_time >= 1.5:
                last_hud_time = now
                status_str = "已夹持水瓶" if has_grasped else "未抓取/张开"
                status_str += latest_valen_hud
                if self._current_phase == 0:
                    self._print(
                        f"📊 [Subtask 1/2: 抓水瓶放盒] 耗时: {elapsed:4.1f}s | "
                        f"右夹爪: {r_gripper:4.2f} rad ({status_str}) | 右肩俯仰: {metrics['r_pitch']:+.2f}"
                    )
                elif self._current_phase == 1:
                    self._print(
                        f"📊 [Subtask 2/2: 盒中取水瓶放桌] 耗时: {elapsed:4.1f}s | "
                        f"右夹爪: {r_gripper:4.2f} rad ({status_str}) | 右肩俯仰: {metrics['r_pitch']:+.2f}"
                    )

            # Phase 0: "pick up the water bottle from the table and place it into the blue box"
            if self._current_phase == 0:
                if r_gripper <= 3.9:
                    grasp_count += 1
                    if grasp_count >= 3:
                        if not has_grasped:
                            has_grasped = True
                            self._print(f"\n✊ [Subtask 1/2] 检测到水瓶已被稳固夹起 (右夹爪: {r_gripper:.2f} rad <= 3.9)")
                else:
                    grasp_count = max(0, grasp_count - 1)

                if has_grasped and r_gripper >= 4.7:
                    release_count += 1
                    if release_count >= 3:
                        reason = f"检测到水瓶在盒中松开释放 (右夹爪: {r_gripper:.2f} rad >= 4.7)"
                        has_grasped = False
                        grasp_count = 0
                        release_count = 0
                        self._phase_start_time = time.time()
                        if auto_home:
                            msg = (
                                f"\n" + "=" * 65 + "\n"
                                f" [Subtask 1/2 动作完成] {reason}\n"
                                f"   平滑回位中: 先平收右臂 ({retract_dur:.1f}s) 避开盒子，再余弦 S 曲线归位 ({homing_dur:.1f}s) 回到默认位置...\n"
                                f"   归位后规划: 到达默认位置后重新观察视野，判断接下来进入 Phase 1 还是 Phase 2 (全程无 Reset)\n"
                                f"=" * 65
                            )
                            self._print(msg)
                            if hasattr(self.controller, "transition_with_homing"):
                                self.controller.transition_with_homing(
                                    None,
                                    duration_s=homing_dur,
                                    retract_first=retract_first,
                                    retract_duration_s=retract_dur,
                                    resume=True,
                                    post_homing_fn=_evaluate_next_phase_at_home,
                                )
                        else:
                            if auto_advance:
                                self._advance_phase(1, reason=reason, with_homing=False)

            # Phase 1: "take the water bottle out of the blue box and place it back on the table"
            elif self._current_phase == 1:
                if r_gripper <= 3.9:
                    grasp_count += 1
                    if grasp_count >= 3:
                        if not has_grasped:
                            has_grasped = True
                            self._print(f"\n✊ [Subtask 2/2] 检测到水瓶已从盒中夹起 (右夹爪: {r_gripper:.2f} rad <= 3.9)")
                else:
                    grasp_count = max(0, grasp_count - 1)

                if has_grasped and r_gripper >= 4.7:
                    release_count += 1
                    if release_count >= 3:
                        self._print("\n" + "=" * 65)
                        self._print("🎉 [Subtasks] 2 阶段抓放水瓶子任务已全部执行完毕！")
                        self._print(
                            "   检测到水瓶已放回桌面并松开，平滑返回默认位置待命...\n"
                            "   (保持当前大模型上下文与环境状态，未自动Reset；如需重置模型上下文可手动按 'r' 或 /reset)"
                        )
                        self._print("=" * 65 + "\n")
                        self._current_phase = 2
                        has_grasped = False
                        grasp_count = 0
                        release_count = 0
                        if auto_home and hasattr(self.controller, "transition_with_homing"):
                            self.controller.transition_with_homing(
                                None,
                                duration_s=homing_dur,
                                retract_first=retract_first,
                                retract_duration_s=retract_dur,
                                resume=False,
                            )

    def _subtask_tracker_loop_3stage(self) -> None:
        """Background thread monitoring kinematics for 3-stage subtask transitions."""
        logger.info("Subtask tracker thread started (3-stage).")
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
    # Transfer mode: cross-table water bottle transport
    # Phase flow:
    #   0 → VLA picking from table A  (JEV detects grasped+lifted → freeze snapshot)
    #   1 → Arm homing while gripping  (transition_with_homing resume=False)
    #   2 → Waiting for /navdone       (navigation walk, upper body fixed at home)
    #   3 → Arm restore to frozen pose (interpolate home→frozen, gripper stays clamped)
    #   4 → VLA placing at table B     (JEV decides box vs table, VLA takes over)
    # ------------------------------------------------------------------

    def _subtask_tracker_loop_transfer(self) -> None:
        """Transfer-mode tracker: detect grasped+lifted → freeze → home → wait nav → restore → VLA place.

        The gripper stays clamped (closed) during homing, navigation, and pose-restore phases.
        The arm joint snapshot is captured the moment JEV confirms the bottle is grasped and
        slightly lifted.  After navigation (signalled via /navdone), the arm is interpolated
        back from the home position to that frozen pose before VLA resumes, preventing a jump.
        """
        import copy
        import math

        logger.info("Transfer-mode subtask tracker started.")
        self._phase_start_time = time.time()
        self._current_phase = 0
        self._nav_done_event.clear()

        # ── Tunable thresholds ────────────────────────────────────────
        grasp_thresh = float(getattr(self._runtime.cfg, "transfer_grasp_thresh", 3.5))
        lift_pitch_thresh = float(getattr(self._runtime.cfg, "transfer_lift_pitch", -0.20))
        grasp_duration = float(getattr(self._runtime.cfg, "transfer_grasp_duration", 0.40))
        grasp_confirm_frames = int(getattr(self._runtime.cfg, "transfer_grasp_frames", max(1, int(round(grasp_duration / 0.05)))))
        homing_dur = float(getattr(self._runtime.cfg, "subtask_homing_duration", 2.5))
        restore_dur = float(getattr(self._runtime.cfg, "transfer_restore_duration", 2.5))
        gripper_closed_val = float(getattr(self._runtime.cfg, "transfer_gripper_closed", 2.0))
        valen_eval_interval = float(getattr(self._runtime.cfg, "valen_eval_interval_s", 0.5))

        hold_after_grasp = getattr(self, "_transfer_hold_after_grasp", False)
        restore_pose_at_table_b = getattr(self, "_transfer_table_b_restore_pose", True)

        # Default placement prompt when JEV is unavailable (exact trained string)
        default_place_box_task = getattr(
            self._runtime.cfg, "transfer_place_box_task",
            "pick up the water bottle from the table and place it into the blue box"
        )
        default_place_table_task = getattr(
            self._runtime.cfg, "transfer_place_table_task",
            "pick up the water bottle from the table and place it into the blue box"
        )

        valen_client = getattr(self, "_valen_client", None)
        valen_auto_advance = True
        raw_vaa = getattr(self._runtime.cfg, "valen_auto_advance", True)
        if isinstance(raw_vaa, str):
            valen_auto_advance = raw_vaa.lower() in ("true", "1", "yes")
        elif isinstance(raw_vaa, bool):
            valen_auto_advance = raw_vaa

        # ── Phase 0 state ─────────────────────────────────────────────
        grasp_count = 0
        transfer_triggered = False
        last_hud_time = 0.0
        valen_last_time = 0.0
        latest_valen_hud = ""

        # ── Helper: capture full joint state snapshot ─────────────────
        def _capture_joint_snapshot() -> dict | None:
            """Return a {joint_key: float} snapshot of the robot's current action-space joints."""
            try:
                robot_wrapper = self.robot_wrapper
                robot = getattr(robot_wrapper, "inner", robot_wrapper)
                if robot is None:
                    return None
                obs = robot_wrapper.get_observation()
                # Keep only keys that appear in the action feature space
                action_keys = set(getattr(robot_wrapper, "action_features", {}).keys())
                snapshot = {k: float(v) for k, v in obs.items() if k in action_keys}
                return snapshot if snapshot else None
            except Exception as exc:
                logger.debug("Failed to capture joint snapshot: %s", exc)
                return None

        # ── Helper: restore arm to frozen pose, gripper stays clamped at real-time value ──
        def _restore_arm_to_frozen_pose(frozen: dict, duration_s: float, gripper_val: float) -> None:
            """Interpolate from current (home) pose to the frozen grasped pose.

            The gripper target is strictly locked to gripper_val (the real-time
            angle captured when the bottle was grasped), ensuring the gripper maintains
            the exact same gripping force and position throughout arm restoration.
            """
            from lerobot.rollout.strategies.core import RolloutStrategy
            try:
                robot_wrapper = self.robot_wrapper
                robot = getattr(robot_wrapper, "inner", robot_wrapper)
                if robot is None or not frozen:
                    return
                current_obs = robot_wrapper.get_observation()
                current_pos = {k: float(v) for k, v in current_obs.items() if k in frozen}

                # Build target: frozen pose, with gripper held at the real-time grasped angle
                target = dict(frozen)
                for grip_key in ("kRightGripper", "right_gripper", "gripper.right",
                                 "kLeftGripper", "left_gripper", "gripper.left"):
                    if grip_key in target:
                        target[grip_key] = gripper_val

                self._print(
                    f"\n── [Phase 3 / Transfer] 开始上肢姿态恢复 ({duration_s:.1f}s)...\n"
                    f"   夹爪保持夹取时的实时测量角度 ({gripper_val:.2f} rad)，恢复完成后 VLA 无缝接管"
                )
                RolloutStrategy._interpolate_motion(
                    robot_wrapper, current_pos, target, duration_s=duration_s
                )
                self._print("✅ [Phase 3 / Transfer] 上肢姿态恢复完成！")
            except Exception as exc:
                logger.warning("Arm pose restore failed: %s", exc)

        # ── Helper: JEV decision at table B ───────────────────────────
        def _decide_placement_task_at_tableB() -> str:
            """Called from post_homing_fn after navigation done + arm restore.

            Waits for /navdone signal, restores arm pose, then queries JEV to
            determine whether to place into box or onto table.
            Returns the subtask prompt string (used as next_task by _homing_transition).
            """
            # ── Phase 2: wait for /navdone signal ──────────────────
            with self._transfer_lock:
                if self._phase2_banner_printed:
                    return getattr(self, "_decided_placement_task", default_place_table_task)
                self._phase2_banner_printed = True

            self._current_phase = 2
            rule_65 = "=" * 65
            if hold_after_grasp:
                self._print(
                    f"\n{rule_65}\n"
                    " 📍 [Transfer] 机械臂已在抓取位置安全保持，夹爪持续夹紧水瓶。\n"
                    "    请移动/导航至桌B（导航过程中外部项目代码将手臂收回身侧默认位）；\n"
                    "    到达桌B后输入 'd' 回车 (或 /navdone) 继续放置任务。\n"
                    "   [TRANSFER_STATE] PHASE2_NAVIGATING\n"
                    f"{rule_65}"
                )
            else:
                self._print(
                    f"\n{rule_65}\n"
                    " 📍 [Transfer] 机器人已回到默认位置，夹爪保持夹紧水瓶。\n"
                    "    请移动/导航至桌B，到达后输入 'd' 回车 (或 /navdone) 继续。\n"
                    "   [TRANSFER_STATE] PHASE2_NAVIGATING\n"
                    f"{rule_65}"
                )
            self._nav_done_event.wait()   # blocks until /navdone or timeout
            self._nav_done_event.clear()

            # ── Phase 3: restore arm to frozen grasped pose or keep default ─
            self._current_phase = 3
            self._print("\n[TRANSFER_STATE] PHASE3_RESTORING")
            if restore_pose_at_table_b:
                frozen = self._frozen_joint_state
                grip_val = self._captured_gripper_val if self._captured_gripper_val is not None else float(getattr(self._runtime.cfg, "transfer_gripper_closed", 3.4))
                if frozen:
                    _restore_arm_to_frozen_pose(frozen, duration_s=restore_dur, gripper_val=grip_val)
                else:
                    self._print("⚠️  [Transfer] 未找到冻结关节快照，跳过姿态恢复。")
                    time.sleep(0.5)
            else:
                self._print(
                    " ⏭️  [Phase 3 / Transfer] 跳过姿态恢复: 机械臂保持在身侧默认位置，"
                    "直接以此动作启动 VLA 放置推理！"
                )
                time.sleep(0.5)

            # ── Phase 4 prep: JEV decides placement target ──────────
            self._current_phase = 4
            self._print("\n[TRANSFER_STATE] PHASE4_PLACING_PREP")
            time.sleep(0.5)  # let camera stabilise

            placement_task = default_place_box_task  # fallback to box
            target_decision_str = "放蓝盒 (place_into_box)"
            if valen_client is not None:
                g_frame, rw_frame = self._get_live_camera_frames()

                if g_frame is not None and rw_frame is not None:
                    ctx_t2 = (
                        "The input image is horizontally stitched from two camera views: "
                        "the left half is the third-person global overview showing Table B; "
                        "the right half is the right-wrist camera close-up showing the robot gripper holding the water bottle. "
                        "The robot is standing in front of Table B to place the water bottle into a container."
                    )
                    q_t2 = {
                        "task_phase": {
                            "type": "choice",
                            "instructions": "Look at Table B in the camera views. What is present on the tabletop?",
                            "criteria": {
                                "place_into_box": "A box, storage container, cardboard box, or bin is present on Table B.",
                                "place_on_table": "Table B is flat and empty, with no box or container present.",
                            },
                        }
                    }
                    v_res = valen_client.evaluate(
                        g_frame, rw_frame, phase=2, custom_questions=q_t2, custom_text=ctx_t2
                    )
                    if getattr(self, "_recorder", None) is not None:
                        self._recorder.record_jev_decision(
                            table=2,
                            phase=2,
                            description="Phase 4 prep Table B placement target decision",
                            result=v_res,
                            questions=q_t2,
                            context_text=ctx_t2,
                        )
                    if v_res.is_success:
                        prob = v_res.probabilities.get(v_res.choice, 0.0)
                        if v_res.choice == "place_into_box" and prob >= 0.50:
                            placement_task = default_place_box_task
                            target_decision_str = f"放蓝盒 (place_into_box, P={prob:.2f}, {v_res.cost_ms:.0f}ms)"
                        elif v_res.choice == "place_on_table" and prob >= 0.50:
                            # 用户目标为有盒子放入盒子，无盒子等待；若放行执行，目标依然对齐转运放蓝盒
                            placement_task = default_place_box_task
                            target_decision_str = f"放蓝盒 (未检出蓝盒 place_on_table, P={prob:.2f} -> 保持放蓝盒目标)"
                        elif v_res.choice == "place_into_box":
                            placement_task = default_place_box_task
                            target_decision_str = f"倾向放蓝盒 (P={prob:.2f}, {v_res.cost_ms:.0f}ms)"
                        else:
                            placement_task = default_place_box_task
                            target_decision_str = f"放蓝盒 (倾向 place_into_box, P={prob:.2f})"
                    else:
                        target_decision_str = f"查询失败 ({v_res.error}) -> 默认放蓝盒"
                else:
                    target_decision_str = "相机不可用 -> 默认放蓝盒"

            with self._transfer_lock:
                if not self._phase4_banner_printed:
                    self._phase4_banner_printed = True
                    self._decided_placement_task = placement_task
                    rule_65 = "=" * 65
                    self._print(
                        f"\n{rule_65}\n"
                        f" 🤖 [Transfer] Jev 视觉判断放置目标: {target_decision_str}\n"
                        f" 🎯 [Transfer Phase 4] VLA 接管放置任务: \"{placement_task}\"\n"
                        f"{rule_65}"
                    )
                else:
                    placement_task = getattr(self, "_decided_placement_task", default_place_table_task)
            return placement_task

        # ── Main Phase 0 detection loop ───────────────────────────────
        show_hud = True
        raw_hud = getattr(self._runtime.cfg, "transfer_hud", None)
        if isinstance(raw_hud, str):
            show_hud = raw_hud.lower() not in ("false", "0", "no")
        elif isinstance(raw_hud, bool):
            show_hud = raw_hud

        clamp_count = 0
        valen_grasp_count = 0
        latest_valen_grasp_str = ""
        clamp_start_pitch = None

        import concurrent.futures
        jev_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        jev_future = None

        try:
            while self._subtask_tracker_running and not self._subtask_stop_event.is_set():
                time.sleep(0.05)

                if not getattr(self.controller, "running", False):
                    continue
                if transfer_triggered:
                    # transition_with_homing is blocking on the serve thread; nothing more to do here
                    break

                now = time.time()
                elapsed = now - self._phase_start_time
                metrics = self._get_robot_metrics()

                if not metrics.get("connected", False):
                    continue

                r_gripper = float(metrics.get("r_gripper", 5.0))
                r_pitch = float(metrics.get("r_pitch", 0.0))

                # ── Detect physical clamping ──────────────────────────────
                # Clamped: right gripper closed around bottle (e.g. <= 3.5 rad)
                is_clamped = (r_gripper <= grasp_thresh)
                if is_clamped:
                    clamp_count += 1
                    if clamp_start_pitch is None:
                        clamp_start_pitch = r_pitch
                else:
                    clamp_count = 0
                    valen_grasp_count = 0
                    clamp_start_pitch = None

                # ── Asynchronous Valen JEV visual evaluation (non-blocking) ──
                if jev_future is not None and jev_future.done():
                    try:
                        v_res = jev_future.result()
                        if getattr(self, "_recorder", None) is not None:
                            self._recorder.record_jev_decision(
                                table=1,
                                phase=3,
                                description="Phase 0 grasp & lift check (held_in_air vs on_table_or_unlifted)",
                                result=v_res,
                            )
                        if v_res.is_success:
                            prob = v_res.probabilities.get(v_res.choice, 0.0)
                            conf = v_res.confidence
                            latest_valen_grasp_str = f"{v_res.choice} (Conf={conf:.3f}, P={prob*100:.1f}%, {v_res.cost_ms:.0f}ms)"
                            latest_valen_hud = f" | Jev: {latest_valen_grasp_str}"

                            choice_tag = "已离地悬空 (held_in_air)" if v_res.choice == "held_in_air" else "未离地/在桌上 (on_table_or_unlifted)"
                            curr_rise = (r_pitch - clamp_start_pitch) if clamp_start_pitch is not None else 0.0
                            self._print(
                                f"🤖 [Phase 0 JEV] 判定: {choice_tag} | "
                                f"Conf: {conf:.3f} (P={prob*100:.1f}%) | "
                                f"右夹爪: {r_gripper:.2f} rad | 肩Pitch: {r_pitch:.2f} rad (抬起量: {curr_rise:+.2f}) | "
                                f"耗时: {v_res.cost_ms:.0f}ms"
                            )

                            # JEV 视觉判定离地：要求高置信度 (prob >= 0.70)，杜绝 52% 弱分类噪点误触发
                            if v_res.choice == "held_in_air" and prob >= 0.70:
                                valen_grasp_count += 1
                            else:
                                valen_grasp_count = 0
                        else:
                            latest_valen_hud = f" | Jev: [{v_res.error or 'err'}]"
                    except Exception as exc:
                        logger.debug("JEV async evaluate failed: %s", exc)
                    jev_future = None

                # Launch new background JEV evaluation if ready and clamped (continuous 2~3Hz pipeline)
                if (
                    valen_client is not None
                    and is_clamped
                    and jev_future is None
                    and now - valen_last_time >= 0.1
                ):
                    valen_last_time = now
                    g_frame, rw_frame = self._get_live_camera_frames()
                    if g_frame is not None and rw_frame is not None:
                        jev_future = jev_executor.submit(valen_client.evaluate, g_frame, rw_frame, 3)

                # ── HUD (only when transfer_hud is explicitly enabled) ──────
                if show_hud and now - last_hud_time >= 1.5:
                    last_hud_time = now
                    grip_str = f"夹紧({clamp_count * 0.05:.2f}s)" if is_clamped else "张开"
                    valen_str = f" [JEV离地: {valen_grasp_count}/2]" if valen_client is not None else ""
                    confirm_str = f" [夹持: {clamp_count}/{grasp_confirm_frames}]"
                    self._print(
                        f"📊 [Transfer Phase 0: 夹水瓶] 耗时: {elapsed:4.1f}s | "
                        f"右夹爪: {r_gripper:.2f} rad ({grip_str}) | "
                        f"右肩: {r_pitch:+.2f} rad{confirm_str}{valen_str}{latest_valen_hud}"
                    )

                # ── Grasp & Lift Trigger Condition ──────────────────────────
                # 双重门禁 (AND 关系，二者必须同时满足)：
                # 1. 夹爪闭合保持达到设定时长 (clamp_count >= grasp_confirm_frames，例如 >= 8 帧 / 0.40s)
                # 2. 并且 JEV 视觉权威确认水瓶离地悬空 (held_in_air, P >= 0.70) 达到 2 次 (valen_grasp_count >= 2)
                # 持续夹持 60 帧 (3.0s) 仅作为网络中断或视觉服务未响应时的安全兜底。
                if valen_client is not None:
                    is_ready_to_home = is_clamped and (
                        ((clamp_count >= grasp_confirm_frames) and (valen_grasp_count >= 2))
                        or (clamp_count >= 60)
                    )
                else:
                    is_ready_to_home = is_clamped and (clamp_count >= grasp_confirm_frames)

                if is_ready_to_home and not transfer_triggered:
                    with self._transfer_lock:
                        if self._transfer_triggered:
                            break
                        self._transfer_triggered = True
                    transfer_triggered = True
                    self._current_phase = 1

                    # 夹起水瓶静止后立即休眠视觉并停止向 JEV 传入图像
                    self._vla_transfer_grasp_completed = True
                    self._vla_switched_out_after_grasp = False
                    self._vla_reentered_after_grasp = False
                    self._stop_transfer_polling()

                    # Capture real-time gripper measurement at the moment of confirmed grasp
                    captured_gripper_val = float(r_gripper)
                    self._captured_gripper_val = captured_gripper_val

                    reason_str = f"夹爪: {captured_gripper_val:.2f} rad, 右肩: {r_pitch:.2f} rad (夹持确认 {clamp_count * 0.05:.2f}s)"
                    if latest_valen_grasp_str:
                        reason_str += f" | 🤖 JEV视觉确认: {latest_valen_grasp_str}"

                    # Capture frozen snapshot BEFORE homing starts
                    snapshot = _capture_joint_snapshot()
                    if snapshot:
                        # Explicitly stamp the real-time grasped gripper angle onto all gripper keys in the snapshot
                        for grip_key in ("kRightGripper", "right_gripper", "gripper.right",
                                         "kLeftGripper", "left_gripper", "gripper.left"):
                            if grip_key in snapshot:
                                snapshot[grip_key] = captured_gripper_val
                        self._frozen_joint_state = snapshot

                    rule_65 = "=" * 65
                    if hold_after_grasp:
                        self._print(
                            f"\n{rule_65}\n"
                            f"✊ [Transfer] 检测到水瓶已夹起 ({reason_str})。\n"
                            f"   已冻结抓取姿态快照，夹爪全程锁定夹紧 ({captured_gripper_val:.2f} rad)。\n"
                            f"   [抓取位保持防碰桌模式]: 保持在当前抓取位置，跳过回身侧动作！\n"
                            f"   [TRANSFER_STATE] PHASE1_HOLDING\n"
                            f"   🔒 [视觉休眠]: 已停止向 JEV 传入图像；导航到达桌B后手柄再次切入 VLA (或按'd') 才会恢复视觉！\n"
                            f"{rule_65}"
                        )
                    else:
                        self._print(
                            f"\n{rule_65}\n"
                            f"✊ [Transfer] 检测到水瓶已夹起 ({reason_str})。\n"
                            f"   已冻结抓取姿态快照，夹爪全程锁定夹紧 ({captured_gripper_val:.2f} rad)。\n"
                            f"   正在平滑回默认位置 ({homing_dur:.1f}s)...\n"
                            f"   [TRANSFER_STATE] PHASE1_RETURNING\n"
                            f"{rule_65}"
                        )

                    # ── Trigger transition (with skip_homing if hold_after_grasp is True) ─────
                    target_override = {}
                    hw_initial = getattr(self.ctx.hardware, "initial_position", {}) or {}
                    for grip_key in ("kRightGripper", "right_gripper", "gripper.right",
                                     "kLeftGripper", "left_gripper", "gripper.left"):
                        if grip_key in hw_initial:
                            target_override[grip_key] = captured_gripper_val

                    ok = self.controller.transition_with_homing(
                        next_task=None,        # task set inside post_homing_fn
                        duration_s=homing_dur,
                        retract_first=False,   # already upright, no obstacle to retract from
                        retract_duration_s=0.0,
                        resume=True,           # _homing_transition will call _run_segment after fn returns
                        post_homing_fn=_decide_placement_task_at_tableB,
                        target_override=target_override if target_override else None,
                        skip_homing=hold_after_grasp,
                    )
                    if not ok:
                        self._print("⚠️  [Transfer] transition_with_homing 被拒绝（控制器已停止？），中止转运。")
                        with self._transfer_lock:
                            self._transfer_triggered = False
                        transfer_triggered = False
                        self._current_phase = 0
                    break
        finally:
            try:
                jev_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        logger.info("Transfer-mode subtask tracker exited (phase=%d).", self._current_phase)

    def _subtask_tracker_loop_transfer_phase4(self) -> None:
        """Monitor Phase 4 (placing at table B): detect release, home to default pose, and wait for user."""
        logger.info("Transfer Phase 4 completion tracker started.")
        self._phase_start_time = time.time()
        has_bottle = False
        release_count = 0
        valen_confirm_count = 0
        valen_last_time = 0.0
        last_hud_time = 0.0
        valen_eval_interval = float(getattr(self._runtime.cfg, "valen_eval_interval_s", 0.4))
        homing_dur = float(getattr(self._runtime.cfg, "subtask_homing_duration", 2.5))
        retract_dur = 1.2
        valen_client = getattr(self, "_valen_client", None)
        task_str = str(self.controller.task or "")
        is_box_task = "box" in task_str.lower()
        show_hud = True
        raw_hud = getattr(self._runtime.cfg, "transfer_hud", None)
        if isinstance(raw_hud, str):
            show_hud = raw_hud.lower() not in ("false", "0", "no")
        elif isinstance(raw_hud, bool):
            show_hud = raw_hud

        while self._subtask_tracker_running and not self._subtask_stop_event.is_set():
            time.sleep(0.05)

            if not getattr(self.controller, "running", False):
                continue

            now = time.time()
            metrics = self._get_robot_metrics()
            if not metrics.get("connected", False):
                continue

            r_gripper = float(metrics.get("r_gripper", 5.0))

            # Track bottle holding state:
            # Robot must be observed holding the bottle (r_gripper <= 3.8 rad) before release can be detected
            if r_gripper <= 3.8:
                has_bottle = True

            # Query Valen evaluator if enabled
            latest_valen_choice = ""
            if valen_client is not None and now - valen_last_time >= valen_eval_interval:
                valen_last_time = now
                g_frame, rw_frame = self._get_live_camera_frames()

                if g_frame is not None and rw_frame is not None:
                    target_phase = 0 if is_box_task else 1
                    v_res = valen_client.evaluate(g_frame, rw_frame, phase=target_phase)
                    if getattr(self, "_recorder", None) is not None:
                        self._recorder.record_jev_decision(
                            table=2,
                            phase=target_phase,
                            description=f"Phase 4 placement completion check ({'box' if is_box_task else 'table'})",
                            result=v_res,
                        )
                    if v_res.is_success:
                        prob = v_res.probabilities.get(v_res.choice, 0.0)
                        latest_valen_choice = f"{v_res.choice} (P={prob:.2f})"
                        expected_choice = "bottle_in_box_completed" if is_box_task else "bottle_on_table_completed"
                        if v_res.choice == expected_choice and prob >= 0.50:
                            valen_confirm_count += 1
                        else:
                            valen_confirm_count = max(0, valen_confirm_count - 1)

                        tag = "已入盒" if v_res.choice == "bottle_in_box_completed" else ("已放回桌面" if v_res.choice == "bottle_on_table_completed" else "放置中")
                        self._print(
                            f"🤖 [Phase 4 JEV] 判定: {tag} ({v_res.choice}, P={prob:.2f}, {v_res.cost_ms:.0f}ms) | "
                            f"右夹爪: {r_gripper:.2f} rad | 确认: {valen_confirm_count}/2"
                        )

            # Periodic HUD every 1.5s
            if show_hud and now - last_hud_time >= 1.5:
                last_hud_time = now
                grip_str = "已松爪释放" if r_gripper >= 4.7 else ("夹持水瓶" if has_bottle else "张开待命")
                self._print(
                    f"📊 [Transfer Phase 4: 放置] 耗时: {now - self._phase_start_time:4.1f}s | "
                    f"右夹爪: {r_gripper:4.2f} rad ({grip_str}) | 放手确认: {release_count}/6 | JEV确认: {valen_confirm_count}/2"
                )

            # Check physical gripper release:
            # ONLY valid if the robot was confirmed holding the bottle (has_bottle is True)
            if has_bottle and r_gripper >= 4.7:
                release_count += 1
            else:
                release_count = max(0, release_count - 1)

            # Confirmed if gripper open sustained for 6 frames (0.3s) after holding bottle,
            # or if Valen confirms 2 frames and gripper is open (>= 4.5 rad)
            is_released = (has_bottle and release_count >= 6) or (valen_confirm_count >= 2 and r_gripper >= 4.5)

            if is_released:
                self._transfer_completed = True
                self._stop_transfer_polling()
                if valen_confirm_count >= 2 and r_gripper >= 4.5:
                    reason = f"视觉确认水瓶入盒完成 (JEV: {latest_valen_choice}) 且右夹爪已松开 ({r_gripper:.2f} rad >= 4.5)"
                else:
                    reason = f"右夹爪已完全松开释放水瓶 (右夹爪: {r_gripper:.2f} rad >= 4.7)"
                    if latest_valen_choice:
                        reason += f" [JEV视觉末次参考: {latest_valen_choice}]"

                rule_65 = "=" * 65
                rule_61 = "-" * 61
                self._print(
                    f"\n{rule_65}\n"
                    "🎉 [Transfer] 跨桌水瓶转运全套任务已圆满完成！\n"
                    f"   状态判定: {reason}\n"
                    "   平滑回位待命:\n"
                    f"     ├─ 水平收手 ({retract_dur:.1f}s) 避开盒子与桌面\n"
                    f"     └─ 余弦 S 曲线归位 ({homing_dur:.1f}s) 平滑回到身侧默认待机位置\n"
                    f"   {rule_61}\n"
                    "   💡 任务完成待命:\n"
                    "     1. 机械臂平稳归位到身侧待命姿态后，将彻底静止不动；\n"
                    "     2. 保持大模型显存与上下文不重置 (Zero-Reset)；\n"
                    "     3. 视觉感知与自动推理已彻底休眠，静待在终端输入 'r' 回车复位；\n"
                    "     4. 亦可按 'q' 回车 退出。\n"
                    f"{rule_65}\n"
                )

                if hasattr(self.controller, "transition_with_homing"):
                    self.controller.transition_with_homing(
                        next_task=None,
                        duration_s=homing_dur,
                        retract_first=True,
                        retract_duration_s=retract_dur,
                        resume=False,
                        post_homing_fn=self._on_transfer_round_completed,
                    )
                else:
                    self._on_transfer_round_completed()
                break

        logger.info("Transfer Phase 4 completion tracker exited.")

    # ------------------------------------------------------------------
    # Command handlers (called from the listener thread)
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            ctrl_state = "RUNNING (运行中)" if self.controller.running else ("RESETTING (复位中)" if self._is_resetting else "IDLE (待命)")
            if self._transfer_mode:
                if getattr(self, "_transfer_completed", False):
                    self._print(
                        "[Transfer 状态] 全套转运任务已圆满完成，机械臂静止待命中 | "
                        "👉 请输入 'r' 回车复位状态，或 'q' 退出"
                    )
                    return
                phase_labels = {
                    0: "Phase 0 (桌A夹水瓶)",
                    1: "Phase 1 (回默认位置)",
                    2: "Phase 2 (等待导航, 输入 d 回车)",
                    3: "Phase 3 (姿态恢复中)",
                    4: "Phase 4 (桌B放置)",
                }
                phase_str = phase_labels.get(self._current_phase, f"Phase {self._current_phase}")
                metrics = self._get_robot_metrics()
                grip_val = metrics.get("r_gripper", 0.0)
                pitch_val = metrics.get("r_pitch", 0.0)
                self._print(
                    f"[Transfer 状态] {phase_str} | 控制器: {ctrl_state} | "
                    f"右夹爪: {grip_val:.2f} rad | 右肩俯仰: {pitch_val:+.2f} rad | "
                    f"指令: 's' 启动, 'd' 导航到达, 'r' 复位, 'q' 退出"
                )
                return
            # User pressed empty Enter: print a one-line quick status
            robot = self.robot_wrapper
            mode = getattr(robot, "current_mode", "N/A")
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            pkts = getattr(robot, "mode_packet_count", 0)
            mode_port = getattr(robot, "mode_port", 6000)
            self._print(f"[当前状态] 模式端口({mode_port}): {mode_val} (收包: {pkts}) | 控制器: {ctrl_state} | 输入 'm' 查看详情, 's' 启动, 'r' 复位")
            return
        cmd = parse_command(stripped)
        if cmd is None:
            self._print("Input not recognized — commands: s (start), d (navdone), r (reset), m (mode), n (next), q (stop), /help.")
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
        if self._transfer_mode and getattr(self, "_transfer_completed", False):
            self._print("⚠️  上一轮转运任务已完成！机械臂保持静止。请先输入 'r' 回车复位状态，再开启新一轮实验。")
            return
        if self._transfer_mode:
            self._stop_transfer_polling()
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

    def _cmd_navdone(self, cmd: InteractiveCommand) -> None:
        """Signal that navigation to table B is complete.

        Sets the nav_done event, which unblocks the post_homing_fn waiting in
        the serve thread.  The serve thread then runs Phase 3 (arm restore) and
        Phase 4 (VLA takeover with JEV-decided prompt) automatically.
        """
        if not self._transfer_mode:
            self._print("Transfer mode is not active — /navdone has no effect.")
            return
        if getattr(self, "_transfer_completed", False):
            self._print("⚠️  上一轮转运任务已完成！请先输入 'r' 回车复位状态。")
            return
        if self._current_phase == 0 and not self.controller.running:
            self._print("👉 [Transfer] 在桌1收到手动 'd' 指令，启动桌1水瓶视觉检测...")
            self._start_transfer_polling(table=1)
            return
        elif self._current_phase != 2:
            self._print(
                f"⚠️  /navdone received but current transfer phase is {self._current_phase} (expected 2: NAVIGATING). "
                "Signal stored; it will unblock the next Phase 2 wait if one occurs."
            )
        else:
            self._stop_transfer_polling()
            self._vla_transfer_grasp_completed = False
            self._vla_switched_out_after_grasp = False
            self._vla_reentered_after_grasp = False
            if getattr(self, "_transfer_table_b_restore_pose", True):
                self._print(
                    "✅ [Transfer] 导航完成信号已接收！\n"
                    "   → 开始 Phase 3: 上肢从身侧默认位置平滑恢复到抓取时的姿态（夹爪保持夹紧）…"
                )
            else:
                self._print(
                    "✅ [Transfer] 导航完成信号已接收！\n"
                    "   → [默认位接管模式]: 机械臂保持在身侧默认位置，直接以此动作启动 VLA 放置推理…"
                )
        self._nav_done_event.set()

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        self._is_starting = False
        self._is_resetting = True
        if self._transfer_mode:
            self._transfer_completed = False
            self._stop_transfer_polling()
        self._stop_subtask_tracker()
        if self._transfer_mode:
            # Also clear transfer-specific state so a re-run starts fresh
            self._nav_done_event.set()   # unblock any waiting post_homing_fn
            self._reset_transfer_state()
            self._last_observed_robot_mode = None
        if self._subtasks_enabled:
            self._current_phase = 0
            self._lift_sustained_seconds = 0.0
            self._phase_start_time = 0.0
            self.controller._initial_task = self._subtask_list[0]
        self.controller.reset()
        if self.controller.stopped:
            self._print("Can't reset — the session has stopped.")
        else:
            if self._transfer_mode:
                self._print(
                    f"♻️ [Reset] 跨桌水瓶转运状态已重置完成！\n"
                    f"   ├─ 任务已恢复为初始阶段: {_format_task(self.controller.initial_task)}\n"
                    "   ├─ 已清空转运生命周期与抓取状态\n"
                    "   └─ 👉 请在桌A重新摆好水瓶后，切入手柄VLA模式或按 's' 回车启动下一轮实验。"
                )
            elif self._subtasks_enabled:
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
        robot = self.robot_wrapper
        mode_port = getattr(robot, "mode_port", getattr(getattr(self.ctx, "runtime", None), "cfg", None).mode_port if hasattr(getattr(self.ctx, "runtime", None), "cfg") else 6000)
        subtask_info = ""
        if self._subtasks_enabled:
            total_phases = len(self._subtask_list)
            seq_desc = " -> ".join(f"[{i+1}] {t}" for i, t in enumerate(self._subtask_list))
            shortcuts = "/".join(str(i+1) for i in range(total_phases))
            auto_home = getattr(self._runtime.cfg, "subtask_auto_home", True)
            auto_adv = getattr(self._runtime.cfg, "subtask_auto_advance", False)
            if auto_home and not auto_adv:
                home_note = " (★ 动作完成自动平滑回位并保持Prompt，按'n'手动切段)"
            elif auto_home and auto_adv:
                home_note = " (★ 阶段完成自动平滑回位并自动切换Prompt)"
            else:
                home_note = ""
            subtask_info = (
                f"Subtasks Sequence: ON ({total_phases} 阶段流转模式已激活{home_note})\n"
                f"  {seq_desc}\n"
                f"  (快捷键: 'n' 跳下一阶段, '{shortcuts}' 选阶段, 'r' 复位, 's' 启动, 'q' 退出)\n"
            )
        transfer_info = ""
        if self._transfer_mode:
            transfer_info = (
                "Task Mode: WATER BOTTLE TRANSFER (跨桌水瓶转运 - JEV双桌感知与自动进入模式)\n"
                f"  • 自动监听 {mode_port} 端口: 手柄切入 VLA 模式时自动触发 JEV 视觉感知，无需手动按 'd'！\n"
                "    ├─ 桌1 (夹爪未抓): JEV 视觉确认桌上有水瓶 -> 自动切入 VLA 夹取 (Phase 0)\n"
                "    └─ 桌2 (夹爪持瓶): JEV 视觉确认桌上有目标盒子 -> 自动切入 VLA 放置 (Phase 4)\n"
                "    └─ 若未检测到目标，机械臂严格保持当前姿态与夹持力！\n"
                "  • 快捷指令: 'd' 强制确认/跳过等待 | 's' 启动 | 'r' 复位 | 'q' 退出\n"
            )
        mode_info = ""
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
            f"{transfer_info}"
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
