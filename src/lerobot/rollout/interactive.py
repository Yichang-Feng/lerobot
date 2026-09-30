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
        # Event set by /navdone command to signal navigation has completed
        self._nav_done_event = threading.Event()
        # Snapshot of robot joint state at the moment the bottle was confirmed grasped+lifted.
        # Used to restore arm pose at table B before VLA takeover (to prevent motion jump).
        self._frozen_joint_state: dict | None = None
        # Real-time gripper angle measured at the moment grasp success was confirmed.
        # Clamped throughout homing, navigation, and pose restore without VLA intervention.
        self._captured_gripper_val: float | None = None
        self._transfer_banner_printed: bool = False

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
            if self._transfer_mode:
                if self._current_phase == 4:
                    self._print(
                        f"Rollout running — [Transfer Phase 4] VLA 已接管放置任务: \"{self.controller.task}\""
                    )
                elif not getattr(self, "_transfer_banner_printed", False):
                    self._transfer_banner_printed = True
                    self._current_phase = 0
                    self.controller.set_task(self.controller.task or "pick up the water bottle from the table")
                    self._print(
                        f"Rollout running — [跨桌水瓶转运] Phase 0: 夹取水瓶 (Task: \"{self.controller.task}\").\n"
                        f"提示: 检测到夹紧并抬起后自动夹持回默认位；导航到达桌B后输入 'd' 回车继续。"
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
            robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
            last_cams = getattr(robot, "_last_camera_frames", {})
            g_frame = last_cams.get("global_view") or last_cams.get("base_0_rgb")
            rw_frame = last_cams.get("right_wrist") or last_cams.get("right_wrist_0_rgb")

            # 1. If Valen evaluator is active, evaluate scene from default position
            if valen_client is not None and g_frame is not None and rw_frame is not None:
                v_res = valen_client.evaluate(g_frame, rw_frame, phase=0)
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
                robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
                last_cams = getattr(robot, "_last_camera_frames", {})
                g_frame = last_cams.get("global_view")
                if g_frame is None:
                    g_frame = last_cams.get("base_0_rgb")
                rw_frame = last_cams.get("right_wrist")
                if rw_frame is None:
                    rw_frame = last_cams.get("right_wrist_0_rgb")

                if g_frame is not None and rw_frame is not None:
                    v_res = valen_client.evaluate(g_frame, rw_frame, phase=self._current_phase)
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
        grasp_confirm_frames = int(getattr(self._runtime.cfg, "transfer_grasp_frames", 8))
        homing_dur = float(getattr(self._runtime.cfg, "subtask_homing_duration", 2.5))
        restore_dur = float(getattr(self._runtime.cfg, "transfer_restore_duration", 2.5))
        gripper_closed_val = float(getattr(self._runtime.cfg, "transfer_gripper_closed", 2.0))
        valen_eval_interval = float(getattr(self._runtime.cfg, "valen_eval_interval_s", 0.5))

        # Default placement prompt when JEV is unavailable
        default_place_box_task = getattr(
            self._runtime.cfg, "transfer_place_box_task",
            "pick up the water bottle and place it into the blue box"
        )
        default_place_table_task = getattr(
            self._runtime.cfg, "transfer_place_table_task",
            "pick up the water bottle and place it on the table"
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
            self._current_phase = 2
            self._print(
                "\n" + "=" * 65 + "\n"
                " 📍 [Transfer] 机器人已回到默认位置，夹爪保持夹紧水瓶。\n"
                "    请移动/导航至桌B，到达后输入 'd' 回车 (或 /navdone) 继续。\n"
                "   [TRANSFER_STATE] PHASE2_NAVIGATING\n"
                "=" * 65
            )
            self._nav_done_event.wait()   # blocks until /navdone or timeout
            self._nav_done_event.clear()

            # ── Phase 3: restore arm to frozen grasped pose ─────────
            self._current_phase = 3
            self._print("\n[TRANSFER_STATE] PHASE3_RESTORING")
            frozen = self._frozen_joint_state
            grip_val = self._captured_gripper_val if self._captured_gripper_val is not None else float(getattr(self._runtime.cfg, "transfer_gripper_closed", 3.4))
            if frozen:
                _restore_arm_to_frozen_pose(frozen, duration_s=restore_dur, gripper_val=grip_val)
            else:
                self._print("⚠️  [Transfer] 未找到冻结关节快照，跳过姿态恢复。")
                time.sleep(0.5)

            # ── Phase 4 prep: JEV decides placement target ──────────
            self._current_phase = 4
            self._print("\n[TRANSFER_STATE] PHASE4_PLACING_PREP")
            time.sleep(0.5)  # let camera stabilise

            placement_task = default_place_table_task  # fallback
            target_decision_str = "放桌上 (place_on_table)"
            if valen_client is not None:
                robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
                last_cams = getattr(robot, "_last_camera_frames", {})
                g_frame = last_cams.get("base_0_rgb") or last_cams.get("global_view")
                rw_frame = last_cams.get("right_wrist_0_rgb") or last_cams.get("right_wrist")

                if g_frame is not None and rw_frame is not None:
                    # phase=2 → JEV server decides: "place_into_box" or "place_on_table"
                    v_res = valen_client.evaluate(g_frame, rw_frame, phase=2)
                    if v_res.is_success:
                        prob = v_res.probabilities.get(v_res.choice, 0.0)
                        if v_res.choice == "place_into_box":
                            placement_task = default_place_box_task
                            target_decision_str = f"放蓝盒 (place_into_box, P={prob:.2f}, {v_res.cost_ms:.0f}ms)"
                        elif v_res.choice == "place_on_table":
                            placement_task = default_place_table_task
                            target_decision_str = f"放桌上 (place_on_table, P={prob:.2f}, {v_res.cost_ms:.0f}ms)"
                        else:
                            placement_task = default_place_table_task
                            target_decision_str = f"未知标签 '{v_res.choice}' -> 默认放桌上"
                    else:
                        target_decision_str = f"查询失败 ({v_res.error}) -> 默认放桌上"
                else:
                    target_decision_str = "相机不可用 -> 默认放桌上"

            self._print(
                "\n" + "=" * 65 + "\n"
                f" 🤖 [Transfer] Jev 视觉判断放置目标: {target_decision_str}\n"
                f" 🎯 [Transfer Phase 4] VLA 接管放置任务: \"{placement_task}\"\n"
                "=" * 65
            )
            return placement_task

        # ── Main Phase 0 detection loop ───────────────────────────────
        show_hud = False
        raw_hud = getattr(self._runtime.cfg, "transfer_hud", False)
        if isinstance(raw_hud, str):
            show_hud = raw_hud.lower() in ("true", "1", "yes")
        elif isinstance(raw_hud, bool):
            show_hud = raw_hud

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

            # ── Valen HUD (only when transfer_hud is explicitly enabled) ─────
            if show_hud and valen_client is not None and now - valen_last_time >= valen_eval_interval:
                valen_last_time = now
                robot = getattr(self.robot_wrapper, "inner", self.robot_wrapper)
                last_cams = getattr(robot, "_last_camera_frames", {})
                g_frame = last_cams.get("base_0_rgb") or last_cams.get("global_view")
                rw_frame = last_cams.get("right_wrist_0_rgb") or last_cams.get("right_wrist")
                if g_frame is not None and rw_frame is not None:
                    v_res = valen_client.evaluate(g_frame, rw_frame, phase=0)
                    if v_res.is_success:
                        prob = v_res.probabilities.get(v_res.choice, 0.0)
                        latest_valen_hud = f" | Jev: {v_res.choice} (P={prob:.2f})"
                    else:
                        latest_valen_hud = f" | Jev: [{v_res.error or 'err'}]"

            # ── Detect grasped + slightly lifted ──────────────────────
            # Clamped: gripper closed around bottle (e.g. <= 3.5 rad)
            # Lifted: right shoulder pitch rotated upwards (r_pitch <= lift_pitch_thresh, e.g. <= -0.15 rad)
            is_clamped = (r_gripper <= grasp_thresh)
            is_lifted = (r_pitch <= lift_pitch_thresh)
            is_grasped_and_lifted = is_clamped and is_lifted

            if is_grasped_and_lifted:
                grasp_count += 1
            else:
                grasp_count = max(0, grasp_count - 1)

            # ── HUD (only when transfer_hud is explicitly enabled) ──────
            if show_hud and now - last_hud_time >= 1.5:
                last_hud_time = now
                grip_str = "夹紧✊" if is_clamped else "张开🤚"
                lift_str = f"已抬起({r_pitch:+.2f})" if is_lifted else f"未抬起({r_pitch:+.2f})"
                confirm_str = f" [确认: {grasp_count}/{grasp_confirm_frames}]" if grasp_count > 0 else ""
                self._print(
                    f"📊 [Transfer Phase 0: 夹水瓶] 耗时: {elapsed:4.1f}s | "
                    f"右夹爪: {r_gripper:.2f} rad ({grip_str}) | "
                    f"右肩: {lift_str}{confirm_str}{latest_valen_hud}"
                )

            if grasp_count >= grasp_confirm_frames and not transfer_triggered:
                transfer_triggered = True
                self._current_phase = 1

                # Capture real-time gripper measurement at the moment of confirmed grasp
                captured_gripper_val = float(r_gripper)
                self._captured_gripper_val = captured_gripper_val

                # Capture frozen snapshot BEFORE homing starts
                snapshot = _capture_joint_snapshot()
                if snapshot:
                    # Explicitly stamp the real-time grasped gripper angle onto all gripper keys in the snapshot
                    for grip_key in ("kRightGripper", "right_gripper", "gripper.right",
                                     "kLeftGripper", "left_gripper", "gripper.left"):
                        if grip_key in snapshot:
                            snapshot[grip_key] = captured_gripper_val
                    self._frozen_joint_state = snapshot
                    self._print(
                        f"\n✊ [Transfer] 检测到水瓶已夹起并抬起 (夹爪: {captured_gripper_val:.2f} rad, 右肩: {r_pitch:.2f} rad)。\n"
                        f"   已冻结抓取姿态快照，夹爪全程锁定夹紧 ({captured_gripper_val:.2f} rad)。\n"
                        f"   正在平滑回默认位置 ({homing_dur:.1f}s)...\n"
                        f"   [TRANSFER_STATE] PHASE1_RETURNING"
                    )
                else:
                    self._print(
                        f"\n✊ [Transfer] 检测到夹紧+抬起 (夹爪: {captured_gripper_val:.2f} rad)，"
                        f"正在回默认位置 ({homing_dur:.1f}s)...\n"
                        f"   [TRANSFER_STATE] PHASE1_RETURNING"
                    )

                # ── Trigger homing, gripper locked to captured_gripper_val ─────
                # Build target_override so return_to_initial_position maintains the exact
                # grasped gripper angle instead of opening back to the default position's 5.0 rad.
                target_override = {}
                hw_initial = getattr(self.ctx.hardware, "initial_position", {}) or {}
                for grip_key in ("kRightGripper", "right_gripper", "gripper.right",
                                 "kLeftGripper", "left_gripper", "gripper.left"):
                    if grip_key in hw_initial:
                        target_override[grip_key] = captured_gripper_val

                # retract_first=False: arm is not forward-extended into a box;
                # the retract waypoint would try to open the gripper (undesired).
                # post_homing_fn blocks the serve thread while waiting for nav + restore + decide.
                ok = self.controller.transition_with_homing(
                    next_task=None,        # task set inside post_homing_fn
                    duration_s=homing_dur,
                    retract_first=False,   # already upright, no obstacle to retract from
                    retract_duration_s=0.0,
                    resume=True,           # _homing_transition will call _run_segment after fn returns
                    post_homing_fn=_decide_placement_task_at_tableB,
                    target_override=target_override if target_override else None,
                )
                if not ok:
                    self._print("⚠️  [Transfer] transition_with_homing 被拒绝（控制器已停止？），中止转运。")
                    transfer_triggered = False
                    self._current_phase = 0
                break

        logger.info("Transfer-mode subtask tracker exited (phase=%d).", self._current_phase)

    # ------------------------------------------------------------------
    # Command handlers (called from the listener thread)
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            ctrl_state = "RUNNING (运行中)" if self.controller.running else ("RESETTING (复位中)" if self._is_resetting else "IDLE (待命)")
            if self._transfer_mode:
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
        if self._current_phase != 2:
            self._print(
                f"⚠️  /navdone received but current transfer phase is {self._current_phase} (expected 2: NAVIGATING). "
                "Signal stored; it will unblock the next Phase 2 wait if one occurs."
            )
        else:
            self._print(
                "✅ [Transfer] 导航完成信号已接收！\n"
                "   → 开始 Phase 3: 上肢从默认位置恢复到抓取时的冻结姿态（夹爪保持夹紧）…"
            )
        self._nav_done_event.set()

    def _cmd_reset(self, cmd: InteractiveCommand) -> None:
        self._is_starting = False
        self._is_resetting = True
        self._stop_subtask_tracker()
        if self._transfer_mode:
            # Also clear transfer-specific state so a re-run starts fresh
            self._nav_done_event.set()   # unblock any waiting post_homing_fn
            self._frozen_joint_state = None
            self._captured_gripper_val = None
            self._current_phase = 0
            self._transfer_banner_printed = False
            self._nav_done_event.clear()
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
                "Task Mode: WATER BOTTLE TRANSFER (跨桌水瓶转运)\n"
                "  • 阶段流程: 桌A夹水瓶 → 自动夹紧回默认位 → 导航至桌B(输入 'd' 回车) → 恢复姿态 → 桌B放置\n"
                "  • 快捷指令: 's' 启动推理 | 'd' 导航到达桌B | 'r' 复位 | 'q' 退出\n"
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
