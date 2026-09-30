# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations

import abc
import contextlib
import logging
import math
from typing import TYPE_CHECKING

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)


class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.

    Lifecycle: ``setup()`` once, then ``run()``, then ``teardown()`` once.
    A strategy whose config declares ``supports_interactive = True`` is also
    driven by ``--interactive=true``, which calls ``run()`` once per
    start/stop segment.  Such a strategy must keep ``run()`` restartable:

    - never finalize the dataset in ``run()`` — that belongs in ``teardown()``;
      at most save a partial tail episode when a segment ends;
    - keep state that must survive a segment on the instance, not in ``run()``
      locals (the ``CycleTimer`` is deliberately the other way round, see ``run()``);
    - never bind keyboard/terminal listeners — stdin belongs to the command prompt;
    - call ``engine.pump_query(obs_processed)`` once at the end of every tick, see
      ``run()``.

    One-shot strategies (``supports_interactive = False``, the default) are
    free to finalize on ``run()`` exit, e.g. via ``VideoEncodingManager``.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self.reset_control_state()
        self._engine.start()
        self._warmup_flushed = False
        logger.info("Inference engine started")

    def reset_control_state(self) -> None:
        """Clear episode-scoped control state so a paused session can restart cleanly.

        Resets the inference engine (policy hidden state, action queues), the action
        interpolator and the cached processed observation; pacing state is untouched.
        ``RolloutController`` calls it on its serve thread before each run segment.
        Only call while the control loop is not running: these resets are not synchronized
        against a live loop.  A caller that resets control state while a loop runs — or a
        strategy that hoists its timer onto the instance — must also call ``timer.restart()``.
        """
        if self._engine is not None:
            self._engine.reset()
        if self._interpolator is not None:
            self._interpolator.reset()
        self._cached_obs_processed = None

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, timer: CycleTimer) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  Warmup ticks are paced through *timer* so the loop cadence
        stays anchored.  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            if not getattr(self, "_warmup_logged", False):
                logger.info(
                    "torch.compile is compiling the policy model in background (typically takes 40~90s on first run). "
                    "Robot will maintain safe stand position until compilation completes..."
                )
                self._warmup_logged = True
            timer.wait()
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            timer.restart()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self.return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def _compute_retract_waypoint(current_pos: dict[str, float]) -> dict[str, float] | None:
        """Compute an intermediate horizontal retraction pose to pull the arm backward away from a container.

        Kinematics for Unitree G1:
        When releasing an object in a container (e.g. blue box):
        - The arm is extended forward (ShoulderPitch ~ -0.90 rad, Elbow ~ +1.25 rad).
        - Direct interpolation to default pose (ShoulderPitch ~ +0.05, Elbow ~ 0.45) drops the shoulder
          downwards and shifts lateral before retreating, hitting the container wall.
        - Conversely, bending the elbow more causes a downward dip because dZ/dElbow is negative.

        Correct 2-stage motion:
        - Stage 1 (Horizontal Retraction Waypoint):
          1. Shoulder pitch retracts from ~ -0.90 to -0.38 rad (raising arm up & backward).
          2. Elbow adjusts to 0.42 rad (retaining forearm horizontally without dipping down).
          3. Shoulder yaw aligns to +0.18 rad (holding the arm centered along the retreat corridor).
          4. Shoulder roll ensures slight clearance (>= 0.14 rad).
          5. Wrist pitch sets to -0.65 rad (slightly up-tilted to prevent finger hook).
          6. Gripper fully open (5.0).
          This achieves a clean horizontal backward translation (X retreats ~7-9 cm, Z rises ~3 cm, dip = 0.0 cm).
        - Stage 2 (To Initial Position):
          From this safely retracted waypoint, the subsequent interpolation moves the arm right (Y -> -0.14)
          and down (Z -> 0.83) to initial_position, completely avoiding the container boundaries.
        """
        waypoint = dict(current_pos)
        retracted_any = False

        # Right arm
        r_pitch_key = next((k for k in ("kRightShoulderPitch", "right_shoulder_pitch", "kRightShoulderPitch.q") if k in current_pos), None)
        r_roll_key = next((k for k in ("kRightShoulderRoll", "right_shoulder_roll", "kRightShoulderRoll.q") if k in current_pos), None)
        r_yaw_key = next((k for k in ("kRightShoulderYaw", "right_shoulder_yaw", "kRightShoulderYaw.q") if k in current_pos), None)
        r_elbow_key = next((k for k in ("kRightElbow", "right_elbow", "kRightElbow.q") if k in current_pos), None)
        r_wroll_key = next((k for k in ("kRightWristRoll", "right_wrist_roll", "kRightWristRoll.q") if k in current_pos), None)
        r_wpitch_key = next((k for k in ("kRightWristPitch", "right_wrist_pitch", "kRightWristPitch.q") if k in current_pos), None)
        r_wyaw_key = next((k for k in ("kRightWristYaw", "right_wrist_yaw", "kRightWristYaw.q") if k in current_pos), None)
        r_grip_key = next((k for k in ("kRightGripper", "gripper.right", "right_gripper") if k in current_pos), None)

        if r_pitch_key is not None and r_elbow_key is not None:
            curr_pitch = current_pos[r_pitch_key]
            curr_elbow = current_pos[r_elbow_key]
            # Arm is extended forward into container if shoulder pitch is forward (< -0.20) or elbow is bent (> 0.65)
            if curr_pitch < -0.20 or curr_elbow > 0.65:
                waypoint[r_pitch_key] = -0.38
                if r_roll_key is not None:
                    waypoint[r_roll_key] = max(current_pos.get(r_roll_key, 0.14), 0.14)
                if r_yaw_key is not None:
                    waypoint[r_yaw_key] = 0.18
                waypoint[r_elbow_key] = 0.42
                if r_wroll_key is not None:
                    waypoint[r_wroll_key] = 0.00
                if r_wpitch_key is not None:
                    waypoint[r_wpitch_key] = -0.65
                if r_wyaw_key is not None:
                    waypoint[r_wyaw_key] = 0.00
                if r_grip_key is not None:
                    waypoint[r_grip_key] = 5.0
                retracted_any = True

        # Left arm (symmetric)
        l_pitch_key = next((k for k in ("kLeftShoulderPitch", "left_shoulder_pitch", "kLeftShoulderPitch.q") if k in current_pos), None)
        l_roll_key = next((k for k in ("kLeftShoulderRoll", "left_shoulder_roll", "kLeftShoulderRoll.q") if k in current_pos), None)
        l_yaw_key = next((k for k in ("kLeftShoulderYaw", "left_shoulder_yaw", "kLeftShoulderYaw.q") if k in current_pos), None)
        l_elbow_key = next((k for k in ("kLeftElbow", "left_elbow", "kLeftElbow.q") if k in current_pos), None)
        l_wroll_key = next((k for k in ("kLeftWristRoll", "left_wrist_roll", "kLeftWristRoll.q") if k in current_pos), None)
        l_wpitch_key = next((k for k in ("kLeftWristPitch", "left_wrist_pitch", "kLeftWristPitch.q") if k in current_pos), None)
        l_wyaw_key = next((k for k in ("kLeftWristYaw", "left_wrist_yaw", "kLeftWristYaw.q") if k in current_pos), None)
        l_grip_key = next((k for k in ("kLeftGripper", "gripper.left", "left_gripper") if k in current_pos), None)

        if l_pitch_key is not None and l_elbow_key is not None:
            curr_pitch = current_pos[l_pitch_key]
            curr_elbow = current_pos[l_elbow_key]
            if curr_pitch < -0.20 or curr_elbow > 0.65:
                waypoint[l_pitch_key] = -0.38
                if l_roll_key is not None:
                    waypoint[l_roll_key] = min(current_pos.get(l_roll_key, -0.14), -0.14)
                if l_yaw_key is not None:
                    waypoint[l_yaw_key] = -0.18
                waypoint[l_elbow_key] = 0.42
                if l_wroll_key is not None:
                    waypoint[l_wroll_key] = 0.00
                if l_wpitch_key is not None:
                    waypoint[l_wpitch_key] = -0.65
                if l_wyaw_key is not None:
                    waypoint[l_wyaw_key] = 0.00
                if l_grip_key is not None:
                    waypoint[l_grip_key] = 5.0
                retracted_any = True

        return waypoint if retracted_any else None

    @staticmethod
    def _interpolate_motion(
        robot: Any,
        start_pos: dict[str, float],
        target_pos: dict[str, float],
        duration_s: float,
        fps: int = 50,
    ) -> None:
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            t = step / steps
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            interp = {}
            for k in target_pos:
                s_val = start_pos.get(k, target_pos[k])
                interp[k] = s_val * (1.0 - alpha) + target_pos[k] * alpha
            # Lock remote chassis axes to 0.0 during arm movement
            for remote_key in ("remote.lx", "remote.ly", "remote.rx", "remote.ry"):
                if remote_key in robot.action_features:
                    interp[remote_key] = 0.0
            robot.send_action(interp)
            precise_sleep(1 / fps)

    @classmethod
    def return_to_initial_position(
        cls,
        hw: HardwareContext,
        duration_s: float = 3.0,
        fps: int = 50,
        retract_first: bool = False,
        retract_duration_s: float = 1.2,
        target_override: dict[str, float] | None = None,
    ) -> bool:
        """Smoothly interpolate the robot back to its initial position using a cosine S-curve.

        If retract_first is True, the robot first retracts any forward-extended arm horizontally
        back toward the torso (avoiding collisions with boxes or containers), and then smoothly
        settles into the default initial position.
        If target_override is provided, those key-value pairs override corresponding joint targets
        (e.g., maintaining the clamped gripper position during homing).

        Returns ``True`` when the interpolation completed, ``False`` when it failed
        partway — the robot is then at an arbitrary pose, so callers must not report
        a completed reset on ``False``.
        """
        robot = hw.robot_wrapper
        target = dict(hw.initial_position) if hw.initial_position else {}
        if target_override:
            target.update(target_override)
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}

            if retract_first:
                retract_waypoint = cls._compute_retract_waypoint(current_pos)
                if retract_waypoint is not None:
                    logger.info("Executing horizontal arm retraction (%.1fs) before homing...", retract_duration_s)
                    cls._interpolate_motion(
                        robot, current_pos, retract_waypoint, duration_s=retract_duration_s, fps=fps
                    )
                    current_pos = retract_waypoint

            cls._interpolate_motion(robot, current_pos, target, duration_s=duration_s, fps=fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)
            return False
        return True

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
        interpolator: ActionInterpolator | None = None,
    ) -> None:
        """Log observation/action telemetry to the visualization backend if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        if interpolator is not None and interpolator.enabled and not interpolator.emitted_policy_action:
            # Skip heavy full-frame image logging on interpolated sub-ticks to protect control loop timing
            return
        try:
            log_visualization_data(
                cfg.display_mode,
                observation=obs_processed,
                action=action_dict,
                compress_images=cfg.display_compressed_images,
            )
        except Exception as e:
            logger.warning("Failed to log telemetry data: %s", e)

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires.

        Implementations must call ``engine.resume()`` before entering their loop
        (async backends start paused, and the interactive controller pauses again at
        the end of every segment), and ``engine.pump_query(obs_processed)`` at the end
        of every tick — the text-query channel only advances through it, and a
        multi-second generation must not sit inside the action path.

        Each ``run()`` call builds its own ``CycleTimer`` and reports it through
        ``timer.log_run_summary()`` from its ``finally``: a fresh timer's start-up
        exemption is what absorbs the interpolator that ``reset_control_state()``
        re-primes at every ``/start``, and each segment gets its own cadence report.
        """

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: finalize dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
    timer: CycleTimer | None = None,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    When *timer* is given, the engine pull and the robot send are timed as the
    ``infer`` and ``send`` steps of its cadence summary, and a tick with no action
    to send is counted there.  Note that on async backends ``infer`` is only a
    queue pull — inference runs off-thread, so its latency surfaces as starved
    ticks rather than as loop-body time.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys
    # ``nullcontext`` accepts (and ignores) the section name, so it stands in for
    # ``timer.section`` verbatim when no timer was passed.
    section = timer.section if timer is not None else contextlib.nullcontext

    if interpolator.needs_new_action():
        with section("infer"):
            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            action_tensor = engine.get_action(obs_frame)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        if timer is not None:
            timer.note_starved_tick()
        if interpolator._prev is not None:
            interp = interpolator._prev
        else:
            return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    with section("send"):
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        ctx.hardware.robot_wrapper.send_action(processed)
    return action_dict
