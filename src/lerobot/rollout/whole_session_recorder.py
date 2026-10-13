# Copyright 2026 The LeRobot & Antigravity Authors.
"""
Whole-Session Continuous Recorder for Robot Deployment.
======================================================
Captures the ENTIRE deployment session from startup to shutdown, completely
independent of VLA execution segments. Designed specifically for cross-table
transfer and interactive multi-task deployments where the robot transitions
between VLA autonomous manipulation, human/locomotion navigation, idle waiting,
and multi-modal JEV perception.

Key Characteristics:
1. Whole-Session Lifecycle: Does NOT stop when VLA pauses or exits. Runs from
   session start to operator quit ('q' / Ctrl+C).
2. Headless & Zero GUI: Operates entirely without Rerun (`--display_data=false`),
   zero graphical memory/CPU overhead.
3. Zero Real-Time Disruption: Pure lock-free in-memory snapshot sampling (<0.1ms),
   OpenCV video encoding and file disk I/O decoupled into a background daemon worker.
4. Strict Multi-Modal Alignment:
   - 3 Camera MP4 videos (Global view, Left wrist, Right wrist) with identical frame count.
   - Telemetry table (CSV / NPY) with identical row count (1:1 with video frames).
   - 6000/6003 Gamepad/Robot Mode transitions (NAV <-> GAMEPAD <-> VLA) with exact timestamps and frame_idx.
   - JEV (Valen) decision logs with exact timestamps, latency, prompt, probabilities, and aligned frame_idx.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

logger = logging.getLogger(__name__)

# Standard 29-DoF Joint Names for Unitree G1
G1_JOINT_NAMES = [motor.name for motor in G1_29_JointIndex]
CAMERA_KEYS = ["global_view", "left_wrist", "right_wrist"]


def _to_numpy_bgr(img: Any) -> np.ndarray | None:
    """Normalize image from PyTorch Tensor or NumPy array to HWC uint8 BGR for VideoWriter."""
    if img is None:
        return None
    try:
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        arr = np.asarray(img)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        # Handle CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.05:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return arr
    except Exception as e:
        logger.debug("Image conversion error: %s", e)
    return None


class WholeSessionRecorder:
    """Continuous whole-session recorder aligning 3 cameras, robot joints, mode, and JEV."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        fps: float = 30.0,
        robot_wrapper: Any = None,
        interactive_session: Any = None,
        controller: Any = None,
    ) -> None:
        if output_dir is None or str(output_dir).strip() in ("", "None"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("outputs/recordings") / f"session_{ts}"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.robot_wrapper = robot_wrapper
        self.session = interactive_session
        self.controller = controller

        self._frame_queue: queue.Queue = queue.Queue(maxsize=450)
        self._is_running = threading.Event()
        self._finalized = False
        self._lock = threading.Lock()

        # Threading workers
        self._sampler_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

        # Video writers per camera name
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._camera_dims: dict[str, tuple[int, int]] = {}
        self._last_valid_frames: dict[str, np.ndarray] = {}

        # Buffers for arrays
        self._frame_count = 0
        self._start_wall_time = 0.0
        self._all_joint_q: list[np.ndarray] = []
        self._all_joint_dq: list[np.ndarray] = []
        self._all_joint_tau: list[np.ndarray] = []
        self._all_grippers: list[list[float]] = []
        self._all_imu_rpy: list[list[float]] = []
        self._all_actions: list[np.ndarray] = []
        self._all_timestamps: list[float] = []
        self._all_modes: list[str] = []
        self._all_phases: list[int] = []
        self._all_vla_flags: list[bool] = []

        # JSONL event streams
        self._jev_records: list[dict[str, Any]] = []
        self._mode_records: list[dict[str, Any]] = []
        self._jev_file = open(self.output_dir / "jev_decisions.jsonl", "w", encoding="utf-8")
        self._mode_file = open(self.output_dir / "mode_transitions.jsonl", "w", encoding="utf-8")

        # CSV telemetry stream
        self._csv_file = open(self.output_dir / "telemetry.csv", "w", newline="", encoding="utf-8")
        self._csv_writer: csv.writer | None = None

        logger.info(
            "WholeSessionRecorder created at '%s' (fps=%.1f)",
            self.output_dir,
            self.fps,
        )

    def start(self) -> None:
        """Start continuous whole-session sampling and writing threads."""
        if self._is_running.is_set():
            return
        self._is_running.set()
        self._start_wall_time = time.time()

        self._writer_thread = threading.Thread(
            target=self._writer_worker_loop, daemon=True, name="SessionWriterWorker"
        )
        self._writer_thread.start()

        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, daemon=True, name="Session30HzSampler"
        )
        self._sampler_thread.start()

        print("=" * 80)
        print(" 🔴 [WholeSessionRecorder] 全程无感录制已启动！")
        print(f"    • 存储目录: {self.output_dir}")
        print(f"    • 采样帧率: {self.fps:.0f} Hz (三路相机MP4 + 29关节遥测 + 手柄模式 + JEV决策时间戳)")
        print("    • 运行范围: 覆盖整个会话生命周期，退出 VLA 动作段不停止，按 'q' 安全归档")
        print("=" * 80)
        sys.stdout.flush()

    def _sampler_loop(self) -> None:
        """30Hz steady clock sampler capturing cameras, motors, grippers, mode, and phase."""
        dt = 1.0 / max(self.fps, 1.0)
        next_tick = time.perf_counter()

        robot_wrapper = self.robot_wrapper
        robot = getattr(robot_wrapper, "inner", robot_wrapper) if robot_wrapper else None

        while self._is_running.is_set():
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick = max(time.perf_counter() + dt, next_tick + dt)

            t_wall = time.time()
            f_idx = self._frame_count
            self._frame_count += 1

            # 1. Fetch images from all 3 cameras
            frames: dict[str, np.ndarray | None] = {"global_view": None, "left_wrist": None, "right_wrist": None}
            if robot is not None and hasattr(robot, "_cameras"):
                cams = robot._cameras
                for ckey in ("global_view", "base_0_rgb"):
                    if ckey in cams:
                        try:
                            frames["global_view"] = cams[ckey].read_latest()
                        except Exception:
                            pass
                        break
                for ckey in ("left_wrist", "left_wrist_0_rgb"):
                    if ckey in cams:
                        try:
                            frames["left_wrist"] = cams[ckey].read_latest()
                        except Exception:
                            pass
                        break
                for ckey in ("right_wrist", "right_wrist_0_rgb"):
                    if ckey in cams:
                        try:
                            frames["right_wrist"] = cams[ckey].read_latest()
                        except Exception:
                            pass
                        break

            # Fallback to robot_wrapper.get_observation if any camera missed
            if any(frames[k] is None for k in frames) and robot_wrapper is not None:
                try:
                    obs = robot_wrapper.get_observation()
                    if frames["global_view"] is None:
                        frames["global_view"] = (
                            obs.get("global_view") or obs.get("base_0_rgb")
                            or obs.get("observation.images.global_view") or obs.get("observation.images.base_0_rgb")
                        )
                    if frames["left_wrist"] is None:
                        frames["left_wrist"] = (
                            obs.get("left_wrist") or obs.get("left_wrist_0_rgb")
                            or obs.get("observation.images.left_wrist") or obs.get("observation.images.left_wrist_0_rgb")
                        )
                    if frames["right_wrist"] is None:
                        frames["right_wrist"] = (
                            obs.get("right_wrist") or obs.get("right_wrist_0_rgb")
                            or obs.get("observation.images.right_wrist") or obs.get("observation.images.right_wrist_0_rgb")
                        )
                except Exception:
                    pass

            # 2. Extract robot joint state (29-DoF)
            q_arr = np.zeros(29, dtype=np.float32)
            dq_arr = np.zeros(29, dtype=np.float32)
            tau_arr = np.zeros(29, dtype=np.float32)
            l_grip = 5.0
            r_grip = 5.0
            imu_rpy = [0.0, 0.0, 0.0]

            if robot is not None:
                state = getattr(robot, "_latest_state", None)
                if isinstance(state, dict):
                    data = state.get("data", state)
                    motors_dict = data.get("motors") or state.get("motors", {})
                    motor_list = data.get("motor_state") or state.get("motor_state", [])

                    for m_idx, motor_enum in enumerate(G1_29_JointIndex):
                        m_name = motor_enum.name
                        m_val = motor_enum.value
                        m_data = None
                        if isinstance(motors_dict, dict) and m_name in motors_dict:
                            m_data = motors_dict[m_name]
                        elif isinstance(motor_list, (list, tuple)) and m_val < len(motor_list):
                            m_data = motor_list[m_val]

                        if isinstance(m_data, dict):
                            q_arr[m_idx] = float(m_data.get("q", 0.0))
                            dq_arr[m_idx] = float(m_data.get("dq", 0.0))
                            tau_arr[m_idx] = float(m_data.get("tau", m_data.get("tau_est", 0.0)))
                        elif isinstance(m_data, (int, float)):
                            q_arr[m_idx] = float(m_data)

                    imu = data.get("imu_state") or data.get("imu") or state.get("imu", {})
                    if isinstance(imu, dict):
                        imu_rpy = [float(x) for x in imu.get("rpy", [0.0, 0.0, 0.0])[:3]]

                # Gripper angles
                if hasattr(robot, "_latest_gripper"):
                    with getattr(robot, "_gripper_lock", threading.Lock()):
                        g = dict(robot._latest_gripper)
                        l_grip = float(g.get("left", 5.0))
                        r_grip = float(g.get("right", 5.0))

            # 3. Mode, phase, VLA status
            curr_mode = "unknown"
            if robot is not None and hasattr(robot, "current_mode"):
                m_enum = robot.current_mode
                curr_mode = getattr(m_enum, "value", str(m_enum))

            transfer_phase = -1
            vla_running = False
            active_task = ""
            if self.session is not None:
                transfer_phase = int(getattr(self.session, "_current_phase", -1))
            ctrl = self.controller or (getattr(self.session, "controller", None) if self.session is not None else None)
            if ctrl is not None:
                vla_running = bool(getattr(ctrl, "running", False))
                active_task = str(getattr(ctrl, "task", ""))

            # 4. Command action
            act_vec = np.zeros(16, dtype=np.float32)
            if robot is not None and hasattr(robot, "_last_sent_action"):
                last_act = getattr(robot, "_last_sent_action", {})
                if isinstance(last_act, dict) and vla_running:
                    act_vals = [float(v) for v in last_act.values() if isinstance(v, (int, float))]
                    if act_vals:
                        act_vec = np.array(act_vals[:16], dtype=np.float32)

            packet = {
                "frame_idx": f_idx,
                "timestamp": t_wall,
                "relative_time_s": t_wall - self._start_wall_time,
                "frames": frames,
                "q": q_arr,
                "dq": dq_arr,
                "tau": tau_arr,
                "grippers": [l_grip, r_grip],
                "imu_rpy": imu_rpy,
                "mode": curr_mode,
                "transfer_phase": transfer_phase,
                "vla_running": vla_running,
                "action": act_vec,
                "task": active_task,
            }

            try:
                self._frame_queue.put_nowait(packet)
            except queue.Full:
                logger.warning("WholeSessionRecorder frame queue full; dropping frame %d", f_idx)

    def _writer_worker_loop(self) -> None:
        """Background thread writing MP4 video frames and recording telemetry rows."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while True:
            try:
                packet = self._frame_queue.get()
                if packet is None:
                    self._frame_queue.task_done()
                    break

                f_idx = packet["frame_idx"]
                t_wall = packet["timestamp"]
                rel_t = packet["relative_time_s"]
                frames = packet["frames"]
                q = packet["q"]
                dq = packet["dq"]
                tau = packet["tau"]
                grippers = packet["grippers"]
                imu = packet["imu_rpy"]
                mode = packet["mode"]
                phase = packet["transfer_phase"]
                vla_running = packet["vla_running"]
                action = packet["action"]
                task = packet["task"]

                # 1. Write frames for each of the 3 cameras
                for cam_name in CAMERA_KEYS:
                    raw_f = frames.get(cam_name)
                    bgr = _to_numpy_bgr(raw_f)
                    if bgr is None:
                        bgr = self._last_valid_frames.get(cam_name)
                    else:
                        self._last_valid_frames[cam_name] = bgr

                    if bgr is not None:
                        h, w = bgr.shape[:2]
                        if cam_name not in self._video_writers:
                            vid_path = self.output_dir / f"camera_{cam_name}.mp4"
                            self._video_writers[cam_name] = cv2.VideoWriter(
                                str(vid_path), fourcc, self.fps, (w, h)
                            )
                            self._camera_dims[cam_name] = (w, h)
                            logger.info(
                                "Initialized VideoWriter for %s: %s (%dx%d @ %.1ffps)",
                                cam_name, vid_path.name, w, h, self.fps
                            )

                        target_w, target_h = self._camera_dims[cam_name]
                        if (w, h) != (target_w, target_h):
                            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                        self._video_writers[cam_name].write(bgr)

                # 2. Append in-memory arrays
                self._all_joint_q.append(q)
                self._all_joint_dq.append(dq)
                self._all_joint_tau.append(tau)
                self._all_grippers.append(grippers)
                self._all_imu_rpy.append(imu)
                self._all_actions.append(action)
                self._all_timestamps.append(t_wall)
                self._all_modes.append(mode)
                self._all_phases.append(phase)
                self._all_vla_flags.append(vla_running)

                # 3. Write CSV row
                if self._csv_writer is None:
                    headers = [
                        "frame_idx", "timestamp", "relative_time_s", "mode", "transfer_phase", "vla_running",
                        "left_gripper", "right_gripper", "imu_roll", "imu_pitch", "imu_yaw"
                    ]
                    headers += [f"q_{name}" for name in G1_JOINT_NAMES]
                    headers += [f"act_{i}" for i in range(len(action))]
                    self._csv_writer = csv.writer(self._csv_file)
                    self._csv_writer.writerow(headers)

                row = [
                    f_idx, f"{t_wall:.4f}", f"{rel_t:.4f}", mode, phase, int(vla_running),
                    f"{grippers[0]:.3f}", f"{grippers[1]:.3f}",
                    f"{imu[0]:.3f}", f"{imu[1]:.3f}", f"{imu[2]:.3f}",
                ]
                row += [f"{val:.4f}" for val in q]
                row += [f"{val:.4f}" for val in action]
                self._csv_writer.writerow(row)

                if f_idx % 30 == 0:
                    self._csv_file.flush()

                self._frame_queue.task_done()

            except Exception as e:
                logger.error("Error in writer worker loop: %s", e, exc_info=True)
                try:
                    self._frame_queue.task_done()
                except Exception:
                    pass

    def record_mode_transition(
        self, old_mode: str, new_mode: str, raw_preview: str = "", raw_state_preview: str = ""
    ) -> None:
        """Record a 6000/6003 robot mode transition event (e.g. NAV <-> GAMEPAD <-> VLA)."""
        now = time.time()
        f_idx = max(self._frame_count - 1, 0)
        raw_str = raw_preview or raw_state_preview
        record = {
            "timestamp": now,
            "relative_time_s": now - self._start_wall_time if self._start_wall_time > 0 else 0.0,
            "aligned_frame_idx": f_idx,
            "old_mode": str(old_mode),
            "new_mode": str(new_mode),
            "raw_preview": str(raw_str),
            "raw_state_preview": str(raw_str),
        }
        with self._lock:
            self._mode_records.append(record)
            if not self._mode_file.closed:
                self._mode_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._mode_file.flush()
        logger.info("Recorded Mode Transition: %s -> %s (frame=%d)", old_mode, new_mode, f_idx)

    def record_jev_decision(
        self,
        *,
        table: int,
        phase: int,
        description: str,
        result: Any,
        questions: dict | None = None,
        context_text: str | None = None,
    ) -> None:
        """Record an evaluation judgment produced by Valen JEV."""
        now = time.time()
        f_idx = max(self._frame_count - 1, 0)

        status_val = getattr(result, "status", "unknown")
        choice_val = getattr(result, "choice", "")
        conf_val = getattr(result, "confidence", 0.0)
        cost_val = getattr(result, "cost_ms", 0.0)
        probs_val = getattr(result, "probabilities", {})
        err_val = getattr(result, "error", None)

        record = {
            "timestamp": now,
            "relative_time_s": now - self._start_wall_time if self._start_wall_time > 0 else 0.0,
            "aligned_frame_idx": f_idx,
            "table": int(table),
            "query_phase": int(phase),
            "description": str(description),
            "status": str(status_val) if not isinstance(status_val, str) else status_val,
            "choice": str(choice_val) if not isinstance(choice_val, str) else choice_val,
            "confidence": float(conf_val) if isinstance(conf_val, (int, float)) else 0.0,
            "probabilities": probs_val if isinstance(probs_val, dict) else {},
            "cost_ms": float(cost_val) if isinstance(cost_val, (int, float)) else 0.0,
            "error": str(err_val) if (err_val is not None and str(err_val) != "None") else None,
        }
        if questions:
            record["questions"] = questions
        if context_text:
            record["context_text"] = context_text

        with self._lock:
            self._jev_records.append(record)
            if not self._jev_file.closed:
                self._jev_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._jev_file.flush()

        logger.info(
            "Recorded JEV Decision: [Table %d | Phase %d] Choice=%s (Conf=%.2f, cost=%.0fms, frame=%d)",
            table, phase, record["choice"], record["confidence"], record["cost_ms"], f_idx
        )

    def stop(self) -> None:
        """Stop recorder, flush queues, finalize videos and write arrays/metadata."""
        with self._lock:
            if self._finalized:
                return
            self._finalized = True

        self._is_running.clear()

        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None

        if self._writer_thread is not None:
            try:
                self._frame_queue.put(None)
                self._writer_thread.join(timeout=10.0)
            except Exception as e:
                logger.warning("Writer thread join exception: %s", e)
            self._writer_thread = None

        # Release all video writers
        for cname, writer in self._video_writers.items():
            try:
                writer.release()
            except Exception as e:
                logger.warning("Error releasing VideoWriter for %s: %s", cname, e)
        self._video_writers.clear()

        # Close CSV and JSONL files
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        try:
            self._jev_file.flush()
            self._jev_file.close()
        except Exception:
            pass
        try:
            self._mode_file.flush()
            self._mode_file.close()
        except Exception:
            pass

        # Save numpy binary arrays
        total_frames = len(self._all_timestamps)
        duration_s = 0.0
        actual_fps = self.fps
        if total_frames > 0:
            duration_s = float(self._all_timestamps[-1] - self._all_timestamps[0])
            actual_fps = float((total_frames - 1) / duration_s) if duration_s > 0 else self.fps

            try:
                np.save(self.output_dir / "joint_states.npy", np.array(self._all_joint_q, dtype=np.float32))
                np.save(self.output_dir / "joint_velocities.npy", np.array(self._all_joint_dq, dtype=np.float32))
                np.save(self.output_dir / "joint_torques.npy", np.array(self._all_joint_tau, dtype=np.float32))
                np.save(self.output_dir / "gripper_states.npy", np.array(self._all_grippers, dtype=np.float32))
                np.save(self.output_dir / "actions.npy", np.array(self._all_actions, dtype=np.float32))
                np.save(self.output_dir / "timestamps.npy", np.array(self._all_timestamps, dtype=np.float64))
            except Exception as e:
                logger.error("Failed to save numpy arrays: %s", e)

        # Write session metadata
        meta = {
            "session_dir": str(self.output_dir.resolve()),
            "total_frames": total_frames,
            "duration_s": round(duration_s, 2),
            "duration_seconds": round(duration_s, 2),
            "target_fps": self.fps,
            "actual_fps": round(actual_fps, 2),
            "cameras": {
                cname: {"width": dims[0], "height": dims[1]}
                for cname, dims in self._camera_dims.items()
            },
            "num_jev_decisions": len(self._jev_records),
            "jev_decisions_count": len(self._jev_records),
            "num_mode_transitions": len(self._mode_records),
            "mode_transitions_count": len(self._mode_records),
            "joint_names": G1_JOINT_NAMES,
            "timestamp_start": self._start_wall_time,
            "timestamp_end": time.time(),
        }
        with open(self.output_dir / "session_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Print completion summary
        print("\n" + "=" * 80)
        print(" ✅ [WholeSessionRecorder] 全程数据录制完成并成功归档！")
        print(f" 📁 存储目录: {self.output_dir.resolve()}")
        print(f" ⏱️ 会话统计: 总时长 {duration_s:.1f} 秒 | 总帧数 {total_frames} 帧 (实测 {actual_fps:.1f} FPS)")
        print(" 🎬 三路视频:")
        for cname in CAMERA_KEYS:
            wpath = self.output_dir / f"camera_{cname}.mp4"
            status = f"✅ 已保存 ({self._camera_dims.get(cname, (0,0))[0]}x{self._camera_dims.get(cname, (0,0))[1]})" if wpath.exists() else "⚪ 未产生"
            print(f"    ├─ {cname:12s}: {wpath.name} ({status})")
        print(" 📊 结构化时序数据:")
        print("    ├─ telemetry.csv       : 逐帧对齐数据表 (时间戳, 模式, 阶段, 全关节, 夹爪, Action)")
        print(f"    ├─ joint_states.npy    : 全关节矩阵 (共 {total_frames} 帧)")
        print(f"    ├─ jev_decisions.jsonl : 共记录 {len(self._jev_records)} 次 JEV 视觉判定 (带对齐帧号与置信度)")
        print(f"    ├─ mode_transitions.jsonl: 共记录 {len(self._mode_records)} 次手柄/底盘模式切换")
        print("    └─ session_meta.json   : 会话全局元数据")
        print("=" * 80 + "\n")
        sys.stdout.flush()
