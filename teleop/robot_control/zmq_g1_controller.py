import ast
import json
import logging
import re
import threading
import time
from enum import IntEnum
from typing import Any, Tuple

import numpy as np
import zmq

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

from teleop.robot_control.robot_arm import (
    G1_29_JointIndex,
    G1_29_JointArmIndex,
    G1_29_JointLegIndex,
)

NUM_MOTORS = 29
ARM_NUM_MOTORS = 14
LEG_NUM_MOTORS = 12

REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")


def parse_gripper_data(raw_data: Any) -> Tuple[float, float] | None:
    """Parse raw bytes, string, multipart list, or dict received on gripper_port into (left, right).
    Aligned with src/lerobot/robots/unitree_g1/unitree_g1_client.py.
    """
    try:
        if isinstance(raw_data, (list, tuple)):
            for part in reversed(raw_data):
                parsed = parse_gripper_data(part)
                if parsed is not None:
                    return parsed
            joined = b" ".join(part if isinstance(part, bytes) else str(part).encode() for part in raw_data)
            return parse_gripper_data(joined)

        if isinstance(raw_data, bytes):
            s = raw_data.decode("utf-8", errors="ignore").strip().strip("\x00\r\n\t ")
        else:
            s = str(raw_data).strip().strip("\x00\r\n\t ")
        if not s:
            return None

        obj = None
        if "{" in s and "}" in s:
            first_b = s.find("{")
            last_b = s.rfind("}")
            candidate = s[first_b : last_b + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                try:
                    obj = ast.literal_eval(candidate)
                except Exception:
                    obj = None

        def _extract_val(v: Any) -> float:
            if isinstance(v, dict):
                for k in ("q", "position", "pos", "angle", "rad", "val", "value"):
                    if k in v:
                        return float(v[k])
                if v:
                    first_v = next(iter(v.values()))
                    if isinstance(first_v, (int, float)):
                        return float(first_v)
                return 0.0
            elif isinstance(v, (list, tuple)):
                return float(v[0])
            return float(v)

        def extract_from_dict(d: Any) -> Tuple[float, float] | None:
            if not isinstance(d, dict):
                return None
            if "data" in d:
                sub = extract_from_dict(d["data"])
                if sub is not None:
                    return sub
                if isinstance(d["data"], (list, tuple)) and len(d["data"]) >= 2:
                    return float(d["data"][0]), float(d["data"][1])
            if "gripper" in d:
                sub = extract_from_dict(d["gripper"])
                if sub is not None:
                    return sub
            if "left" in d or "right" in d:
                l = _extract_val(d.get("left", d.get("kLeftGripper", 0.0)))
                r = _extract_val(d.get("right", d.get("kRightGripper", 0.0)))
                return l, r
            return None

        if isinstance(obj, dict):
            res = extract_from_dict(obj)
            if res is not None:
                return res

        m_left = re.search(r"left[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        m_right = re.search(r"right[\'\"]?\s*[:=]\s*([+-]?\d*\.?\d+)", s, re.IGNORECASE)
        if m_left or m_right:
            l = float(m_left.group(1)) if m_left else 0.0
            r = float(m_right.group(1)) if m_right else 0.0
            return l, r
    except Exception as e:
        logger_mp.debug(f"Error parsing gripper data: {e}")
    return None


class G1_29_ZMQ_Controller:
    """ZeroMQ-based controller for Unitree G1 (29-DoF) + Dex1 gripper.
    
    Provides the same API interface as G1_29_ArmController and Dex1_1_Gripper_Controller,
    allowing seamless drop-in replacement with standard LeRobot VLA communication:
      - Subscribes to 29-DoF lowstate on tcp://{robot_ip}:{state_port} (default 6001)
      - Pushes 14-DoF arms + chassis velocity + gripper action to tcp://{robot_ip}:{action_port} (default 6002)
      - Subscribes to {left, right} gripper state on tcp://{robot_ip}:{gripper_port} (default 6004)
    """

    def __init__(
        self,
        robot_ip: str = "192.168.123.164",
        action_ip: str = "",
        state_port: int = 6001,
        action_port: int = 6002,
        gripper_port: int = 6004,
        frequency: float = 30.0,
        enable_gripper: bool = True,
        left_gripper_value_in = None,
        right_gripper_value_in = None,
        dual_gripper_data_lock = None,
        dual_gripper_state_array = None,
        dual_gripper_action_array = None,
        input_mode: str = "controller",
        gripper_min: float = 2.70,
        gripper_max: float = 5.00,
        delta_gripper_open: float = 0.15,
        delta_gripper_close: float = 0.08,
    ):
        logger_mp.info(f"Initialize G1_29_ZMQ_Controller (IP={robot_ip}, State={state_port}, Action={action_port}, Gripper={gripper_port}, Range=[{gripper_min}, {gripper_max}], OpenStep={delta_gripper_open}, CloseStep={delta_gripper_close})...")
        self.robot_ip = robot_ip
        self.action_ip = action_ip if action_ip else robot_ip
        self.state_port = state_port
        self.action_port = action_port
        self.gripper_port = gripper_port
        self.frequency = frequency
        self.control_dt = 1.0 / frequency
        self.enable_gripper = enable_gripper
        self.input_mode = input_mode

        # Gripper shared buffers from teleop_hand_and_arm.py
        self.left_gripper_value_in = left_gripper_value_in
        self.right_gripper_value_in = right_gripper_value_in
        self.dual_gripper_data_lock = dual_gripper_data_lock
        self.dual_gripper_state_array = dual_gripper_state_array
        self.dual_gripper_action_array = dual_gripper_action_array

        self.arm_velocity_limit = 10.0

        # State storage
        self._state_lock = threading.Lock()
        self._latest_state = None
        self._all_motor_q = np.zeros(NUM_MOTORS)
        self._all_motor_dq = np.zeros(NUM_MOTORS)

        # Gripper state
        self._gripper_lock = threading.Lock()
        self._latest_gripper = {"left": 0.0, "right": 0.0}
        self._first_gripper_packet = False

        # Target command storage
        self.ctrl_lock = threading.Lock()
        self.q_target = np.zeros(ARM_NUM_MOTORS)
        self.tauff_target = np.zeros(ARM_NUM_MOTORS)
        self.chassis_action = [0.0, 0.0, 0.0]  # [vx, vy, vyaw]
        self.gripper_action = {"left": gripper_max, "right": gripper_max}

        # Gripper mapping parameters (Dex1: 2.7 rad = close, 5.0 rad = open)
        self.LEFT_MAPPED_MIN = gripper_min
        self.RIGHT_MAPPED_MIN = gripper_min
        self.LEFT_MAPPED_MAX = gripper_max
        self.RIGHT_MAPPED_MAX = gripper_max
        self.delta_gripper_open = delta_gripper_open
        self.delta_gripper_close = delta_gripper_close
        self.current_gripper_cmd = np.array([gripper_max, gripper_max], dtype=float)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._shutdown_event = threading.Event()

        # 1. State Subscriber (Port 6001)
        self._state_sub = self._ctx.socket(zmq.SUB)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._state_sub.connect(f"tcp://{self.robot_ip}:{self.state_port}")

        # 2. Action Pusher (Port 6002)
        self._action_push = self._ctx.socket(zmq.PUSH)
        self._action_push.setsockopt(zmq.CONFLATE, 1)
        self._action_push.connect(f"tcp://{self.action_ip}:{self.action_port}")

        # 3. Gripper Subscriber (Port 6004)
        if self.enable_gripper and self.gripper_port > 0:
            self._gripper_sub = self._ctx.socket(zmq.SUB)
            self._gripper_sub.setsockopt(zmq.CONFLATE, 1)
            self._gripper_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self._gripper_sub.connect(f"tcp://{self.robot_ip}:{self.gripper_port}")
        else:
            self._gripper_sub = None

        # Background threads
        self._state_thread = threading.Thread(target=self._state_reader_loop, daemon=True)
        self._state_thread.start()

        if self._gripper_sub is not None:
            self._gripper_thread = threading.Thread(target=self._gripper_reader_loop, daemon=True)
            self._gripper_thread.start()
        else:
            self._gripper_thread = None

        self._publish_thread = threading.Thread(target=self._ctrl_action_loop, daemon=True)
        self._publish_thread.start()

        # Wait briefly for initial state connection
        self._wait_for_initial_state()
        logger_mp.info("Initialize G1_29_ZMQ_Controller OK!")

    def _wait_for_initial_state(self, timeout: float = 3.0):
        start_t = time.time()
        while time.time() - start_t < timeout:
            with self._state_lock:
                if self._latest_state is not None:
                    logger_mp.info("✅ G1_29_ZMQ_Controller received initial robot state successfully.")
                    return
            time.sleep(0.05)
        logger_mp.warning(f"⚠️ G1_29_ZMQ_Controller: No state received from tcp://{self.robot_ip}:{self.state_port} within {timeout}s. Will keep waiting in background.")

    def _state_reader_loop(self):
        while not self._shutdown_event.is_set():
            if self._state_sub is None:
                time.sleep(0.05)
                continue
            try:
                if self._state_sub.poll(timeout=100):
                    payload = self._state_sub.recv(zmq.NOBLOCK)
                    msg = json.loads(payload.decode("utf-8"))
                    self._parse_and_update_state(msg)
            except zmq.ContextTerminated:
                break
            except (zmq.ZMQError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger_mp.debug(f"Exception in _state_reader_loop: {e}")
                time.sleep(0.05)

    def _parse_and_update_state(self, msg: dict):
        data = msg.get("data", msg) if isinstance(msg, dict) else {}
        motors_dict = data.get("motors") or (msg.get("motors") if isinstance(msg, dict) else None)
        motor_list = data.get("motor_state") or (msg.get("motor_state") if isinstance(msg, dict) else None)

        q_arr = np.zeros(NUM_MOTORS)
        dq_arr = np.zeros(NUM_MOTORS)

        for motor in G1_29_JointIndex:
            idx = motor.value
            if idx >= NUM_MOTORS:
                continue
            m_data = {}
            if motors_dict and motor.name in motors_dict:
                m_data = motors_dict[motor.name]
            elif motor_list and idx < len(motor_list):
                m_data = motor_list[idx]

            if isinstance(m_data, dict):
                q_arr[idx] = float(m_data.get("q", 0.0))
                dq_arr[idx] = float(m_data.get("dq", 0.0))
            elif isinstance(m_data, (int, float)):
                q_arr[idx] = float(m_data)

        is_first_state = False
        with self._state_lock:
            if self._latest_state is None:
                is_first_state = True
            self._latest_state = msg
            self._all_motor_q = q_arr
            self._all_motor_dq = dq_arr

        if is_first_state:
            arm_indices = [member.value for member in G1_29_JointArmIndex]
            init_arm_q = q_arr[arm_indices]
            if np.any(init_arm_q != 0.0):
                with self.ctrl_lock:
                    self.q_target = init_arm_q.copy()

        # Check if gripper state is bundled in the state packet
        if "gripper" in data or "gripper" in msg:
            g_obj = data.get("gripper") or msg.get("gripper")
            if isinstance(g_obj, dict):
                with self._gripper_lock:
                    if "left" in g_obj:
                        self._latest_gripper["left"] = float(g_obj["left"])
                    if "right" in g_obj:
                        self._latest_gripper["right"] = float(g_obj["right"])
                self._sync_gripper_to_shared_arrays()

    def _gripper_reader_loop(self):
        while not self._shutdown_event.is_set():
            sock = self._gripper_sub
            if sock is None:
                time.sleep(0.05)
                continue
            try:
                if sock.poll(timeout=100):
                    newest_parts = None
                    while True:
                        try:
                            parts = sock.recv_multipart(zmq.NOBLOCK)
                            newest_parts = parts
                        except zmq.Again:
                            break
                    if newest_parts is None:
                        continue

                    parsed = parse_gripper_data(newest_parts)
                    if parsed is not None:
                        l_val, r_val = parsed
                        with self._gripper_lock:
                            self._latest_gripper["left"] = float(l_val)
                            self._latest_gripper["right"] = float(r_val)
                        self._sync_gripper_to_shared_arrays()
                        if not self._first_gripper_packet:
                            self._first_gripper_packet = True
                            logger_mp.info(f"📡 [ZMQ Gripper] First packet received on port {self.gripper_port}: left={l_val:.2f}, right={r_val:.2f}")
            except zmq.ContextTerminated:
                break
            except (zmq.ZMQError, OSError):
                time.sleep(0.05)
            except Exception as e:
                logger_mp.debug(f"Exception in _gripper_reader_loop: {e}")
                time.sleep(0.05)

    def _sync_gripper_to_shared_arrays(self):
        """Sync latest gripper state and action to shared multiprocessing arrays if provided."""
        if self.dual_gripper_state_array is not None and self.dual_gripper_data_lock is not None:
            with self.dual_gripper_data_lock:
                self.dual_gripper_state_array[0] = self._latest_gripper["left"]
                self.dual_gripper_state_array[1] = self._latest_gripper["right"]
                self.dual_gripper_action_array[0] = self.current_gripper_cmd[0]
                self.dual_gripper_action_array[1] = self.current_gripper_cmd[1]

    def _ctrl_action_loop(self):
        DELTA_GRIPPER_OPEN = self.delta_gripper_open    # Faster release (default 0.15: 4.5 rad/s at 30Hz, ~0.51s for full 2.3 rad open)
        DELTA_GRIPPER_CLOSE = self.delta_gripper_close  # Smooth closing (default 0.08: 2.4 rad/s at 30Hz, ~0.95s for full close)
        while not self._shutdown_event.is_set():
            start_t = time.time()

            # 1. Update gripper target from shared XR input if available
            if self.left_gripper_value_in is not None and self.right_gripper_value_in is not None:
                with self.left_gripper_value_in.get_lock():
                    l_in = self.left_gripper_value_in.value
                with self.right_gripper_value_in.get_lock():
                    r_in = self.right_gripper_value_in.value

                if self.input_mode == "controller":
                    l_target = np.interp(l_in, [0.0, 10.0], [self.LEFT_MAPPED_MIN, self.LEFT_MAPPED_MAX])
                    r_target = np.interp(r_in, [0.0, 10.0], [self.RIGHT_MAPPED_MIN, self.RIGHT_MAPPED_MAX])
                else:
                    l_target = np.interp(l_in, [5.0, 7.0], [self.LEFT_MAPPED_MIN, self.LEFT_MAPPED_MAX])
                    r_target = np.interp(r_in, [5.0, 7.0], [self.RIGHT_MAPPED_MIN, self.RIGHT_MAPPED_MAX])

                target_gripper = np.array([l_target, r_target])
                target_diff = target_gripper - self.current_gripper_cmd
                step = np.where(target_diff > 0, np.minimum(target_diff, DELTA_GRIPPER_OPEN), np.maximum(target_diff, -DELTA_GRIPPER_CLOSE))
                self.current_gripper_cmd += step
                self.current_gripper_cmd = np.clip(
                    self.current_gripper_cmd,
                    [self.LEFT_MAPPED_MIN, self.RIGHT_MAPPED_MIN],
                    [self.LEFT_MAPPED_MAX, self.RIGHT_MAPPED_MAX],
                )
                with self.ctrl_lock:
                    self.gripper_action["left"] = float(self.current_gripper_cmd[0])
                    self.gripper_action["right"] = float(self.current_gripper_cmd[1])

            self._sync_gripper_to_shared_arrays()

            # 2. Snapshot target values
            with self.ctrl_lock:
                arm_q = self.q_target.copy()
                chassis = list(self.chassis_action)
                gripper = dict(self.gripper_action)

            # 3. Build action payload strictly compatible with Groot / GripperBridge
            action_dict = {}
            for idx, motor in enumerate(G1_29_JointArmIndex):
                if idx < len(arm_q):
                    action_dict[f"{motor.name}.q"] = float(arm_q[idx])

            # Remote locomotion axes: chassis = [vx, vy, vyaw]
            # LeRobot convention: cmd_vel[0]=remote.ly, cmd_vel[1]=-remote.lx, cmd_vel[2]=-remote.rx
            vx, vy, vyaw = chassis if len(chassis) >= 3 else [0.0, 0.0, 0.0]
            action_dict["remote.ly"] = float(vx)
            action_dict["remote.lx"] = float(-vy)
            action_dict["remote.rx"] = float(-vyaw)
            action_dict["remote.ry"] = 0.0

            # Gripper payload: strictly aligned with Groot GripperBridge protocol
            # {"gripper": {"right": {"q": ...}, "left": {"q": ...}}}
            if self.enable_gripper:
                l_grip = float(np.clip(gripper.get("left", self.LEFT_MAPPED_MAX), self.LEFT_MAPPED_MIN, self.LEFT_MAPPED_MAX))
                r_grip = float(np.clip(gripper.get("right", self.RIGHT_MAPPED_MAX), self.RIGHT_MAPPED_MIN, self.RIGHT_MAPPED_MAX))
                action_dict["gripper"] = {
                    "right": {"q": r_grip},
                    "left":  {"q": l_grip},
                }

            msg = {
                "cmd": "action",
                "action": action_dict,
                "timestamp": time.time(),
            }

            try:
                payload = json.dumps(msg).encode("utf-8")
                self._action_push.send(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                logger_mp.debug(f"Failed to send ZMQ action: {e}")

            elapsed = time.time() - start_t
            sleep_time = max(0.0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    # ================== API Methods compatible with ArmController ==================

    def get_current_motor_q(self) -> np.ndarray:
        """Return 29-DoF joint positions."""
        with self._state_lock:
            return self._all_motor_q.copy()

    def get_current_motor_dq(self) -> np.ndarray:
        """Return 29-DoF joint velocities."""
        with self._state_lock:
            return self._all_motor_dq.copy()

    def get_current_dual_arm_q(self) -> np.ndarray:
        """Return 14-DoF dual arm joint positions."""
        with self._state_lock:
            arm_indices = [member.value for member in G1_29_JointArmIndex]
            return self._all_motor_q[arm_indices].copy()

    def get_current_dual_arm_dq(self) -> np.ndarray:
        """Return 14-DoF dual arm joint velocities."""
        with self._state_lock:
            arm_indices = [member.value for member in G1_29_JointArmIndex]
            return self._all_motor_dq[arm_indices].copy()

    def get_current_leg_q(self) -> np.ndarray:
        """Return 12-DoF leg joint positions."""
        with self._state_lock:
            leg_indices = [member.value for member in G1_29_JointLegIndex]
            return self._all_motor_q[leg_indices].copy()

    def get_current_leg_dq(self) -> np.ndarray:
        """Return 12-DoF leg joint velocities."""
        with self._state_lock:
            leg_indices = [member.value for member in G1_29_JointLegIndex]
            return self._all_motor_dq[leg_indices].copy()

    def get_current_gripper_state(self) -> Tuple[float, float]:
        """Return (left, right) gripper state."""
        with self._gripper_lock:
            return self._latest_gripper["left"], self._latest_gripper["right"]

    def ctrl_dual_arm(self, q_target: np.ndarray, tauff_target: np.ndarray = None):
        """Update arm target angles."""
        with self.ctrl_lock:
            self.q_target = np.array(q_target, dtype=np.float64)
            if tauff_target is not None:
                self.tauff_target = np.array(tauff_target, dtype=np.float64)

    def ctrl_chassis(self, vx: float, vy: float, vyaw: float):
        """Update chassis velocity."""
        with self.ctrl_lock:
            self.chassis_action = [float(vx), float(vy), float(vyaw)]

    def ctrl_gripper(self, left: float, right: float):
        """Update gripper positions."""
        with self.ctrl_lock:
            self.gripper_action = {"left": float(left), "right": float(right)}

    def ctrl_dual_arm_go_home(self):
        """Smoothly reset arms to zero."""
        logger_mp.info("[G1_29_ZMQ_Controller] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(ARM_NUM_MOTORS)
            self.chassis_action = [0.0, 0.0, 0.0]
        try:
            reset_msg = json.dumps({"cmd": "reset", "timestamp": time.time()}).encode("utf-8")
            self._action_push.send(reset_msg, flags=zmq.NOBLOCK)
        except Exception:
            pass

    def close(self):
        """Clean up sockets and threads."""
        logger_mp.info("[G1_29_ZMQ_Controller] Closing...")
        self._shutdown_event.set()
        try:
            stop_msg = json.dumps({"cmd": "stop", "timestamp": time.time()}).encode("utf-8")
            self._action_push.send(stop_msg, flags=zmq.NOBLOCK)
        except Exception:
            pass
        time.sleep(0.1)
        if self._state_sub:
            self._state_sub.close(linger=0)
        if self._action_push:
            self._action_push.close(linger=0)
        if self._gripper_sub:
            self._gripper_sub.close(linger=0)
        logger_mp.info("[G1_29_ZMQ_Controller] Closed.")
