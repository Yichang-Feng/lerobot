#!/usr/bin/env python

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

import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = deque(maxlen=self._window_size)

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        return data_array.T @ self._weights

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(
            new_data, self._data_queue[-1]
        ):  # skip duplicate data
            return

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data


def rot6d_to_matrix(col1: np.ndarray, col2: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt orthonormalization of 6D rotation (two 3D columns) into 3x3 SO(3) matrix.
    col1: first column of rotation matrix
    col2: second column of rotation matrix (prior to orthogonalization)
    """
    c1 = np.asarray(col1, dtype=np.float64).flatten()
    c2 = np.asarray(col2, dtype=np.float64).flatten()
    n1 = np.linalg.norm(c1)
    if n1 > 1e-8:
        c1 = c1 / n1
    else:
        c1 = np.array([1.0, 0.0, 0.0])
    proj = np.dot(c1, c2)
    c2_ortho = c2 - proj * c1
    n2 = np.linalg.norm(c2_ortho)
    if n2 > 1e-8:
        c2 = c2_ortho / n2
    else:
        c2 = np.array([0.0, 1.0, 0.0]) if abs(c1[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        c2 = c2 - np.dot(c1, c2) * c1
        c2 = c2 / np.linalg.norm(c2)
    c3 = np.cross(c1, c2)
    return np.column_stack([c1, c2, c3])


class G1_29_ArmIK:  # noqa: N801
    def __init__(self, unit_test=False):
        import casadi
        import pinocchio as pin
        from huggingface_hub import snapshot_download
        from pinocchio import casadi as cpin

        self._pin = pin
        self.unit_test = unit_test

        local_candidates = [
            "/home/yichangfeng/unitree-g1-mujoco",
            "/home/yichangfeng/lerobot/unitree-g1-mujoco",
            "./unitree-g1-mujoco",
        ]
        local_found = next(
            (p for p in local_candidates if os.path.isfile(os.path.join(p, "assets", "g1_body29_hand14.urdf"))),
            None,
        )
        if local_found:
            self.repo_path = local_found
        else:
            self.repo_path = snapshot_download("lerobot/unitree-g1-mujoco")

        urdf_path = os.path.join(self.repo_path, "assets", "g1_body29_hand14.urdf")
        mesh_dir = os.path.join(self.repo_path, "assets")

        self.robot = self._pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # Arm joint names in G1 motor order (G1_29_JointArmIndex)
        self._arm_joint_names_g1 = [
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
        # Pinocchio uses its own joint order in q; build index mapping.
        self._arm_joint_names_pin = sorted(
            self._arm_joint_names_g1,
            key=lambda name: self.reduced_robot.model.idx_qs[self.reduced_robot.model.getJointId(name)],
        )
        logger.info(f"Pinocchio arm joint order: {self._arm_joint_names_pin}")
        self._arm_reorder_g1_to_pin = [
            self._arm_joint_names_g1.index(name) for name in self._arm_joint_names_pin
        ]
        # Inverse mapping to return tau in G1 motor order.
        self._arm_reorder_pin_to_g1 = np.argsort(self._arm_reorder_g1_to_pin)

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )
        # Re-create pinocchio data so frame buffer contains added frames
        self.reduced_robot.data = self.reduced_robot.model.createData()

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 20, "tol": 1e-3},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)

    def compute_fk(self, q_14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute end-effector 4x4 transformation matrices for left and right arm joints (14-DoF).

        Args:
            q_14: Joint angles in G1 motor order (14-DoF).
        """
        q = np.asarray(q_14, dtype=np.float64).flatten()
        if len(q) < 14:
            q_full = np.zeros(14, dtype=np.float64)
            q_full[:len(q)] = q
            q = q_full
        else:
            q = q[:14]

        # G1 motor order → Pinocchio internal q order
        q = q[self._arm_reorder_g1_to_pin]

        self._pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        self._pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        T_L = self.reduced_robot.data.oMf[self.L_hand_id].homogeneous.copy()
        T_R = self.reduced_robot.data.oMf[self.R_hand_id].homogeneous.copy()
        return T_L, T_R

    def compute_proprio_23d(
        self,
        raw_state: np.ndarray,
        default_grippers: tuple[float, float] = (4.0, 4.0),
    ) -> np.ndarray:
        """
        Compute 23-DoF proprioception [L_xyz (3), L_6d (6), R_xyz (3), R_6d (6), grippers (2), waist (3)]
        from robot joint state (29-DoF or 14-DoF).
        """
        s = np.asarray(raw_state, dtype=np.float64).flatten()
        if len(s) >= 29:
            q_left = s[15:22]
            q_right = s[22:29]
            waist = s[12:15]
        elif len(s) >= 14:
            q_left = s[0:7]
            q_right = s[7:14]
            waist = np.zeros(3, dtype=np.float64)
        else:
            q_left = np.zeros(7, dtype=np.float64)
            q_right = np.zeros(7, dtype=np.float64)
            waist = np.zeros(3, dtype=np.float64)

        q_14 = np.concatenate([q_left, q_right])
        T_L, T_R = self.compute_fk(q_14)

        L_xyz = T_L[:3, 3]
        L_rot = T_L[:3, :3]
        L_6d = np.concatenate([L_rot[:, 0], L_rot[:, 1]])

        R_xyz = T_R[:3, 3]
        R_rot = T_R[:3, :3]
        R_6d = np.concatenate([R_rot[:, 0], R_rot[:, 1]])

        grippers = np.array(default_grippers, dtype=np.float64)
        return np.concatenate([L_xyz, L_6d, R_xyz, R_6d, grippers, waist]).astype(np.float32)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            # Convert from G1 motor order to Pinocchio internal order
            self.init_data = np.asarray(current_lr_arm_motor_q, dtype=np.float64)[self._arm_reorder_g1_to_pin]
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        converged = True
        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
        except Exception as e:
            converged = False
            logger.debug(f"IK convergence warning: {e}")
            sol_q = self.opti.debug.value(self.var_q)

        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q

        if not converged:
            # current_lr_arm_motor_q is already in G1 order; sol_q is in Pinocchio order
            fallback = current_lr_arm_motor_q if current_lr_arm_motor_q is not None else sol_q[self._arm_reorder_pin_to_g1]
            return fallback, np.zeros(self.reduced_robot.model.nv)

        sol_tauff = self._pin.rnea(
            self.reduced_robot.model,
            self.reduced_robot.data,
            sol_q,
            np.zeros(self.reduced_robot.model.nv),
            np.zeros(self.reduced_robot.model.nv),
        )

        # Convert from Pinocchio order back to G1 motor order
        return sol_q[self._arm_reorder_pin_to_g1], sol_tauff[self._arm_reorder_pin_to_g1]

    def solve_ik_chunk(
        self,
        T_L_seq: list[np.ndarray] | np.ndarray,
        T_R_seq: list[np.ndarray] | np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Sequentially solve IK for a chunk of poses with warm starting between consecutive steps.

        Args:
            q_init: Initial joint seed in G1 motor order (14-DoF). If None, uses internal state.

        Returns: (num_steps, 14) array of joint angles in G1 motor order.
        """
        num_steps = min(len(T_L_seq), len(T_R_seq))
        sol_seq = np.zeros((num_steps, 14), dtype=np.float32)
        # self.init_data is in Pinocchio order; convert to G1 motor order for solve_ik interface
        cur_q = q_init if q_init is not None else self.init_data[self._arm_reorder_pin_to_g1]

        for t in range(num_steps):
            cur_q, _ = self.solve_ik(T_L_seq[t], T_R_seq[t], current_lr_arm_motor_q=cur_q)
            sol_seq[t] = cur_q

        return sol_seq

    def solve_tau(self, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        try:
            q_g1 = np.array(current_lr_arm_motor_q, dtype=float)
            if q_g1.shape[0] != len(self._arm_joint_names_g1):
                raise ValueError(f"Expected {len(self._arm_joint_names_g1)} arm joints, got {q_g1.shape[0]}")
            q_pin = q_g1[self._arm_reorder_g1_to_pin]
            sol_tauff = self._pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                q_pin,
                np.zeros(self.reduced_robot.model.nv),
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_tauff[self._arm_reorder_pin_to_g1]

        except Exception as e:
            logger.error(f"ERROR in convergence, plotting debug info.{e}")
            return np.zeros(self.reduced_robot.model.nv)
