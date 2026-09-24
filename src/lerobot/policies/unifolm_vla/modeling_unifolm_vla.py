# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
from __future__ import annotations

import logging
import os
import sys
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_unifolm_vla import UnifolmVLAConfig

logger = logging.getLogger(__name__)

# 确保能加载 unifolm_vla 核心库
UNIFOLM_SRC = "/home/yichangfeng/unifolm-vla/src"
if UNIFOLM_SRC not in sys.path:
    sys.path.insert(0, UNIFOLM_SRC)

try:
    from unifolm_vla.model.framework.base_framework import baseframework
    from qwen_vl_utils import process_vision_info
    from unifolm_vla.rlds_dataloader.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        NormalizationType,
    )
    HAS_UNIFOLM = True
except ImportError as e:
    logger.warning(f"无法直接导入 unifolm_vla: {e}，将在首次加载权重时尝试修复环境。")
    HAS_UNIFOLM = False


def _ensure_unifolm():
    global baseframework, process_vision_info, ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType, HAS_UNIFOLM
    if HAS_UNIFOLM:
        return
    try:
        from unifolm_vla.model.framework.base_framework import baseframework
        from qwen_vl_utils import process_vision_info
        from unifolm_vla.rlds_dataloader.constants import (
            ACTION_PROPRIO_NORMALIZATION_TYPE,
            NormalizationType,
        )
        HAS_UNIFOLM = True
    except ImportError as e:
        raise ImportError(
            f"无法导入 unifolm_vla: {e}。请确保当前环境中已安装 omegaconf, qwen-vl-utils, json_numpy 等依赖。"
        ) from e


class UnifolmVLAPolicy(PreTrainedPolicy):
    """
    UnifoLM-VLA 适配器策略模型。
    将 UnifoLM-VLA (23-DoF EE/Proprio 动作模型) 无缝桥接到 LeRobot 的 Unitree G1 运行时中 (18-DoF: 14-DoF 手臂 + 4-DoF 摇杆)。
    """

    config_class = UnifolmVLAConfig
    name = "unifolm_vla"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | os.PathLike[str],
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> UnifolmVLAPolicy:
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        policy = cls(config, **kwargs)
        policy.to(config.device)
        policy.eval()
        return policy

    def __init__(self, config: UnifolmVLAConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self._action_queue = deque()

        _ensure_unifolm()

        logger.info(f"[UnifoLM-VLA] 正在从 {config.unifolm_ckpt_path} 载入模型权重...")
        self.vla = baseframework.from_pretrained(
            pretrained_checkpoint=config.unifolm_ckpt_path,
            vlm_pretrained_path=config.unifolm_vlm_path,
        )

        if config.dtype == "bfloat16":
            self.vla = self.vla.to(torch.bfloat16)

        target_device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.vla = self.vla.to(target_device).eval()
        self.processor = self.vla.qwen_vl_interface.processor

        # 载入默认任务反归一化统计量
        self.task_name = config.task_name
        if self.task_name in self.vla.norm_stats:
            self.norm_stats_action = self.vla.norm_stats[self.task_name]["action"]
            self.norm_stats_proprio = self.vla.norm_stats[self.task_name]["proprio"]
            logger.info(f"[UnifoLM-VLA] 成功加载任务 '{self.task_name}' 的动作/状态统计量")
        else:
            first_key = next(iter(self.vla.norm_stats.keys()))
            logger.warning(f"[UnifoLM-VLA] 未找到任务 '{self.task_name}'，降级使用 '{first_key}'")
            self.norm_stats_action = self.vla.norm_stats[first_key]["action"]
            self.norm_stats_proprio = self.vla.norm_stats[first_key]["proprio"]

        logger.info(f"[UnifoLM-VLA] 模型载入完成！工作模式: 控制 {config.arm_side} 臂")

        # 初始化 G1 运动学求解器 (IK / FK)
        try:
            from lerobot.robots.unitree_g1.g1_kinematics import G1_29_ArmIK, rot6d_to_matrix
            self.ik = G1_29_ArmIK()
            self._rot6d_to_matrix = rot6d_to_matrix
            logger.info("[UnifoLM-VLA] G1 运动学求解器 (IK / FK) 初始化成功")
        except Exception as e:
            logger.warning(f"[UnifoLM-VLA] 无法初始化 G1_29_ArmIK: {e}，将降级处理。")
            self.ik = None
            self._rot6d_to_matrix = None

        # 记录初始物理臂关节位置，用于非活跃臂固定锁位（防止重力下垂漂移）
        self._hold_left_arm_q: np.ndarray | None = None
        self._hold_right_arm_q: np.ndarray | None = None
        self._last_solved_q14: np.ndarray | None = None
        self._first_vision_logged: bool = False

    def supports_rtc(self) -> bool:
        """支持 Real-Time Chunking 异步推理模式"""
        return True

    def reset(self):
        """重置动作缓存队列与非活跃臂锁定基准"""
        self._action_queue.clear()
        self._hold_left_arm_q = None
        self._hold_right_arm_q = None
        self._last_solved_q14 = None

    def get_optim_params(self) -> dict:
        return self.vla.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return torch.tensor(0.0, device=self.config.device), {}

    def _resolve_task_norm_key(self, task_desc: str) -> str:
        """根据自然语言指令或任务名自动解析最匹配的 norm_stats key"""
        if self.config.task_name in self.vla.norm_stats:
            default_key = self.config.task_name
        else:
            default_key = next(iter(self.vla.norm_stats.keys()))

        desc = task_desc.lower().replace("_", " ").strip()
        if task_desc in self.vla.norm_stats:
            return task_desc

        for key in self.vla.norm_stats.keys():
            clean_key = key.replace("g1_", "").replace("_", " ")
            if clean_key in desc or desc in clean_key:
                return key

        keywords_map = {
            "clean_table": ["clean", "table"],
            "wipe_table": ["wipe", "table"],
            "stack_block": ["stack", "block"],
            "pack_pencilbox": ["pencil", "box"],
            "erase_board": ["erase", "board"],
            "bag_insert": ["bag", "insert"],
            "pour_medicine": ["pour", "medicine"],
            "pack_pingpong": ["pingpong", "ping"],
            "organize_tools": ["tool", "organize"],
            "prepare_fruit": ["fruit"],
            "fold_towel": ["towel", "fold"],
        }
        for task_id, words in keywords_map.items():
            candidate = f"g1_{task_id}"
            if candidate in self.vla.norm_stats and all(w in desc for w in words):
                return candidate

        return default_key

    def _get_norm_stats_for_task(self, task_str: str) -> tuple[dict[str, Any], dict[str, Any]]:
        key = self._resolve_task_norm_key(task_str)
        return self.vla.norm_stats[key]["proprio"], self.vla.norm_stats[key]["action"]

    def _normalize_proprio(self, proprio: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
        """将状态归一化到 [-1, 1] 区间"""
        if "q99" in norm_stats and "q01" in norm_stats:
            high = np.array(norm_stats["q99"], dtype=np.float32)
            low = np.array(norm_stats["q01"], dtype=np.float32)
        elif "max" in norm_stats and "min" in norm_stats:
            high = np.array(norm_stats["max"], dtype=np.float32)
            low = np.array(norm_stats["min"], dtype=np.float32)
        else:
            return proprio

        # 维度对齐保护 (若传入维数与统计量维数不一致，则对齐填充)
        if proprio.shape[-1] != low.shape[-1]:
            target_dim = low.shape[-1]
            padded_proprio = np.array(norm_stats.get("mean", (high + low) * 0.5), dtype=np.float32).copy()
            copy_len = min(proprio.shape[-1], target_dim)
            padded_proprio[:copy_len] = proprio[:copy_len]
            proprio = padded_proprio

        mask = norm_stats.get("mask", np.ones_like(low, dtype=bool))
        return np.clip(
            np.where(mask, 2 * (proprio - low) / (high - low + 1e-8) - 1, proprio),
            -1.0,
            1.0,
        )

    def _unnormalize_action(self, norm_actions: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
        """将 [-1, 1] 动作反归一化"""
        if "q99" in norm_stats and "q01" in norm_stats:
            high = np.array(norm_stats["q99"], dtype=np.float32)
            low = np.array(norm_stats["q01"], dtype=np.float32)
        elif "max" in norm_stats and "min" in norm_stats:
            high = np.array(norm_stats["max"], dtype=np.float32)
            low = np.array(norm_stats["min"], dtype=np.float32)
        else:
            return norm_actions

        mask = norm_stats.get("mask", np.ones_like(low, dtype=bool))
        return np.where(mask, 0.5 * (norm_actions + 1) * (high - low + 1e-8) + low, norm_actions)

    def _tensor_to_pil(self, img_tensor: Any) -> Image.Image:
        """Helper to convert tensor/array into 224x224 RGB PIL image."""
        if isinstance(img_tensor, torch.Tensor):
            t = img_tensor.detach().cpu()
            if t.ndim == 4:
                t = t[0]  # 取 batch[0]
            if t.ndim == 3 and t.shape[0] == 3:  # (3, H, W)
                t = t.permute(1, 2, 0)  # (H, W, 3)
            img_np = t.numpy()
            if np.issubdtype(img_np.dtype, np.floating):
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(img_tensor)
            if np.issubdtype(img_np.dtype, np.floating):
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_np = img_np.clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        pil_img = Image.fromarray(img_np).convert("RGB")
        return pil_img.resize((224, 224), Image.Resampling.BILINEAR)

    def _extract_images(self, batch: dict[str, Any]) -> list[Image.Image]:
        """
        从 LeRobot batch 中提取相机图像并转换为 224x224 RGB PIL 图像列表。
        支持单相机 (global_view) 与三相机 (global_view + left_wrist + right_wrist) 自动适配。
        """
        images = []

        # 1. 查找主全局相机 (Primary Camera)
        primary_tensor = None
        for k in [
            "observation.images.global_view",
            "observation.images.cam_left_high",
            "observation.images.cam_high",
            "observation.images.head_camera",
            "observation.image",
        ]:
            if k in batch:
                primary_tensor = batch[k]
                break

        if primary_tensor is None:
            for k, v in batch.items():
                if "image" in k.lower() and "wrist" not in k.lower():
                    primary_tensor = v
                    break

        if primary_tensor is not None:
            images.append(self._tensor_to_pil(primary_tensor))
        else:
            images.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        # 只有在启用手腕相机 (use_wrist_image=True) 时，才提取左右手腕视角
        # 顺序严格对齐 UnifoLM-VLA-Base 训练基线: [主视角/全局, 左手腕, 右手腕]
        if getattr(self.config, "use_wrist_image", True):
            # 2. 查找左腕部相机 (Left Wrist Camera)
            left_wrist_tensor = None
            for k in [
                "observation.images.left_wrist",
                "observation.images.cam_left_wrist",
                "observation.images.wrist_left",
                "cam_left_wrist",
                "left_wrist",
            ]:
                if k in batch:
                    left_wrist_tensor = batch[k]
                    break
            if left_wrist_tensor is not None:
                images.append(self._tensor_to_pil(left_wrist_tensor))
            else:
                # 缺失时使用零值填充以保持 3 画面输入对齐
                images.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

            # 3. 查找右腕部相机 (Right Wrist Camera)
            right_wrist_tensor = None
            for k in [
                "observation.images.right_wrist",
                "observation.images.cam_right_wrist",
                "observation.images.wrist_right",
                "cam_right_wrist",
                "right_wrist",
            ]:
                if k in batch:
                    right_wrist_tensor = batch[k]
                    break
            if right_wrist_tensor is not None:
                images.append(self._tensor_to_pil(right_wrist_tensor))
            else:
                # 缺失时使用零值填充以保持 3 画面输入对齐
                images.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        return images

    def _extract_image(self, batch: dict[str, Any]) -> Image.Image:
        """兼容旧接口"""
        return self._extract_images(batch)[0]

    def _extract_arm_states(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        从 29-DoF (或 14-DoF) 状态中提取左臂 (7 维) 与右臂 (7 维) 关节角度。
        """
        state_tensor = batch.get("observation.state", batch.get("state", None))
        if state_tensor is None:
            left_arm = np.zeros(7, dtype=np.float32)
            right_arm = np.zeros(7, dtype=np.float32)
            return left_arm, right_arm

        if isinstance(state_tensor, torch.Tensor):
            s = state_tensor.detach().cpu().numpy()
        else:
            s = np.array(state_tensor)

        if s.ndim > 1:
            s = s[0]  # 取 batch 0

        # G1 全机 29 关节映射：
        # Left arm: 15..21 (7 自由度)
        # Right arm: 22..28 (7 自由度)
        if len(s) >= 29:
            left_arm = s[15:22].astype(np.float32)
            right_arm = s[22:29].astype(np.float32)
        elif len(s) >= 14:
            left_arm = s[0:7].astype(np.float32)
            right_arm = s[7:14].astype(np.float32)
        else:
            left_arm = np.zeros(7, dtype=np.float32)
            right_arm = np.zeros(7, dtype=np.float32)

        return left_arm, right_arm

    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        inference_delay: int = 0,
        prev_chunk_left_over: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        核心推理入口：接收 LeRobot 的 batch，调用 UnifoLM-VLA，输出 (B, chunk_size, 18) 动作块
        """
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # 1. 提取任务提示词并匹配最优统计量
        task_str = batch.get("task", self.config.default_task)
        if isinstance(task_str, (list, tuple)):
            task_str = task_str[0]
        elif isinstance(task_str, torch.Tensor):
            task_str = self.config.default_task
        task_str = str(task_str)

        norm_stats_proprio, norm_stats_action = self._get_norm_stats_for_task(task_str)

        # 2. 提取相机图像 (支持单视角或三视角: 全局 + 左手腕 + 右手腕)
        pil_images = self._extract_images(batch)
        if not getattr(self, "_first_vision_logged", False):
            self._first_vision_logged = True
            logger.info(
                f"[UnifoLM-VLA] 视觉输入就绪: 包含 {len(pil_images)} 路画面 "
                f"(顺序: 主全局视角 -> 左手腕视角 -> 右手腕视角, use_wrist_image={getattr(self.config, 'use_wrist_image', True)})"
            )

        # 3. 提取并映射关节状态 (23 维 EE / Proprio)
        left_arm, right_arm = self._extract_arm_states(batch)

        # 锁定非活跃臂位置，避免重力下垂与状态跟随导致正反馈漂移 ("左手缓慢放下")
        if np.any(left_arm != 0.0):
            if self._hold_left_arm_q is None:
                self._hold_left_arm_q = left_arm.copy()
                logger.info(f"[UnifoLM-VLA] 成功锁定左臂初始静止姿态: {np.round(self._hold_left_arm_q, 3)}")

        if np.any(right_arm != 0.0):
            if self._hold_right_arm_q is None:
                self._hold_right_arm_q = right_arm.copy()
                logger.info(f"[UnifoLM-VLA] 成功锁定右臂初始静止姿态: {np.round(self._hold_right_arm_q, 3)}")

        hold_left = self._hold_left_arm_q if self._hold_left_arm_q is not None else left_arm
        hold_right = self._hold_right_arm_q if self._hold_right_arm_q is not None else right_arm

        raw_state = batch.get("observation.state", batch.get("state", None))
        if isinstance(raw_state, torch.Tensor):
            raw_s = raw_state.detach().cpu().numpy()
            if raw_s.ndim > 1:
                raw_s = raw_s[0]
        elif raw_state is not None:
            raw_s = np.array(raw_state)
            if raw_s.ndim > 1:
                raw_s = raw_s[0]
        else:
            raw_s = np.zeros(0, dtype=np.float32)

        proprio_dim = len(norm_stats_proprio.get("q01", norm_stats_proprio.get("min", [0] * 23)))

        # 提取实测当前夹爪角度 (若有，范围 0.0 ~ 5.0)
        r_grip = batch.get("gripper.right", batch.get("observation.right_gripper", batch.get("kRightGripper.q", 5.0)))
        l_grip = batch.get("gripper.left", batch.get("observation.left_gripper", batch.get("kLeftGripper.q", 5.0)))
        if isinstance(r_grip, (torch.Tensor, np.ndarray)):
            r_grip = float(r_grip.item() if hasattr(r_grip, "item") and r_grip.numel() == 1 else r_grip.flatten()[0])
        if isinstance(l_grip, (torch.Tensor, np.ndarray)):
            l_grip = float(l_grip.item() if hasattr(l_grip, "item") and l_grip.numel() == 1 else l_grip.flatten()[0])
        r_grip = float(np.clip(r_grip, 0.0, 5.0))
        l_grip = float(np.clip(l_grip, 0.0, 5.0))

        # 优先使用正运动学 (FK) 从 29-DoF (或 14-DoF) 实测关节计算真实的 23 维末端 proprio
        if len(raw_s) >= 14 and self.ik is not None:
            try:
                # default_grippers: (r_grip, l_grip) 对应 index 18 (右爪) 和 index 19 (左爪)
                proprio_23d = self.ik.compute_proprio_23d(raw_s, default_grippers=(r_grip, l_grip))
            except Exception as e:
                logger.debug(f"[UnifoLM-VLA] FK proprio 计算异常: {e}，回退使用先验均值")
                proprio_23d = np.array(norm_stats_proprio.get("mean", np.zeros(proprio_dim)), dtype=np.float32)
        elif len(raw_s) == proprio_dim:
            proprio_23d = raw_s.astype(np.float32)
        else:
            # 使用数据集统计量的均值作为基准先验
            proprio_23d = np.array(
                norm_stats_proprio.get("mean", np.zeros(proprio_dim)),
                dtype=np.float32,
            ).copy()

        normed_proprio = self._normalize_proprio(proprio_23d, norm_stats_proprio)

        # 4. 构建 Qwen2.5-VL 视觉语言批次输入
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": f'The task is "{task_str.lower()}".'},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch_input["state"] = torch.from_numpy(normed_proprio).unsqueeze(0).to(device)
        for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]:
            batch_input[k] = batch_input[k].to(device)

        # 5. 执行前向动作扩散预测
        with torch.no_grad():
            output = self.vla.predict_action(qwen_inputs=batch_input)

        normed_actions = output["normalized_actions"][0]  # shape: (25, 23)
        # 诊断日志：打印 VLA 原始输出的统计量，用于判断模型是否产生了变化的输出
        logger.debug(
            f"[VLA 原始输出] mean={normed_actions.mean():.4f}, "
            f"std={normed_actions.std():.4f}, "
            f"range=[{normed_actions.min():.4f}, {normed_actions.max():.4f}]"
        )
        unnorm_actions = self._unnormalize_action(normed_actions, norm_stats_action)  # shape: (25, 23)

        # 6. 将 23 维笛卡尔末端动作映射为 LeRobot 动作空间 (18 维: 14臂+4摇杆; 或 20 维: 14臂+2爪+4摇杆)
        chunk_size = getattr(self.config, "chunk_size", 16)
        num_steps = min(chunk_size, unnorm_actions.shape[0])

        from lerobot.utils.constants import ACTION
        use_gripper = getattr(self.config, "use_gripper", False)
        if hasattr(self.config, "output_features") and ACTION in self.config.output_features:
            if self.config.output_features[ACTION].shape[0] == 20:
                use_gripper = True

        out_dim = 20 if use_gripper else 18
        action_out = np.zeros((num_steps, out_dim), dtype=np.float32)

        # 提取夹爪目标角度 (index 18 为右爪, index 19 为左爪，范围 0.0 ~ 5.0)
        if unnorm_actions.shape[1] > 19:
            pred_r_grip = np.clip(unnorm_actions[:num_steps, 18], 0.0, 5.0)
            pred_l_grip = np.clip(unnorm_actions[:num_steps, 19], 0.0, 5.0)
        else:
            pred_r_grip = np.full(num_steps, 5.0, dtype=np.float32)
            pred_l_grip = np.full(num_steps, 5.0, dtype=np.float32)

        # 检查输出动作空间：UnifoLM-VLA 预测 23 维笛卡尔末端位姿与控制
        # 0..3: L_xyz, 3..9: L_6d_rot, 9..12: R_xyz, 12..18: R_6d_rot, 18: R_grip, 19: L_grip, 20..23: waist
        is_ee_23d = (unnorm_actions.shape[1] >= 18) and (self.ik is not None) and (self._rot6d_to_matrix is not None)

        if is_ee_23d:
            T_L_seq = []
            T_R_seq = []
            for t in range(num_steps):
                # 恢复左手 4x4 变换矩阵
                L_xyz = unnorm_actions[t, 0:3]
                L_rot_6d = unnorm_actions[t, 3:9]
                R_L = self._rot6d_to_matrix(L_rot_6d[:3], L_rot_6d[3:6])
                T_L = np.eye(4, dtype=np.float64)
                T_L[:3, :3] = R_L
                T_L[:3, 3] = L_xyz
                T_L_seq.append(T_L)

                # 恢复右手 4x4 变换矩阵
                R_xyz = unnorm_actions[t, 9:12]
                R_rot_6d = unnorm_actions[t, 12:18]
                R_R = self._rot6d_to_matrix(R_rot_6d[:3], R_rot_6d[3:6])
                T_R = np.eye(4, dtype=np.float64)
                T_R[:3, :3] = R_R
                T_R[:3, 3] = R_xyz
                T_R_seq.append(T_R)

            # 初始关节初值种子（优先使用上一 chunk 结尾或当前实测关节角）
            if self._last_solved_q14 is not None:
                q_seed = self._last_solved_q14
            else:
                q_seed = np.concatenate([left_arm, right_arm])

            # 快速逆运动学求解 (14 自由度: 左 7 + 右 7)
            sol_q14 = self.ik.solve_ik_chunk(T_L_seq, T_R_seq, q_init=q_seed)
            self._last_solved_q14 = sol_q14[-1].copy()

            if self.config.arm_side == "right":
                # 左臂牢牢保持在初始静止位置 (切断重力下垂正反馈闭环)
                action_out[:, 0:7] = hold_left
                # 右臂执行 IK 解算的 7 自由度关节目标角
                action_out[:, 7:14] = sol_q14[:, 7:14]
                if use_gripper:
                    action_out[:, 14] = pred_r_grip
                    action_out[:, 15] = 5.0  # 非受控左夹爪维持全开
            elif self.config.arm_side == "left":
                # 左臂执行 IK 求解动作，右臂保持初始位置
                action_out[:, 0:7] = sol_q14[:, 0:7]
                action_out[:, 7:14] = hold_right
                if use_gripper:
                    action_out[:, 14] = 5.0  # 非受控右夹爪维持全开
                    action_out[:, 15] = pred_l_grip
            else:
                # 双臂控制 (both / dual)
                action_out[:, 0:7] = sol_q14[:, 0:7]
                action_out[:, 7:14] = sol_q14[:, 7:14]
                if use_gripper:
                    action_out[:, 14] = pred_r_grip
                    action_out[:, 15] = pred_l_grip

        else:
            # 降级模式：针对直接在关节空间训练的模型
            dim_slice = min(7, unnorm_actions.shape[1])
            if self.config.arm_side == "right":
                action_out[:, 0:7] = hold_left
                action_out[:, 7:7+dim_slice] = unnorm_actions[:num_steps, 0:dim_slice]
                if use_gripper:
                    action_out[:, 14] = pred_r_grip
                    action_out[:, 15] = 5.0
            elif self.config.arm_side == "left":
                action_out[:, 0:dim_slice] = unnorm_actions[:num_steps, 0:dim_slice]
                action_out[:, 7:14] = hold_right
                if use_gripper:
                    action_out[:, 14] = 5.0
                    action_out[:, 15] = pred_l_grip
            else:
                action_out[:, 0:7] = unnorm_actions[:num_steps, 0:7]
                if unnorm_actions.shape[1] >= 14:
                    action_out[:, 7:14] = unnorm_actions[:num_steps, 7:14]
                if use_gripper:
                    action_out[:, 14] = pred_r_grip
                    action_out[:, 15] = pred_l_grip

        # 摇杆 4 轴默认归零
        if use_gripper:
            action_out[:, 16:20] = 0.0
        else:
            action_out[:, 14:18] = 0.0

        # 返回 (1, chunk_size, out_dim) 的 Tensor
        action_tensor = torch.from_numpy(action_out).unsqueeze(0).to(device=device, dtype=torch.float32)
        return action_tensor

    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        """单步执行接口：从内部动作队列取动作，若队列为空则触发一次 predict_action_chunk"""
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)  # (1, chunk_size, 18)
            chunk_cpu = chunk.detach().cpu()
            target_steps = min(self.config.n_action_steps, chunk_cpu.shape[1])
            self._action_queue.extend(chunk_cpu[:, :target_steps].transpose(0, 1))

        action = self._action_queue.popleft()
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        return action.to(device)
