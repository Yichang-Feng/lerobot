# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("unifolm_vla")
@dataclass
class UnifolmVLAConfig(PreTrainedConfig):
    """
    UnifoLM-VLA 模型在 LeRobot 体系中的配置类。
    适配 Unitree G1 (14-DoF 手臂关节 + 1 视角相机，无夹爪)
    """

    unifolm_ckpt_path: str = "/home/yichangfeng/unifolm-vla/model/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt"
    unifolm_vlm_path: str = "/home/yichangfeng/unifolm-vla/model/UnifoLM-VLM-Base"
    task_name: str = "g1_stack_block"
    arm_side: str = "right"  # "right" 或 "left"：指定当前 7-DoF 动作控制作用于哪一侧手臂
    default_task: str = "stack the red block on the blue block"

    chunk_size: int = 16
    n_action_steps: int = 16
    n_obs_steps: int = 1

    device: str = "cuda"
    dtype: str = "bfloat16"

    # 是否启用夹爪控制 (True: 20 维动作: 14 臂 + 2 爪 + 4 摇杆; False: 18 维动作)
    use_gripper: bool = False

    # 是否启用手腕部双相机 (True: 3 相机: 全局 + 左手腕 + 右手腕，对齐 UnifoLM-VLA-Base 训练参数)
    use_wrist_image: bool = True

    # RTC (Real-Time Chunking) 异步推理支持
    use_rtc: bool = True
    rtc_config: Any | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # 训练预设参数
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        self.validate_features()

    def validate_features(self) -> None:
        # 默认匹配 UnitreeG1Client 提供的状态 (29 维) 与相机
        if not self.input_features:
            self.input_features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
                "observation.images.global_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            }
            if self.use_wrist_image:
                self.input_features["observation.images.left_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                self.input_features["observation.images.right_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        action_dim = 20 if self.use_gripper else 18
        if not self.output_features:
            self.output_features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            }
        elif ACTION in self.output_features:
            if self.use_gripper and self.output_features[ACTION].shape[0] == 18:
                self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(20,))
            elif self.output_features[ACTION].shape[0] == 20:
                self.use_gripper = True

        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "UnifoLM-VLA requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        return None
