# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
from __future__ import annotations

from typing import Any
import torch

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_unifolm_vla import UnifolmVLAConfig


def make_unifolm_vla_pre_post_processors(
    config: UnifolmVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    构造 UnifoLM-VLA 的前处理和后处理流水线。
    注：UnifoLM-VLA 内部自带专用动作与状态反归一化模块 (dataset_statistics.json)，
    故流水线主要负责批次维度增加和设备迁移。
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
