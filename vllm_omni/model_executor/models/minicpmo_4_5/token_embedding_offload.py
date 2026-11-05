# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sparse token lookup from a CPU-pinned embedding table through CUDA UVA."""

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _copy_to_pinned_cpu(weight: torch.Tensor) -> torch.Tensor:
    # Complete the startup D2H copy before publishing the CUDA view. Inference
    # then reads only the selected rows; it never copies the table per token.
    host_weight = torch.empty_like(weight, device="cpu", pin_memory=True)
    host_weight.copy_(weight.detach())
    return host_weight


def _offload_embedding_weight(embedding: nn.Module) -> int:
    weight = embedding.weight
    if getattr(weight, "_vllm_is_uva_offloaded", False):
        return 0
    host_weight = _copy_to_pinned_cpu(weight)
    device_view = get_accelerator_view_from_cpu_tensor(host_weight)
    # Retain Parameter identity, its loader metadata, dtype and requires_grad.
    # Keep an explicit host owner as well as the official view's storage owner.
    weight.data = device_view
    weight._vllm_is_uva_offloaded = True
    embedding._cpu_token_embedding_host_weight = host_weight
    return weight.numel() * weight.element_size()


def enable_token_embedding_uva(embedding: nn.Module, lm_head: nn.Module, vllm_config: "VllmConfig") -> int:
    """Move an untied, unquantized TP1 token table after checkpoint loading.

    This allocation is separate from ``cpu_offload_gb`` for decoder layers.
    A caller budgeting total host-resident weights must account for both.
    The CUDA view preserves ``weight.device`` for multimodal device discovery.
    """
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("cpu_token_embedding_uva requires tensor_parallel_size=1 and pipeline_parallel_size=1")
    if getattr(vllm_config, "lora_config", None) is not None:
        raise ValueError("cpu_token_embedding_uva does not support LoRA embedding adapters")
    if not current_platform.is_cuda() or not is_uva_available():
        raise ValueError("cpu_token_embedding_uva requires CUDA with pinned-memory UVA support")
    weight = getattr(embedding, "weight", None)
    output_weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        raise ValueError("cpu_token_embedding_uva requires a two-dimensional embedding Parameter")
    if weight.requires_grad:
        raise ValueError("cpu_token_embedding_uva requires inference-only embedding weights")
    if isinstance(output_weight, torch.Tensor) and (
        output_weight is weight
        or (output_weight.device == weight.device and output_weight.data_ptr() == weight.data_ptr())
    ):
        raise ValueError("cpu_token_embedding_uva requires input embeddings not tied to lm_head")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("cpu_token_embedding_uva requires unquantized floating-point embedding weights")
    if weight.device.type != "cuda":
        raise ValueError("cpu_token_embedding_uva expects loaded CUDA embedding weights")
    return _offload_embedding_weight(embedding)
