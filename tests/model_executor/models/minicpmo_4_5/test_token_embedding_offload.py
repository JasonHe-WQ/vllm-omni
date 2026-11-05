# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minicpmo_4_5 import token_embedding_offload as offload
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _embedding():
    embedding = nn.Embedding(12, 4, dtype=torch.bfloat16)
    embedding.weight.requires_grad_(False)
    return embedding


def _config(**overrides):
    values = {
        "parallel_config": SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        "lora_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_embedding_offload_preserves_parameter_rows_and_host_lifetime(monkeypatch):
    embedding = _embedding()
    weight = embedding.weight
    weight.weight_loader = object()
    loader = weight.weight_loader
    ids = torch.tensor([8, 1, 8, 11])
    expected = embedding(ids).clone()
    host_refs = []

    def cpu_copy(value):
        host = value.detach().clone()
        host_refs.append(weakref.ref(host))
        return host

    # Exercise ownership and Parameter assignment on CPU. CUDA registration
    # and the real CUDA embedding kernel require a separate GPU smoke.
    monkeypatch.setattr(offload, "_copy_to_pinned_cpu", cpu_copy)
    monkeypatch.setattr(offload, "get_accelerator_view_from_cpu_tensor", lambda host: host.detach())
    moved = offload._offload_embedding_weight(embedding)
    assert moved == 12 * 4 * 2
    assert embedding.weight is weight
    assert weight.weight_loader is loader
    assert weight.dtype == torch.bfloat16 and not weight.requires_grad
    assert weight._vllm_is_uva_offloaded
    gc.collect()
    assert host_refs[0]() is embedding._cpu_token_embedding_host_weight
    assert weight.data_ptr() == host_refs[0]().data_ptr()
    torch.testing.assert_close(embedding(ids), expected, rtol=0, atol=0)
    assert offload._offload_embedding_weight(embedding) == 0
    assert len(host_refs) == 1


def test_failed_view_creation_leaves_loaded_parameter_unchanged(monkeypatch):
    embedding = _embedding()
    weight = embedding.weight
    pointer = weight.data_ptr()
    monkeypatch.setattr(offload, "_copy_to_pinned_cpu", lambda value: value.detach().clone())

    def fail(_):
        raise RuntimeError("view creation failed")

    monkeypatch.setattr(offload, "get_accelerator_view_from_cpu_tensor", fail)
    with pytest.raises(RuntimeError, match="view creation failed"):
        offload._offload_embedding_weight(embedding)
    assert embedding.weight is weight and weight.data_ptr() == pointer
    assert not hasattr(weight, "_vllm_is_uva_offloaded")
    assert not hasattr(embedding, "_cpu_token_embedding_host_weight")


@pytest.mark.parametrize("unsupported", ["tp", "pp", "lora", "platform", "uva", "tied", "trainable", "cpu"])
def test_unsupported_embedding_modes_fail_before_offloading(monkeypatch, unsupported):
    config = _config()
    embedding, head = _embedding(), _embedding()
    monkeypatch.setattr(offload, "current_platform", SimpleNamespace(is_cuda=lambda: unsupported != "platform"))
    monkeypatch.setattr(offload, "is_uva_available", lambda: unsupported != "uva")
    if unsupported == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif unsupported == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "lora":
        config.lora_config = object()
    elif unsupported == "tied":
        head.weight = embedding.weight
    elif unsupported == "trainable":
        embedding.weight.requires_grad_(True)
    expected = {
        "tp": "tensor_parallel_size",
        "pp": "pipeline_parallel_size",
        "lora": "LoRA",
        "platform": "CUDA",
        "uva": "UVA",
        "tied": "not tied",
        "trainable": "inference-only",
        "cpu": "loaded CUDA",
    }
    with pytest.raises(ValueError, match=expected[unsupported]):
        offload.enable_token_embedding_uva(embedding, head, config)
    assert not hasattr(embedding.weight, "_vllm_is_uva_offloaded")


@pytest.mark.parametrize("stage,enabled,expected_calls", [("llm", False, 0), ("tts", True, 0), ("llm", True, 1)])
def test_post_load_hook_is_opt_in_and_thinker_only(monkeypatch, stage, enabled, expected_calls):
    llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=object()), lm_head=object())
    model = SimpleNamespace(
        model_stage=stage,
        thinker=SimpleNamespace(llm=llm),
        vllm_config=SimpleNamespace(additional_config={"cpu_token_embedding_uva": enabled}),
    )
    calls = []
    monkeypatch.setattr(offload, "enable_token_embedding_uva", lambda *args: calls.append(args) or 0)
    MiniCPMO45OmniForConditionalGeneration.process_weights_after_loading(model)
    assert len(calls) == expected_calls
    if calls:
        assert calls[0] == (llm.model.embed_tokens, llm.lm_head, model.vllm_config)


def test_post_load_hook_rejects_string_boolean():
    model = SimpleNamespace(
        model_stage="llm", vllm_config=SimpleNamespace(additional_config={"cpu_token_embedding_uva": "false"})
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        MiniCPMO45OmniForConditionalGeneration.process_weights_after_loading(model)
