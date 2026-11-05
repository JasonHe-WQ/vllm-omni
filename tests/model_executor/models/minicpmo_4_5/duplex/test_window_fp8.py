# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex import window_kv
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import PositionReanchor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
FP8 = torch.float8_e4m3fn


def _reference(raw, delta, inv_freq, scale):
    # Independent reference: explicitly dequantize, rotate physical keys, then
    # requantize with the SAME non-unit per-layer scale used by attention.
    physical = raw.view(FP8).float() * scale
    first, second = physical.chunk(2, dim=-1)
    angle = delta * inv_freq.float()
    rotated = torch.cat(
        (first * angle.cos() + second * angle.sin(), second * angle.cos() - first * angle.sin()), dim=-1
    )
    return (rotated / scale).clamp(-448, 448).to(FP8).view(torch.uint8)


def _pool(layout):
    # Five physical pages, GQA's two KV heads, four tokens/page, head_dim=4.
    generator = torch.Generator().manual_seed(19)
    keys = torch.randn(5, 2, 4, 4, generator=generator).to(FP8).view(torch.uint8)
    values = torch.randn(5, 2, 4, 4, generator=generator).to(FP8).view(torch.uint8)
    if layout == "packed":
        storage = torch.cat((keys, values), dim=-1)
        return storage, storage[..., :4], storage[..., 4:]
    if layout == "head_major":
        return keys, keys, values
    if layout == "token_major":
        storage = keys.transpose(1, 2).contiguous()
        return storage, storage.transpose(1, 2), values
    if layout == "split_first":
        storage = torch.stack((keys, values))
        return storage, storage[0], storage[1]
    if layout == "split_second":
        storage = torch.stack((keys, values), dim=1)
        return storage, storage[:, 0], storage[:, 1]
    raise AssertionError(layout)


@pytest.mark.parametrize("layout", ["packed", "head_major", "token_major", "split_first", "split_second"])
@pytest.mark.parametrize("scale", [0.125, 2.0])
def test_fp8_reanchor_matches_scaled_reference_and_preserves_other_storage(layout, scale):
    storage, keys, values = _pool(layout)
    original = keys.clone()
    original_values = values.clone()
    pointer, shape, stride = storage.data_ptr(), storage.shape, storage.stride()
    inv_freq = torch.tensor([1.0, 0.01])
    plan = PositionReanchor(delta=4, moved_from=8, sink_blocks=1)
    # Logical rows 8..11 become rows 4..7, which still reside in physical page 2.
    for _ in range(2):
        expected = _reference(keys[2].transpose(0, 1), plan.delta, inv_freq, scale).transpose(0, 1)
        count = window_kv.rotate_cached_keys(
            storage,
            block_ids=[0, 2],
            positions=torch.arange(8, 12),
            plan=plan,
            inv_freq=inv_freq,
            block_size=4,
            fp8_dtype=FP8,
        )
        assert count == 4
        assert torch.equal(keys[2], expected)
        assert torch.equal(keys[[0, 1, 3, 4]], original[[0, 1, 3, 4]])
        assert torch.equal(values, original_values)
        assert (storage.data_ptr(), storage.shape, storage.stride()) == (pointer, shape, stride)


def test_fp8_view_storage_uses_byte_indexing_and_saturates_overflow():
    storage = torch.full((3, 2, 4, 4), 448.0).to(FP8)
    pointer = storage.data_ptr()
    window_kv.rotate_cached_keys(
        storage,
        block_ids=[0, 2],
        positions=torch.arange(8, 12),
        plan=PositionReanchor(delta=4, moved_from=8, sink_blocks=1),
        inv_freq=torch.full((2,), math.pi / 16),
        block_size=4,
        fp8_dtype=FP8,
    )
    assert storage.data_ptr() == pointer
    assert torch.isfinite(storage.float()).all()
    assert storage.float()[2].abs().max() == 448


def test_raw_byte_cache_without_encoding_is_rejected_before_mutation():
    storage, _, _ = _pool("packed")
    original = storage.clone()
    with pytest.raises(ValueError, match="requires an explicit"):
        window_kv.rotate_cached_keys(
            storage,
            block_ids=[0, 2],
            positions=torch.arange(8, 12),
            plan=PositionReanchor(delta=4, moved_from=8, sink_blocks=1),
            inv_freq=torch.tensor([1.0, 0.01]),
            block_size=4,
        )
    assert torch.equal(storage, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unquantized_reanchor_keeps_original_dtype_and_value_pages(dtype):
    pool = torch.randn(5, 2, 4, 8, generator=torch.Generator().manual_seed(23)).to(dtype)
    original = pool.clone()
    inv_freq = torch.tensor([1.0, 0.01])
    keys = original[2, ..., :4].transpose(0, 1)
    # Independent reference using the original dtype-rounded sin/cos and
    # elementwise intermediates, with angles still evaluated in FP32.
    angle = 4 * inv_freq
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    a, b = keys.chunk(2, dim=-1)
    expected = torch.cat((a * cos + b * sin, b * cos - a * sin), dim=-1).transpose(0, 1)
    window_kv.rotate_cached_keys(
        pool,
        block_ids=[0, 2],
        positions=torch.arange(8, 12),
        plan=PositionReanchor(delta=4, moved_from=8, sink_blocks=1),
        inv_freq=inv_freq,
        block_size=4,
    )
    assert pool.dtype == dtype
    assert torch.equal(pool[2, ..., :4], expected)
    assert torch.equal(pool[..., 4:], original[..., 4:])
    assert torch.equal(pool[[0, 1, 3, 4]], original[[0, 1, 3, 4]])


@pytest.mark.parametrize(
    "cache_dtype", ["fp8_per_token_head", "fp8_e5m2", "int8_per_token_head", "int4_per_token_head"]
)
def test_unsupported_quantized_layout_fails_at_startup(cache_dtype):
    geometry = window_kv.duplex_window_geometry(prefix_tokens=4, window_tokens=16, block_size=4, max_model_len=64)
    with pytest.raises(ValueError, match="per-tensor"):
        window_kv.validate_duplex_window_install(
            SimpleNamespace(cache_dtype=cache_dtype, enable_prefix_caching=False, block_size=4),
            SimpleNamespace(max_model_len=64),
            geometry,
        )


def test_worker_propagates_actual_fp8_encoding_and_applies_once(monkeypatch, caplog):
    from vllm import platforms

    monkeypatch.setattr(platforms, "current_platform", SimpleNamespace(fp8_dtype=lambda: FP8))
    storage, keys, values = _pool("packed")
    before = keys.clone()
    before_v = values.clone()
    inv_freq = torch.tensor([1.0, 0.01])
    command = {"delta": 4, "moved_from": 8, "sink_blocks": 1, "old_computed_tokens": 12, "reanchor_id": "roll-1"}
    info = {"duplex": {"stage0_reanchor": command.copy()}}
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(block_size=4, cache_dtype="fp8_e4m3"),
        _duplex_inv_freq=inv_freq,
        kv_caches=[storage],
        requests={"req": SimpleNamespace(block_ids=([0, 2],), mrope_positions=None)},
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["req"], num_computed_tokens_cpu=[8]),
        model_intermediate_buffer={"req": info},
    )
    with caplog.at_level(logging.INFO, logger=window_kv.__name__):
        window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)
    assert torch.equal(keys[2], _reference(before[2].transpose(0, 1), 4, inv_freq, 0.125).transpose(0, 1))
    assert torch.equal(values, before_v)
    assert "dtype=torch.float8_e4m3fn delta=4 layers=1 rows_per_layer=4" in caplog.text
    after = storage.clone()
    info["duplex"]["stage0_reanchor"] = command.copy()
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner)
    assert torch.equal(storage, after)
