# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.step_audio2 import eager_scratch

CFM = pytest.importorskip("cosyvoice2.flow.flow_matching").CausalConditionalCFM
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Estimator(nn.Module):
    """Small estimator with the upstream borrowed-output buffer contract."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cnn_cache_buffer", torch.zeros(1, 2, 2, 1), persistent=False)
        self.register_buffer("att_cache_buffer", torch.zeros(1, 2, 1, 32, 2), persistent=False)
        self.use_cuda_graph = False
        self.graph_chunk = {}
        self.max_size_chunk = {}
        self.fail = False

    def forward_chunk(self, x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        if self.fail:
            raise RuntimeError("estimator failure")
        batch, _, frames = x.shape
        old_frames = 0 if att_cache is None else att_cache.shape[3]
        marker = x.mean((1, 2)) + mu.mean((1, 2))
        if cnn_cache is not None:
            marker = marker + cnn_cache.mean((0, 2, 3))
        self.cnn_cache_buffer.copy_(marker.reshape(1, batch, 1, 1))
        self.att_cache_buffer[..., :frames, :].copy_(marker.reshape(1, batch, 1, 1, 1))
        if old_frames:
            self.att_cache_buffer[..., frames : frames + old_frames, :].copy_(att_cache)
        return (
            x * 0.1 + marker[:, None, None],
            self.cnn_cache_buffer,
            self.att_cache_buffer[..., : frames + old_frames, :],
        )


class _SmallCFM(CFM):
    def __init__(self):
        # Reuse the real upstream Euler solver, without its 2 GiB constructor.
        nn.Module.__init__(self)
        self.estimator = _Estimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("cnn_cache_buffer", torch.zeros(6, 1, 2, 2, 1), persistent=False)
        self.register_buffer("att_cache_buffer", torch.zeros(6, 1, 2, 1, 32, 2), persistent=False)


def _run(decoder, *, steps=2, frames=4, marker=1.0, cache=None):
    return decoder.solve_euler_chunk(
        x=torch.full((1, 1, frames), marker),
        t_span=torch.linspace(0, 1, steps + 1),
        mu=torch.ones(1, 1, frames),
        spks=torch.ones(1, 1),
        cond=torch.zeros(1, 1, frames),
        cnn_cache=None if cache is None else cache[1],
        att_cache=None if cache is None else cache[2],
    )


def _snapshot(outputs, steps):
    return outputs[0].clone(), outputs[1][:steps].clone(), outputs[2][:steps].clone()


def _assert_released(decoder):
    assert all(
        module._buffers[name].numel() == 0
        for module in (decoder, decoder.estimator)
        for name in ("cnn_cache_buffer", "att_cache_buffer")
    )


def test_eager_scratch_matches_solver_and_preserves_interleaved_request_caches():
    baseline, decoder = _SmallCFM(), _SmallCFM()
    assert eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    assert eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    _assert_released(decoder)
    first = _run(decoder)
    expected_first = _snapshot(_run(baseline), 2)
    torch.testing.assert_close(first, expected_first)
    assert first[2].shape == (2, 1, 2, 1, 4, 2)
    _assert_released(decoder)

    second = _run(decoder, steps=3, frames=7, marker=5.0)
    expected_second = _snapshot(_run(baseline, steps=3, frames=7, marker=5.0), 3)
    torch.testing.assert_close(second, expected_second)
    torch.testing.assert_close(first, expected_first)
    assert first[2].untyped_storage().data_ptr() != second[2].untyped_storage().data_ptr()
    _assert_released(decoder)

    continued = _run(decoder, frames=3, cache=first)
    expected = _snapshot(_run(baseline, frames=3, cache=expected_first), 2)
    torch.testing.assert_close(continued, expected)
    torch.testing.assert_close(first, expected_first)
    assert continued[2].shape[4] == 7
    _assert_released(decoder)


def test_eager_scratch_releases_module_references_on_failure():
    decoder = _SmallCFM()
    eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    decoder.estimator.fail = True
    with pytest.raises(RuntimeError, match="estimator failure"):
        _run(decoder)
    _assert_released(decoder)


def test_capture_scratch_addresses_survive_eager_calls(monkeypatch):
    decoder = _SmallCFM()
    eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    capturing = True
    monkeypatch.setattr(eager_scratch, "_capturing", lambda _: capturing)
    captured = _run(decoder)
    pointers = [tensor.data_ptr() for pair in decoder._omni_chunk_scratch.graph_buffers.values() for tensor in pair]
    snapshot = _snapshot(captured, 2)
    capturing = False
    _run(decoder, steps=3, frames=9, marker=7.0)
    torch.testing.assert_close(captured, snapshot)
    capturing = True
    repeated = _run(decoder)
    assert repeated[2].data_ptr() == captured[2].data_ptr()
    assert pointers == [
        tensor.data_ptr() for pair in decoder._omni_chunk_scratch.graph_buffers.values() for tensor in pair
    ]
    _assert_released(decoder)


def test_existing_native_graph_buffers_are_not_replaced():
    decoder = _SmallCFM()
    decoder.estimator.graph_chunk = {4: object()}
    original = decoder.att_cache_buffer
    assert not eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    assert decoder.att_cache_buffer is original


def test_dropped_or_unknown_workspaces_are_not_adapted():
    decoder = _SmallCFM()
    del decoder.estimator._buffers["att_cache_buffer"]
    assert not eager_scratch.enable_dynamic_chunk_scratch(SimpleNamespace(decoder=decoder))
    assert not hasattr(decoder, "_omni_chunk_scratch")
