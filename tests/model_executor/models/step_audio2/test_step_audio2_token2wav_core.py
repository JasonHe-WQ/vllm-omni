# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference setup ownership and the shared eager encoder path."""

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.utils.torch_compile import _undecorate_dynamo

try:
    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
        StepAudio2Token2WavCore,
        _StreamState,
    )
except ModuleNotFoundError as exc:
    if exc.name.split(".")[0] not in {"s3tokenizer", "onnxruntime", "hyperpyyaml", "cosyvoice2", "torchaudio"}:
        raise
    pytest.skip(f"Step-Audio2 extras not installed ({exc.name})", allow_module_level=True)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Encoder(torch.nn.Module):
    def forward_chunk(self, x):
        return x.sin() + 1


def test_only_eager_dynamo_backend_is_unwrapped():
    encoder = _Encoder()
    encoder.forward_chunk = torch.compile(encoder.forward_chunk, backend="eager")
    x = torch.tensor([0.5, 1.5])
    expected = _Encoder().forward_chunk(x)
    assert _undecorate_dynamo(encoder, "forward_chunk")
    torch.testing.assert_close(encoder.forward_chunk(x), expected, rtol=0, atol=0)
    assert not _undecorate_dynamo(encoder, "forward_chunk")


@pytest.mark.parametrize("backend", ["inductor", "aot_eager"])
def test_real_compiler_wrapper_is_preserved(backend):
    encoder = _Encoder()
    compiled = torch.compile(encoder.forward_chunk, backend=backend)
    encoder.forward_chunk = compiled
    assert not _undecorate_dynamo(encoder, "forward_chunk")
    assert encoder.forward_chunk is compiled


def test_plain_and_unknown_decorators_are_preserved():
    encoder = _Encoder()
    assert not _undecorate_dynamo(encoder, "forward_chunk")

    @functools.wraps(encoder.forward_chunk)
    def wrapper(x):
        return x + 8

    encoder.forward_chunk = wrapper
    assert not _undecorate_dynamo(encoder, "forward_chunk")
    assert encoder.forward_chunk is wrapper
    wrapper._torchdynamo_orig_callable = wrapper.__wrapped__
    assert not _undecorate_dynamo(encoder, "forward_chunk")
    assert encoder.forward_chunk is wrapper


class _Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.decoder = SimpleNamespace(inference_cfg_rate=0.7)
        self.up_rate = 2
        self.setup_calls = 0
        self.scratch = torch.zeros(2, 3)

    def setup_cache(self, tokens, mels, speaker, *, n_timesteps):
        self.setup_calls += 1
        # Model the upstream implementation's reusable output workspace.
        self.scratch.fill_(mels.mean())
        return {"conformer_cnn_cache": self.scratch, "estimator_att_cache": self.scratch[:, :2]}


def _core(monkeypatch, *, prompt_cache_size=4, setup_cache_size=1):
    core = StepAudio2Token2WavCore(
        "/fake/model",
        device="cpu",
        prompt_cache_size=prompt_cache_size,
        setup_cache_size=setup_cache_size,
    )
    core._flow = _Flow()
    core._models_loaded = True
    calls = []

    def prepare(path):
        calls.append(path)
        value = float(Path(path).read_text())
        return (
            torch.ones(1, 3, dtype=torch.int64),
            torch.tensor([3]),
            torch.ones(1, 2),
            torch.full((1, 6, 2), value),
            torch.tensor([6]),
        )

    monkeypatch.setattr(core, "_prepare_prompt", prepare)
    return core, calls


def _prompt(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(str(value))
    return str(path)


def test_setup_reuses_reference_but_isolates_work_states(monkeypatch, tmp_path):
    core, prepare_calls = _core(monkeypatch, setup_cache_size=2)
    path = _prompt(tmp_path, "voice.wav", 2)
    first, second, third = _StreamState(), _StreamState(), _StreamState(finished=True)
    core.setup_stream_for(path, first)
    first.stream_cache["conformer_cnn_cache"].fill_(99)
    core.setup_stream_for(path, second)
    torch.testing.assert_close(second.stream_cache["conformer_cnn_cache"], torch.full((2, 3), 2.0))
    assert first.stream_cache["estimator_att_cache"].data_ptr() != second.stream_cache["estimator_att_cache"].data_ptr()
    # A second reference overwrites the upstream workspace, not our saved base.
    core.setup_stream_for(_prompt(tmp_path, "other.wav", 4), _StreamState())
    core.setup_stream_for(path, third)
    torch.testing.assert_close(third.stream_cache["conformer_cnn_cache"], torch.full((2, 3), 2.0))
    assert core.flow.setup_calls == 2
    assert prepare_calls.count(path) == 1
    assert third.setup_done and not third.finished
    assert third.hift_cache_dict is not second.hift_cache_dict
    core.reset_stream_for(second)
    assert third.stream_cache is not None


def test_setup_lru_and_feature_eviction_are_bounded(monkeypatch, tmp_path):
    core, _ = _core(monkeypatch, prompt_cache_size=2, setup_cache_size=1)
    paths = [_prompt(tmp_path, f"{index}.wav", index) for index in range(3)]
    for path in [paths[0], paths[1], paths[0], paths[2]]:
        core.setup_stream_for(path, _StreamState())
    assert list(core.cache) == [paths[0], paths[2]]
    assert list(core._setup_cache) == [paths[2]]
    assert set(core._prompt_versions) == set(core.cache)
    assert core.flow.setup_calls == 4


@pytest.mark.parametrize("changed", ["steps", "precision", "flow_dtype", "cfg", "file"])
def test_setup_invalidates_when_configuration_or_reference_changes(monkeypatch, tmp_path, changed):
    core, calls = _core(monkeypatch)
    path = _prompt(tmp_path, "voice.wav", 2)
    core.setup_stream_for(path, _StreamState())
    if changed == "steps":
        core.n_timesteps = 5
    elif changed == "precision":
        core.float16 = True
    elif changed == "flow_dtype":
        core.flow.double()
    elif changed == "cfg":
        core.flow.decoder.inference_cfg_rate = 0.5
    else:
        stat = os.stat(path)
        Path(path).write_text("3")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    core.setup_stream_for(path, _StreamState())
    assert core.flow.setup_calls == 2
    assert len(calls) == 2
    assert len(core._setup_cache) == 1


@pytest.mark.parametrize("prompt_capacity,setup_capacity", [(0, 1), (1, 0), (0, 0)])
def test_zero_capacity_disables_setup_reuse(monkeypatch, tmp_path, prompt_capacity, setup_capacity):
    core, _ = _core(monkeypatch, prompt_cache_size=prompt_capacity, setup_cache_size=setup_capacity)
    path = _prompt(tmp_path, "voice.wav", 2)
    first = _StreamState()
    core.setup_stream_for(path, first)
    core.setup_stream_for(path, _StreamState())
    assert core.flow.setup_calls == 2
    assert not core._setup_cache
    core.flow.scratch.zero_()
    torch.testing.assert_close(first.stream_cache["conformer_cnn_cache"], torch.full((2, 3), 2.0))


def test_negative_capacity_is_rejected():
    with pytest.raises(ValueError, match="capacities"):
        StepAudio2Token2WavCore("/fake", device="cpu", setup_cache_size=-1)


def test_concurrent_setup_reuses_one_base_and_isolates_requests(monkeypatch, tmp_path):
    core, calls = _core(monkeypatch)
    path = _prompt(tmp_path, "voice.wav", 2)
    states = [_StreamState() for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda state: core.setup_stream_for(path, state), states))
    assert core.flow.setup_calls == len(calls) == 1
    assert len({state.stream_cache["estimator_att_cache"].data_ptr() for state in states}) == len(states)


def test_active_stream_keeps_conditioning_after_reference_change_and_eviction(monkeypatch, tmp_path):
    core, calls = _core(monkeypatch, prompt_cache_size=1)
    path = _prompt(tmp_path, "voice.wav", 2)
    active = _StreamState()
    core.setup_stream_for(path, active)
    original_features = active.prompt_features
    Path(path).write_text("33")
    core.setup_stream_for(path, _StreamState())
    core.setup_stream_for(_prompt(tmp_path, "other.wav", 4), _StreamState())
    Path(path).unlink()
    before = len(calls)
    observed_speakers = []

    def decode(**kwargs):
        observed_speakers.append(kwargs["spk"])
        return torch.zeros(1, 2, 2), {"estimator_att_cache": torch.zeros(1, 1, 2, 1, 1, 1)}

    monkeypatch.setattr(core.flow, "inference_chunk", decode, raising=False)
    core._hift = SimpleNamespace(inference=lambda *args: (torch.ones(1, 8), torch.zeros(1, 1, 8)))
    core.stream_chunk_for([1, 2], path, True, active)
    assert len(calls) == before
    assert active.prompt_features is original_features
    assert observed_speakers[0] is original_features[2]
    torch.testing.assert_close(active.prompt_features[3], torch.full((1, 6, 2), 2.0))
    core.reset_stream_for(active)
    assert active.prompt_features is active.config_signature is None


def test_active_stream_rejects_configuration_change(monkeypatch, tmp_path):
    core, _ = _core(monkeypatch)
    path = _prompt(tmp_path, "voice.wav", 2)
    state = _StreamState()
    core.setup_stream_for(path, state)
    core.n_timesteps += 1
    with pytest.raises(RuntimeError, match="configuration changed"):
        core.stream_chunk_for([1], path, True, state)


def test_active_stream_rejects_different_reference(monkeypatch, tmp_path):
    core, _ = _core(monkeypatch)
    path = _prompt(tmp_path, "voice.wav", 2)
    state = _StreamState()
    core.setup_stream_for(path, state)
    with pytest.raises(ValueError, match="reference changed"):
        core.stream_chunk_for([1], "different.wav", True, state)


def test_core_loader_bypasses_eager_encoder(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_token2wav as module

    (tmp_path / "flow.yaml").write_text("fake flow")
    flow = _Flow()
    flow.encoder = _Encoder()
    flow.encoder.forward_chunk = torch.compile(flow.encoder.forward_chunk, backend="eager")
    hift = torch.nn.Linear(2, 2)

    def load_tokenizer(_):
        assert torch.empty(1).device.type == "cpu"
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(module.s3tokenizer, "load_model", load_tokenizer)
    monkeypatch.setattr(module, "load_hyperpyyaml", lambda _: {"flow": flow})
    monkeypatch.setattr(module, "_build_hift", lambda: hift)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, **kwargs: flow.state_dict() if path.endswith("flow.pt") else hift.state_dict(),
    )
    core = StepAudio2Token2WavCore(str(tmp_path), device="cpu")
    with torch.device("meta"):
        core._ensure_models_loaded()
    assert not hasattr(core.flow.encoder.forward_chunk, "_torchdynamo_orig_callable")
    torch.testing.assert_close(core.flow.encoder.forward_chunk(torch.zeros(1)), torch.ones(1))
    assert not core._drop_chunk_att_buffers


def test_minicpm_facade_keeps_complete_stream_state(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import MiniCPMO45Token2wav

    core, _ = _core(monkeypatch)
    facade = MiniCPMO45Token2wav.__new__(MiniCPMO45Token2wav)
    facade._core = core
    facade.stream_cache = facade._stream_state = None
    facade.hift_cache_dict = {}
    path = _prompt(tmp_path, "voice.wav", 2)
    facade.set_stream_cache(path)
    state = facade._stream_state
    seen = []

    def stream(tokens, path, last_chunk, received_state):
        seen.append(received_state)
        assert received_state.prompt_features is not None
        assert received_state.config_signature is not None
        return torch.zeros(8)

    monkeypatch.setattr(core, "stream_chunk_for", stream)
    facade.stream([1, 2], path, return_waveform=True)
    facade.stream([3, 4], path, last_chunk=True, return_waveform=True)
    assert seen == [state, state]


def test_lazy_speaker_initialization_does_not_invalidate_first_setup(monkeypatch, tmp_path):
    core, calls = _core(monkeypatch)
    original_prepare = core._prepare_prompt

    def prepare(path):
        core._spk_model = SimpleNamespace()
        return original_prepare(path)

    monkeypatch.setattr(core, "_prepare_prompt", prepare)
    path = _prompt(tmp_path, "voice.wav", 2)
    core.setup_stream_for(path, _StreamState())
    core.setup_stream_for(path, _StreamState())
    assert core.flow.setup_calls == len(calls) == 1


def test_half_flow_setup_uses_autocast_for_float_conditioning(monkeypatch, tmp_path):
    core, _ = _core(monkeypatch)
    core.float16 = True
    core.flow.half()
    projection = torch.nn.Linear(2, 2).half()
    original_setup = core.flow.setup_cache

    def setup(tokens, mels, speaker, **kwargs):
        assert speaker.dtype == torch.float32
        assert projection(speaker).dtype == torch.float16
        return original_setup(tokens, mels, speaker, **kwargs)

    monkeypatch.setattr(core.flow, "setup_cache", setup)
    core.setup_stream_for(_prompt(tmp_path, "voice.wav", 2), _StreamState())


def test_overlap_size_change_rebuilds_window_and_setup(monkeypatch, tmp_path):
    core, _ = _core(monkeypatch)
    path = _prompt(tmp_path, "voice.wav", 2)
    core.setup_stream_for(path, _StreamState())
    core.source_cache_len = 16
    core.setup_stream_for(path, _StreamState())
    assert core.speech_window.numel() == 32
    assert core.flow.setup_calls == 2


def test_reference_tokenizer_can_stay_on_cpu_while_results_move_to_flow_device(monkeypatch):
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_token2wav as module

    # Meta models a distinct destination without using a GPU in this CPU test.
    core = StepAudio2Token2WavCore("/fake", device="meta", audio_tokenizer_device="cpu")
    core._models_loaded = True
    core._flow = _Flow()

    class Tokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def quantize(self, mels, lengths):
            assert mels.device.type == lengths.device.type == self.weight.device.type == "cpu"
            self.calls += 1
            return torch.tensor([[1, 2, 3]], dtype=torch.int64), torch.tensor([3], dtype=torch.int32)

    tokenizer = Tokenizer()
    core._audio_tokenizer = tokenizer
    monkeypatch.setattr(module, "load_audio", lambda *args, **kwargs: (np.ones(1600, dtype=np.float32), 16000))
    monkeypatch.setattr(module, "AudioResampler", lambda **kwargs: SimpleNamespace(resample=lambda audio, **kw: audio))
    monkeypatch.setattr(module.s3tokenizer, "log_mel_spectrogram", lambda _: torch.ones(128, 12))
    monkeypatch.setattr(module.s3tokenizer, "padding", lambda _: (torch.ones(1, 128, 12), torch.tensor([12])))
    monkeypatch.setattr(module.kaldi, "fbank", lambda *args, **kwargs: torch.ones(8, 80))
    monkeypatch.setattr(core, "_forward_spk_embedding", lambda _: torch.ones(1, 192, device="meta"))
    monkeypatch.setattr(module, "mel_spectrogram", lambda *args, **kwargs: torch.ones(1, 80, 6))
    first = core._prepare_prompt("/fake/reference.wav")
    second = core._prepare_prompt("/fake/reference.wav")
    assert tokenizer.calls == 2
    assert tokenizer.weight.device.type == "cpu"
    assert all(tensor.device.type == "meta" for tensor in (*first, *second))
    assert first[0].dtype == torch.int64 and first[1].dtype == torch.int32


def test_reference_tokenizer_device_defaults_to_flow_device():
    core = StepAudio2Token2WavCore("/fake", device="meta")
    assert core.audio_tokenizer_device == core.device
