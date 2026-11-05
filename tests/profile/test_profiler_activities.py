# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configured activities must reach the worker's device profiler unchanged."""

from types import SimpleNamespace

import pytest
from vllm.config import ProfilerConfig

import vllm_omni.profiler.omni_torch_profiler as profiler_module
from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def profiler_calls(monkeypatch):
    calls = []

    def profile(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(profiler_module.torch.profiler, "profile", profile)
    return calls


@pytest.mark.parametrize(
    ("configured", "explicit", "expected"),
    [
        (["CUDA"], None, ["CUDA"]),
        (["CPU"], None, ["CPU"]),
        (["CPU"], ["CUDA"], ["CUDA"]),
        (["CUDA"], ["CPU"], ["CPU"]),
        (["PrivateUse1"], None, ["PrivateUse1"]),
        (None, None, ["CPU", "CUDA"]),
    ],
)
def test_activity_selection_reaches_torch_profiler(tmp_path, profiler_calls, configured, explicit, expected):
    config = ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path), torch_profiler_activities=configured)
    selected = OmniTorchProfilerWrapper(config, "worker0", 0, activities=explicit)

    assert profiler_calls[0]["activities"] == [
        profiler_module.TorchProfilerActivityMap[activity] for activity in expected
    ]
    assert selected._activities == expected
    assert selected.dump_cpu_time_total is (expected == ["CPU"])
    assert selected._has_cuda_like_activity() is ("CUDA" in expected)


@pytest.mark.parametrize("configured", [None, ["CPU"]])
def test_platform_activity_default_is_used_only_when_config_is_unset(tmp_path, profiler_calls, configured):
    class PlatformProfiler(OmniTorchProfilerWrapper):
        def _get_default_activities(self):
            return ["CPU", "XPU"]

    config = ProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path), torch_profiler_activities=configured)
    selected = PlatformProfiler(config, "worker0", 0)
    expected = ["CPU", "XPU"] if configured is None else configured
    assert selected._activities == expected
    assert profiler_calls[0]["activities"] == [
        profiler_module.TorchProfilerActivityMap[activity] for activity in expected
    ]


def test_private_use_config_keeps_npu_memory_history_selection(tmp_path, profiler_calls, monkeypatch):
    memory = object()
    monkeypatch.setattr(
        profiler_module.torch, "npu", SimpleNamespace(is_available=lambda: True, memory=memory), raising=False
    )
    config = ProfilerConfig(
        profiler="torch", torch_profiler_dir=str(tmp_path), torch_profiler_activities=["PrivateUse1"]
    )
    selected = OmniTorchProfilerWrapper(config, "worker0", 0)
    assert selected._resolve_memory_history_backend() == ("NPU", memory)
