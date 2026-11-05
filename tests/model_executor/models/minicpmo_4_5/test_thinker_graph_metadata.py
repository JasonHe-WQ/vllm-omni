# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Thinker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, **kwargs):
        return torch.ones(input_ids.numel(), 4)


def _model():
    model = object.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model.have_multimodal_outputs = True
    model._use_v2_model_runner = False
    model.thinker = _Thinker()
    return model


def _info(prompt, *, prefill=True):
    return {
        "_omni_is_prefill": prefill,
        "_omni_num_computed_tokens": len(prompt) - 1,
        "_omni_prompt_len": len(prompt),
        "duplex": {"duplex_prompt_token_ids": prompt, "special_token_ids": {"tts_bos_token_id": 91}},
    }


def _runner(model, infos):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner._build_model_kwargs_extra = lambda: {"runtime_additional_information": infos}
    return runner


def _forward(model, infos=None):
    return model.forward(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, 4),
        runtime_additional_information=infos,
    )


def test_cold_graph_replay_publishes_one_token_prefill_metadata(monkeypatch):
    model = _model()
    captured = _forward(model)
    assert "duplex_prompt_token_ids" not in captured.multimodal_outputs
    runner = _runner(model, [_info([11, 12, 13])])
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: tuple(captured))
    output = runner._model_forward(input_ids=torch.tensor([13]), positions=torch.tensor([2]))
    assert isinstance(output, OmniOutput)
    assert output.multimodal_outputs["duplex_prompt_token_ids"] == [[11, 12, 13]]
    assert output.multimodal_outputs["meta"]["tts_bos_token_id"][0].item() == 91
    torch.testing.assert_close(output.multimodal_outputs["latent_input_ids"], torch.tensor([[13]]))
    torch.testing.assert_close(output.multimodal_outputs["latent_positions"], torch.tensor([[2]]))
    assert "duplex_prompt_token_ids" not in captured.multimodal_outputs


def test_replay_between_sessions_replaces_only_live_snapshot(monkeypatch):
    model = _model()
    captured = _forward(model, [_info([1, 2])])
    infos = [_info([21, 22, 23])]
    runner = _runner(model, infos)
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: tuple(captured))
    output = runner._model_forward(input_ids=torch.tensor([23]), positions=torch.tensor([2]))
    assert output.multimodal_outputs["duplex_prompt_token_ids"] == [[21, 22, 23]]
    infos[:] = [_info([31, 32, 33, 34])]
    following = runner._model_forward(input_ids=torch.tensor([34]), positions=torch.tensor([3]))
    assert following.multimodal_outputs["duplex_prompt_token_ids"] == [[31, 32, 33, 34]]
    assert output.multimodal_outputs["duplex_prompt_token_ids"] == [[21, 22, 23]]
    assert captured.multimodal_outputs["duplex_prompt_token_ids"] == [[1, 2]]


def test_decode_replay_drops_capture_metadata_and_refreshes_ledger(monkeypatch):
    model = _model()
    captured = _forward(model, [_info([1, 2])])
    runner = _runner(model, [_info([21, 22, 23], prefill=False)])
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: tuple(captured))
    output = runner._model_forward(input_ids=torch.tensor([41]), positions=torch.tensor([[3]]))
    assert "duplex_prompt_token_ids" not in output.multimodal_outputs
    assert "meta" not in output.multimodal_outputs
    torch.testing.assert_close(output.multimodal_outputs["latent_input_ids"], torch.tensor([[41]]))
    torch.testing.assert_close(output.multimodal_outputs["latent_positions"], torch.tensor([[3]]))


def test_eager_and_direct_forward_contract_remains_unchanged(monkeypatch):
    model = _model()
    infos = [_info([11, 12], prefill=False)]
    eager = _forward(model, infos)
    assert eager.multimodal_outputs["duplex_prompt_token_ids"] == [[11, 12]]
    runner = _runner(model, infos)
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: eager)
    monkeypatch.setattr(model, "refresh_omni_output_after_replay", lambda *args, **kwargs: pytest.fail("eager hook"))
    assert runner._model_forward() is eager


@pytest.mark.parametrize("stage,v2", [("tts", False), ("llm", True)])
def test_replay_refresh_does_not_change_talker_or_mrv2(stage, v2):
    output = OmniOutput(torch.ones(1, 4), {"meta": {"keep": torch.tensor([1])}})
    model = SimpleNamespace(model_stage=stage, _use_v2_model_runner=v2)
    actual = MiniCPMO45OmniForConditionalGeneration.refresh_omni_output_after_replay(
        model, output, input_ids=None, positions=None, runtime_additional_information=[]
    )
    assert actual is output
