# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.omni_config import OmniStageModelConfig, VllmOmniConfig
from vllm_omni.config.stage_config import DeployConfig, merge_pipeline_deploy
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_code2wav_input_width_survives_typed_and_legacy_transport(monkeypatch):
    deploy = DeployConfig()
    config = VllmOmniConfig.from_pipeline_config(MINICPMO_4_5_PIPELINE, user_deploy_config=deploy)
    legacy = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, deploy)
    for sid in (0, 1):
        assert config.stage_by_id(sid).model_config.inputs_embeds_size is None
        assert "inputs_embeds_size" not in legacy[sid].yaml_engine_args
    stage = ForkingPickler.loads(ForkingPickler.dumps(config.stage_by_id(2)))
    args = build_engine_args_dict_from_omni_stage_config(stage, model="/fake/model")
    assert stage.model_config.inputs_embeds_size == args["inputs_embeds_size"] == 1
    assert legacy[2].yaml_engine_args["inputs_embeds_size"] == 1

    # Exercise the final EngineArgs -> OmniModelConfig handoff without loading
    # a checkpoint, tokenizer or GPU model in this configuration test.
    base = object()
    monkeypatch.setattr(EngineArgs, "create_model_config", lambda self: base)
    monkeypatch.setattr(OmniEngineArgs, "_ensure_omni_models_registered", lambda self: True)
    monkeypatch.setattr(OmniEngineArgs, "_patch_empty_hf_config", lambda self, model_type: None)
    monkeypatch.setattr(
        OmniModelConfig,
        "from_vllm_model_config",
        lambda model_config, **kwargs: SimpleNamespace(base=model_config, **kwargs),
    )
    terminal = OmniEngineArgs(**args).create_model_config()
    assert terminal.base is base
    assert terminal.inputs_embeds_size == 1


@pytest.mark.parametrize("hidden_size", [4096, 768])
def test_unspecified_input_width_keeps_checkpoint_geometry(hidden_size):
    config = object.__new__(OmniModelConfig)
    config.inputs_embeds_size = None
    config.hf_config_name = None
    config.hf_text_config = SimpleNamespace(hidden_size=hidden_size)
    config.model_arch_config = SimpleNamespace(hidden_size=hidden_size)
    assert config.get_inputs_embeds_size() == hidden_size
    assert config.get_hidden_size() == hidden_size


def test_code2wav_width_matches_real_input_adapter_without_changing_hidden_size():
    config = object.__new__(OmniModelConfig)
    config.inputs_embeds_size = MINICPMO_4_5_PIPELINE.get_stage(2).inputs_embeds_size
    config.hf_config_name = None
    config.hf_config = config.hf_text_config = SimpleNamespace(hidden_size=4096)
    config.model_arch_config = SimpleNamespace(hidden_size=4096)
    tokens = torch.arange(7)
    embeddings = MiniCPMO45Code2Wav.embed_input_ids(object(), tokens)
    assert embeddings.shape == (tokens.numel(), config.get_inputs_embeds_size()) == (7, 1)
    assert config.hf_config.hidden_size == config.get_hidden_size() == 4096
    # Preserve 8192-token staging capacity, changing only its unused width.
    assert 8192 * config.get_inputs_embeds_size() * 2 == 16 * 1024


def test_existing_stage_embedding_size_override_is_preserved():
    config = object.__new__(OmniModelConfig)
    config.inputs_embeds_size = None
    config.hf_config_name = "tts_config"
    config.hf_config = SimpleNamespace(tts_config=SimpleNamespace(embedding_size=1024))
    config.hf_text_config = SimpleNamespace(hidden_size=768)
    config.model_arch_config = SimpleNamespace(hidden_size=768)
    assert config.get_inputs_embeds_size() == 1024
    assert config.get_hidden_size() == 768


@pytest.mark.parametrize("width", [0, -1])
def test_declared_input_width_must_be_positive(width):
    with pytest.raises(ValidationError):
        OmniStageModelConfig(inputs_embeds_size=width)
