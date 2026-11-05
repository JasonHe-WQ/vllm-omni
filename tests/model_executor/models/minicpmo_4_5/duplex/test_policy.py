# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import (
    MiniCPMO45DuplexPolicy,
    MiniCPMO45DuplexWindowConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("hf_config", "high", "low", "resident_bound"),
    [
        (None, 8000, 6000, 8097),
        (SimpleNamespace(), 8000, 6000, 8097),
        (SimpleNamespace(duplex_window_high_tokens=16000, duplex_window_low_tokens=12000), 16000, 12000, 16097),
    ],
)
def test_deployment_geometry_and_session_defaults_share_watermarks(hf_config, high, low, resident_bound):
    config = MiniCPMO45DuplexWindowConfig.from_hf_config(hf_config)
    geometry = config.startup_geometry(block_size=16, max_model_len=16384)

    assert (config.basic_window_high_tokens, config.basic_window_low_tokens) == (high, low)
    assert (geometry.trigger_tokens, geometry.window_tokens) == (high, low)
    # Session watermarks count the prefix; startup keeps its historical extra
    # prefix allowance, so default 8k admission/cache sizing stays unchanged.
    assert geometry.prefix_tokens + geometry.trigger_tokens + geometry.sample_room == resident_bound


@pytest.mark.parametrize(
    ("high", "low"),
    [(True, 6000), (16000, False), (0, 6000), (16000, 0), (16000, 16000), (16000, 17000)],
)
def test_invalid_deployment_watermarks_fail_before_model_setup(high, low):
    with pytest.raises(ValueError):
        MiniCPMO45DuplexWindowConfig.from_hf_config(
            SimpleNamespace(duplex_window_high_tokens=high, duplex_window_low_tokens=low)
        )


@pytest.mark.parametrize(
    ("samples", "tokens"),
    [(0, 0), (1280, 0), (1281, 1), (1599, 1), (1600, 1), (2880, 1), (2881, 2), (96256, 60), (97281, 61)],
)
def test_reference_audio_token_budget_preserves_partial_frames(samples, tokens):
    assert MiniCPMO45DuplexPolicy.audio_token_count(samples) == tokens


@pytest.mark.parametrize("samples", [1281, 1599, 1600, 2880, 2881, 96256])
def test_reference_audio_budget_matches_feature_extractor_and_pooling(samples):
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor

    processor = WhisperFeatureExtractor(feature_size=80)
    features = processor(np.zeros(samples, dtype=np.float32), sampling_rate=16000, return_attention_mask=True)
    mel_frames = int(np.asarray(features["attention_mask"]).sum())
    # Exercise the encoder's actual shape-changing operations, including the
    # CNN padding that makes clips just below 100 ms produce one audio token.
    conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
    with torch.no_grad():
        pooled = torch.nn.functional.avg_pool1d(conv(torch.zeros(1, 1, mel_frames)), kernel_size=5, stride=5)
    assert MiniCPMO45DuplexPolicy.audio_token_count(samples) == pooled.shape[-1]


def test_seed_tts_initial_text_is_part_of_native_duplex_context():
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        "Speak exactly.",
        True,
        "The quick brown fox.",
    )

    assert prefix == "<|im_start|>system\nSpeak exactly.\n<|audio_start|>"
    assert suffix == (
        "<|audio_end|><|im_end|>\n<|im_start|>user\nThe quick brown fox.<|im_end|>\n<|im_start|>assistant\n"
    )
