# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow helpers for upstream modules that trace an otherwise eager method."""

import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def _undecorate_dynamo(module: nn.Module, method: str) -> bool:
    """Bypass only an identifiable ``torch.compile(backend="eager")`` wrapper.

    CosyVoice's chunk encoder uses that backend, which adds tracing/guards but
    executes the original operations. Leave real compilation and unfamiliar
    wrapper layouts intact. Do not unwrap ordinary decorators such as
    ``inference_mode`` merely because they expose ``__wrapped__``.
    """
    bound = getattr(module, method, None)
    original = getattr(bound, "_torchdynamo_orig_callable", None)
    function = getattr(bound, "__func__", bound)
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None)
    if original is None or code is None or closure is None:
        return False

    cells = dict(zip(code.co_freevars, closure))
    callback_cell = cells.get("callback")
    if callback_cell is None:
        return False
    try:
        backend = callback_cell.cell_contents
    except ValueError:
        return False
    from torch._dynamo.backends.registry import lookup_backend

    eager = lookup_backend("eager")
    seen: set[int] = set()
    while backend is not None and id(backend) not in seen:
        seen.add(id(backend))
        if getattr(backend, "compiler_fn", None) is eager:
            original_function = getattr(original, "__func__", original)
            module.__dict__[method] = original_function.__get__(module, type(module))
            logger.debug("Bypassed eager TorchDynamo wrapper on %s.%s", type(module).__name__, method)
            return True
        backend = getattr(backend, "_torchdynamo_orig_backend", None)
    return False
