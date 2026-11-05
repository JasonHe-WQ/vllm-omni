# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned workspaces for the upstream CosyVoice chunk solver."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from types import MethodType

import torch
from torch import nn


def _capturing(reference: torch.Tensor) -> bool:
    capture_query = getattr(torch.get_device_module(reference.device), "is_current_stream_capturing", None)
    return bool(capture_query is not None and capture_query())


class _ChunkScratch:
    def __init__(self, decoder: nn.Module, estimator: nn.Module) -> None:
        self.decoder = decoder
        self.estimator = estimator
        self.solve = decoder.solve_euler_chunk
        self.step = estimator.forward_chunk
        self.lock = RLock()
        self.shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        # Captures retain their own fixed addresses. Eager calls never borrow
        # these buffers, including after a captured call has completed.
        self.graph_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        for tag, module in (("solver", decoder), ("estimator", estimator)):
            for name in ("cnn_cache_buffer", "att_cache_buffer"):
                buffer = module._buffers[name]
                self.shapes[tag, name] = tuple(buffer.shape)
                # Keep registered empty templates so .to() / .half() still
                # supply the upstream workspace's device and precision.
                module._buffers[name] = buffer.new_empty((0,))

    @contextmanager
    def workspace(
        self, tag: str, module: nn.Module, reference: torch.Tensor, batch: int, frames: int, steps: int | None = None
    ) -> Iterator[None]:
        with self.lock:
            names = ("cnn_cache_buffer", "att_cache_buffer")
            templates = tuple(module._buffers[name] for name in names)
            shapes = [list(self.shapes[tag, name]) for name in names]
            for shape in shapes:
                shape[1 if steps is None else 2] = batch
                if steps is not None:
                    shape[0] = steps
            shapes[1][-2] = frames
            graph_key = (tag, *(tuple(shape) for shape in shapes), reference.device, templates[0].dtype)
            capturing = _capturing(reference)
            buffers = self.graph_buffers.get(graph_key) if capturing else None
            if buffers is None:
                buffers = tuple(template.new_empty(shape) for template, shape in zip(templates, shapes, strict=True))
                if capturing:
                    self.graph_buffers[graph_key] = buffers
            try:
                for name, buffer in zip(names, buffers, strict=True):
                    module._buffers[name] = buffer
                yield
            finally:
                # Returned caches keep the call's allocations alive. Removing
                # these module references frees scratch on errors too, without
                # overwriting tensors still owned by another request.
                for name, template in zip(names, templates, strict=True):
                    module._buffers[name] = template


def _solve_euler_chunk(self, x, t_span, mu, spks, cond, cnn_cache=None, att_cache=None):
    scratch = self._omni_chunk_scratch
    old_frames = 0 if att_cache is None else int(att_cache.shape[4])
    with scratch.workspace("solver", self, x, 2 * int(x.shape[0]), old_frames + int(x.shape[2]), len(t_span) - 1):
        return scratch.solve(x, t_span, mu, spks, cond, cnn_cache=cnn_cache, att_cache=att_cache)


def _forward_chunk(self, x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
    scratch = self._omni_chunk_scratch
    old_frames = 0 if att_cache is None else int(att_cache.shape[3])
    frames = int(x.shape[2])
    # Native DiT graphs own separate static buffers. Leave both those buffers
    # and their replay path alone; only its eager fallback needs scratch.
    if (
        getattr(self, "use_cuda_graph", False)
        and att_cache is not None
        and frames in self.graph_chunk
        and old_frames <= self.max_size_chunk[frames]
    ):
        with scratch.lock:
            return scratch.step(x, mu, t, spks, cond, cnn_cache=cnn_cache, att_cache=att_cache)
    with scratch.workspace("estimator", self, x, int(x.shape[0]), old_frames + frames):
        return scratch.step(x, mu, t, spks, cond, cnn_cache=cnn_cache, att_cache=att_cache)


def enable_dynamic_chunk_scratch(flow: nn.Module) -> bool:
    """Adapt the supported upstream solver before it serves any requests.

    The batched MiniCPM backend drops these buffers entirely and needs no
    adapter. Unknown implementations and already-initialized native graphs
    are left unchanged rather than resizing storage a graph may reference.
    """
    decoder = getattr(flow, "decoder", None)
    estimator = getattr(decoder, "estimator", None)
    if not isinstance(decoder, nn.Module) or not isinstance(estimator, nn.Module):
        return False
    if hasattr(decoder, "_omni_chunk_scratch"):
        return True
    if not callable(getattr(decoder, "solve_euler_chunk", None)) or not callable(
        getattr(estimator, "forward_chunk", None)
    ):
        return False
    if getattr(estimator, "graph_chunk", None) or getattr(estimator, "use_cuda_graph", False):
        return False
    for module, dims in ((decoder, (5, 6)), (estimator, (4, 5))):
        for name, ndim in zip(("cnn_cache_buffer", "att_cache_buffer"), dims, strict=True):
            buffer = module._buffers.get(name)
            if not isinstance(buffer, torch.Tensor) or buffer.ndim != ndim:
                return False
    scratch = _ChunkScratch(decoder, estimator)
    decoder._omni_chunk_scratch = scratch
    estimator._omni_chunk_scratch = scratch
    decoder.solve_euler_chunk = MethodType(_solve_euler_chunk, decoder)
    estimator.forward_chunk = MethodType(_forward_chunk, estimator)
    return True
