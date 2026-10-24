# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay an unchanged, stateless single-row greedy output head.

This owns execution buffers, not a cache of tokens or hidden-state results.
Every replay overwrites the complete input and returns an independent token
tensor, since several draft results stay live until the proposer stacks them.
"""

from collections.abc import Callable

import torch


class GreedyHeadGraph:
    def __init__(self, compute_logits: Callable[[torch.Tensor], torch.Tensor]):
        self._compute_logits = compute_logits
        self._graph = None
        self._input = None
        self._output = None
        self._signature = None
        self._stream = None

    def _eager(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._compute_logits(hidden_states).argmax(dim=-1)

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Never nest capture or change execution for training, CPU, batches or
        # unusual layouts. The proposer also guards the model and TP/PP size.
        if (
            not hidden_states.is_cuda
            or hidden_states.ndim != 2
            or hidden_states.shape[0] != 1
            or not hidden_states.is_contiguous()
            or hidden_states.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            or torch.is_grad_enabled()
        ):
            return self._eager(hidden_states)

        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                return self._eager(hidden_states)
            stream = torch.cuda.current_stream(hidden_states.device)
            signature = (
                tuple(hidden_states.shape),
                hidden_states.dtype,
                hidden_states.device,
            )
            if self._graph is not None:
                # Bound storage to one signature and one owning stream. An
                # unexpected caller uses eager execution, not a second cache.
                if signature != self._signature or stream != self._stream:
                    return self._eager(hidden_states)
                self._input.copy_(hidden_states)
            else:
                static_input = torch.empty_like(hidden_states)
                static_input.copy_(hidden_states)
                capture_stream = torch.cuda.Stream(device=hidden_states.device)
                capture_stream.wait_stream(stream)
                # Warm the exact head operations on a side stream before
                # capture. In this resident, synthetic boot requests reach
                # this path before it reports ready to the adapter.
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        self._eager(static_input)
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    static_output = self._eager(static_input)
                stream.wait_stream(capture_stream)
                # Publish only a completed capture. Capture failures propagate;
                # no failed CUDA capture is silently treated as usable state.
                self._input = static_input
                self._output = static_output
                self._signature = signature
                self._stream = stream
                self._graph = graph

            self._graph.replay()
            # Without this snapshot, subsequent drafts would overwrite every
            # entry in draft_token_ids_list with the most recent token.
            return self._output.clone()
