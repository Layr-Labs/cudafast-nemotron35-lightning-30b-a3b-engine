# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage graphs of the current draft screen, under a serial model lifetime.

Stage zero copies its transient gathered row into graph-owned input storage.
Later stages bind the live producer view and use eager computation on a
signature mismatch. Each output is consumed by the proposer's final stack
before that stage can be replayed by the next proposal.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class _ScreenGraphEntry:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor
    token_ids: torch.Tensor
    signature: tuple[object, ...]


class MTPScreenStageGraphs:
    def __init__(
        self,
        sample_tokens: Callable[[torch.Tensor], torch.Tensor],
        screen_owner: object,
        model_owner: object,
        num_stages: int,
    ) -> None:
        self._sample_tokens = sample_tokens
        self.screen_owner = screen_owner
        # Retain the producer wrapper/pool, not only a weak output alias.
        self.model_owner = model_owner
        self._entries: list[_ScreenGraphEntry | None] = [None] * num_stages

    def sample(self, hidden_states: torch.Tensor, draft_index: int) -> torch.Tensor:
        if (
            not 0 <= draft_index < len(self._entries)
            or not hidden_states.is_cuda
            or hidden_states.shape != (1, 2688)
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or torch.is_grad_enabled()
        ):
            return self._sample_tokens(hidden_states)

        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                return self._sample_tokens(hidden_states)
            stream = torch.cuda.current_stream(hidden_states.device)
            staged = draft_index == 0
            address = () if staged else (
                hidden_states.data_ptr(),
                hidden_states.untyped_storage().data_ptr(),
                hidden_states.storage_offset(),
            )
            signature = (
                tuple(hidden_states.shape), tuple(hidden_states.stride()),
                hidden_states.dtype, hidden_states.device, address,
                stream.cuda_stream, torch.is_inference_mode_enabled(),
            )
            entry = self._entries[draft_index]
            if entry is not None and signature != entry.signature:
                return self._sample_tokens(hidden_states)
            if entry is None:
                bound_hidden = hidden_states.clone() if staged else hidden_states
                capture_stream = torch.cuda.Stream(device=hidden_states.device)
                capture_stream.wait_stream(stream)
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        self._sample_tokens(bound_hidden)
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                # Do not share private pools between live stage outputs.
                with torch.cuda.graph(graph, stream=capture_stream):
                    token_ids = self._sample_tokens(bound_hidden)
                stream.wait_stream(capture_stream)
                entry = _ScreenGraphEntry(
                    graph, bound_hidden, token_ids, signature
                )
                self._entries[draft_index] = entry
            if staged:
                entry.hidden_states.copy_(hidden_states)
            # Also replay after first capture; a capture is not an evaluation.
            entry.graph.replay()
            return entry.token_ids
