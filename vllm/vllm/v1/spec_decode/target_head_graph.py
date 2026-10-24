# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One fixed-stream graph for the original target head and greedy rejection.

Inputs are copied from the current original gather/metadata snapshot. Logits
are borrowed only within one serial runner frame; token IDs are always cloned.
This helper requires the single-device UniProcExecutor's serial worker calls.
It does not make GPUModelRunner a concurrent or multi-stream API.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, tensor.device, tensor.storage_offset(),
    )


def metadata_signature(metadata: SpecDecodeMetadata) -> tuple[object, ...]:
    return (
        id(metadata), tuple(metadata.num_draft_tokens), metadata.max_spec_len,
        *(_tensor_signature(tensor) for tensor in (
            metadata.draft_token_ids, metadata.cu_num_draft_tokens,
            metadata.cu_num_sampled_tokens, metadata.target_logits_indices,
            metadata.bonus_logits_indices, metadata.logits_indices,
        )),
    )


@dataclass
class _TargetHeadEntry:
    graph: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    metadata: SpecDecodeMetadata
    logits: torch.Tensor
    tokens: torch.Tensor
    stream: torch.cuda.Stream
    signature: tuple[object, ...]
    # Retain wrapper/model/head ownership, including their graph pools.
    model_owner: object
    rejection_owner: object
    captured_owners: tuple[object, ...]


@dataclass
class TargetHeadTicket:
    sequence: int
    entry: _TargetHeadEntry
    scheduler_output: object
    request_ids: tuple[str, ...]
    metadata: SpecDecodeMetadata
    metadata_key: tuple[object, ...]
    owner_key: tuple[object, ...]
    logits: torch.Tensor
    tokens: torch.Tensor
    source_hidden: torch.Tensor
    used: bool = False
    allow_tokens: bool = True


class TargetHeadGraph:
    """Bounded entry; incompatible calls retain the complete eager path."""

    def __init__(self, depth: int, device: torch.device) -> None:
        self.depth = depth
        self.device = device
        self.entry: _TargetHeadEntry | None = None
        self.frame: TargetHeadTicket | None = None
        self.disabled = False
        self.sequence = 0
        # A partial capture must stay owned until synchronized shutdown too.
        self._building: tuple[object, ...] | None = None

    def _metadata_ok(self, metadata: SpecDecodeMetadata) -> bool:
        d = self.depth
        tensors = (
            (metadata.draft_token_ids, (d,), torch.int32),
            (metadata.cu_num_draft_tokens, (1,), torch.int32),
            (metadata.cu_num_sampled_tokens, (1,), torch.int32),
            (metadata.target_logits_indices, (d,), torch.int32),
            (metadata.bonus_logits_indices, (1,), torch.int32),
            (metadata.logits_indices, (d + 1,), torch.int64),
        )
        return (
            metadata.num_draft_tokens == [d] and metadata.max_spec_len == d
            and all(t.shape == shape and t.dtype == dtype
                    and t.device == self.device and t.is_contiguous()
                    for t, shape, dtype in tensors)
        )

    def prepare(
        self,
        hidden: torch.Tensor,
        metadata: SpecDecodeMetadata,
        scheduler_output: object,
        request_ids: tuple[str, ...],
        owner_key: tuple[object, ...],
        model_owner: object,
        rejection_owner: object,
        captured_owners: tuple[object, ...],
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        greedy_sample: Callable[[torch.Tensor, SpecDecodeMetadata], SamplerOutput],
    ) -> tuple[torch.Tensor, TargetHeadTicket] | None:
        if self.disabled:
            return None
        if self.frame is not None:
            # An abandoned execute/sample frame cannot lend scratch again.
            self.poison()
            raise RuntimeError("Target head graph has an unretired frame")
        if (
            not self._metadata_ok(metadata) or not hidden.is_cuda
            or hidden.device != self.device or hidden.ndim != 2
            or hidden.shape[0] != self.depth + 1 or not hidden.is_contiguous()
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing()
        ):
            return None
        stream = torch.cuda.current_stream(self.device)
        signature = (
            tuple(hidden.shape), tuple(hidden.stride()), hidden.dtype,
            hidden.device, stream.cuda_stream, torch.is_inference_mode_enabled(),
            owner_key,
        )
        entry = self.entry
        if entry is not None and signature != entry.signature:
            return None
        try:
            if entry is None:
                static_hidden = torch.empty_like(hidden)
                static_draft = torch.empty_like(metadata.draft_token_ids)
                static_hidden.copy_(hidden)
                static_draft.copy_(metadata.draft_token_ids)
                # These private tables encode only the host-proven index
                # program. They never contain a previous request's answer.
                d = self.depth
                private_metadata = SpecDecodeMetadata(
                    static_draft, [d],
                    torch.tensor([d], dtype=torch.int32, device=self.device),
                    torch.tensor([d + 1], dtype=torch.int32, device=self.device),
                    torch.arange(d, dtype=torch.int32, device=self.device),
                    torch.tensor([d], dtype=torch.int32, device=self.device),
                    torch.arange(d + 1, dtype=torch.int64, device=self.device),
                )
                capture_stream = torch.cuda.Stream(device=self.device)
                graph = torch.cuda.CUDAGraph()
                self._building = (
                    graph, static_hidden, private_metadata, capture_stream,
                    model_owner, rejection_owner, captured_owners,
                )
                capture_stream.wait_stream(stream)
                # Actual first-use work, inside the ordinary invocation. Only
                # the original stateless head/rejection body is repeated.
                with torch.cuda.stream(capture_stream):
                    warm_logits = compute_logits(static_hidden)
                    if warm_logits.dtype not in (
                        torch.float16, torch.bfloat16, torch.float32
                    ):
                        stream.wait_stream(capture_stream)
                        self.disabled = True
                        return None
                    greedy_sample(warm_logits, private_metadata)
                    for _ in range(2):
                        greedy_sample(compute_logits(static_hidden), private_metadata)
                stream.wait_stream(capture_stream)
                with torch.cuda.graph(graph, stream=capture_stream):
                    # Retain logits as a live graph output, not a pool-reusable
                    # temporary hidden behind the sampler's output.
                    static_logits = compute_logits(static_hidden)
                    static_tokens = greedy_sample(
                        static_logits, private_metadata
                    ).sampled_token_ids
                stream.wait_stream(capture_stream)
                entry = _TargetHeadEntry(
                    graph, static_hidden, private_metadata, static_logits,
                    static_tokens, stream, signature, model_owner, rejection_owner,
                    captured_owners,
                )
                self._building = (entry,)
            else:
                entry.hidden.copy_(hidden)
                entry.metadata.draft_token_ids.copy_(metadata.draft_token_ids)

            # Capture is not an evaluation. First use also explicitly replays.
            entry.graph.replay()
            # Never lend graph tokens to the existing asynchronous D2H stream.
            token_snapshot = entry.tokens.clone()
            self.sequence += 1
            ticket = TargetHeadTicket(
                self.sequence, entry, scheduler_output, request_ids, metadata,
                metadata_signature(metadata), owner_key, entry.logits,
                token_snapshot, hidden,
            )
            # Publish only a fully captured/replayed current frame.
            self.entry = entry
            self.frame = ticket
            self._building = None
            return entry.logits, ticket
        except BaseException:
            self.poison()
            # Capture/replay errors remain visible; eager retry could execute
            # against an invalid CUDA capture state.
            raise

    def resolve_logits(
        self, ticket: TargetHeadTicket, logits: torch.Tensor,
        scheduler_output: object, metadata: SpecDecodeMetadata | None,
        request_ids: tuple[str, ...], owner_key: tuple[object, ...] | None,
    ) -> torch.Tensor:
        if (
            self.disabled or self.frame is not ticket or ticket.used
            or self.entry is not ticket.entry or logits is not ticket.logits
            or scheduler_output is not ticket.scheduler_output
            or metadata is not ticket.metadata or request_ids != ticket.request_ids
            or metadata_signature(ticket.metadata) != ticket.metadata_key
        ):
            self.poison()
            raise RuntimeError("Stale target head graph frame")
        current = torch.cuda.current_stream(self.device)
        changed_stream = current.cuda_stream != ticket.entry.stream.cuda_stream
        # Unknown owner/config changes must not pass borrowed storage to new
        # callbacks. A private snapshot is safe for the original fallback.
        if changed_stream or owner_key != ticket.owner_key:
            with torch.cuda.stream(ticket.entry.stream):
                snapshot = logits.clone()
                ready = torch.cuda.Event()
                ready.record(ticket.entry.stream)
            if changed_stream:
                current.wait_event(ready)
                snapshot.record_stream(current)
            ticket.allow_tokens = False
            return snapshot
        return logits

    def take_tokens(
        self, ticket: TargetHeadTicket, logits: torch.Tensor,
        metadata: SpecDecodeMetadata, owner_key: tuple[object, ...] | None,
        allowed: bool,
    ) -> SamplerOutput | None:
        if self.frame is not ticket or ticket.used:
            self.poison()
            raise RuntimeError("Target head graph ticket already consumed")
        ticket.used = True
        if (
            self.disabled or not allowed or not ticket.allow_tokens
            or logits is not ticket.logits or metadata is not ticket.metadata
            or owner_key != ticket.owner_key
            or metadata_signature(metadata) != ticket.metadata_key
            or torch.cuda.current_stream(self.device).cuda_stream
            != ticket.entry.stream.cuda_stream
        ):
            return None
        return SamplerOutput(ticket.tokens, None)

    def finish_frame(self) -> None:
        if self.frame is not None:
            self.frame.used = True
        self.frame = None

    def poison(self) -> None:
        self.disabled = True
        if self.frame is not None:
            self.frame.allow_tokens = False

    def release_after_sync(self) -> None:
        self.poison()
        self.finish_frame()
        self.entry = None
        self._building = None
