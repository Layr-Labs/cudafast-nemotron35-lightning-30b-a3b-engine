# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass(frozen=True)
class UniformSpecDecodeTables:
    """Read-only index programs for one request with exactly D+1 decode rows.

    Only shape-derived indices/counts are retained. Every metadata instance
    receives its own current draft-token snapshot and Python count list.
    """

    depth: int
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
    ready_event: torch.cuda.Event
    creation_stream: torch.cuda.Stream

    @classmethod
    def create(cls, depth: int, device: torch.device) -> "UniformSpecDecodeTables":
        assert 1 <= depth <= 6 and device.type == "cuda"
        with torch.cuda.device(device):
            # Match _calc_spec_decode_metadata exactly. Its cumulative
            # scheduled-token sum is int64, so only logits_indices is int64.
            cu_draft = torch.tensor([depth], dtype=torch.int32, device=device)
            cu_sampled = torch.tensor([depth + 1], dtype=torch.int32, device=device)
            target = torch.tensor(list(range(depth)), dtype=torch.int32, device=device)
            bonus = torch.tensor([depth], dtype=torch.int32, device=device)
            logits = torch.tensor(
                list(range(depth + 1)), dtype=torch.int64, device=device
            )
            stream = torch.cuda.current_stream(device)
            ready = torch.cuda.Event()
            ready.record(stream)
        return cls(
            depth,
            cu_draft,
            cu_sampled,
            target,
            bonus,
            logits,
            ready,
            stream,
        )

    def make_metadata(self, input_ids: torch.Tensor) -> SpecDecodeMetadata:
        stream = torch.cuda.current_stream(input_ids.device)
        if stream.cuda_stream != self.creation_stream.cuda_stream:
            stream.wait_event(self.ready_event)
        # _prepare_input_ids has already queued current input writes on this
        # stream. Clone isolates the consumer from the next step's overwrite.
        draft_token_ids = input_ids[1 : self.depth + 1].clone()
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=[self.depth],
            cu_num_draft_tokens=self.cu_num_draft_tokens,
            cu_num_sampled_tokens=self.cu_num_sampled_tokens,
            target_logits_indices=self.target_logits_indices,
            bonus_logits_indices=self.bonus_logits_indices,
            logits_indices=self.logits_indices,
        )
