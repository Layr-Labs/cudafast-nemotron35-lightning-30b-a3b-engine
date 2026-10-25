# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens_list = [0] + list(itertools.accumulate(chunk_lens))
        # Final boundary must equal total tokens (check on host to avoid a sync)
        assert cu_chunk_seqlens_list[-1] == total
    else:
        cu_chunk_seqlens_list = [0]
    cu_chunk_seqlens = async_tensor_h2d(
        cu_chunk_seqlens_list, dtype=torch.int32, device=device
    )

    # last_chunk_indices is empty when there are no sequences (len(starts) == 0).
    last_chunk_indices_t = async_tensor_h2d(
        last_chunk_indices, dtype=torch.int32, device=device
    )
    seq_idx_chunks_t = async_tensor_h2d(
        seq_idx_chunks, dtype=torch.int32, device=device
    )
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None


_META_REUSE = os.environ.get("VLLM_MAMBA2_META_REUSE", "1") != "0"
_META_REUSE_CHECK = os.environ.get("VLLM_MAMBA2_META_REUSE_CHECK", "0") == "1"
_SHARED_ACCEPTED: dict[tuple, torch.Tensor] = {}
_STEP: dict[str, Any] = {"qsl": None, "sig": None, "acc_qsl": None}


def _tensor_key(t: torch.Tensor | None) -> tuple | None:
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _step_signature(
    common_attn_metadata: CommonAttentionMetadata, kwargs: dict[str, Any]
) -> tuple | None:
    """Inputs that decide a decode build's structure, or None when not a reusable
    pure-decode step. Computed once per step for all groups."""
    m = common_attn_metadata
    if _STEP["qsl"] is m.query_start_loc:
        return _STEP["sig"]
    sig = None
    acc = kwargs.get("num_accepted_tokens")
    isp = m.is_prefilling
    if (
        acc is not None
        and kwargs.get("prev_last_scheduled_idx") is None
        and isp is not None
        and not isp.is_cuda
        and m.query_start_loc_cpu is not None
    ):
        n = m.num_reqs
        if n > 0 and not bool(isp.numpy()[:n].any()):
            sig = (
                n,
                m.num_actual_tokens,
                m.max_query_len,
                tuple(m.query_start_loc_cpu.tolist()),
                _tensor_key(m.query_start_loc),
                _tensor_key(m.seq_lens),
                _tensor_key(acc),
            )
    _STEP["qsl"] = m.query_start_loc
    _STEP["sig"] = sig
    return sig


def _structure_diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and _tensor_key(a) == _tensor_key(b)
            and a.device == b.device
        ):
            out.append(path)
        return
    if type(a) is not type(b):
        out.append(path)
        return
    if hasattr(a, "__dataclass_fields__"):
        for f in fields(a):
            _structure_diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}", out)
        return
    if a != b:
        out.append(path)


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size
        self._meta_reuse = (
            _META_REUSE
            and self.use_spec_decode
            and vllm_config.cache_config.mamba_cache_mode == "none"
            and not self.use_replayssm
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self._decode_cache: tuple | None = None
        self._capture_build = False
        if self._meta_reuse:
            buf = self.decode_num_accepted_tokens
            key = (str(buf.device), tuple(buf.shape), buf.dtype)
            shared = _SHARED_ACCEPTED.setdefault(key, buf)
            self.decode_num_accepted_tokens = shared

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Mamba2AttentionMetadata:
        self._decode_cache = None
        self._capture_build = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capture_build = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        if not self._meta_reuse or self._capture_build or common_prefix_len != 0:
            return self._build_full(
                common_prefix_len, common_attn_metadata, fast_build, **kwargs
            )
        step_sig = _step_signature(common_attn_metadata, kwargs)
        if step_sig is None:
            self._decode_cache = None
            return self._build_full(
                common_prefix_len, common_attn_metadata, fast_build, **kwargs
            )
        sig = (step_sig, _tensor_key(common_attn_metadata.block_table_tensor))
        cached = self._decode_cache
        if cached is not None and cached[0] == sig:
            md = cached[1]
            nd = md.num_decodes
            bt = common_attn_metadata.block_table_tensor
            self.state_indices_tensor_d[:nd].copy_(
                bt[:nd, : 1 + self.num_spec_tokens], non_blocking=True
            )
            qsl = common_attn_metadata.query_start_loc
            if _STEP["acc_qsl"] is not qsl:
                acc = kwargs["num_accepted_tokens"]
                self.decode_num_accepted_tokens[:nd].copy_(acc[:nd], non_blocking=True)
                _STEP["acc_qsl"] = qsl
            if _META_REUSE_CHECK:
                full = self._build_full(
                    common_prefix_len, common_attn_metadata, fast_build, **kwargs
                )
                diffs: list[str] = []
                _structure_diff(md, full, "mamba2", diffs)
                if diffs:
                    raise RuntimeError(f"Mamba2 decode metadata reuse differs: {diffs}")
            return md
        md = self._build_full(
            common_prefix_len, common_attn_metadata, fast_build, **kwargs
        )
        if (
            md.num_prefills == 0
            and md.num_decodes == md.num_reqs
            and md.num_decodes <= self.decode_cudagraph_max_bs
            and md.state_indices_tensor_d is not None
            and md.num_accepted_tokens is not None
            and md.state_indices_tensor_d.data_ptr()
            == self.state_indices_tensor_d.data_ptr()
        ):
            self._decode_cache = (sig, md)
            _STEP["acc_qsl"] = common_attn_metadata.query_start_loc
        else:
            self._decode_cache = None
        return md

    def _build_full(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            prev_last_scheduled_idx=kwargs.get("prev_last_scheduled_idx"),
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )
