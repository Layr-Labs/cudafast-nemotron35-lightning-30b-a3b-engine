# SPDX-License-Identifier: Apache-2.0
"""LOCAL PATCH (cudafast-nemotron35): one-copy, one-kernel target decode prep.

In the steady single-request speculative decode step, ``_prepare_inputs`` (and
``_prepare_input_ids``) issue about 24 tiny GPU operations before the target
graph: nine pinned H2D copies of per-step metadata, a fill, the async-spec
``num_computed_tokens`` correction kernel, four kernels for ``positions`` and
``seq_lens`` and four H2D copies, two gathers and two scatters for the input
ids. Each costs ~2.5-5 us of serialized GPU time on GB10 and ~5-15 us of host
time, and they sit between the drafter's last replay and the target graph.

This path stages the same host values into ONE pinned buffer, sends it with
ONE H2D copy and runs ONE single-warp Triton kernel that writes every one of
those device buffers with the same values the stock operations write:

* plain copies of ``query_start_loc`` (whole buffer), ``discard_request_mask``,
  ``prev_positions``, ``prev_num_draft_tokens`` (whole buffer), ``req_indices``,
  ``query_pos``, ``num_scheduled_tokens``;
* ``num_accepted_tokens`` filled with 1, then the correction of
  ``update_num_computed_tokens_for_batch_change`` (same gather, same
  ``where``) for ``num_computed_tokens[0]`` and ``num_accepted_tokens[0]``;
* ``positions[t] = int64(num_computed_tokens[req_indices[t]]) + query_pos[t]``
  and ``seq_lens = num_computed_tokens + num_scheduled_tokens`` (rest zero);
* the input-id scatters: the previous step's sampled token and its drafts.

Integer moves and integer adds only: every byte equals the stock path's. The
path is taken only for one request that was row 0 of the previous step; any
other shape keeps the stock code. ``VLLM_DECODE_PREP_FUSED=0`` disables it;
``VLLM_DECODE_PREP_CHECK=1`` runs the stock path as well and compares every
written buffer bitwise (testing only: it synchronizes).
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_DECODE_PREP_FUSED", "1") != "0"
CHECK = os.environ.get("VLLM_DECODE_PREP_CHECK", "0") == "1"
SKIP_CPU_IDS = os.environ.get("VLLM_DECODE_PREP_SKIP_CPU_IDS", "1") != "0"
# Read the staged values from the pinned buffer (mapped host memory) in the
# kernel instead of after an H2D copy (VLLM_PREP_ZERO_COPY=0 keeps the copy).
ZERO_COPY = os.environ.get("VLLM_PREP_ZERO_COPY", "1") != "0"
# Also compute the attention group's slot mapping (same formula as
# block_table._compute_slot_mapping_kernel, same PAD fill) and a copy of the
# draft ids for the spec-decode metadata (the values input_ids[1..D] gets),
# replacing two more small kernels (VLLM_DECODE_PREP_SLOT=0 keeps them).
SLOT = os.environ.get("VLLM_DECODE_PREP_SLOT", "1") != "0"

_ARGS = ["S", "Q", "P", "A", "T", "LS", "D", "PP", "DRAFT_START", "TOK_STRIDE",
         "qsl", "disc", "pp_out", "pnd", "ri", "qp", "ns", "acc", "nct", "vstc",
         "pos", "seq", "ids", "prev_tok", "draft",
         "bt", "slot", "MAXTOK", "bsz", "dout"]


@triton.jit(do_not_specialize=_ARGS)
def _decode_prep_kernel(S, Q, P, A, T, LS, D, PP, DRAFT_START, TOK_STRIDE,
                        qsl, disc, pp_out, pnd, ri, qp, ns, acc, nct, vstc,
                        pos, seq, ids, prev_tok, draft,
                        bt, slot, MAXTOK, bsz, dout,
                        KV_CACHE_BLOCK_SIZE: tl.constexpr, BLOCKS_PER_KV_BLOCK: tl.constexpr,
                        TOTAL_CP_WORLD_SIZE: tl.constexpr, TOTAL_CP_RANK: tl.constexpr,
                        CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr, PAD_ID: tl.constexpr,
                        DO_SLOT: tl.constexpr, PAD_BLOCK: tl.constexpr,
                        BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    off_disc = Q
    off_pp = Q + 1
    off_pnd = Q + 2
    off_cpu = Q + 2 + P
    off_ri = off_cpu + 1
    off_qp = off_ri + T
    off_ns = off_qp + T
    # Plain copies (the stock copy_to_gpu extents).
    v = tl.load(S + o, mask=o < Q, other=0)
    tl.store(qsl + o, v.to(tl.int32), mask=o < Q)
    tl.store(disc + o, tl.load(S + off_disc + o, mask=o < 1, other=0) != 0, mask=o < 1)
    tl.store(pp_out + o, tl.load(S + off_pp + o, mask=o < 1, other=0), mask=o < 1)
    v = tl.load(S + off_pnd + o, mask=o < P, other=0)
    tl.store(pnd + o, v.to(tl.int32), mask=o < P)
    qpv = tl.load(S + off_qp + o, mask=o < T, other=0)
    tl.store(ri + o, tl.load(S + off_ri + o, mask=o < T, other=0), mask=o < T)
    tl.store(qp + o, qpv, mask=o < T)
    nsv = tl.load(S + off_ns).to(tl.int32)
    tl.store(ns + o, nsv + o * 0, mask=o < 1)
    # update_num_computed_tokens_for_batch_change, one row, prev row PP >= 0.
    g = tl.maximum(PP, 0)
    pd = tl.load(S + off_pnd + g)
    cpu_nc = tl.load(S + off_cpu).to(tl.int32)
    vc = tl.load(vstc + g)
    pc = tl.load(nct + g)
    part = (PP >= 0) & (pd > 0)
    newnc = tl.where(part, pc + vc.to(tl.int32), cpu_nc)
    acc0 = tl.where(part, vc, 1).to(tl.int32)
    tl.store(nct + o, newnc + o * 0, mask=o < 1)
    accv = tl.where(o == 0, acc0, 1)
    tl.store(acc + o, accv, mask=o < A)
    # positions (req_indices are all 0 for one request) and seq_lens.
    posv = newnc.to(tl.int64) + qpv
    tl.store(pos + o, posv, mask=o < T)
    tl.store(seq + o, tl.where(o == 0, newnc + nsv, 0), mask=o < LS)
    # Input ids: sampled token at T-1-D, then the D drafts.
    tok = tl.load(prev_tok + PP * TOK_STRIDE).to(tl.int32)
    tl.store(ids + (T - 1 - D) + o, tok + o * 0, mask=o < 1)
    dv = tl.load(draft + DRAFT_START + o, mask=o < D, other=0)
    tl.store(ids + (T - D) + o, dv.to(tl.int32), mask=o < D)
    if DO_SLOT:
        # spec-decode draft ids: input_ids[1 .. D] for one request (T == D + 1)
        tl.store(dout + o, dv.to(tl.int32), mask=o < D)
        # _compute_slot_mapping_kernel, request 0 (tokens 0..T-1), then the PAD
        # fill of [T, MAXTOK) its last program writes.
        virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
        vbi = posv // virtual_block_size
        vbo = posv - vbi * virtual_block_size
        is_local = (vbo // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        lbo = (vbo // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
               ) * CP_KV_CACHE_INTERLEAVE_SIZE + (vbo % CP_KV_CACHE_INTERLEAVE_SIZE)
        bidx = vbi * BLOCKS_PER_KV_BLOCK + lbo // bsz
        bn = tl.load(bt + bidx, mask=(o < T) & is_local, other=0).to(tl.int64)
        sid = bn * bsz + lbo % bsz
        sid = tl.where(is_local, sid, PAD_ID)
        tl.store(slot + o, sid, mask=o < T)
        op = tl.arange(0, PAD_BLOCK)
        for i in range(T, MAXTOK, PAD_BLOCK):
            tl.store(slot + i + op, tl.full([PAD_BLOCK], PAD_ID, tl.int64), mask=i + op < MAXTOK)


class DecodePrep:
    """Pinned staging, its device twin and the guard event."""

    def __init__(self, runner) -> None:
        r = runner
        self.Q = r.query_start_loc.gpu.numel()
        self.P = r.prev_num_draft_tokens.gpu.numel()
        self.A = r.num_accepted_tokens.gpu.numel()
        self.LS = r.seq_lens.numel()
        self.tmax = r.num_spec_tokens + 1
        n = self.Q + self.P + 5 + 2 * self.tmax
        self.cpu = torch.zeros(n, dtype=torch.int64, pin_memory=True)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(n, dtype=torch.int64, device=r.device)
        self.event = torch.cuda.Event()
        self.recorded = False
        self.block = triton.next_power_of_2(max(self.Q, self.P, self.A, self.LS, self.tmax, 2))
        self.dummy = torch.zeros(1, dtype=torch.int32, device=r.device)
        self.dummy64 = torch.zeros(1, dtype=torch.int64, device=r.device)
        self.used = 0
        self.slot_bt = None
        self.draft_out = torch.zeros(max(self.tmax, 1), dtype=torch.int32, device=r.device)
        self.draft_ready = -1
        if SLOT:
            from vllm.v1.worker.block_table import SlotMappingMode
            tabs = [b for b in r.input_batch.block_table.block_tables
                    if b.slot_mapping_mode != SlotMappingMode.NONE]
            if (len(tabs) == 1 and tabs[0].slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT
                    and tabs[0].block_table.gpu.dtype == torch.int32
                    and tabs[0].slot_mapping.gpu.dtype == torch.int64):
                self.slot_bt = tabs[0]
        self.do_slot = self.slot_bt is not None
        self._precompile(r.device)
        # The prep copy batch (copy_batch.py) is used from the first fused decode
        # step on; build it (pinned staging only, nothing request-specific) and
        # compile its kernel now, with this first prep (the warm-up prefill),
        # outside any graph capture, so no compile can land in a timed decode step.
        from vllm.v1.worker import copy_batch
        if copy_batch.ENABLED and not torch.cuda.is_current_stream_capturing():
            copy_batch.get(r.device)

    def _consts(self):
        b = self.slot_bt
        if b is None:
            return dict(KV_CACHE_BLOCK_SIZE=1, BLOCKS_PER_KV_BLOCK=1, TOTAL_CP_WORLD_SIZE=1,
                        TOTAL_CP_RANK=0, CP_KV_CACHE_INTERLEAVE_SIZE=1, PAD_ID=-1,
                        DO_SLOT=False, PAD_BLOCK=256)
        from vllm.v1.worker.block_table import PAD_SLOT_ID
        return dict(KV_CACHE_BLOCK_SIZE=b.kv_cache_block_size,
                    BLOCKS_PER_KV_BLOCK=b.blocks_per_kv_block,
                    TOTAL_CP_WORLD_SIZE=b.dcp_world_size, TOTAL_CP_RANK=b.dcp_rank,
                    CP_KV_CACHE_INTERLEAVE_SIZE=b.cp_kv_cache_interleave_size,
                    PAD_ID=PAD_SLOT_ID, DO_SLOT=True, PAD_BLOCK=256)

    def slot_args(self):
        b = self.slot_bt
        if b is None:
            return (self.dummy, self.dummy64, 0, 1, self.dummy)
        return (b.block_table.gpu, b.slot_mapping.gpu, b.max_num_batched_tokens,
                b.block_size, self.draft_out)

    def _precompile(self, device) -> None:
        """Compile every dtype variant now (boot), never inside a timed step:
        the valid-count, sampled-token and draft tensors are int32 or int64
        depending on the sampler path. Writes only scratch tensors."""
        import itertools
        n = max(self.block, self.gpu.numel())

        def z(dt):
            return torch.zeros(n, dtype=dt, device=device)

        i32, i64 = torch.int32, torch.int64
        for dv, dtk, ddr in itertools.product((i32, i64), repeat=3):
            _decode_prep_kernel[(1,)](
                self.cpu if ZERO_COPY else self.gpu, self.Q, self.P, self.A, 1, self.LS, 0, 0, 0, 1,
                z(i32), z(torch.bool), z(i64), z(i32), z(i64), z(i64), z(i32),
                z(i32), z(i32), z(dv), z(i64), z(i32), z(i32), z(dtk), z(ddr),
                z(i32), z(i64), 0, 1, z(i32),
                BLOCK=self.block, num_warps=1, **self._consts(),
            )
        torch.cuda.synchronize()

    def eligible(self, r, num_reqs: int, total: int, scheduler_output) -> int:
        """-1 when the fused path does not apply, else the draft count D."""
        if num_reqs != 1 or total > self.tmax or self.LS < 1:
            return -1
        ib = r.input_batch
        if (
            ib.prev_sampled_token_ids is None
            or not ib.prev_req_id_to_index
            or r.valid_sampled_token_count_gpu is None
            or not r.use_async_spec_decode
            or r.mamba_prev_last_scheduled_idx is not None
            or r.uses_mrope or r.uses_xdrope_dim > 0
            or r.enable_prompt_embeds or ib.req_prompt_embeds
            or r.lora_config
        ):
            return -1
        if r.num_accepted_tokens_event is not None and not (
            r.use_async_scheduling and r.cache_config.mamba_cache_mode != "align"
        ):
            return -1
        req_id = ib.req_ids[0]
        prev = ib.prev_req_id_to_index.get(req_id, -1)
        if prev != 0:
            return -1
        d = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        if total != d + 1:
            return -1
        pst = ib.prev_sampled_token_ids
        if not pst.is_cuda or pst.dim() != 2 or pst.shape[0] < 1:
            return -1
        if d > 0:
            dt = r._draft_token_ids
            if (not isinstance(dt, torch.Tensor) or not dt.is_cuda or not dt.is_contiguous()
                    or dt.numel() < d or dt.dtype not in (torch.int32, torch.int64)):
                return -1
        return d

    def run(self, r, num_reqs: int, total: int, d: int) -> None:
        Q, P = self.Q, self.P
        if self.recorded:
            # The previous step's copy read this buffer; it ran long ago.
            self.event.synchronize()
        s = self.np
        s[:Q] = r.query_start_loc.np
        s[Q] = int(r.discard_request_mask.np[0])
        s[Q + 1] = r.prev_positions.np[0]
        s[Q + 2:Q + 2 + P] = r.prev_num_draft_tokens.np
        o = Q + 2 + P
        s[o] = int(r.input_batch.num_computed_tokens_cpu_tensor[0])
        s[o + 1:o + 1 + total] = r.req_indices.np[:total]
        s[o + 1 + total:o + 1 + 2 * total] = r.query_pos.np[:total]
        s[o + 1 + 2 * total] = r.num_scheduled_tokens.np[0]
        n = o + 2 + 2 * total
        if not ZERO_COPY:
            self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
            self.event.record()
        pst = r.input_batch.prev_sampled_token_ids
        draft = r._draft_token_ids if d > 0 else self.dummy
        pp = int(r.prev_positions.np[0])
        _decode_prep_kernel[(1,)](
            self.cpu if ZERO_COPY else self.gpu, Q, P, self.A, total, self.LS, d, pp,
            pp * r.prev_num_spec_tokens, pst.stride(0),
            r.query_start_loc.gpu, r.discard_request_mask.gpu, r.prev_positions.gpu,
            r.prev_num_draft_tokens.gpu, r.req_indices.gpu, r.query_pos.gpu,
            r.num_scheduled_tokens.gpu, r.num_accepted_tokens.gpu,
            r.num_computed_tokens, r.valid_sampled_token_count_gpu,
            r.positions, r.seq_lens, r.input_ids.gpu, pst, draft,
            *self.slot_args(),
            BLOCK=self.block, num_warps=1, **self._consts(),
        )
        self.draft_ready = d if self.do_slot else -1
        if ZERO_COPY:
            self.event.record()  # the kernel read the pinned buffer
        self.recorded = True
        self.used += 1


def check_buffers(r, total: int):
    """Clone every buffer either path writes (VLLM_DECODE_PREP_CHECK)."""
    return [
        r.query_start_loc.gpu.clone(), r.discard_request_mask.gpu[:1].clone(),
        r.prev_positions.gpu[:1].clone(), r.prev_num_draft_tokens.gpu.clone(),
        r.req_indices.gpu[:total].clone(), r.query_pos.gpu[:total].clone(),
        r.num_scheduled_tokens.gpu[:1].clone(), r.num_accepted_tokens.gpu.clone(),
        r.num_computed_tokens.clone(), r.positions[:total].clone(),
        r.seq_lens.clone(), r.input_ids.gpu[:total].clone(),
    ] + ([r._decode_prep.slot_bt.slot_mapping.gpu.clone()]
         if getattr(r, "_decode_prep", None) is not None and r._decode_prep.do_slot else [])


def state_buffers(r):
    """Device buffers the prep reads AND writes (restored between paths)."""
    return [r.num_computed_tokens, r.num_accepted_tokens.gpu]


_check_stats = {"steps": 0, "diffs": 0}


def compare(ref, got) -> bool:
    ok = all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.view(torch.uint8) if a.dtype == torch.bool else a,
        b.view(torch.uint8) if b.dtype == torch.bool else b) for a, b in zip(ref, got))
    _check_stats["steps"] += 1
    if not ok:
        _check_stats["diffs"] += 1
        bad = [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)]
        print(f"decode_prep CHECK MISMATCH buffers {bad}: ref {[ref[i].tolist() for i in bad]} "
              f"got {[got[i].tolist() for i in bad]}", flush=True)
    if _check_stats["steps"] % 200 == 0 or not ok:
        print(f"decode_prep CHECK steps {_check_stats['steps']} diffs {_check_stats['diffs']}", flush=True)
    return ok


__all__ = ["ENABLED", "CHECK", "DecodePrep", "check_buffers", "state_buffers", "compare"]
