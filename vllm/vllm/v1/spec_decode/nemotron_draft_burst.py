# SPDX-License-Identifier: Apache-2.0
"""Load-qualified three-step native MTP graph with exact native plan categories.

Only structural planner programs and fixed-address operational buffers persist.
Every token, hidden row, position, sequence length and KV page ID is current-call
data. Unknown categories/identities decline before touching request state.
"""
from __future__ import annotations

import copy
import logging
import os
import statistics
import time

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.nemotron_decode_plan import _SCALARS, _eligible, _tensor
from vllm.v1.spec_decode.nemotron_full_step import maybe_run as original_model_graph
from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["length"])
def _stage_current(pos_in, seq_in, table_in, hidden_in, token_in,
                   pos, seq, table, hidden, token, cpu_length,
                   length, H: tl.constexpr, PAGES: tl.constexpr,
                   BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(table + i, tl.load(table_in + i, mask=i < PAGES, other=0), mask=i < PAGES)
    tl.store(hidden + i, tl.load(hidden_in + i, mask=i < H, other=0), mask=i < H)
    if tl.program_id(0) == 0:
        tl.store(pos, tl.load(pos_in))
        tl.store(seq, tl.load(seq_in))
        tl.store(token, tl.load(token_in))
        tl.store(cpu_length, length)


@triton.jit
def _install_step(position, seq, table, hidden, token, cpu_length,
                  prefix_addresses, native_pages, native_residues,
                  int_workspace, indptr, indices, last,
                  out_position, out_slots, out_hidden, out_token,
                  STEP: tl.constexpr, PREFIX_BYTES: tl.constexpr,
                  H: tl.constexpr, PAGES: tl.constexpr, PADDED: tl.constexpr,
                  PAGE_SIZE: tl.constexpr, MAX_LEN: tl.constexpr,
                  UPDATE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(cpu_length) + STEP
    # Addresses name strongly owned, immutable native planner snapshots, never
    # request/model outputs. Runtime category selection already bounds the key.
    source = tl.load(prefix_addresses + length).to(tl.pointer_type(tl.uint8))
    tl.store(int_workspace + i, tl.load(source + i, mask=i < PREFIX_BYTES, other=0),
             mask=i < PREFIX_BYTES)
    pages = tl.load(native_pages + length)
    residue = tl.load(native_residues + length)
    tl.store(indices + i, tl.load(table + i, mask=i < pages, other=0), mask=i < pages)
    if UPDATE:
        tl.store(out_hidden + i, tl.load(hidden + i, mask=i < H, other=0), mask=i < H)
        # Only CTA0 reads or writes the position/sequence words. Other CTAs
        # never derive their work from a word that this CTA updates in place.
        if tl.program_id(0) == 0:
            new_position = tl.load(position) + 1
            exceeds_max = new_position >= MAX_LEN
            clamped = tl.where(exceeds_max, 0, new_position)
            block = tl.minimum(clamped // PAGE_SIZE, PAGES - 1)
            page_id = tl.load(table + block)
            slot = tl.where(exceeds_max, -1, page_id * PAGE_SIZE + clamped % PAGE_SIZE)
            new_seq = tl.minimum(tl.where(exceeds_max, 1, tl.load(seq) + 1), MAX_LEN)
            tl.store(out_position, clamped)
            tl.store(seq, new_seq)
            tl.store(out_token, tl.load(token))
            j = tl.arange(0, BLOCK)
            tl.store(out_slots + j, tl.where(j == 0, slot, -1), mask=j < PADDED)
    if tl.program_id(0) == 0:
        tl.store(indptr, 0)
        tl.store(indptr + 1, pages)
        tl.store(last, residue)


def _entry_category(entry):
    plan, prefix, scalars, _meta = entry
    return tuple(plan), tuple(scalars), prefix.numel()


def _wrapper_fields(wrapper, entry):
    wrapper._plan_info = list(entry[0])
    for name, value in entry[2]:
        setattr(wrapper, name, value)


def _independent_cpu_shadows(common):
    ptrs = []
    for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
        t = getattr(common, name, None)
        if t is not None:
            if t.device.type != "cpu" or t.dtype != torch.int32 or tuple(t.shape) != (1,):
                return False
            ptrs.append(t.data_ptr())
    return len(ptrs) == len(set(ptrs))


def _primitive_options(wrapper):
    # Native run can use additional host launch hints beyond the fields that
    # the planner bank restores. Any such change declines this captured graph.
    return tuple((k, type(v).__name__, v) for k, v in sorted(vars(wrapper).items())
                 if k not in _SCALARS and k != "_plan_info"
                 and type(v) in (int, float, bool, str, type(None)))


def _builder_identity(builder, wrapper):
    return (id(wrapper), id(wrapper._cached_module),
            tuple((name, _tensor(getattr(wrapper, name))) for name in (
                "_float_workspace_buffer", "_int_workspace_buffer", "_qo_indptr_buf",
                "_paged_kv_indptr_buf", "_paged_kv_indices_buf",
                "_paged_kv_last_page_len_buf")),
            tuple(_tensor(getattr(builder, name).gpu) for name in (
                "paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len")),
            wrapper._backend, wrapper.is_cuda_graph_enabled,
            getattr(wrapper, "_enable_pdl", None),
            getattr(wrapper, "_kv_layout", None),
            getattr(wrapper, "_use_fp16_qk_reduction", None),
            tuple(getattr(builder, n, None) for n in (
                "num_qo_heads", "num_kv_heads", "head_dim", "page_size",
                "q_data_type_decode", "kv_cache_dtype", "sm_scale", "window_left",
                "logits_soft_cap", "decode_fixed_split_size", "disable_split_kv")))


class Burst:
    def __init__(self, p, builder, wrapper, bank, padded):
        self.p, self.builder, self.wrapper, self.bank = p, builder, wrapper, bank
        self.bank_entries = bank.entries
        self.padded = padded
        self.blocks = (p.max_model_len + p.block_size - 1) // p.block_size
        self.screen = p._draft_head_screen
        self.screen_graph = self.screen.graph
        self.screen_owners = (self.screen._h, self.screen._out)
        self.screen_io = tuple(_tensor(x) for x in self.screen_owners)
        self.model = p.model
        self.identity = _builder_identity(builder, wrapper)
        self.options = _primitive_options(wrapper)
        self.inputs = tuple(_tensor(x) for x in (
            p.input_ids, p.positions, p.hidden_states, p._slot_mapping_buffer))
        self.kv = tuple((n, _tensor(p.compilation_config.static_forward_context[n].kv_cache))
                        for n in sorted(p._draft_attn_layer_names))
        self.entries = {}
        bank_identity = None
        for key, value in bank.entries.items():
            starts, residue, constants, identity = key
            if bank_identity is None:
                bank_identity = (constants, identity)
            if (constants, identity) != bank_identity or starts[0] != 0:
                raise ValueError("mixed native plan identities")
            length = (starts[1] - 1) * p.block_size + residue
            meta = (starts[1], residue)
            self.entries[length] = (*value, meta)
        self.bank_identity = bank_identity
        cap = max(self.entries)
        addresses, pages, residues = [0] * (cap + 1), [0] * (cap + 1), [0] * (cap + 1)
        for length, entry in self.entries.items():
            addresses[length] = entry[1].data_ptr()
            pages[length], residues[length] = entry[3]
        self.prefix_addresses = torch.tensor(addresses, dtype=torch.int64, device=p.device)
        self.native_pages = torch.tensor(pages, dtype=torch.int32, device=p.device)
        self.native_residues = torch.tensor(residues, dtype=torch.int32, device=p.device)
        self.cpu_length = torch.empty(1, dtype=torch.int32, device=p.device)
        self.scalar_choices = {tuple(v[2]) for v in self.entries.values()}
        self.categories = {}
        for length in sorted(self.entries):
            if all(length + i in self.entries for i in (1, 2, 3)):
                triple = tuple(_entry_category(self.entries[length + i]) for i in (1, 2, 3))
                self.categories.setdefault(triple, length)
        if not self.categories or len(self.categories) > 16:
            raise ValueError("native triple graph category budget")
        self.table = torch.empty((1, self.blocks), dtype=torch.int32, device=p.device)
        self.seq = torch.empty(1, dtype=torch.int32, device=p.device)
        self.pos = torch.empty(1, dtype=p.positions.dtype, device=p.device)
        self.hidden = torch.empty((1, p.hidden_states.shape[-1]),
                                  dtype=p.hidden_states.dtype, device=p.device)
        self.token = torch.empty(1, dtype=torch.int32, device=p.device)
        self.row_zero = torch.zeros(1, dtype=torch.long, device=p.device)
        self.graphs = {}
        self.enabled = False

    def triple(self, length):
        values = [self.entries.get(length + i) for i in (1, 2, 3)]
        if any(x is None for x in values):
            return None, None
        return tuple(_entry_category(x) for x in values), values

    def stage(self, positions, common, hidden, token, length):
        _stage_current[(triton.cdiv(max(self.hidden.numel(), self.blocks), 256),)](
            positions, common.seq_lens, common.block_table_tensor, hidden, token,
            self.pos, self.seq, self.table, self.hidden, self.token, self.cpu_length,
            length=length, H=self.hidden.numel(), PAGES=self.blocks,
            BLOCK=256, num_warps=4)

    def install(self, position, hidden, token, step, prefix_bytes, update=True):
        p = self.p
        _install_step[(triton.cdiv(max(prefix_bytes, self.hidden.numel(), self.blocks), 256),)](
            position, self.seq, self.table, hidden, token, self.cpu_length,
            self.prefix_addresses, self.native_pages, self.native_residues,
            self.wrapper._int_workspace_buffer.view(torch.uint8),
            self.builder.paged_kv_indptr.gpu, self.builder.paged_kv_indices.gpu,
            self.builder.paged_kv_last_page_len.gpu, p.positions,
            p._slot_mapping_buffer, p.hidden_states, p.input_ids,
            STEP=step, PREFIX_BYTES=prefix_bytes, H=self.hidden.numel(),
            PAGES=self.blocks, PADDED=self.padded, PAGE_SIZE=p.block_size,
            MAX_LEN=p.max_model_len, UPDATE=update, BLOCK=256, num_warps=4)

    def metadata(self, length):
        qcpu = torch.tensor([0, 1], dtype=torch.int32)
        scpu = torch.tensor([length], dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=qcpu.to(self.p.device), query_start_loc_cpu=qcpu,
            seq_lens=self.seq, num_reqs=1, num_actual_tokens=1, max_query_len=1,
            max_seq_len=length, block_table_tensor=self.table,
            slot_mapping=self.p._slot_mapping_buffer[:1], _seq_lens_cpu=scpu,
            _num_computed_tokens_cpu=scpu - 1, seq_lens_cpu_upper_bound=scpu.clone(),
            causal=True)

    def capture(self, triple, length):
        p = self.p
        _, values = self.triple(length)
        # Exact native snapshots stay strongly owned; direct table loads remove
        # category-local refresh buffers and per-key tiny device allocations.
        sources = [v[1] for v in values]
        metas = [v[3] for v in values]
        output = torch.empty((1, 3), dtype=torch.int32, device=p.device)
        md = []
        for step, entry in enumerate(values):
            common = self.metadata(length + step + 1)
            _, per_layer = p.build_per_group_and_layer_attn_metadata(common, draft_index=step + 1)
            per_layer = {name: copy.copy(m) for name, m in per_layer.items()}
            for m in per_layer.values():
                m.plan_ready_event = None
            md.append(per_layer)
        # Every actual native kernel is warmed before capture. Each step gets
        # its exact native Python launch vector, not another graph's constants.
        # Assign category-specific fields on each unrolled capture iteration.
        self._capture_entries = values
        self._body_with_plans(md, sources, metas, output)
        torch.cuda.synchronize(p.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            final_hidden = self._body_with_plans(md, sources, metas, output)
        torch.cuda.synchronize(p.device)
        self.graphs[triple] = graph, sources, metas, output, md, final_hidden

    def _body_with_plans(self, metadata, sources, metas, output):
        # Native launch-vector assignment is deliberately inside the Python
        # capture loop before each model's kernels are recorded.
        p = self.p
        position, hidden, token = self.pos, self.hidden, self.token
        kwargs = dict(input_ids=p.input_ids[:self.padded],
                      positions=p._get_positions(self.padded), inputs_embeds=None,
                      hidden_states=p.hidden_states[:self.padded])
        for step in range(3):
            _wrapper_fields(self.wrapper, self._capture_entries[step])
            self.install(position, hidden, token, step + 1, sources[step].numel())
            position = p.positions[:1]
            with set_forward_context(
                    metadata[step], p.vllm_config, num_tokens=self.padded,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    slot_mapping=p._get_slot_mapping(self.padded)):
                ret = p.model(**kwargs)
            hidden = ret[:1]
            # The original greedy core, without its otherwise-unneeded clone.
            # Store this ID before the next screen overwrites the shared output.
            self.screen._h.copy_(hidden[0])
            self.screen.graph.replay()
            token = self.screen._out
            output[:, step].copy_(token)
        return ret

    def host_finish(self, common, length, values):
        # No pinned buffer is consumed by the burst, but honor original staging
        # retirement before writing its CPU-visible final per-depth metadata.
        p, b = self.p, self.builder
        sets = p._hostfast_draft_sets()
        for step, entry in enumerate(values):
            sets.before(step + 1)
            pages = (length + step) // p.block_size + 1
            residue = (length + step) % p.block_size + 1
            b.paged_kv_indptr.np[1] = pages
            b.paged_kv_indptr_cpu_buffer[:2].copy_(b.paged_kv_indptr.cpu[:2])
            b.paged_kv_last_page_len.np[0] = residue
            sets.after()
        _wrapper_fields(self.wrapper, values[-1])
        self.bank.last_restore = True
        common.max_seq_len = min(common.max_seq_len + 3, p.max_model_len)
        for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
            value = getattr(common, name, None)
            if value is not None:
                value.add_(3)
        common.slot_mapping = p._slot_mapping_buffer[:1]

    def execute(self, positions, common, hidden, token, length):
        triple, values = self.triple(length)
        graph, sources, metas, output, _md, _hidden = self.graphs[triple]
        self.stage(positions, common, hidden, token, length)
        # Retire/write original pinned staging before replay. These buffers are
        # not consumed by this graph; device-native snapshots are its sources.
        self.host_finish(common, length, values)
        graph.replay()
        common.seq_lens.copy_(self.seq)
        return output.clone()


def maybe_run(p, positions, common, hidden, token, batch_size, padded, greedy, gated):
    state = getattr(p, "_nemo_burst", None)
    if (state is None or not state.enabled or batch_size != 1 or padded != state.padded
            or p.num_speculative_tokens != 4 or p.parallel_drafting
            or not greedy or gated is None or torch.cuda.is_current_stream_capturing()
            or p.model is not state.model or p._draft_head_screen is not state.screen
            or state.screen.graph is not state.screen_graph
            or tuple(_tensor(x) for x in (state.screen._h, state.screen._out)) != state.screen_io
            or p.draft_attn_groups[0].get_metadata_builder() is not state.builder
            or state.builder._decode_wrappers_cudagraph.get(1) is not state.wrapper
            or getattr(state.wrapper, "_nemo_native_plan", None) is not state.bank
            or not state.bank.enabled or state.bank.force_native
            or state.bank.entries is not state.bank_entries
            or _builder_identity(state.builder, state.wrapper) != state.identity
            or _primitive_options(state.wrapper) != state.options
            or tuple((n, getattr(state.wrapper, n)) for n in _SCALARS
                     if hasattr(state.wrapper, n)) not in state.scalar_choices
            or tuple(_tensor(x) for x in (
                p.input_ids, p.positions, p.hidden_states, p._slot_mapping_buffer)) != state.inputs
            or tuple((n, _tensor(p.compilation_config.static_forward_context[n].kv_cache))
                     for n in sorted(p._draft_attn_layer_names)) != state.kv
            or p._hostfast_draft_sets() is None
            or gated is not state.gate or gated.dtype != torch.int32 or gated.numel() != 4
            or common.num_reqs != 1 or common.num_actual_tokens != 1
            or common.max_query_len != 1 or not common.causal
            or not _independent_cpu_shadows(common)
            or common._seq_lens_cpu is None or common._seq_lens_cpu.dtype != torch.int32
            or common._seq_lens_cpu.device.type != "cpu"
            or tuple(common._seq_lens_cpu.shape) != (1,)
            or common.seq_lens.dtype != torch.int32 or tuple(common.seq_lens.shape) != (1,)
            or common.seq_lens.device != state.seq.device
            or tuple(common.block_table_tensor.shape) != (1, state.blocks)
            or common.block_table_tensor.dtype != torch.int32
            or common.block_table_tensor.device != state.table.device
            or not common.block_table_tensor.is_contiguous()
            or tuple(positions.shape) != (1,) or positions.dtype != state.pos.dtype
            or positions.device != state.pos.device
            or tuple(hidden.shape) != tuple(state.hidden.shape) or hidden.dtype != state.hidden.dtype
            or hidden.device != state.hidden.device or not hidden.is_contiguous()
            or tuple(token.shape) != (1,) or token.dtype != torch.int32
            or token.device != state.token.device
            or any(t.data_ptr() % 16 for t in (
                positions, common.seq_lens, common.block_table_tensor, hidden, token))):
        return None
    length = int(common._seq_lens_cpu[0])
    if length < 1 or length + 3 > p.max_model_len:
        return None
    triple, _ = state.triple(length)
    if triple not in state.graphs:
        return None
    return state.execute(positions, common, hidden, token, length)


def _raw(t):
    return t.contiguous().view(torch.uint8).flatten().clone()


@torch.no_grad()
def prepare(p):
    if (getattr(p, "_nemo_burst_attempted", False)
            or os.environ.get("NEMO_MTP_DRAFT_BURST", "1") == "0"
            or not _eligible(p) or p.model_returns_tuple()
            or torch.cuda.is_current_stream_capturing()):
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    wrapper = getattr(builder, "_decode_wrappers_cudagraph", {}).get(1)
    bank = getattr(wrapper, "_nemo_native_plan", None)
    screen = getattr(p, "_draft_head_screen", None)
    sets = p._hostfast_draft_sets()
    mode, padded, _ = p._determine_batch_execution_and_padding(1)
    if (bank is None or not getattr(bank, "enabled", False) or not bank.entries
            or screen is None or screen.graph is None or sets is None
            or not screen.cascade or not screen.fused_topk or not screen.reuse
            or padded not in (1, 5) or len(sets.builders) != 1):
        return
    if any(getattr(wrapper, wname).data_ptr() != getattr(builder, bname).gpu.data_ptr()
           for wname, bname in (
               ("_paged_kv_indptr_buf", "paged_kv_indptr"),
               ("_paged_kv_indices_buf", "paged_kv_indices"),
               ("_paged_kv_last_page_len_buf", "paged_kv_last_page_len"))):
        return
    p._nemo_burst_attempted = True
    p._nemo_burst = None
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    gate = kernels.depth_gate_init(p.device)
    if gate is None or gate.numel() != 4:
        return
    caches = [p.compilation_config.static_forward_context[n].kv_cache
              for n in sorted(p._draft_attn_layer_names)]
    if any(not isinstance(t, torch.Tensor) or t.shape[0] < 4 for t in caches):
        return
    # Save exact current aliases as well as contents; per-depth pinned-buffer
    # installation must not leak a private fixture's aliases into later setup.
    refs = (builder.paged_kv_indptr.cpu, builder.paged_kv_indptr.np,
            builder.paged_kv_indptr_cpu_buffer, builder.paged_kv_last_page_len.cpu,
            builder.paged_kv_last_page_len.np, wrapper._pin_memory_int_workspace_buffer)
    old_current, old_events = sets._current, list(sets.events)
    old_plan = list(wrapper._plan_info)
    old_scalars = [(n, hasattr(wrapper, n), getattr(wrapper, n, None)) for n in _SCALARS]
    old_bank_restore = bank.last_restore
    old_bank_key = bank.last_native_key
    saved, seen = [], set()
    state, admitted = None, False
    try:
        for i in range(len(sets.sets)):
            sets.before(i)
        tensors = [p.input_ids[:padded], p.positions[:padded], p.hidden_states[:padded],
                   p._slot_mapping_buffer[:padded], gate]
        tensors.extend(t[:4] for t in caches)
        tensors.extend(t[:max(5, padded)] for t in p.model._moe_row_src_buffers())
        for name in ("_h", "_out", "scores", "refine_scores", "cand_logits", "_coarse_cand",
                     "_cand", "_hc", "_hr", "_hs"):
            t = getattr(screen, name, None)
            if isinstance(t, torch.Tensor):
                tensors.append(t)
        for name in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
            tensors.extend((getattr(builder, name).cpu, getattr(builder, name).gpu))
        tensors.extend((refs[0], refs[2], refs[3], refs[5]))
        for staging_list in sets.sets:
            for staging in staging_list:
                tensors.extend((staging.indptr_cpu, staging.indptr_buf, staging.last_page_len_cpu))
                tensors.extend(pair[1] for pair in staging.workspaces.values())
        for name in ("_int_workspace_buffer", "_float_workspace_buffer", "_qo_indptr_buf",
                     "_kv_lens_buffer", "_pin_memory_int_workspace_buffer"):
            t = getattr(wrapper, name, None)
            if isinstance(t, torch.Tensor):
                tensors.append(t)
        torch.cuda.synchronize(p.device)
        for tensor in tensors:
            key = _tensor(tensor)
            if key not in seen:
                saved.append((tensor, tensor.clone()))
                seen.add(key)
        state = Burst(p, builder, wrapper, bank, padded)
        state.gate = gate
        features = torch.arange(p.hidden_states.shape[-1], device=p.device)
        blocks = state.blocks
        table = (torch.arange(blocks, device=p.device, dtype=torch.int32) % 2).view(1, -1)
        row_zero = state.row_zero
        token = torch.zeros(1, dtype=torch.int32, device=p.device)
        pos = torch.zeros(1, dtype=p.positions.dtype, device=p.device)
        hidden = torch.zeros_like(state.hidden)

        def restore():
            for target, original in saved:
                target.copy_(original)
            wrapper._plan_info = list(old_plan)
            for name, present, value in old_scalars:
                if present:
                    setattr(wrapper, name, value)
                elif hasattr(wrapper, name):
                    delattr(wrapper, name)

        def initialize(length, case):
            restore()
            for cache in caches:
                cache[:2].zero_()
                cache[1].view(torch.uint8).fill_(0x20 + case)
            table.copy_(((torch.arange(blocks, device=p.device, dtype=torch.int32) + case) % 2).view(1, -1))
            token.fill_(case + 1)
            pos.fill_(length - 1)
            pattern = ((features * 7 + case * 13) % 127 - 63).float() / 128
            hidden.copy_(pattern)
            if case == 0:
                hidden.zero_()
            gate.copy_(torch.tensor(
                [1 if case == 2 else 0, 1, 0 if case == 3 else 1,
                 0x3F333333 if case else kernels.GATE_PROD_ONE],
                dtype=torch.int32, device=p.device))
            p.model.set_moe_row_src(padded, row_zero)
            state.pos.copy_(pos)
            state.hidden.copy_(hidden)
            state.token.copy_(token)
            state.table.copy_(table)
            state.seq.fill_(length)
            state.cpu_length.fill_(length)
            common = state.metadata(length)
            common.seq_lens = torch.tensor([length], dtype=torch.int32, device=p.device)
            common.block_table_tensor = table
            if case == 3:
                common._num_computed_tokens_cpu = None
                common.seq_lens_cpu_upper_bound = None
            return common

        # Capture every bounded exact triple on private data before requests.
        graph_bytes_before = torch.cuda.memory_allocated(p.device)
        for triple, length in state.categories.items():
            common = initialize(length, 0)
            state.capture(triple, length)
            if torch.cuda.memory_allocated(p.device) - graph_bytes_before > 512 * (1 << 20):
                raise ValueError("burst graph memory budget exceeded")

        # Exhaustive CURRENT metadata comparison for every recorded length.
        for length, entry in state.entries.items():
            common = state.metadata(length)
            state.table.copy_(table)
            _, _md = p.build_per_group_and_layer_attn_metadata(common, draft_index=1)
            pages = (length + p.block_size - 1) // p.block_size
            a = [builder.paged_kv_indptr.gpu[:2].clone(),
                 builder.paged_kv_indices.gpu[:pages].clone(),
                 builder.paged_kv_last_page_len.gpu[:1].clone(),
                 wrapper._int_workspace_buffer.view(torch.uint8)[:entry[1].numel()].clone()]
            state.cpu_length.fill_(length - 1)
            state.install(state.pos, state.hidden, state.token, 1, entry[1].numel(), update=False)
            b = [builder.paged_kv_indptr.gpu[:2], builder.paged_kv_indices.gpu[:pages],
                 builder.paged_kv_last_page_len.gpu[:1],
                 wrapper._int_workspace_buffer.view(torch.uint8)[:entry[1].numel()]]
            if not all(torch.equal(x, y) for x, y in zip(a, b)):
                raise ValueError("burst live page metadata differs from native builder")

        # Isolate the new fused input/position handoff against the actual eagle
        # kernel, including GPU positions/lengths independent of the CPU key.
        # All arguments remain private, and every dynamic producer is queued
        # immediately before its consumer on the actual stream.
        probe_length = sorted(state.categories.values())[0]
        entry = state.entries[probe_length + 1]
        for position_value in (0, p.block_size - 1, p.max_model_len - 2,
                               p.max_model_len - 1, p.max_model_len):
            for seq_value in (1, probe_length, p.max_model_len):
                common = initialize(probe_length, (position_value + seq_value) % 4)
                pos.fill_(position_value)
                common.seq_lens.fill_(seq_value)
                state.stage(pos, common, hidden, token, probe_length)
                eagle_step_update_slot_mapping_and_metadata(
                    pos, table, common.seq_lens, p.block_size, p.max_model_len,
                    p.positions[:1], p._slot_mapping_buffer[:padded], padded)
                p.hidden_states[:1].copy_(hidden)
                p.input_ids[:1].copy_(token)
                expected = [_raw(x) for x in (
                    p.positions[:padded], p._slot_mapping_buffer[:padded],
                    p.hidden_states[:padded], p.input_ids[:padded], common.seq_lens)]
                # Inputs in state were copied before the original kernel. The
                # candidate must produce the same state even at reset/saturation.
                state.install(state.pos, state.hidden, state.token, 1, entry[1].numel())
                actual = [_raw(x) for x in (
                    p.positions[:padded], p._slot_mapping_buffer[:padded],
                    p.hidden_states[:padded], p.input_ids[:padded], state.seq)]
                if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
                    raise ValueError("fused current-input/eagle handoff bytes differ")

        # Exercise the actual runtime guard and handoff against the original,
        # using independent simulated caller buffers, never request data.
        state.enabled = True
        p._nemo_burst = state

        def original(common):
            position, h, t = pos, hidden, token
            out = []
            for step in range(3):
                position = p._update_positions_dependent_metadata(
                    position, common, 1, padded, p.block_size)
                side = p._draft_plan_side_stream()
                if side is not None:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        _, md = p.build_per_group_and_layer_attn_metadata(common, draft_index=step + 1)
                        ready = p._draft_plan_ready[step]
                        ready.record(side)
                    for m in {id(x): x for x in md.values()}.values():
                        m.plan_ready_event = ready
                else:
                    _, md = p.build_per_group_and_layer_attn_metadata(common, draft_index=step + 1)
                p.input_ids[:1].copy_(t.int())
                p.hidden_states[:1].copy_(h)
                kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                              inputs_embeds=None, hidden_states=p.hidden_states[:padded])
                with set_forward_context(md, p.vllm_config, num_tokens=padded,
                                         cudagraph_runtime_mode=mode,
                                         slot_mapping=p._get_slot_mapping(padded)):
                    ret = original_model_graph(p, kwargs, md, padded, 1)
                    if ret is None:
                        ret = p.model(**kwargs)
                h = ret[:1]
                t = screen.greedy(h)
                out.append(t)
            return torch.stack(out, 1), ret

        def observe(common, out, final_hidden):
            result = dict(tokens=_raw(out), gate=_raw(gate), positions=_raw(p.positions[:padded]),
                          inputs=_raw(p.input_ids[:padded]), hidden=_raw(p.hidden_states[:padded]),
                          seq=_raw(common.seq_lens), slots=_raw(p._slot_mapping_buffer[:padded]),
                          kv=torch.cat([_raw(t[:4]) for t in caches]),
                          metadata=torch.cat([_raw(getattr(builder, n).gpu) for n in (
                              "paged_kv_indptr", "paged_kv_last_page_len")]),
                          indices=_raw(builder.paged_kv_indices.gpu[:(int(common._seq_lens_cpu[0]) + p.block_size - 1) // p.block_size]),
                          final_hidden=_raw(final_hidden[:1]),
                          maxlen=torch.tensor([common.max_seq_len]),
                          plan=torch.tensor(list(wrapper._plan_info)),
                          plan_scalars=torch.tensor(list(repr(tuple(
                              (n, hasattr(wrapper, n), getattr(wrapper, n, None))
                              for n in _SCALARS)).encode()), dtype=torch.uint8))
            for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
                t = getattr(common, name, None)
                result[name] = torch.empty(0, dtype=torch.uint8) if t is None else _raw(t)
            for name in ("scores", "refine_scores", "cand_logits", "_coarse_cand", "_cand", "_h", "_out", "_hc", "_hr", "_hs"):
                result[name] = _raw(getattr(screen, name))
            for i, staging_list in enumerate(sets.sets):
                for j, staging in enumerate(staging_list):
                    for name in ("indptr_cpu", "indptr_buf", "last_page_len_cpu"):
                        result[f"staging{i}_{j}_{name}"] = _raw(getattr(staging, name))
            return result

        samples = set(state.categories.values())
        cap = max(state.entries)
        for pages in (1, 2, max(1, cap // p.block_size // 2), max(1, cap // p.block_size - 1)):
            for residue in (1, max(1, p.block_size // 2), p.block_size - 1, p.block_size):
                length = (pages - 1) * p.block_size + residue
                triple, _ = state.triple(length)
                if triple in state.graphs and length + 3 <= p.max_model_len:
                    samples.add(length)
        ratios = []
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=p.device)
        for length in sorted(samples):
            for case in range(4):
                results, times = {}, {}
                for fused in ((False, True) if (length + case) % 2 else (True, False)):
                    common = initialize(length, case)
                    torch.cuda.synchronize(p.device)
                    start = time.perf_counter()
                    if fused:
                        out = maybe_run(p, pos, common, hidden, token, 1, padded, True, gate)
                        if out is None:
                            raise ValueError("burst actual runtime guard declined private case")
                        final_hidden = state.graphs[state.triple(length)[0]][5]
                    else:
                        out, final_hidden = original(common)
                    torch.cuda.synchronize(p.device)
                    times[fused] = time.perf_counter() - start
                    results[fused] = observe(common, out, final_hidden)
                if results[False].keys() != results[True].keys() or not all(
                        torch.equal(results[False][n], results[True][n]) for n in results[False]):
                    raise ValueError("burst changes private three-step model/KV/metadata/gate state")
                ratios.append((case, times[True] / times[False]))
        # Repeat complete tail timing on changing private data, both cold/warm.
        for cold in (False, True):
            times = {False: [], True: []}
            length = sorted(samples)[len(samples) // 2]
            for repeat in range(6):
                for fused in ((False, True) if repeat % 2 else (True, False)):
                    common = initialize(length, repeat % 4)
                    if cold:
                        flush.zero_()
                    torch.cuda.synchronize(p.device)
                    start = time.perf_counter()
                    out = (maybe_run(p, pos, common, hidden, token, 1, padded, True, gate)
                           if fused else original(common))
                    if out is None:
                        raise ValueError("burst timed runtime guard declined")
                    torch.cuda.synchronize(p.device)
                    times[fused].append(time.perf_counter() - start)
            ratios.append((-1, statistics.median(times[True]) / statistics.median(times[False])))
        if (statistics.median([r for _, r in ratios]) >= 0.97
                or statistics.median([r for c, r in ratios if c != 2]) >= 0.98
                or statistics.median([r for c, r in ratios if c == 2]) >= 1.01):
            raise ValueError("burst full-tail latency admission declined")
        admitted = True
    except Exception as exc:
        logger.warning("Native three-step burst kept original loop: %r", exc)
    finally:
        p._nemo_burst = None
        if state is not None:
            state.enabled = False
        torch.cuda.synchronize(p.device)
        for target, original in saved:
            target.copy_(original)
        (builder.paged_kv_indptr.cpu, builder.paged_kv_indptr.np,
         builder.paged_kv_indptr_cpu_buffer, builder.paged_kv_last_page_len.cpu,
         builder.paged_kv_last_page_len.np, wrapper._pin_memory_int_workspace_buffer) = refs
        sets._current, sets.events = old_current, old_events
        wrapper._plan_info = old_plan
        for name, present, value in old_scalars:
            if present:
                setattr(wrapper, name, value)
            elif hasattr(wrapper, name):
                delattr(wrapper, name)
        bank.last_restore, bank.last_native_key = old_bank_restore, old_bank_key
        torch.cuda.synchronize(p.device)
    if admitted:
        state.enabled = True
        del state._capture_entries
        p._nemo_burst = state
        logger.info("Native three-step burst admitted: %d exact plan triples", len(state.graphs))
