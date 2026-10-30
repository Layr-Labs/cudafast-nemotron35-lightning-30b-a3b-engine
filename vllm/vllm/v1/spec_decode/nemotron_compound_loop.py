# SPDX-License-Identifier: Apache-2.0
"""Boot-qualified native MTP loop graph with original device-selected plans.

Only structural native planner bytes are retained. Request block IDs are copied
from the live request on each invocation. No token, KV or prompt is cached.
All qualification and compilation runs on the official remote box.
"""
import os
import statistics
import time
from types import SimpleNamespace

import torch

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode import mtp_glue
from vllm.v1.spec_decode.nemotron_full_step import (
    _inputs, _metadata, _tensor, maybe_run as model_graph,
)
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

logger = init_logger(__name__)
ENABLED = os.environ.get("NEMO_MTP_COMPOUND_LOOP", "1") != "0"


@triton.jit
def _stage(token, ids, hsrc, hdst, hstride, positions, seq,
           table, slots, indptr, indices, last_page, workspace,
           prefix_ptrs, prefix_sizes,
           HID: tl.constexpr, PADDED: tl.constexpr, PAGE: tl.constexpr,
           MAX_LEN: tl.constexpr, MAX_PAGES: tl.constexpr,
           MAX_PREFIX: tl.constexpr, PAD: tl.constexpr,
           BLOCK: tl.constexpr):
    # Original fused MTP input/position/slot arithmetic, batch one. Runtime
    # guards exclude clamp boundaries, but retain the original clamp itself.
    position = tl.load(positions)
    old_seq = tl.load(seq)
    tl.debug_barrier()
    new_position = position + 1
    exceeds = new_position >= MAX_LEN
    clamped = tl.where(exceeds, 0, new_position)
    block = tl.minimum(clamped // PAGE, MAX_PAGES - 1)
    slot = tl.load(table + block) * PAGE + clamped % PAGE
    tl.store(ids, tl.load(token).to(tl.int32))
    source_bits = hsrc.to(tl.pointer_type(tl.uint16))
    dest_bits = hdst.to(tl.pointer_type(tl.uint16))
    offsets = tl.arange(0, BLOCK)
    for start in range(0, HID, BLOCK):
        mask = start + offsets < HID
        tl.store(dest_bits + start + offsets,
                 tl.load(source_bits + start + offsets, mask=mask), mask=mask)
    tl.store(positions, clamped)
    padded = tl.arange(0, 8)
    tl.store(slots + padded, tl.where(padded == 0, tl.where(exceeds, PAD, slot), PAD),
             mask=padded < PADDED)
    length = tl.minimum(tl.where(exceeds, 1, old_seq + 1), MAX_LEN)
    tl.store(seq, length)

    # Same current-page values as FlashInferMetadataBuilder, and the exact
    # used integer prefix the installed native planner produced for length.
    pages = (length + PAGE - 1) // PAGE
    tl.store(indptr, 0)
    tl.store(indptr + 1, pages)
    residue = length % PAGE
    tl.store(last_page, tl.where(residue == 0, PAGE, residue))
    page_offsets = tl.arange(0, MAX_PAGES)
    tl.store(indices + page_offsets, tl.load(table + page_offsets,
             mask=page_offsets < pages, other=0), mask=page_offsets < pages)
    pointer = tl.load(prefix_ptrs + length).to(tl.uint64)
    source = pointer.to(tl.pointer_type(tl.uint8))
    size = tl.load(prefix_sizes + length)
    for start in range(0, MAX_PREFIX, BLOCK):
        mask = start + offsets < size
        tl.store(workspace + start + offsets,
                 tl.load(source + start + offsets, mask=mask, other=0), mask=mask)


def _launch_stage(s, p, token, hidden):
    _stage[(1,)](
        token, p.input_ids, hidden, p.hidden_states, hidden.stride(0),
        p.positions, s.seq, s.table, p._slot_mapping_buffer,
        s.builder.paged_kv_indptr.gpu, s.builder.paged_kv_indices.gpu,
        s.builder.paged_kv_last_page_len.gpu,
        s.wrapper._int_workspace_buffer.view(torch.uint8),
        s.prefix_ptrs, s.prefix_sizes,
        HID=p.hidden_size, PADDED=s.padded, PAGE=p.block_size,
        MAX_LEN=p.max_model_len, MAX_PAGES=s.max_pages,
        MAX_PREFIX=s.max_prefix, PAD=PADDING_SLOT_ID, BLOCK=1024, num_warps=4,
    )


def _family(key):
    return key[2:]


def _length(key, page):
    starts, residue = key[:2]
    return (starts[1] - 1) * page + residue


def _state(p, builder, wrapper, bank, layout):
    if p.hidden_states.dtype != torch.bfloat16:
        return None
    wanted_plan, wanted_scalars = layout[1:]
    families = {}
    for key, (plan, prefix, scalars) in bank.entries.items():
        if (plan == wanted_plan and scalars == wanted_scalars
                and prefix.dtype == torch.uint8 and prefix.is_cuda
                and prefix.is_contiguous() and 0 < prefix.numel() <= 65536):
            length = _length(key, p.block_size)
            if 1 <= length <= p.max_model_len:
                families.setdefault(_family(key), {})[length] = (key, prefix)
    if not families:
        return None
    family, entries = max(families.items(), key=lambda item: len(item[1]))
    passes = int(p.num_speculative_tokens) - 1
    if passes < 2:
        return None
    starts = sorted(n for n in entries
                    if all(n + i in entries for i in range(1, passes))
                    and n + passes - 1 < p.max_model_len)
    if not starts:
        return None
    # Strong prefix owners make the device pointer lookup immutable. These
    # are copied ORIGINAL planner prefixes, never reconstructed schedules.
    pointers, sizes = [0] * (p.max_model_len + 1), [0] * (p.max_model_len + 1)
    for length, (_, prefix) in entries.items():
        pointers[length], sizes[length] = prefix.data_ptr(), prefix.numel()
    return SimpleNamespace(
        builder=builder, wrapper=wrapper, bank=bank, family=family,
        entries=entries, starts=frozenset(starts), seed=starts[0],
        prefix_owners=tuple(v[1] for v in entries.values()),
        prefix_ptrs=torch.tensor(pointers, dtype=torch.int64, device=p.device),
        prefix_sizes=torch.tensor(sizes, dtype=torch.int32, device=p.device),
        seq=torch.ones(1, dtype=torch.int32, device=p.device),
        max_prefix=max(sizes),
        max_pages=triton.next_power_of_2(
            (p.max_model_len + p.block_size - 1) // p.block_size),
        padded=p._nemo_full[1],
        table=torch.zeros((1, triton.next_power_of_2(
            (p.max_model_len + p.block_size - 1) // p.block_size)),
            dtype=torch.int32, device=p.device),
        plan=wanted_plan, scalars=wanted_scalars,
        signature=p._nemo_full[2], inputs=p._nemo_full[3], graph=None,
        passes=passes,
    )


def _advance_cpu(cad, steps):
    # Preserve the original field order, including any CPU aliasing.
    for _ in range(steps):
        cad.max_seq_len += 1
        for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu",
                     "seq_lens_cpu_upper_bound"):
            value = getattr(cad, name)
            if value is not None:
                value += 1


def _ready(metadata):
    # The first original plan can be on the plan side stream. Its writes
    # must precede the compound graph's first attention and its later writes.
    seen = set()
    for md in metadata.values():
        event = getattr(md, "plan_ready_event", None)
        if event is not None and id(event) not in seen:
            torch.cuda.current_stream().wait_event(event)
            seen.add(id(event))


def _check(s, p, kwargs, metadata, cad, padded, batch_size):
    if (batch_size != 1 or padded != s.padded or p.num_speculative_tokens != s.passes + 1
            or p._nemo_full is None or s.wrapper._nemo_native_plan is not s.bank
            or not s.bank.enabled or s.bank.force_native
            or _inputs(kwargs) != s.inputs or _metadata(p, metadata) != s.signature
            or tuple(s.wrapper._plan_info) != s.plan
            or s.bank.last_restore_key is None
            or _family(s.bank.last_restore_key) != s.family):
        return None
    cpu = cad._seq_lens_cpu
    if (cpu is None or cpu.device.type != "cpu" or cpu.dtype != torch.int32
            or cpu.numel() != 1 or cad.seq_lens.dtype != torch.int32
            or not cad.seq_lens.is_cuda or cad.seq_lens.numel() != 1
            or not cad.seq_lens.is_contiguous()
            or cad.max_seq_len + s.passes - 1 >= p.max_model_len):
        return None
    length = int(cpu[0])
    shadows = [getattr(cad, name) for name in (
        "_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound")
        if getattr(cad, name) is not None]
    if (any(t.device.type != "cpu" or t.dtype != torch.int32 or t.numel() != 1
            for t in shadows)
            or len({t.data_ptr() for t in shadows}) != len(shadows)):
        return None
    if (length not in s.starts
            or s.bank.last_restore_key != s.entries[length][0]
            or any(s.bank.entries.get(s.entries[n][0], (None, None))[1]
                   is not s.entries[n][1] for n in range(length, length + s.passes))):
        return None
    table = cad.block_table_tensor
    pages = (length + s.passes - 1 + p.block_size - 1) // p.block_size
    if (not table.is_cuda or table.dtype != torch.int32 or table.ndim != 2
            or table.shape[0] != 1 or table.shape[1] < pages
            or table.stride(1) != 1 or table.device != s.table.device
            or pages > s.max_pages
            or pages > s.builder.paged_kv_indices.gpu.numel()):
        return None
    if tuple((n, getattr(s.wrapper, n)) for n, _ in s.scalars) != s.scalars:
        return None
    return pages


def _replay(s, p, kwargs, metadata, cad, padded, batch_size):
    pages = _check(s, p, kwargs, metadata, cad, padded, batch_size)
    if pages is None:
        return None
    _ready(metadata)
    s.seq.copy_(cad.seq_lens)
    s.table[0, :pages].copy_(cad.block_table_tensor[0, :pages])
    s.graph.replay()
    # The caller's original seq buffer is preserved as an observable output,
    # although the captured metadata stages use a private stable sequence ptr.
    cad.seq_lens.copy_(s.seq)
    _advance_cpu(cad, s.passes - 1)
    s.bank.last_restore_key = s.entries[int(cad._seq_lens_cpu[0])][0]
    return s.tokens


def maybe_run(p, kwargs, metadata, cad, padded, batch_size, sampling_metadata):
    s = getattr(p, "_nemo_compound_loop", None)
    if (s is None or not sampling_metadata.all_greedy
            or torch.cuda.is_current_stream_capturing()):
        return None
    return _replay(s, p, kwargs, metadata, cad, padded, batch_size)


def _bindings(builder, wrappers):
    return (
        builder.paged_kv_indptr.cpu, builder.paged_kv_indptr.np,
        builder.paged_kv_indptr_cpu_buffer,
        builder.paged_kv_last_page_len.cpu, builder.paged_kv_last_page_len.np,
        tuple((w, w._pin_memory_int_workspace_buffer) for w in wrappers),
    )


def _restore_bindings(builder, values):
    (builder.paged_kv_indptr.cpu, builder.paged_kv_indptr.np,
     builder.paged_kv_indptr_cpu_buffer,
     builder.paged_kv_last_page_len.cpu, builder.paged_kv_last_page_len.np,
     workspaces) = values
    for wrapper, buffer in workspaces:
        wrapper._pin_memory_int_workspace_buffer = buffer


def prepare(p):
    """Compound-loop qualification; a plan bank admitted only for this loop
    (guard_failed, depth >= 5) is removed again when the loop is not admitted,
    so the original native planner runs exactly as without the bank."""
    try:
        _prepare(p)
    finally:
        if getattr(p, "_nemo_compound_loop", None) is None:
            try:
                builder = p.draft_attn_groups[0].get_metadata_builder()
                wrapper = builder._decode_wrappers_cudagraph.get(1)
            except (AttributeError, IndexError):
                wrapper = None
            bank = getattr(wrapper, "_nemo_native_plan", None)
            if bank is not None and getattr(bank, "guard_failed", False):
                bank.enabled = False
                wrapper._nemo_native_plan = None
                logger.info("Native MTP plan bank removed: compound loop not admitted")


def _prepare(p):
    """Complete original-loop versus compound-loop qualification, remote boot."""
    if (not ENABLED or getattr(p, "_nemo_compound_attempted", False)
            or torch.cuda.is_current_stream_capturing()
            or getattr(p, "_nemo_full", None) is None
            or getattr(p, "_nemo_full_native_layout", None) is None):
        return
    from vllm.v1.spec_decode.nemotron_decode_plan import _eligible, _SCALARS
    if not _eligible(p):
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    layout = p._nemo_full_native_layout
    wrapper = builder._decode_wrappers_cudagraph.get(1)
    bank = getattr(wrapper, "_nemo_native_plan", None)
    screen = getattr(p, "_draft_head_screen", None)
    sets = p._hostfast_draft_sets()
    if (wrapper is not layout[0] or bank is None or not bank.enabled
            or bank.force_native or screen is None or screen.graph is None
            or sets is None or len(sets.sets) < 4):
        return
    p._nemo_compound_attempted = True
    p._nemo_compound_loop = None
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    from vllm.v1.attention.backend import CommonAttentionMetadata
    gate = kernels.depth_gate_init(p.device)
    if gate is None:
        return
    saved, seen, admitted, s = [], set(), False, None
    wrappers = sets.wrappers(builder)
    bindings = _bindings(builder, wrappers)
    old_plan = list(wrapper._plan_info)
    old_scalars = tuple((n, hasattr(wrapper, n), getattr(wrapper, n, None)) for n in _SCALARS)
    old_restore = (bank.last_restore, bank.last_restore_key, bank.last_native_key)
    old_force_native = bank.force_native
    old_set_index = sets._current
    caches = [p.compilation_config.static_forward_context[n].kv_cache
              for n in sorted(p._draft_attn_layer_names)]
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or t.shape[0] < 4
           for t in caches):
        return
    values = [p.input_ids[:5], p.positions[:5], p.hidden_states[:5],
              p._slot_mapping_buffer[:5], gate]
    values.extend(t[:4] for t in caches)
    values.extend(t[:5] for t in p.model._moe_row_src_buffers())
    for name in ("_h", "_out", "scores", "refine_scores", "cand_logits"):
        value = getattr(screen, name, None)
        if isinstance(value, torch.Tensor):
            values.append(value)
    for name in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        values.extend((getattr(builder, name).cpu, getattr(builder, name).gpu))
    values.extend((builder.paged_kv_indptr_cpu_buffer,
                   wrapper._int_workspace_buffer, wrapper._float_workspace_buffer))
    for per_step in sets.sets:
        for staging in per_step:
            values.extend((staging.indptr_cpu, staging.indptr_buf,
                           staging.last_page_len_cpu))
            values.extend(v[1] for v in staging.workspaces.values())
    values.extend(w._pin_memory_int_workspace_buffer for w in wrappers)
    torch.cuda.synchronize(p.device)
    try:
        for value in values:
            key = _tensor(value)
            if key not in seen:
                saved.append((value, value.clone()))
                seen.add(key)
        s = _state(p, builder, wrapper, bank, layout)
        if s is None:
            return
        row_zero = torch.zeros(1, dtype=torch.long, device=p.device)
        s.table.copy_((torch.arange(s.max_pages, device=p.device,
                                   dtype=torch.int32) % 2).view(1, -1))
        # Distinct from captured staging: private timing must include the same
        # live-request-table copy as production, never a self-copy no-op.
        private_live_table = s.table.clone()
        features = torch.arange(p.hidden_size, device=p.device)
        kwargs = dict(input_ids=p.input_ids[:s.padded],
                      positions=p._get_positions(s.padded), inputs_embeds=None,
                      hidden_states=p.hidden_states[:s.padded])

        def common(length):
            cpu = torch.tensor([length], dtype=torch.int32)
            query = torch.tensor([0, 1], dtype=torch.int32)
            return CommonAttentionMetadata(
                query_start_loc=query.to(p.device), query_start_loc_cpu=query,
                seq_lens=cpu.to(p.device), _seq_lens_cpu=cpu,
                _num_computed_tokens_cpu=cpu - 1,
                seq_lens_cpu_upper_bound=cpu.clone(),
                num_reqs=1, num_actual_tokens=1, max_query_len=1,
                max_seq_len=length, block_table_tensor=private_live_table,
                slot_mapping=p._slot_mapping_buffer[:1], causal=True,
            )

        def restore():
            torch.cuda.synchronize(p.device)
            for target, original in saved:
                target.copy_(original)
            _restore_bindings(builder, bindings)
            wrapper._plan_info = list(old_plan)
            for name, present, value in old_scalars:
                if present:
                    setattr(wrapper, name, value)
                elif hasattr(wrapper, name):
                    delattr(wrapper, name)

        def setup(length, case):
            restore()
            private_live_table.copy_(((torch.arange(s.max_pages, device=p.device,
                                dtype=torch.int32) + case) % 2).view(1, -1))
            s.table.copy_(private_live_table)
            for cache in caches:
                cache[:2].zero_()
                cache[1].view(torch.uint8).fill_(32 + case)
            p.input_ids[:s.padded].fill_(case + 1)
            p.positions[:s.padded].fill_(length - 1)
            pattern = (((features * 7 + case * 13) % 127 - 63).float() / 128)
            p.hidden_states[:s.padded].copy_(pattern)
            p._slot_mapping_buffer[:s.padded].fill_(PADDING_SLOT_ID)
            pos = length - 1
            p._slot_mapping_buffer[0] = (
                private_live_table[0, pos // p.block_size] * p.block_size
                + pos % p.block_size
            )
            gate.copy_(torch.tensor(
                [1 if case == 2 else 0, 1, 1, kernels.GATE_PROD_ONE],
                dtype=torch.int32, device=p.device))
            p.model.set_moe_row_src(s.padded, row_zero)
            cad = common(length)
            _, md = p.build_per_group_and_layer_attn_metadata(cad, draft_index=1)
            s.seq.copy_(cad.seq_lens)
            return cad, md

        # Exhaustive structural stage proof for every admitted native length.
        # Writes only private synthetic page IDs plus saved MTP work buffers.
        private_hidden = torch.full((1, p.hidden_size), 0.125,
                                    dtype=p.dtype, device=p.device)
        bit_cases = torch.tensor([0, -32768, 32640, -128, 32641, -127, 1, -32767],
                                 dtype=torch.int16, device=p.device)
        private_hidden.view(torch.int16).flatten().copy_(
            bit_cases.repeat((p.hidden_size + 7) // 8)[:p.hidden_size])
        private_token = torch.tensor([17], dtype=torch.int32, device=p.device)
        raw = wrapper._int_workspace_buffer.view(torch.uint8)
        for length in sorted(s.entries):
            if length <= 1:
                continue
            p.positions[0].fill_(length - 2)
            s.seq.fill_(length - 1)
            prefix = s.entries[length][1]
            tail = raw[prefix.numel():prefix.numel() + 16]
            tail.fill_(0xD5)
            _launch_stage(s, p, private_token, private_hidden)
            pages = (length + p.block_size - 1) // p.block_size
            expected_indices = s.table[0, :pages]
            expected_slot = ((length - 1) // p.block_size % 2) * p.block_size + (length - 1) % p.block_size
            torch.cuda.synchronize(p.device)
            checks = {
                "prefix": torch.equal(raw[:prefix.numel()], prefix),
                "tail": bool(torch.all(tail == 0xD5)),
                "indices": torch.equal(builder.paged_kv_indices.gpu[:pages], expected_indices),
                "position": int(p.positions[0]) == length - 1,
                "seq": int(s.seq[0]) == length,
                "input": int(p.input_ids[0]) == 17,
                "slot": int(p._slot_mapping_buffer[0]) == expected_slot,
                "pad_slots": bool(torch.all(p._slot_mapping_buffer[1:s.padded] == PADDING_SLOT_ID)),
                "indptr": builder.paged_kv_indptr.gpu[:2].tolist() == [0, pages],
                "last_page": int(builder.paged_kv_last_page_len.gpu[0]) == ((length - 1) % p.block_size + 1),
                "hidden": torch.equal(p.hidden_states[:1].view(torch.int16),
                                      private_hidden.view(torch.int16)),
            }
            if not all(checks.values()):
                bad = [k for k, v in checks.items() if not v]
                extra = ""
                if "prefix" in bad:
                    neq = (raw[:prefix.numel()] != prefix).nonzero().flatten()
                    extra = f" prefix_bytes={prefix.numel()} first_bad={neq[:8].tolist()} n_bad={neq.numel()}"
                raise ValueError(f"compound native metadata stage differs: length={length} "
                                 f"failed={bad}{extra}")

        from vllm.v1.spec_decode import nemotron_step_cond as step_cond
        cond = getattr(p, "_nemo_step", None)

        last_hidden = [None, None]
        final_hidden_of = {}

        def body(md):
            tokens = []
            for step in range(s.passes):
                if cond is not None and torch.cuda.is_current_stream_capturing():
                    hidden, token = step_cond.capture_pass(p, gate)
                elif cond is not None:
                    hidden, token = step_cond.maybe_run(p, kwargs, md, s.padded, 1) or (None, None)
                else:
                    hidden = model_graph(p, kwargs, md, s.padded, 1)
                    token = screen.greedy(hidden[:1]) if isinstance(hidden, torch.Tensor) else None
                if hidden is None or not isinstance(hidden, torch.Tensor):
                    raise ValueError("full model graph declined compound capture")
                tokens.append(token)
                if step < s.passes - 1:
                    _launch_stage(s, p, token, hidden[:1])
            last_hidden[1] = hidden
            return tuple(tokens)

        # Warm Triton, the existing model/head child graphs and clone outputs
        # before capturing the three-pass parent. Each token clone has its own
        # strong owner within this parent graph, including across replays.
        for _ in range(2):
            cad, md = setup(s.seed, 0)
            body(md)
        cad, capture_md = setup(s.seed, 0)
        torch.cuda.synchronize(p.device)
        capture_stream = torch.cuda.Stream(device=p.device)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream,
                              pool=torch.cuda.graph_pool_handle()):
            tokens = body(capture_md)
        torch.cuda.synchronize(p.device)
        if len({t.data_ptr() for t in tokens}) != s.passes:
            raise ValueError("compound draft output slots alias")
        s.graph, s.tokens = graph, tokens
        final_hidden_of[1] = last_hidden[1]
        s.owners = (kwargs, capture_md, row_zero, capture_stream)

        fallbacks = [0]

        def original(cad, md):
            tokens = []
            mode, _, across = p._determine_batch_execution_and_padding(1)
            for step in range(s.passes):
                token = None
                with set_forward_context(
                    md, p.vllm_config, num_tokens=s.padded,
                    num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
                    slot_mapping=p._get_slot_mapping(s.padded),
                ):
                    res = None if cond is None else step_cond.maybe_run(
                        p, kwargs, md, s.padded, 1)
                    if res is not None:
                        hidden, token = res
                    elif cond is not None:
                        fallbacks[0] += 1
                    else:
                        hidden = model_graph(p, kwargs, md, s.padded, 1)
                if hidden is None:
                    raise ValueError("original full graph declined private comparison")
                if token is None:
                    token = screen.greedy(hidden[:1])
                tokens.append(token)
                last_hidden[0] = hidden
                if step < s.passes - 1:
                    p._update_positions_dependent_metadata(
                        p.positions[:1], cad, 1, s.padded, p.block_size,
                        fused=(token, hidden[:1]),
                    )
                    side = p._draft_plan_side_stream()
                    if side is None:
                        _, md = p.build_per_group_and_layer_attn_metadata(
                            cad, draft_index=step + 2)
                    else:
                        main = torch.cuda.current_stream()
                        side.wait_stream(main)
                        with torch.cuda.stream(side):
                            _, md = p.build_per_group_and_layer_attn_metadata(
                                cad, draft_index=step + 2)
                            ready = p._draft_plan_ready[step + 1]
                            ready.record(side)
                        for value in {id(v): v for v in md.values()}.values():
                            value.plan_ready_event = ready
            return tuple(tokens)

        def observation(cad, tokens):
            result = [torch.stack(tokens), p.positions[:s.padded],
                      p.input_ids[:s.padded], p.hidden_states[:s.padded],
                      p._slot_mapping_buffer[:s.padded], cad.seq_lens, gate]
            result.extend(cache[:4] for cache in caches)
            result.extend(getattr(cad, n) for n in (
                "_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"))
            pages = (int(cad._seq_lens_cpu[0]) + p.block_size - 1) // p.block_size
            result.extend((builder.paged_kv_indices.gpu[:pages],
                           builder.paged_kv_indptr.gpu[:2],
                           builder.paged_kv_last_page_len.gpu[:1]))
            result.extend(getattr(screen, n) for n in (
                "_h", "_out", "scores", "refine_scores", "cand_logits")
                          if isinstance(getattr(screen, n, None), torch.Tensor))
            return ([t.contiguous().view(torch.uint8).clone() for t in result],
                    cad.max_seq_len)

        starts = sorted(s.starts)
        selected = {starts[0], starts[-1], starts[len(starts) // 2]}
        for page in (1, 2, 3, max(1, starts[-1] // p.block_size // 2)):
            selected.update(n for n in (page * p.block_size - 1,
                                       page * p.block_size,
                                       page * p.block_size + 1) if n in s.starts)
        pressure = torch.empty(64 << 20, dtype=torch.uint8, device=p.device)
        ratios, active_ratios = [], []
        # Actual three-pass production path: current plan first, normal side
        # stream staging later, full model child graphs and original head.
        # Compare complete private state, then wall latency including host
        # submission and CPU metadata work, in alternating order.
        for length in sorted(selected):
            for case in range(3):
                for cold in (False, True):
                    samples = [[], []]
                    for trial in range(5):
                        observed = {}
                        for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                            cad, md = setup(length, case)
                            if cold:
                                pressure.fill_(trial + case)
                            torch.cuda.synchronize(p.device)
                            begin = time.perf_counter()
                            out = (original(cad, md) if which == 0 else
                                   _replay(s, p, kwargs, md, cad, s.padded, 1))
                            if out is None:
                                raise ValueError("compound runtime guard declined private fixture")
                            torch.cuda.synchronize(p.device)
                            elapsed = time.perf_counter() - begin
                            if not elapsed > 0:
                                raise ValueError("invalid compound private timing")
                            samples[which].append(elapsed)
                            observed[which] = [observation(cad, out)]
                            # Exercise a subsequent ORIGINAL builder and model
                            # after the compound path, including reuse of a
                            # different pinned staging set. This is outside the
                            # timed three-pass comparison.
                            if length + 3 < p.max_model_len:
                                final_hidden = (last_hidden[0] if which == 0
                                                else final_hidden_of[1])
                                if not isinstance(final_hidden, torch.Tensor):
                                    raise ValueError("compound continuation hidden is not a tensor")
                                p._update_positions_dependent_metadata(
                                    p.positions[:1], cad, 1, s.padded, p.block_size,
                                    fused=(out[-1], final_hidden[:1]),
                                )
                                # Also exercise the opaque original-planner
                                # boundary, not just another bank hit.
                                previous_force = bank.force_native
                                bank.force_native = case == 1
                                try:
                                    _, extra_md = p.build_per_group_and_layer_attn_metadata(
                                        cad, draft_index=0)
                                finally:
                                    bank.force_native = previous_force
                                extra_hidden = model_graph(p, kwargs, extra_md, s.padded, 1)
                                if extra_hidden is None:
                                    raise ValueError("original continuation graph declined")
                                extra_token = screen.greedy(extra_hidden[:1])
                                torch.cuda.synchronize(p.device)
                                observed[which].append(observation(cad, (extra_token,)))
                        a, b = observed[0], observed[1]
                        if (len(a) != len(b) or any(
                                x[1] != y[1] or len(x[0]) != len(y[0])
                                or not all(torch.equal(u, v) for u, v in zip(x[0], y[0]))
                                for x, y in zip(a, b))):
                            where = []
                            for oi, (x, y) in enumerate(zip(a, b)):
                                if x[1] != y[1]:
                                    where.append((oi, "max_seq_len", x[1], y[1]))
                                for ti, (u, v) in enumerate(zip(x[0], y[0])):
                                    if u.shape != v.shape or not torch.equal(u, v):
                                        nd = int((u != v).sum()) if u.shape == v.shape else -1
                                        where.append((oi, ti, tuple(u.shape), nd))
                            raise ValueError(
                                f"compound full private state differs from original loop: "
                                f"length={length} case={case} cold={cold} diffs={where[:12]} "
                                f"orig_cond_fallbacks={fallbacks[0]}")
                    ratio = statistics.median(samples[1]) / statistics.median(samples[0])
                    ratios.append(ratio)
                    if case != 2:
                        active_ratios.append(ratio)
        if (not active_ratios or statistics.median(ratios) >= 0.98
                or statistics.median(active_ratios) >= 0.98
                or max(active_ratios) >= 1.01):
            raise ValueError("compound graph did not improve complete private loop latency")
        s.private_ratios = tuple(ratios)
        admitted = True
    except Exception as exc:
        logger.warning("Native MTP compound loop retained original path: %r", exc)
    finally:
        torch.cuda.synchronize(p.device)
        for target, original in saved:
            target.copy_(original)
        _restore_bindings(builder, bindings)
        wrapper._plan_info = old_plan
        for name, present, value in old_scalars:
            if present:
                setattr(wrapper, name, value)
            elif hasattr(wrapper, name):
                delattr(wrapper, name)
        bank.last_restore, bank.last_restore_key, bank.last_native_key = old_restore
        bank.force_native = old_force_native
        sets._current = old_set_index
        torch.cuda.synchronize(p.device)
        p._nemo_compound_loop = s if admitted else None
    if admitted:
        logger.info("Native MTP compound loop admitted: lengths=%d ratios=%s",
                    len(s.starts), s.private_ratios)


nowait_stats = [0, 0]


def maybe_run_nowait(p, verify_seq_lens_cpu, cad, token, fused_first, gate,
                     sampling_metadata, padded, batch_size, pre=None):
    out = _run_nowait(p, verify_seq_lens_cpu, cad, token, fused_first, gate,
                      sampling_metadata, padded, batch_size, pre)
    nowait_stats[0 if out is not None else 1] += 1
    return out


def _run_nowait(p, verify_seq_lens_cpu, cad, token, fused_first, gate,
                sampling_metadata, padded, batch_size, pre):
    """Passes two to depth of an admitted compound loop with pass two's input
    and plan staged on the device (the admitted _stage, already proven for
    every admitted length against the original fused step and native plan)
    instead of by the host from the rejection result. Every length the
    rejection can produce must be admitted; the host never needs to know
    which one it is. Returns the draft tokens of passes two..depth, or None
    (the caller then runs the original path, which waits)."""
    s = getattr(p, "_nemo_compound_loop", None)
    if (s is None or s.graph is None or batch_size != 1 or padded != s.padded
            or not sampling_metadata.all_greedy
            or torch.cuda.is_current_stream_capturing()
            or p.num_speculative_tokens != s.passes + 1
            or p._nemo_full is None or s.wrapper._nemo_native_plan is not s.bank
            or not s.bank.enabled or s.bank.force_native
            or tuple(s.wrapper._plan_info) != s.plan
            or tuple((n, getattr(s.wrapper, n)) for n, _ in s.scalars) != s.scalars):
        return None
    cpu = verify_seq_lens_cpu
    if (cpu is None or cpu.device.type != "cpu" or cpu.numel() != 1
            or cad.seq_lens.dtype != torch.int32 or not cad.seq_lens.is_cuda
            or cad.seq_lens.numel() != 1 or not cad.seq_lens.is_contiguous()):
        return None
    upper = int(cpu[0])
    lower = upper - p.num_speculative_tokens
    if lower < 1 or upper + s.passes + 1 >= p.max_model_len:
        return None
    for length in range(lower + 1, upper + 2):
        if (length not in s.starts
                or any(s.bank.entries.get(s.entries[n][0], (None, None))[1]
                       is not s.entries[n][1] for n in range(length, length + s.passes))):
            return None
    target = getattr(p, "emit_target", None)
    if target is not None and int(target) - upper - 2 < p.num_speculative_tokens:
        return None
    table = cad.block_table_tensor
    pages = (upper + s.passes + p.block_size) // p.block_size
    if (not table.is_cuda or table.dtype != torch.int32 or table.ndim != 2
            or table.shape[0] != 1 or table.stride(1) != 1
            or table.device != s.table.device
            or pages > s.max_pages or pages > table.shape[1]
            or pages > s.builder.paged_kv_indices.gpu.numel()):
        return None
    positions, hidden = fused_first[1], fused_first[2]
    if positions.numel() < 1 or hidden.shape[0] < 1:
        return None
    if pre is not None:
        pre()
    if gate is not None:
        p._mtp_step_prefetch(gate)
    p.positions[:1].copy_(positions[:1])
    s.seq.copy_(cad.seq_lens)
    s.table[0, :pages].copy_(table[0, :pages])
    _launch_stage(s, p, token.view(-1)[:1], hidden[:1])
    s.graph.replay()
    cad.seq_lens.copy_(s.seq)
    cad.slot_mapping = p._slot_mapping_buffer[:batch_size]
    cad.max_seq_len = min(upper + s.passes, p.max_model_len)
    return s.tokens
