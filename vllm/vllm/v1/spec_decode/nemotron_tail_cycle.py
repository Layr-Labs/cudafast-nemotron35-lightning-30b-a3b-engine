"""Boot-admitted, same-page spans of two or three native Nemotron MTP steps.

The original first-tail input update and native attention plan run on every
request. Each graph removes its subsequent identical FA2 plans and combines the
unchanged model/screen calls. Two-step spans also cover page-edge prefixes
and suffixes; every group still starts with the original native planner.
Every token, state update and KV write is still computed; only executable graph structure and scratch buffers persist.
"""
from __future__ import annotations

import copy
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

TAIL_SPANS = (3, 2)
logger = logging.getLogger(__name__)
_LAST_PAGE_KERNEL = None


def supported_flashinfer():
    try:
        return version("flashinfer-python") == "0.6.16.post3"
    except PackageNotFoundError:
        return False


def same_page(first_tail_length: int, page_size: int, max_model_len: int, steps: int) -> bool:
    """Length is after the original first update of this span."""
    if (steps not in TAIL_SPANS
            or not all(type(x) is int for x in (first_tail_length, page_size, max_model_len))):
        return False
    last = first_tail_length + steps - 1
    return (page_size > 0 and 0 < first_tail_length <= last <= max_model_len
            and (first_tail_length - 1) // page_size == (last - 1) // page_size)


def tensor_signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _cpu_shadow_advance(common, max_model_len, steps):
    for _ in range(steps - 1):
        common.max_seq_len = min(common.max_seq_len + 1, max_model_len)
        for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
            value = getattr(common, name)
            if value is not None:
                value += 1
                setattr(common, name, value)


def _make_last_page_kernel():
    global _LAST_PAGE_KERNEL, tl
    if _LAST_PAGE_KERNEL is not None:
        return _LAST_PAGE_KERNEL
    from vllm.triton_utils import tl, triton

    @triton.jit
    def update(seq, last_page, PAGE: tl.constexpr):
        length = tl.load(seq)
        tl.store(last_page, ((length - 1) % PAGE) + 1)

    _LAST_PAGE_KERNEL = update
    return _LAST_PAGE_KERNEL


def plan_signature(p, metadata):
    try:
        return _plan_signature(p, metadata)
    except (AttributeError, TypeError, ValueError):
        return None


def _plan_signature(p, metadata):
    """Guard all graph-bound identities AND the actual scalar plan vector."""
    from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata
    if len(p.draft_attn_groups) != 1 or not metadata:
        return None
    builder = p.draft_attn_groups[0].get_metadata_builder()
    result = []
    for name, md in sorted(metadata.items()):
        if (not isinstance(md, FlashInferMetadata) or not isinstance(md.decode, FIDecode)
                or md.num_actual_tokens != 1 or md.num_decodes != 1
                or md.num_decode_tokens != 1 or md.num_prefills or md.use_cascade
                or md.causal is not True):
            return None
        wrapper = md.decode.wrapper
        if (wrapper._backend != "fa2" or not wrapper.use_tensor_cores
                or not wrapper.is_cuda_graph_enabled
                or getattr(wrapper, "_q_len_per_req", 1) != 1
                or wrapper._fixed_batch_size != 1
                or any(getattr(wrapper, n, None) is not None for n in (
                    "_jit_module", "_custom_mask_buf", "_mask_indptr_buf",
                    "_prefix_len_ptr", "_token_pos_in_items_ptr", "_max_item_len_ptr"))):
            return None
        cache = p.compilation_config.static_forward_context[name].kv_cache
        tensor_names = ("_float_workspace_buffer", "_int_workspace_buffer",
                        "_paged_kv_indptr_buf", "_paged_kv_indices_buf",
                        "_paged_kv_last_page_len_buf", "_qo_indptr_buf")
        result.append((name, id(wrapper), id(wrapper._cached_module), tuple(wrapper._plan_info),
                       md.q_data_type_decode, tensor_signature(md.slot_mapping),
                       tensor_signature(cache),
                       tuple((n, tensor_signature(getattr(wrapper, n))) for n in tensor_names),
                       wrapper._window_left, wrapper._logits_soft_cap, wrapper._sm_scale,
                       wrapper._pos_encoding_mode, wrapper._rope_scale, wrapper._rope_theta,
                       wrapper._kv_layout,
                       tuple((n, getattr(wrapper, n, None)) for n in (
                           "_use_fp16_qk_reduction", "_token_pos_in_items_len",
                           "_enable_pdl", "_causal"))))
    return (tuple(result), builder.page_size, builder.num_qo_heads, builder.num_kv_heads,
            builder.head_dim, builder.decode_fixed_split_size, builder.disable_split_kv,
            tuple(tensor_signature(getattr(builder, n).gpu) for n in
                  ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len")))


class SamePageTail:
    """Fail-closed experimental graph; successful admission is mandatory."""

    def __init__(self, p, private_common, private_metadata, padded: int, steps: int):
        import torch
        if (type(p.model).__name__ != "NemotronHMTP" or p.method != "mtp"
                or p.num_speculative_tokens != 4 or padded not in (1, 5)
                or p.constant_draft_positions or p.parallel_drafting
                or p.model_returns_tuple() or not p.pass_hidden_states_to_model
                or p.supports_mm_inputs or p.use_heterogeneous_vocab
                or p.eplb_state is not None or p.uses_mrope or p.uses_xdrope_dim
                or p.vllm_config.lora_config is not None
                or p.vllm_config.parallel_config.tensor_parallel_size != 1
                or p.vllm_config.parallel_config.data_parallel_size != 1
                or p.vllm_config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("unsupported proposer")
        if steps not in TAIL_SPANS:
            raise ValueError("unsupported tail span")
        self.p, self.padded, self.steps = p, padded, steps
        self.screen = getattr(p, "_draft_head_screen", None)
        if self.screen is None or self.screen.graph is None:
            raise ValueError("the existing admitted screen graph is required")
        self.signature = plan_signature(p, private_metadata)
        if self.signature is None:
            raise ValueError("unsupported FA2 metadata")
        builder = p.draft_attn_groups[0].get_metadata_builder()
        if builder.use_dcp or builder.use_trtllm_decode_attention:
            raise ValueError("unsupported attention mode")
        if builder.page_size != p.block_size:
            raise ValueError("page and slot sizes differ")
        self.last_page = builder.paged_kv_last_page_len.gpu[:1]
        self.seq = torch.empty_like(private_common.seq_lens[:1])
        self.block_table = torch.empty_like(private_common.block_table_tensor[:1])
        self.tokens = torch.empty((1, steps), dtype=torch.int32, device=p.device)
        self.token_slots = tuple(self.tokens[0, i:i + 1] for i in range(steps))
        self.metadata = {name: copy.copy(md) for name, md in private_metadata.items()}
        for md in self.metadata.values():
            md.plan_ready_event = None
        self.kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                           inputs_embeds=None, hidden_states=p.hidden_states[:padded])
        self.input_signature = tuple((k, tensor_signature(v) if v is not None else None)
                                     for k, v in sorted(self.kwargs.items()))
        self.update_last_page = _make_last_page_kernel()
        self.graph = None
        self.pool = None
        self.final_hidden = None
        self.hidden_steps = []
        self.admitted = False
        self.admission_cases = []

    def _stage(self, common):
        self.seq.copy_(common.seq_lens[:1])
        self.block_table.copy_(common.block_table_tensor[:1])

    def _body(self):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata
        p = self.p
        self.hidden_steps = []
        for i in range(self.steps):
            with set_forward_context(self.metadata, p.vllm_config, num_tokens=self.padded,
                                     cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                     slot_mapping=p._get_slot_mapping(self.padded)):
                hidden = p.model(**self.kwargs)
            self.hidden_steps.append(hidden)
            self.screen._h.copy_(hidden[0])
            self.screen._greedy_row(self.screen._h, gated=True, out=self.token_slots[i])
            if i + 1 < self.steps:
                eagle_step_update_slot_mapping_and_metadata(
                    positions_1d=p.positions[:1], block_table_tensor=self.block_table,
                    seq_lens=self.seq, block_size=p.block_size, max_model_len=p.max_model_len,
                    out_clamped_positions=p.positions[:1],
                    out_slot_mapping=p._slot_mapping_buffer[:self.padded],
                    input_batch_size=self.padded)
                self.update_last_page[(1,)](self.seq, self.last_page, PAGE=p.block_size)
                p.input_ids[:1].copy_(self.token_slots[i])
                p.hidden_states[:1].copy_(hidden[:1])
        self.final_hidden = hidden

    def capture_private(self, install_private_fixture: Callable):
        """Installer returns (common, metadata), after restoring ORIGINAL plan.

        This callback also restores private input/KV/gate buffers. Required only
        at startup; caller must ensure there is no active user/model request.
        """
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot nest graph capture")
        self.admitted = False
        self.graph = None
        try:
            for _ in range(2):
                common, metadata = install_private_fixture()
                assert plan_signature(self.p, metadata) == self.signature
                self._stage(common)
                torch.cuda.synchronize(self.p.device)
                self._body()
            common, metadata = install_private_fixture()
            self._stage(common)
            torch.cuda.synchronize(self.p.device)
            graph = torch.cuda.CUDAGraph()
            self.pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self.pool):
                self._body()
            torch.cuda.synchronize(self.p.device)
            if len({tensor_signature(h) for h in self.hidden_steps}) != self.steps:
                raise RuntimeError("per-step outputs alias; comparison needs distinct storage")
            self.graph = graph
        finally:
            install_private_fixture()

    def admit_private_case(self, name, install_private_fixture, original_tail, observe):
        """The original_tail callback executes the original per-step code.

        observe(tokens, hidden_steps) MUST return cloned byte tensors covering:
        tokens, every hidden step, gate, touched KV pages and untouched sentinels,
        positions, seq_lens, slot_mapping, p.input_ids, p.hidden_states, last_page.
        It must NOT observe only token equality or the same aliased output twice.
        Call this on zero/nonzero inputs, gate active/stopped, and page edges.
        """
        import torch
        self.admitted = False
        ok = False
        try:
            common, metadata = install_private_fixture()
            assert plan_signature(self.p, metadata) == self.signature
            ref_tokens, ref_hidden_steps = original_tail()
            reference = observe(ref_tokens, ref_hidden_steps)
            common, metadata = install_private_fixture()
            assert plan_signature(self.p, metadata) == self.signature
            self._stage(common)
            torch.cuda.synchronize(self.p.device)
            self.graph.replay()
            common.seq_lens[:1].copy_(self.seq)
            _cpu_shadow_advance(common, self.p.max_model_len, self.steps)
            actual = observe(self.tokens, self.hidden_steps)
            torch.cuda.synchronize(self.p.device)
            required = {"tokens", "hidden_steps", "gate", "kv_pages", "kv_sentinels",
                        "positions", "seq_lens", "slot_mapping", "input_ids", "hidden_states",
                        "last_page", "cpu_shadows", "max_seq_len", "plan_workspace"}
            ok = (set(reference) == set(actual) == required
                  and all(torch.equal(reference[k], actual[k]) for k in required))
            return ok
        finally:
            self.admission_cases.append((name, ok))
            install_private_fixture()

    def seal_admission(self, required_case_names):
        done = dict(self.admission_cases)
        self.admitted = bool(required_case_names) and all(done.get(n) is True
                                                        for n in required_case_names)
        return self.admitted

    def maybe_run(self, common, metadata, first_tail_length: int, first_tail_position: int,
                  sampling_metadata, current_signature):
        """Return stable draft slots or None; no capture/retry on requests.

        CPU length/position arguments must be freshly proven values from this
        cycle (post-rejection), never max_seq_len/upper-bound substitutes. The
        caller consumes output with its existing final stack before next reuse.
        """
        import torch
        p = self.p
        if (not self.admitted or self.graph is None or not sampling_metadata.all_greedy
                or p.num_speculative_tokens != 4 or common.num_reqs != 1
                or common.num_actual_tokens != 1
                or not same_page(first_tail_length, p.block_size, p.max_model_len, self.steps)
                or type(first_tail_position) is not int or first_tail_position < 0
                or first_tail_position + self.steps - 1 >= p.max_model_len
                or common.block_table_tensor[:1].shape != self.block_table.shape
                or common.seq_lens.dtype != self.seq.dtype
                or common.block_table_tensor.dtype != self.block_table.dtype
                or current_signature != self.signature):
            return None
        current = dict(input_ids=p.input_ids[:self.padded], positions=p._get_positions(self.padded),
                       inputs_embeds=None, hidden_states=p.hidden_states[:self.padded])
        if tuple((k, tensor_signature(v) if v is not None else None)
                 for k, v in sorted(current.items())) != self.input_signature:
            return None
        self._stage(common)
        for md in {id(v): v for v in metadata.values()}.values():
            ready = getattr(md, "plan_ready_event", None)
            if ready is not None:
                torch.cuda.current_stream().wait_event(ready)
        self.graph.replay()
        common.seq_lens[:1].copy_(self.seq)
        _cpu_shadow_advance(common, p.max_model_len, self.steps)
        return self.tokens


def exact_cpu_length(common):
    """Only the actual shadow refreshed by the original first-tail planner.

    Accessing the private field never triggers a new GPU readback. The caller
    runs after the original rejection-count wait, position update and plan.
    An optimistic upper bound alone is insufficient for the same-page guard.
    """
    import torch
    value = getattr(common, "_seq_lens_cpu", None)
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or tuple(value.shape) != (1,) or value.dtype != torch.int32):
        return None
    for name in ("_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound"):
        other = getattr(common, name, None)
        if isinstance(other, torch.Tensor) and other.data_ptr() == value.data_ptr():
            return None
    return int(value[0])


def _save(values):
    seen, result = set(), []
    for value in values:
        key = tensor_signature(value)
        if key not in seen:
            seen.add(key)
            result.append((value, value.clone()))
    return result


def _restore(saved):
    for value, original in saved:
        value.copy_(original)


def _bytes(value):
    import torch
    return value.contiguous().view(torch.uint8).flatten().clone()


def _eligible(p):
    parallel = p.vllm_config.parallel_config
    return (
        os.environ.get("NEMO_MTP_TAIL_CYCLE", "1") != "0"
        and type(p.model).__name__ == "NemotronHMTP" and p.method == "mtp"
        and p.speculative_config.num_speculative_tokens == 4
        and not p.speculative_config.enforce_eager
        and not p.parallel_drafting and not p.constant_draft_positions
        and not p.needs_extra_input_slots
        and p.pass_hidden_states_to_model and not p.supports_mm_inputs
        and not p.use_heterogeneous_vocab and not p.uses_mrope
        and not p.uses_xdrope_dim and p.eplb_state is None
        and not p._share_mtp_indices and p.vllm_config.lora_config is None
        and parallel.tensor_parallel_size == parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and len(p.draft_attn_groups) == len(p._draft_attn_layer_names) == 1
    )


def prepare(p):
    """Prepare and compare private fixtures at boot, never during a request."""
    import torch
    if getattr(p, "_nemo_tail_attempted", False):
        return
    if not _eligible(p) or not supported_flashinfer() or torch.cuda.is_current_stream_capturing():
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    screen = getattr(p, "_draft_head_screen", None)
    if (type(builder).__name__ != "FlashInferMetadataBuilder"
            or not builder.enable_cuda_graph or builder.use_dcp
            or builder.use_trtllm_decode_attention or builder.is_kvcache_nvfp4
            or builder.page_size != p.block_size or p.block_size < 4
            or screen is None or screen.graph is None):
        return
    original_mode, padded, _ = p._determine_batch_execution_and_padding(1)
    if padded not in (1, 5):
        return
    caches = [p.compilation_config.static_forward_context[name].kv_cache
              for name in sorted(p._draft_attn_layer_names)]
    if any(not isinstance(c, torch.Tensor) or not c.is_cuda
           or c.dim() != 4 or c.shape[0] < 4 for c in caches):
        return
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.spec_decode.nemotron_full_step import maybe_run as existing_model_graph

    gate = kernels.depth_gate_init(p.device)
    if gate is None:
        return
    p._nemo_tail_attempted = True
    p._nemo_tail = None
    values = [p.input_ids[:padded], p.positions[:padded], p.hidden_states[:padded],
              p._slot_mapping_buffer[:padded], gate]
    values.extend(c[:4] for c in caches)
    values.extend(b[:max(5, padded)] for b in p.model._moe_row_src_buffers())
    for name in ("_h", "_out", "scores", "refine_scores", "cand_logits"):
        value = getattr(screen, name, None)
        if isinstance(value, torch.Tensor):
            values.append(value)
    for name in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        values.extend((getattr(builder, name).cpu, getattr(builder, name).gpu))
    values.append(builder.paged_kv_indptr_cpu_buffer)
    wrappers = []
    for name in ("_decode_wrappers_cudagraph", "_spec_prefill_wrappers_cudagraph"):
        wrappers.extend(getattr(builder, name, {}).values())
    for wrapper in wrappers:
        for name in ("_int_workspace_buffer", "_pin_memory_int_workspace_buffer",
                     "_qo_indptr_buf", "_kv_lens_buffer"):
            value = getattr(wrapper, name, None)
            if isinstance(value, torch.Tensor):
                values.append(value)
    old_plans = [(w, list(w._plan_info)) for w in wrappers
                 if getattr(w, "_plan_info", None) is not None]
    torch.cuda.synchronize(p.device)
    saved = _save(values)
    bank, failed, required_by_key, succeeded = {}, set(), {}, False
    try:
        blocks = (p.max_model_len + p.block_size - 1) // p.block_size
        table = (torch.arange(blocks, device=p.device, dtype=torch.int32) % 2).view(1, -1)
        row_zero = torch.zeros(1, dtype=torch.long, device=p.device)
        features = torch.arange(p.hidden_states.shape[1], device=p.device)
        qcpu = torch.tensor([0, 1], dtype=torch.int32)
        span_lengths = [(steps, sorted({1, p.block_size - steps + 1,
                         p.block_size + 1, 2 * p.block_size - steps + 1,
                         p.max_model_len - steps + 1})) for steps in TAIL_SPANS]
        kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                      inputs_embeds=None, hidden_states=p.hidden_states[:padded])
        for steps, lengths in span_lengths:
            for length in lengths:
                if not same_page(length, p.block_size, p.max_model_len, steps):
                    continue
                fixture_state = {"case": 0}

                def install():
                    case = fixture_state["case"]
                    for cache in caches:
                        cache[:2].zero_()
                    for cache, original in saved:
                        if any(cache.data_ptr() == c.data_ptr() for c in caches):
                            cache[2:4].copy_(original[2:4])
                    p.input_ids[:padded].fill_(case + 1)
                    p.positions[:padded].fill_(length - 1)
                    pattern = ((features * 7 + case * 13) % 127 - 63).float() / 128
                    p.hidden_states[:padded].copy_(pattern)
                    if case == 0:
                        p.hidden_states[:padded].zero_()
                    p._slot_mapping_buffer[:padded].fill_(-1)
                    slot = (((length - 1) // p.block_size) % 2) * p.block_size + (length - 1) % p.block_size
                    p._slot_mapping_buffer[:1].fill_(slot)
                    gate.copy_(torch.tensor([1 if case == 2 else 0, 1, 1],
                                            dtype=torch.int32, device=p.device))
                    p.model.set_moe_row_src(padded, row_zero)
                    scpu = torch.tensor([length], dtype=torch.int32)
                    common = CommonAttentionMetadata(
                        query_start_loc=qcpu.to(p.device), query_start_loc_cpu=qcpu,
                        seq_lens=scpu.to(p.device), num_reqs=1, num_actual_tokens=1,
                        max_query_len=1, max_seq_len=length,
                        block_table_tensor=table, slot_mapping=p._slot_mapping_buffer[:1],
                        _seq_lens_cpu=scpu, _num_computed_tokens_cpu=scpu - 1,
                        seq_lens_cpu_upper_bound=scpu.clone(), causal=True)
                    _, metadata = p.build_per_group_and_layer_attn_metadata(
                        common, draft_index=1)
                    torch.cuda.synchronize(p.device)
                    fixture_state["common"], fixture_state["metadata"] = common, metadata
                    return common, metadata

                common, metadata = install()
                signature = plan_signature(p, metadata)
                key = (steps, signature)
                if signature is None or key in failed:
                    continue
                candidate = bank.get(key)
                if candidate is None:
                    candidate = SamePageTail(p, common, metadata, padded, steps)
                    candidate.gate_signature = tensor_signature(gate)
                    candidate.capture_private(install)
                    bank[key] = candidate
                    required_by_key[key] = []

                def original_tail(collect=True):
                    common = fixture_state["common"]
                    md = fixture_state["metadata"]
                    token_rows, hidden_rows = [], []
                    for step in range(steps):
                        if step:
                            p._update_positions_dependent_metadata(
                                p.positions[:1], common, 1, padded, p.block_size)
                            side = p._draft_plan_side_stream()
                            if side is not None:
                                main = torch.cuda.current_stream()
                                side.wait_stream(main)
                                with torch.cuda.stream(side):
                                    _, md = p.build_per_group_and_layer_attn_metadata(
                                        common, draft_index=step + 1)
                                    ready = p._draft_plan_ready[step]
                                    ready.record(side)
                                for item in {id(m): m for m in md.values()}.values():
                                    item.plan_ready_event = ready
                            else:
                                _, md = p.build_per_group_and_layer_attn_metadata(
                                    common, draft_index=step + 1)
                            p.input_ids[:1].copy_(token_rows[-1].int())
                            p.hidden_states[:1].copy_(hidden[:1])
                        with set_forward_context(md, p.vllm_config, num_tokens=padded,
                                                 cudagraph_runtime_mode=original_mode,
                                                 slot_mapping=p._get_slot_mapping(padded)):
                            hidden = existing_model_graph(p, kwargs, md, padded, 1)
                            if hidden is None:
                                hidden = p.model(**kwargs)
                        if collect:
                            hidden_rows.append(hidden[:1].clone())
                        token_rows.append(screen.greedy(hidden[:1]))
                    if collect:
                        return torch.stack(token_rows, dim=1), hidden_rows
                    return token_rows, hidden

                def observe(tokens, hidden_rows):
                    common = fixture_state["common"]
                    shadows = [getattr(common, n) for n in (
                        "_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound")]
                    return {
                        "tokens": _bytes(tokens),
                        "hidden_steps": torch.cat([_bytes(h[:1]) for h in hidden_rows]),
                        "gate": _bytes(gate),
                        "kv_pages": torch.cat([_bytes(c[:2]) for c in caches]),
                        "kv_sentinels": torch.cat([_bytes(c[2:4]) for c in caches]),
                        "positions": _bytes(p.positions[:padded]),
                        "seq_lens": _bytes(common.seq_lens),
                        "slot_mapping": _bytes(p._slot_mapping_buffer[:padded]),
                        "input_ids": _bytes(p.input_ids[:padded]),
                        "hidden_states": _bytes(p.hidden_states[:padded]),
                        "last_page": _bytes(builder.paged_kv_last_page_len.gpu[:1]),
                        "cpu_shadows": torch.cat([_bytes(v) for v in shadows]),
                        "max_seq_len": torch.tensor([common.max_seq_len]),
                        "plan_workspace": torch.cat([
                            _bytes(md.decode.wrapper._int_workspace_buffer)
                            for md in {id(m): m for m in fixture_state["metadata"].values()}.values()
                        ]),
                    }

                cases = ("zero-active", "nonuniform-active", "stopped", "repeated-active")
                required = tuple(f"{length}:{name}" for name in cases)
                required_by_key[key].extend(required)
                for case, name in enumerate(required):
                    fixture_state["case"] = case
                    if not candidate.admit_private_case(name, install, original_tail, observe):
                        failed.add(key)
                        break
                candidate.owners = (table, row_zero, candidate.metadata)
                if key not in failed and candidate.seal_admission(required_by_key[key]):
                    if not _qualify_speed(p, key, candidate, install, original_tail,
                                          fixture_state):
                        failed.add(key)
        bank = {key: candidate for key, candidate in bank.items()
                if key not in failed and candidate.seal_admission(required_by_key[key])}
        succeeded = True
    except Exception as exc:
        logger.warning("Native MTP tail-cycle graph unavailable: %r", exc)
    finally:
        torch.cuda.synchronize(p.device)
        _restore(saved)
        for wrapper, plan in old_plans:
            wrapper._plan_info = plan
        torch.cuda.synchronize(p.device)
    if succeeded and bank:
        p._nemo_tail = bank
        logger.info("Native MTP two/three-step tail graphs admitted: %d", len(bank))
    elif succeeded:
        logger.info("Native MTP tail-span admission kept the original path")


def _qualify_speed(p, key, candidate, install, original_tail, fixture_state):
    """Remote startup only: admit the complete guarded dispatch if it is faster.

    Twelve alternating paired trials use fresh private inputs and actual model,
    native plans, head screening and gate state. Exclude fixture installation
    equally; time through device completion, so CPU launch/plan gaps count.
    Original timing has neither diagnostic clones nor per-step device fences.
    This is an admission filter, not a claim about the official whole score.
    """
    import statistics
    import time
    from types import SimpleNamespace
    import torch
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels

    previous = getattr(p, "_nemo_tail", None)
    p._nemo_tail = {key: candidate}
    records = []
    sampling = SimpleNamespace(all_greedy=True)
    try:
        for trial in range(12):
            fixture_state["case"] = trial % 4
            pair = {}
            order = ("original", "span") if trial % 2 == 0 else ("span", "original")
            for path in order:
                common, metadata = install()
                torch.cuda.synchronize(p.device)
                start = time.perf_counter()
                if path == "original":
                    original_tail(collect=False)
                else:
                    result = maybe_run(p, common, metadata, candidate.padded, 1,
                                       sampling, kernels.depth_gate_init(p.device),
                                       candidate.steps)
                    if result is None:
                        return False
                torch.cuda.synchronize(p.device)
                pair[path] = time.perf_counter() - start
            if min(pair.values()) <= 0:
                return False
            records.append((trial % 4, pair["span"] / pair["original"]))
        ratios = [ratio for _, ratio in records]
        active = [ratio for case, ratio in records if case != 2]
        stopped = [ratio for case, ratio in records if case == 2]
        candidate.private_speed_ratios = tuple(records)
        return (statistics.median(ratios) < 0.98
                and statistics.median(active) < 0.99
                and statistics.median(stopped) < 0.99)
    finally:
        p._nemo_tail = previous
        install()


def maybe_run(p, common, metadata, padded, batch_size, sampling_metadata, depth_gate,
              remaining):
    """Prefer the largest admitted span fitting this page and remaining loop.

    Perform the complete dynamic plan signature exactly once per attempted
    span. Unknown plans and page crossings take the original per-step path.
    No length, token or request output is cached in the bank.
    """
    import torch
    bank = getattr(p, "_nemo_tail", None)
    if (not bank or batch_size != 1 or depth_gate is None or remaining < 2
            or not sampling_metadata.all_greedy or torch.cuda.is_current_stream_capturing()):
        return None
    length = exact_cpu_length(common)
    if length is None:
        return None
    spans = [steps for steps in TAIL_SPANS if steps <= remaining
             and same_page(length, p.block_size, p.max_model_len, steps)]
    if not spans:
        return None
    current_plan = plan_signature(p, metadata)
    if current_plan is None:
        return None
    for steps in spans:
        candidate = bank.get((steps, current_plan))
        if (candidate is None or candidate.padded != padded
                or tensor_signature(depth_gate) != candidate.gate_signature):
            continue
        result = candidate.maybe_run(common, metadata, length, length - 1,
                                     sampling_metadata, current_plan)
        if result is not None:
            return [result[:, i] for i in range(steps)], candidate.final_hidden
    return None
