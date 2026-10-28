# SPDX-License-Identifier: Apache-2.0
"""Remote-boot-qualified native FA2 decode planning state for native MTP.

Cache only the original planner's executable metadata and integer workspace.
Current page indices and last-page lengths are still rebuilt by the original
builder on every step. No model output, token or KV content enters this bank.
"""
from __future__ import annotations

import logging
import os
import statistics
import time
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)
_SCALARS = ("_q_len_per_req", "_pos_encoding_mode", "_window_left", "_logits_soft_cap",
            "_sm_scale", "_rope_scale", "_rope_theta")


def _tensor(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _bytes(t):
    import torch
    return t.contiguous().view(torch.uint8).flatten().clone()


def _extent(plan, capacity):
    """Used integer prefix of the native FA2 PrefillPlanInfo 15-word schema.

    Read original native offsets, never synthesize a schedule. Each prefix
    and vector comes from the native planner for its exact CPU structural inputs.
    """
    if len(plan) != 15 or not all(type(v) is int and v >= 0 for v in plan):
        return None
    padded, rows = plan[:2]
    if not 1 <= padded <= 4096 or rows != 1 or plan[13] != 1 or plan[14] not in (0, 1):
        return None
    ends = [plan[4] + 4 * padded, plan[5] + 4 * padded, plan[6] + 4 * padded,
            plan[8] + 8, plan[9] + 4, plan[2] + 4]
    if plan[14]:
        ends.extend((plan[7] + 4 * (rows + 1), plan[12] + padded))
    size = max(ends)
    return size if 0 < size <= min(capacity, 65536) else None


def _signature(wrapper, kw):
    import torch
    if (wrapper._backend != "fa2" or not wrapper.use_tensor_cores
            or not wrapper.is_cuda_graph_enabled or wrapper._fixed_batch_size != 1
            or getattr(wrapper, "_q_len_per_req", 1) != 1
            or any(getattr(wrapper, n, None) is not None for n in (
                "_jit_module", "_custom_mask_buf", "_mask_indptr_buf",
                "_prefix_len_ptr", "_token_pos_in_items_ptr", "_max_item_len_ptr"))):
        return None
    indptr, last, indices = kw["indptr"], kw["last_page_len"], kw["indices"]
    if (indptr.device.type != "cpu" or last.device.type != "cpu"
            or indptr.dtype != torch.int32 or last.dtype != torch.int32
            or tuple(indptr.shape) != (2,) or tuple(last.shape) != (1,)
            or not indptr.is_contiguous() or not last.is_contiguous()
            or indices.dtype != torch.int32 or not indices.is_cuda
            or indices.ndim != 1 or indices.stride(0) != 1
            or indices.data_ptr() != wrapper._paged_kv_indices_buf.data_ptr()):
        return None
    starts = tuple(indptr.tolist())
    page = kw["page_size"]
    residue = int(last[0])
    if (starts[0] != 0 or starts[1] < 1 or starts[1] > indices.numel()
            or type(page) is not int or page <= 0 or not 1 <= residue <= page):
        return None
    scalar_items = [(k, v) for k, v in sorted(kw.items())
                    if k not in ("indptr", "indices", "last_page_len")]
    if any(type(v) not in (int, float, bool, str, type(None))
           and not isinstance(v, torch.dtype) for _, v in scalar_items):
        return None
    constants = tuple((k, type(v).__name__, str(v)) for k, v in scalar_items)
    names = ("_float_workspace_buffer", "_int_workspace_buffer", "_qo_indptr_buf",
             "_paged_kv_indptr_buf", "_paged_kv_indices_buf", "_paged_kv_last_page_len_buf")
    identity = (id(wrapper), id(wrapper._cached_module),
                tuple((n, _tensor(getattr(wrapper, n))) for n in names),
                getattr(wrapper, "_kv_layout", None),
                getattr(wrapper, "_use_fp16_qk_reduction", None),
                getattr(wrapper, "_enable_pdl", None))
    return starts, residue, constants, identity


class NativePlanBank:
    def __init__(self, wrapper):
        from flashinfer.decode import fast_decode_plan
        self.wrapper = wrapper
        self.native = fast_decode_plan
        self.entries = {}
        self.enabled = False
        self.force_native = False
        self.last_native_key = None
        self.last_restore = False
        self.last_restore_key = None
        # Strong references prevent pointer reuse from admitting a new arena.
        self.owners = tuple(getattr(wrapper, n) for n in (
            "_float_workspace_buffer", "_int_workspace_buffer", "_qo_indptr_buf",
            "_paged_kv_indptr_buf", "_paged_kv_indices_buf", "_paged_kv_last_page_len_buf"))

    def __call__(self, wrapper, **kw):
        import torch
        self.last_restore = False
        self.last_restore_key = None
        signed = None
        try:
            if not torch.cuda.is_current_stream_capturing():
                signed = _signature(wrapper, kw)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        key = signed
        entry = self.entries.get(key) if key is not None else None
        if self.enabled and not self.force_native and entry is not None:
            plan, prefix, scalars = entry
            # Only executable native integer metadata is restored. Current
            # indices and last-page length were refreshed by the builder.
            from vllm.v1.worker import copy_batch
            copy_batch.copy(
                wrapper._int_workspace_buffer.view(torch.uint8)[:prefix.numel()], prefix)
            wrapper._plan_info = list(plan)
            for name, value in scalars:
                setattr(wrapper, name, value)
            self.last_restore = True
            self.last_restore_key = key
            return
        # A cache miss enters an opaque native planner. Publish any recorded
        # page metadata before it can consume device buffers or launch work.
        from vllm.v1.worker import copy_batch
        copy_batch.materialize()
        self.native(wrapper, **kw)
        self.last_native_key = key

    def snapshot(self):
        import torch
        plan = tuple(int(v) for v in self.wrapper._plan_info)
        raw = self.wrapper._int_workspace_buffer.view(torch.uint8)
        size = _extent(plan, raw.numel())
        if size is None or self.last_native_key is None:
            raise ValueError("unsupported native plan schema or identities")
        return (plan, raw[:size].clone(),
                tuple((n, getattr(self.wrapper, n)) for n in _SCALARS
                      if hasattr(self.wrapper, n)))


def _eligible(p):
    parallel = p.vllm_config.parallel_config
    try:
        package_ok = version("flashinfer-python") == "0.6.16.post3"
    except PackageNotFoundError:
        package_ok = False
    return (os.environ.get("NEMO_MTP_NATIVE_PLAN_BANK", "1") != "0" and package_ok
            and type(p.model).__name__ == "NemotronHMTP" and p.method == "mtp"
            and (p.speculative_config.num_speculative_tokens == 4
                 or (p.speculative_config.num_speculative_tokens == 5
                     and getattr(getattr(p, "_draft_head_screen", None),
                                 "_tail_pf_table", None) is not None))
            and not p.speculative_config.enforce_eager
            and not p.parallel_drafting and not p.constant_draft_positions
            and not p.needs_extra_input_slots and p.pass_hidden_states_to_model
            and not p.supports_mm_inputs and not p.use_heterogeneous_vocab
            and not p.uses_mrope and not p.uses_xdrope_dim
            and p.eplb_state is None and not p._share_mtp_indices
            and p.vllm_config.lora_config is None
            and parallel.tensor_parallel_size == parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and len(p.draft_attn_groups) == len(p._draft_attn_layer_names) == 1)


def prepare(p):
    """Build/check native planning state on private fixtures, remote boot only."""
    import torch
    if (getattr(p, "_nemo_plan_attempted", False) or not _eligible(p)
            or torch.cuda.is_current_stream_capturing()):
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    wrapper = getattr(builder, "_decode_wrappers_cudagraph", {}).get(1)
    screen = getattr(p, "_draft_head_screen", None)
    if (type(builder).__name__ != "FlashInferMetadataBuilder"
            or not builder.enable_cuda_graph or builder.use_dcp
            or builder.use_trtllm_decode_attention or builder.is_kvcache_nvfp4
            or builder.page_size != p.block_size or p.block_size < 4
            or wrapper is None or getattr(wrapper, "vllm_first_call", True)
            or screen is None or screen.graph is None):
        return
    mode, padded, _ = p._determine_batch_execution_and_padding(1)
    if padded not in (1, 5):
        return
    caches = [p.compilation_config.static_forward_context[n].kv_cache
              for n in sorted(p._draft_attn_layer_names)]
    if any(not isinstance(t, torch.Tensor) or not t.is_cuda or t.dim() != 4
           or t.shape[0] < 4 for t in caches):
        return
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.spec_decode.nemotron_full_step import maybe_run as model_graph

    gate = kernels.depth_gate_init(p.device)
    if gate is None:
        return
    p._nemo_plan_attempted = True
    values = [p.input_ids[:padded], p.positions[:padded], p.hidden_states[:padded],
              p._slot_mapping_buffer[:padded], gate]
    values.extend(t[:4] for t in caches)
    values.extend(b[:max(5, padded)] for b in p.model._moe_row_src_buffers())
    for name in ("_h", "_out", "scores", "refine_scores", "cand_logits"):
        t = getattr(screen, name, None)
        if isinstance(t, torch.Tensor):
            values.append(t)
    for name in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        values.extend((getattr(builder, name).cpu, getattr(builder, name).gpu))
    values.append(builder.paged_kv_indptr_cpu_buffer)
    for name in ("_float_workspace_buffer", "_int_workspace_buffer",
                 "_pin_memory_int_workspace_buffer", "_qo_indptr_buf", "_kv_lens_buffer"):
        t = getattr(wrapper, name, None)
        if isinstance(t, torch.Tensor):
            values.append(t)
    old_plan = list(wrapper._plan_info)
    old_scalars = [(n, hasattr(wrapper, n), getattr(wrapper, n, None)) for n in _SCALARS]
    old_hook = getattr(wrapper, "_nemo_native_plan", None)
    torch.cuda.synchronize(p.device)
    saved, seen = [], set()
    bank, admitted = None, False
    try:
        for t in values:
            key = _tensor(t)
            if key not in seen:
                saved.append((t, t.clone()))
                seen.add(key)
        bank = NativePlanBank(wrapper)
        wrapper._nemo_native_plan = bank
        blocks = (p.max_model_len + p.block_size - 1) // p.block_size
        table = (torch.arange(blocks, dtype=torch.int32, device=p.device) % 2).view(1, -1)
        qcpu = torch.tensor([0, 1], dtype=torch.int32)
        row_zero = torch.zeros(1, dtype=torch.long, device=p.device)
        features = torch.arange(p.hidden_states.shape[1], device=p.device)

        def common_for(length):
            scpu = torch.tensor([length], dtype=torch.int32)
            return CommonAttentionMetadata(
                query_start_loc=qcpu.to(p.device), query_start_loc_cpu=qcpu,
                seq_lens=scpu.to(p.device), num_reqs=1, num_actual_tokens=1,
                max_query_len=1, max_seq_len=length, block_table_tensor=table,
                slot_mapping=p._slot_mapping_buffer[:1], _seq_lens_cpu=scpu,
                _num_computed_tokens_cpu=scpu - 1, seq_lens_cpu_upper_bound=scpu.clone(),
                causal=True)

        # Bound boot work and memory independently of request tokens. Plans for
        # pages outside this generic prefix use the original planner.
        # Depth five uses this bank only to choose private full-loop test
        # fixtures; production planning is restored after that comparison.
        # Keep its calibration bounded to a few pages rather than building a
        # request-time cache that will never be installed.
        fixture_only = p.speculative_config.num_speculative_tokens == 5
        max_pages = min(blocks, 4 if fixture_only else 512)
        timed_pages = min(blocks, 4 if fixture_only else 128)
        calibrated_pages = 0
        raw_float = wrapper._float_workspace_buffer.view(torch.uint8)
        raw_int = wrapper._int_workspace_buffer.view(torch.uint8)
        # Actual installed planner, not a version-string assumption: no
        # model/attention executes during this exhaustive planning sequence.
        # Verify its float arena is untouched and every integer write lies
        # inside the native 15-word schema's copied prefix.
        raw_float.fill_(0xA5)
        total_plan_bytes = 0
        for pages in range(1, max_pages + 1):
            page_entries, page_bytes = {}, 0
            budget_full = False
            for residue in range(1, p.block_size + 1):
                length = (pages - 1) * p.block_size + residue
                if length > p.max_model_len:
                    break
                common = common_for(length)
                raw_int.fill_(0xB6)
                p.build_per_group_and_layer_attn_metadata(common, draft_index=1)
                current = bank.snapshot()
                torch.cuda.synchronize(p.device)
                if not bool(torch.all(raw_int[current[1].numel():] == 0xB6).item()):
                    raise ValueError("native integer writes exceed captured prefix")
                size = current[1].numel()
                if total_plan_bytes + page_bytes + size > 64 * (1 << 20):
                    budget_full = True
                    break
                page_entries[bank.last_native_key] = current
                page_bytes += size
            if budget_full:
                # Keep complete earlier pages; never disable their valid bank
                # merely because a later optional page would exceed budget.
                break
            if page_entries:
                bank.entries.update(page_entries)
                total_plan_bytes += page_bytes
                calibrated_pages = pages
        if not bank.entries:
            raise ValueError("no native plans")
        if not bool(torch.all(raw_float == 0xA5).item()):
            raise ValueError("native planner mutates uncached float workspace")
        bank.enabled = True
        kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                      inputs_embeds=None, hidden_states=p.hidden_states[:padded])
        timed_pages = min(timed_pages, calibrated_pages)
        sample_pages = sorted({1, min(2, timed_pages), max(1, timed_pages // 2),
                               timed_pages, max(1, calibrated_pages // 2), calibrated_pages})
        residues = sorted({1, max(1, p.block_size // 2), p.block_size})
        ratios = []
        # Warm the original stream/event control before either timed arm.
        p._draft_plan_side_stream()

        def run_case(length, case, native):
            # Both arms get identical private model inputs/KV and integer
            # workspaces. Installation and diagnostic clones are not timed.
            for target, original in saved:
                target.copy_(original)
            for cache in caches:
                cache[:2].zero_()
                cache[1].view(torch.uint8).fill_(0x20 + case)
            table.copy_(((torch.arange(blocks, device=p.device, dtype=torch.int32)
                          + case) % 2).view(1, -1))
            p.input_ids[:padded].fill_(case + 1)
            p.positions[:padded].fill_(length - 1)
            pattern = ((features * 7 + case * 13) % 127 - 63).float() / 128
            p.hidden_states[:padded].copy_(pattern)
            if case == 0:
                p.hidden_states[:padded].zero_()
            p._slot_mapping_buffer[:padded].fill_(-1)
            slot = (((length - 1) // p.block_size) % 2) * p.block_size + (length - 1) % p.block_size
            p._slot_mapping_buffer[:1].fill_(slot)
            gate.copy_(torch.tensor([1 if case == 2 else 0, 1, 1, kernels.GATE_PROD_ONE],
                                    device=p.device, dtype=torch.int32))
            p.model.set_moe_row_src(padded, row_zero)
            common = common_for(length)
            bank.force_native = native
            # The original arm calls the unmodified native planner directly,
            # without timing cache-key construction or a forced cache miss.
            wrapper._nemo_native_plan = None if native else bank
            torch.cuda.synchronize(p.device)
            start = time.perf_counter()
            side = p._draft_plan_side_stream()
            if side is not None:
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    _, metadata = p.build_per_group_and_layer_attn_metadata(common, draft_index=1)
                    ready = p._draft_plan_ready[0]
                    ready.record(side)
                for md in {id(v): v for v in metadata.values()}.values():
                    md.plan_ready_event = ready
            else:
                _, metadata = p.build_per_group_and_layer_attn_metadata(common, draft_index=1)
            if not native and not bank.last_restore:
                raise ValueError("private plan restore declined")
            with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                     cudagraph_runtime_mode=mode,
                                     slot_mapping=p._get_slot_mapping(padded)):
                hidden = model_graph(p, kwargs, metadata, padded, 1)
                if hidden is None:
                    hidden = p.model(**kwargs)
            token = screen.greedy(hidden[:1])
            torch.cuda.synchronize(p.device)
            elapsed = time.perf_counter() - start
            extent = _extent(tuple(int(v) for v in wrapper._plan_info),
                             wrapper._int_workspace_buffer.view(torch.uint8).numel())
            if extent is None:
                raise ValueError("private native plan extent changed")
            observation = {
                "hidden": _bytes(hidden[:1]), "token": _bytes(token), "gate": _bytes(gate),
                "kv": torch.cat([_bytes(t[:2]) for t in caches]),
                "sentinels": torch.cat([_bytes(t[2:4]) for t in caches]),
                "positions": _bytes(p.positions[:padded]), "seq": _bytes(common.seq_lens),
                "slots": _bytes(p._slot_mapping_buffer[:padded]),
                "input": _bytes(p.input_ids[:padded]),
                "input_hidden": _bytes(p.hidden_states[:padded]),
                "last_page": _bytes(builder.paged_kv_last_page_len.gpu[:1]),
                "plan": torch.tensor(list(wrapper._plan_info)),
                "plan_scalars": torch.tensor(list(repr(tuple(
                    (n, hasattr(wrapper, n), getattr(wrapper, n, None))
                    for n in _SCALARS)).encode()), dtype=torch.uint8),
                "cpu_shadows": torch.cat([_bytes(getattr(common, n)) for n in (
                    "_seq_lens_cpu", "_num_computed_tokens_cpu", "seq_lens_cpu_upper_bound")]),
                "max_seq_len": torch.tensor([common.max_seq_len]),
                # Native uninitialized alignment/padded-CTA bytes are opaque
                # scratch, not observations. All integer writes were bounded
                # at calibration; actual model/KV state and live page metadata
                # are compared instead of assuming unused padding is equal.
                "page_indices": _bytes(builder.paged_kv_indices.gpu[
                    :(length + p.block_size - 1) // p.block_size]),
            }
            return observation, elapsed

        trial = 0
        for pages in sample_pages:
            for residue in residues:
                length = (pages - 1) * p.block_size + residue
                if length > p.max_model_len:
                    continue
                for case in range(4):
                    order = (True, False) if trial % 2 == 0 else (False, True)
                    results = {native: run_case(length, case, native) for native in order}
                    original, original_time = results[True]
                    actual, cached_time = results[False]
                    torch.cuda.synchronize(p.device)
                    if (set(actual) != set(original)
                            or not all(torch.equal(actual[k], original[k]) for k in actual)):
                        raise ValueError("native plan restore changes private model state")
                    if min(original_time, cached_time) <= 0:
                        raise ValueError("invalid private timing")
                    # Keep the original latency gate's coverage. Extended
                    # endpoints add full state/bytes checks, not a new timing
                    # entitlement or a weaker comparison threshold.
                    if pages <= timed_pages:
                        ratios.append((case, cached_time / original_time))
                    trial += 1
        if (not ratios or statistics.median([r for _, r in ratios]) >= 0.98
                or statistics.median([r for case, r in ratios if case != 2]) >= 0.99
                or statistics.median([r for case, r in ratios if case == 2]) >= 0.99):
            raise ValueError("native plan restore did not reduce guarded model-step latency")
        bank.private_speed_ratios = tuple(ratios)
        admitted = True
    except Exception as exc:
        logger.warning("Native MTP plan bank unavailable: %r", exc)
    finally:
        torch.cuda.synchronize(p.device)
        for target, original in saved:
            target.copy_(original)
        wrapper._plan_info = old_plan
        for name, present, value in old_scalars:
            if present:
                setattr(wrapper, name, value)
            elif hasattr(wrapper, name):
                delattr(wrapper, name)
        if bank is not None:
            bank.force_native = False
            bank.enabled = admitted
        wrapper._nemo_native_plan = bank if admitted else old_hook
        torch.cuda.synchronize(p.device)
    if admitted:
        logger.info("Native MTP FA2 exact structural plans admitted: %d", len(bank.entries))
