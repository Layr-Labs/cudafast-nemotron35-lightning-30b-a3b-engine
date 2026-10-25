# SPDX-License-Identifier: Apache-2.0
"""Boot-admitted whole graph for the five-row native MTP first pass.

Every request still runs the original input preparation and attention planner.
Only an exact match to a startup-captured native plan replays. The one-row tail
graph is independent. No prompt, model output or KV state is cached here.
"""
import dataclasses
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _tensor(value):
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device)


def _inputs(kwargs):
    return tuple((name, _tensor(value) if isinstance(value, torch.Tensor) else value)
                 for name, value in sorted(kwargs.items()))


def _eligible(p):
    parallel = p.vllm_config.parallel_config
    return (
        os.environ.get("NEMO_MTP_FIRST_PASS", "1") != "0"
        and type(p.model).__name__ == "NemotronHMTP" and p.method == "mtp"
        and p.speculative_config.num_speculative_tokens == 4
        and not p.speculative_config.enforce_eager and not p.parallel_drafting
        and p.pass_hidden_states_to_model and not p.supports_mm_inputs
        and not p.use_heterogeneous_vocab and not p.uses_mrope
        and not p.uses_xdrope_dim and p.eplb_state is None
        and not p._share_mtp_indices and p.vllm_config.lora_config is None
        and parallel.tensor_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and len(p.draft_attn_groups) == 1 and len(p._draft_attn_layer_names) == 1
    )


def _normal_metadata(p, metadata):
    # KV writes consume forward_context.slot_mapping. Bind the metadata field
    # to that same stable buffer too, rather than retaining the runner's fresh
    # first-pass slot-mapping view in a captured graph.
    slots = p._get_slot_mapping(5)
    return {name: dataclasses.replace(value, slot_mapping=slots[name])
            for name, value in metadata.items()}, slots


def _signature(p, metadata):
    from vllm.v1.attention.backends.flashinfer import FIPrefill, FlashInferMetadata

    if set(metadata) != set(p._draft_attn_layer_names):
        return None
    records = []
    for name, md in sorted(metadata.items()):
        if (type(md) is not FlashInferMetadata or type(md.prefill) is not FIPrefill
                or md.decode is not None or md.num_actual_tokens != 5
                or md.num_prefills != 1 or md.num_prefill_tokens != 5
                or md.num_decodes != 0 or md.num_decode_tokens != 0
                or md.use_cascade or type(md.causal) is not bool or not md.causal):
            return None
        wrapper = md.prefill.wrapper
        if (not getattr(wrapper, "_vllm_spec_cg", False)
                or not wrapper.is_cuda_graph_enabled
                or getattr(wrapper, "_backend", None) != "fa2"
                or any(getattr(wrapper, attr, None) is not None for attr in (
                    "_jit_module", "_custom_mask_buf", "_mask_indptr_buf",
                    "_prefix_len_ptr", "_token_pos_in_items_ptr", "_max_item_len_ptr"))):
            return None
        plan = getattr(wrapper, "_plan_info", None)
        if not isinstance(plan, (list, tuple)) or not all(type(v) is int for v in plan):
            return None
        tensors = []
        for attr in ("_float_workspace_buffer", "_int_workspace_buffer",
                     "_qo_indptr_buf", "_paged_kv_indptr_buf",
                     "_paged_kv_indices_buf", "_paged_kv_last_page_len_buf",
                     "_kv_lens_buffer"):
            value = getattr(wrapper, attr, None)
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                return None
            tensors.append((attr, _tensor(value)))
        cache = p.compilation_config.static_forward_context[name].kv_cache
        if (not isinstance(cache, torch.Tensor) or not cache.is_cuda
                or cache.dim() != 4 or cache.shape[0] < 2):
            return None
        # plan_info includes the scalar launch arguments; workspace addresses
        # alone would not establish that a different plan can replay safely.
        constants = tuple((attr, getattr(wrapper, attr, None)) for attr in (
            "_causal", "_window_left", "_logits_soft_cap", "_sm_scale",
            "_rope_scale", "_rope_theta", "_pos_encoding_mode", "_fixed_batch_size",
            "_kv_layout", "_use_fp16_qk_reduction", "_token_pos_in_items_len",
            "_enable_pdl", "_max_total_num_rows", "_qo_indptr_last"))
        records.append((name, id(wrapper), id(wrapper._cached_module), tuple(plan),
                        tuple(tensors), constants, md.q_data_type_prefill,
                        _tensor(md.slot_mapping), _tensor(cache)))
    rows = tuple(_tensor(b) for b in p.model._moe_row_src_buffers())
    return tuple(records), rows


def _byte_copy(value):
    if not isinstance(value, torch.Tensor) or value.shape[0] != 5:
        raise ValueError("native MTP first pass must return all five hidden rows")
    return value.contiguous().view(torch.uint8).clone()


def _save(values):
    """Preserve aliased buffers once; restoring each original view is exact."""
    seen, saved = set(), []
    for value in values:
        signature = _tensor(value)
        if signature not in seen:
            seen.add(signature)
            saved.append((value, value.clone()))
    return saved


def _restore(saved):
    for value, snapshot in saved:
        value.copy_(snapshot)


def prepare(p):
    """Called only from the existing startup graph-capture path."""
    if getattr(p, "_nemo_first_attempted", False):
        return
    if not _eligible(p) or torch.cuda.is_current_stream_capturing():
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    if (type(builder).__name__ != "FlashInferMetadataBuilder"
            or not builder._spec_prefill_cudagraph or builder._spec_q_len != 5
            or builder.use_dcp or builder.use_trtllm_decode_attention
            or builder.is_kvcache_nvfp4
            or p.block_size < 5 or p.max_model_len < 32):
        return
    original_mode, padded, _ = p._determine_batch_execution_and_padding(5)
    if padded != 5:
        return
    p._nemo_first_attempted = True
    p._nemo_first = None

    from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    from vllm.v1.attention.backend import CommonAttentionMetadata

    caches = [p.compilation_config.static_forward_context[name].kv_cache
              for name in sorted(p._draft_attn_layer_names)]
    if any(not isinstance(c, torch.Tensor) or not c.is_cuda
           or c.dim() != 4 or c.shape[0] < 2 for c in caches):
        return
    gate = kernels.depth_gate_init(p.device)
    values = [p.input_ids[:5], p.positions[:5], p.hidden_states[:5],
              p._slot_mapping_buffer[:5]]
    values.extend(b[:5] for b in p.model._moe_row_src_buffers())
    values.extend(c[:2] for c in caches)
    if gate is not None:
        values.append(gate)
    for attr in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        buf = getattr(builder, attr)
        values.extend((buf.cpu, buf.gpu))
    values.append(builder.paged_kv_indptr_cpu_buffer)
    # Snapshot existing wrapper staging too. New wrappers created below have
    # no previous contents to preserve. The direct startup-only builder call
    # never switches DraftStagingSets or replaces its installed CPU buffers.
    wrappers = []
    for attr in ("_decode_wrappers_cudagraph", "_spec_prefill_wrappers_cudagraph"):
        wrappers.extend(getattr(builder, attr, {}).values())
    for wrapper in wrappers:
        workspace = getattr(wrapper, "_pin_memory_int_workspace_buffer", None)
        if isinstance(workspace, torch.Tensor):
            values.append(workspace)
    # Finish prior startup copies before snapshotting reusable pinned buffers.
    torch.cuda.synchronize(p.device)
    saved = _save(values)
    bank = []
    succeeded = False
    try:
        if gate is not None:
            gate.zero_()
        kwargs = dict(input_ids=p.input_ids[:5], positions=p._get_positions(5),
                      inputs_embeds=None, hidden_states=p.hidden_states[:5])
        input_signature = _inputs(kwargs)
        blocks = (p.max_model_len + p.block_size - 1) // p.block_size
        # Synthetic logical pages alternate between two private physical pages.
        # All KV reads/writes are confined to the two pages backed up above.
        table = (torch.arange(blocks, device=p.device, dtype=torch.int32) % 2).view(1, -1)
        qcpu = torch.tensor([0, 5], dtype=torch.int32)
        row_index = torch.zeros(1, dtype=torch.long, device=p.device)
        feature = torch.arange(p.hidden_states.shape[1], device=p.device)
        row_offsets = 17 * torch.arange(5, device=p.device)[:, None]
        lengths = sorted({5, min(p.block_size + 3, p.max_model_len),
                          min(512, p.max_model_len), p.max_model_len})
        for length in lengths:
            scpu = torch.tensor([length], dtype=torch.int32)
            private = CommonAttentionMetadata(
                query_start_loc=qcpu.to(p.device), query_start_loc_cpu=qcpu,
                seq_lens=scpu.to(p.device), num_reqs=1, num_actual_tokens=5,
                max_query_len=5, max_seq_len=p.max_model_len,
                block_table_tensor=table, slot_mapping=p._slot_mapping_buffer[:5],
                _seq_lens_cpu=scpu, _num_computed_tokens_cpu=scpu - 5,
                seq_lens_cpu_upper_bound=scpu.clone(), causal=True)
            md = builder.build_for_drafting(private, draft_index=0)
            raw_metadata = {name: md for name in p._draft_attn_layer_names}
            # This boot-only call deliberately bypasses the request staging
            # switch. Complete H2D reads before another private plan can write
            # the same pinned buffers, including a duplicate-key early exit.
            torch.cuda.synchronize(p.device)
            metadata, slots = _normal_metadata(p, raw_metadata)
            signature = _signature(p, metadata)
            if signature is None:
                continue
            if any(entry["signature"] == signature for entry in bank):
                continue

            def set_case(case):
                p.input_ids[:5].copy_(torch.arange(5, device=p.device) + case + 1)
                p.positions[:5].copy_(torch.arange(length - 5, length, device=p.device))
                p.hidden_states[:5].copy_(
                    ((7 * feature[None, :] + row_offsets + 13 * case) % 127 - 63)
                    .to(torch.float32) / 128)
                pos = p.positions[:5]
                physical = ((pos // p.block_size) % 2) * p.block_size + pos % p.block_size
                p._slot_mapping_buffer[:5].copy_(physical)
                row_index.fill_(case)
                p.model.set_moe_row_src(5, row_index)

            def reset_private_cache():
                for cache in caches:
                    cache[:2].zero_()

            def run_model(mode, callback):
                with set_forward_context(metadata, p.vllm_config, num_tokens=5,
                                         cudagraph_runtime_mode=mode, slot_mapping=slots):
                    return callback(**kwargs)

            set_case(0)
            for _ in range(2):
                reset_private_cache()
                run_model(CUDAGraphMode.FULL, p.model)
            torch.cuda.synchronize(p.device)
            graph = CUDAGraphWrapper(p.model, p.vllm_config, CUDAGraphMode.FULL,
                                    CUDAGraphOptions(weak_ref_output=False))
            graph.graph_pool = torch.cuda.graph_pool_handle()
            reset_private_cache()
            outputs = run_model(CUDAGraphMode.FULL, graph)
            torch.cuda.synchronize(p.device)
            if BatchDescriptor(num_tokens=5) not in graph.concrete_cudagraph_entries:
                continue
            valid = True
            # Change the selected row through all five positions and overwrite
            # every input. Compare all hidden rows and every touched KV byte.
            for case in range(5):
                set_case(case)
                reset_private_cache()
                actual = _byte_copy(run_model(CUDAGraphMode.FULL, graph))
                actual_kv = [c[:2].contiguous().view(torch.uint8).clone() for c in caches]
                reset_private_cache()
                expected = _byte_copy(run_model(original_mode, p.model))
                expected_kv = [c[:2].contiguous().view(torch.uint8).clone() for c in caches]
                if (not torch.equal(actual, expected)
                        or not all(torch.equal(a, b) for a, b in zip(actual_kv, expected_kv))):
                    valid = False
                    break
            if valid:
                bank.append(dict(graph=graph, signature=signature,
                                 inputs=input_signature,
                                 owners=(outputs, private, metadata, slots, row_index, table)))
        succeeded = True
    except Exception as exc:
        logger.warning("Native MTP first-pass graph unavailable: %r", exc)
    finally:
        # No synthetic KV, input, gate, row selection or pinned staging values
        # survive this startup probe. Errors in restoration must propagate.
        torch.cuda.synchronize(p.device)
        _restore(saved)
        torch.cuda.synchronize(p.device)
    if succeeded and bank:
        p._nemo_first = bank
        logger.info("Native MTP five-row first-pass graphs admitted: %d", len(bank))


def maybe_run(p, kwargs, metadata, padded, batch_size):
    bank = getattr(p, "_nemo_first", None)
    if not bank or padded != 5 or batch_size != 1:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        normalized, slots = _normal_metadata(p, metadata)
        signature = _signature(p, normalized)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    inputs = _inputs(kwargs)
    for entry in bank:
        if signature == entry["signature"] and inputs == entry["inputs"]:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import BatchDescriptor, set_forward_context
            graph = entry["graph"]
            if BatchDescriptor(num_tokens=5) not in graph.concrete_cudagraph_entries:
                return None
            with set_forward_context(normalized, p.vllm_config, num_tokens=5,
                                     cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                     slot_mapping=slots):
                return graph(**kwargs)
    return None
