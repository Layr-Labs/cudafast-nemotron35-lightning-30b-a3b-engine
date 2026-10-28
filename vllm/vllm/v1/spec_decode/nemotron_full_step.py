# SPDX-License-Identifier: Apache-2.0
"""Native Nemotron MTP decode-step graph, prepared only during remote startup."""
import os

import torch

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)
# NEMO_MTP_DEEP_LOOP=0 restores the depth-four-only gate: at a declared depth of
# five neither this graph nor the native plan bank nor the compound draft loop
# is prepared, and follow-on draft passes take the per-pass path.
DEEP_LOOP = os.environ.get("NEMO_MTP_DEEP_LOOP", "1") != "0"


def deep_depths():
    """Declared depths the depth-four single-row loop family also serves."""
    return (4, 5) if DEEP_LOOP else (4,)


def deep_rows(p, padded):
    """Whether this padded row count may run the family at a declared depth of five.

    The whitelist is a shape guard, not a numeric one: the layer code, the plan
    restore and the three stage kernels all derive their rows from the call
    shape, so a row count the family never qualified on is the only thing it
    has to refuse. Depth four serves a single row at capture and five rows for
    a uniform decode batch; depth five serves a single row at capture and can
    serve six rows for a uniform decode batch of the full verify width. Both
    are admitted here and nothing else is, so an unexpected shape still
    declines to the per-pass path.
    """
    n = p.speculative_config.num_speculative_tokens
    return DEEP_LOOP and n == 5 and padded in (1, n + 1)


def _tensor(x):
    return (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device)


def _inputs(kwargs):
    return tuple((k, _tensor(v) if isinstance(v, torch.Tensor) else v)
                 for k, v in sorted(kwargs.items()))


def _metadata(p, metadata):
    from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata
    result = []
    for name, m in sorted(metadata.items()):
        if (not isinstance(m, FlashInferMetadata) or not isinstance(m.decode, FIDecode)
                or m.num_actual_tokens != 1 or m.num_decodes != 1 or m.num_prefills
                or m.num_decode_tokens != 1 or m.use_cascade
                or type(m.causal) is not bool or not m.causal):
            return None
        cache = p.compilation_config.static_forward_context[name].kv_cache
        if not isinstance(cache, torch.Tensor) or not cache.is_cuda or not cache.numel():
            return None
        result.append((name, id(m.decode.wrapper), m.q_data_type_decode,
                       _tensor(m.slot_mapping), _tensor(cache)))
    buffers = tuple(_tensor(getattr(g.get_metadata_builder(), name).gpu)
                    for g in p.draft_attn_groups
                    for name in ('paged_kv_indptr', 'paged_kv_indices', 'paged_kv_last_page_len'))
    return tuple(result), buffers


def prepare(p):
    """Build the graph outside requests, preserving the existing padded model shape."""
    if getattr(p, '_nemo_full_attempted', False):
        return
    if (os.environ.get('NEMO_MTP_FULL_STEP', '1') == '0'
            or torch.cuda.is_current_stream_capturing()
            or type(p.model).__name__ != 'NemotronHMTP' or p.method != 'mtp'
            or p.speculative_config.enforce_eager or p.supports_mm_inputs
            or not p.pass_hidden_states_to_model or p.use_heterogeneous_vocab
            or p.eplb_state is not None or p.uses_mrope or p.uses_xdrope_dim
            or p.vllm_config.lora_config is not None
            or p.vllm_config.parallel_config.tensor_parallel_size != 1
            or p.vllm_config.parallel_config.data_parallel_size != 1
            or len(p.draft_attn_groups) != 1):
        return
    builder = p.draft_attn_groups[0].get_metadata_builder()
    if (type(builder).__name__ != 'FlashInferMetadataBuilder'
            or not builder.enable_cuda_graph or builder.use_trtllm_decode_attention
            or builder.use_dcp):
        return
    p._nemo_full_attempted = True
    p._nemo_full = None
    saved_gate = None
    gate = None
    padded = 0
    try:
        from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
        gate = kernels.depth_gate_init(p.device)
        if gate is not None:
            saved_gate = gate.clone()
            gate.zero_()
        _, padded, _ = p._determine_batch_execution_and_padding(1)
        if padded not in (1, 5) and not deep_rows(p, padded):
            return
        if p.block_size <= 0:
            return
        # Startup has no requests. Capture's padding slots suppress KV writes;
        # initialize the cache used by the private one-token attention warmup.
        for name in p._draft_attn_layer_names:
            cache = p.compilation_config.static_forward_context[name].kv_cache
            if not isinstance(cache, torch.Tensor) or not cache.is_cuda or not cache.numel():
                return
            cache.zero_()
        p.input_ids[:padded].zero_()
        p.positions[:padded].zero_()
        p.hidden_states[:padded].zero_()
        p._slot_mapping_buffer[:padded].fill_(-1)
        zero_row = torch.zeros(1, dtype=torch.long, device=p.device)
        p.model.set_moe_row_src(padded, zero_row)
        qcpu = torch.tensor([0, 1], dtype=torch.int32)
        scpu = torch.ones(1, dtype=torch.int32)
        private = CommonAttentionMetadata(
            query_start_loc=qcpu.to(p.device), query_start_loc_cpu=qcpu,
            seq_lens=scpu.to(p.device), num_reqs=1, num_actual_tokens=1,
            max_query_len=1, max_seq_len=p.max_model_len,
            block_table_tensor=torch.zeros(
                (1, (p.max_model_len + p.block_size - 1) // p.block_size),
                dtype=torch.int32, device=p.device),
            slot_mapping=p._slot_mapping_buffer[:1],
            _seq_lens_cpu=scpu, _num_computed_tokens_cpu=torch.zeros_like(scpu),
            seq_lens_cpu_upper_bound=scpu.clone())
        _, metadata = p.build_per_group_and_layer_attn_metadata(private, draft_index=1)
        signature = _metadata(p, metadata)
        if signature is None:
            return
        kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                      inputs_embeds=None, hidden_states=p.hidden_states[:padded])
        slots = p._get_slot_mapping(padded)
        # FULL bypasses inner PIECEWISE wrappers. Warm all actual attention/math
        # launches before capture; no sampler or gate counter is included.
        with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                 cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                 slot_mapping=slots):
            for _ in range(2):
                p.model(**kwargs)
            torch.cuda.synchronize(p.device)
            graph = CUDAGraphWrapper(p.model, p.vllm_config, CUDAGraphMode.FULL,
                                    CUDAGraphOptions(weak_ref_output=False))
            graph.graph_pool = torch.cuda.graph_pool_handle()
            outputs = graph(**kwargs)
        torch.cuda.synchronize(p.device)
        key = BatchDescriptor(num_tokens=padded)
        if key not in graph.concrete_cudagraph_entries:
            return
        def used_rows(value):
            if isinstance(value, torch.Tensor):
                return [value[:1].contiguous().view(torch.uint8).clone()]
            return [t for v in value for t in used_rows(v)]
        # Compare updated private inputs as well as the initial zero-input case.
        original_mode, _, _ = p._determine_batch_execution_and_padding(1)
        for case in range(3 if gate is not None else 2):
            if gate is not None:
                gate.zero_()
                if case == 2:
                    gate[0].fill_(1)
            p.input_ids[:padded].fill_(case % 2)
            p.hidden_states[:padded].fill_(0.125 * (case % 2))
            with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                     cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                     slot_mapping=slots):
                actual = used_rows(graph(**kwargs))
            with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                     cudagraph_runtime_mode=original_mode,
                                     slot_mapping=slots):
                for _ in range(2):
                    expected = p.model(**kwargs)
                reference = used_rows(expected)
            if len(actual) != len(reference) or not all(
                    torch.equal(a, b) for a, b in zip(actual, reference)):
                logger.warning('Native MTP full-step private rows differ; original path retained')
                return
        # Own strong output references and a private pool throughout replay.
        p._nemo_full = (graph, padded, signature, _inputs(kwargs),
                        (outputs, private, metadata, zero_row))
        # A compound graph may reuse this exact captured executable layout,
        # never substitute a different native plan's scalar launch arguments.
        from vllm.v1.spec_decode.nemotron_decode_plan import _SCALARS
        wrapper = builder._decode_wrappers_cudagraph[1]
        p._nemo_full_native_layout = (
            wrapper, tuple(wrapper._plan_info),
            tuple((n, getattr(wrapper, n)) for n in _SCALARS if hasattr(wrapper, n)),
        )
        logger.info('Native MTP full-step graph prepared at padded=%d', padded)
    except Exception as exc:
        logger.warning('Native MTP full-step graph unavailable: %r', exc)
    finally:
        if saved_gate is not None:
            gate.copy_(saved_gate)
        p.model.set_moe_row_src(min(p.max_num_tokens, max(padded, 5)), None)


def maybe_run(p, kwargs, metadata, padded, batch_size):
    state = getattr(p, '_nemo_full', None)
    if state is None or batch_size != 1 or padded != state[1]:
        return None
    graph, _, signature, inputs, _outputs = state
    if (BatchDescriptor(num_tokens=padded) not in graph.concrete_cudagraph_entries
            or _inputs(kwargs) != inputs or _metadata(p, metadata) != signature):
        return None
    with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                             cudagraph_runtime_mode=CUDAGraphMode.FULL,
                             slot_mapping=p._get_slot_mapping(padded)):
        return graph(**kwargs)
