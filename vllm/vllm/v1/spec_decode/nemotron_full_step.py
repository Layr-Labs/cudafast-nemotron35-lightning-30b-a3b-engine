# SPDX-License-Identifier: Apache-2.0
"""Native Nemotron MTP decode-step graph, prepared only during remote startup."""
import os
import subprocess
import sys

import torch

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


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
    try:
        from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
        gate = kernels.depth_gate_init(p.device)
        if gate is not None:
            saved_gate = gate.clone()
            gate.zero_()
        _, padded, _ = p._determine_batch_execution_and_padding(1)
        if padded not in (1, 5) or p.block_size <= 0:
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
        skip = None
        if gate is not None and os.environ.get('NEMO_MTP_DRAFT_SKIP', '1') != '0':
            skip = _skip_graph(p, graph, kwargs, metadata, slots, gate, padded,
                               used_rows, original_mode)
        # Own strong output references and a private pool throughout replay.
        p._nemo_full = (graph, padded, signature, _inputs(kwargs),
                        (outputs, private, metadata, zero_row), skip)
        logger.info('Native MTP full-step graph prepared at padded=%d', padded)
    except Exception as exc:
        logger.warning('Native MTP full-step graph unavailable: %r', exc)
    finally:
        if saved_gate is not None:
            gate.copy_(saved_gate)
        p.model.set_moe_row_src(min(p.max_num_tokens, 5), None)


def _replay_us(fn, reps=8, rounds=7):
    """Median GPU time per replay in microseconds (event-timed back-to-back replays)."""
    fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return sorted(ms)[rounds // 2] * 1000.0 / reps


_IF_PROBE = """
import importlib.util, sys, torch, triton
import triton.language as tl

@triton.jit
def add1(src, dst, N: tl.constexpr, B: tl.constexpr, launch_pdl: tl.constexpr = False):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(dst + i, tl.load(src + i, mask=i < N) + 1.0, mask=i < N)

l2pf = None
if sys.argv[1] != "-":
    spec = importlib.util.spec_from_file_location("l2pf", sys.argv[1])
    l2pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(l2pf)
dev = torch.device("cuda")
pdl = torch.cuda.get_device_capability(dev)[0] >= 9
gate = torch.zeros(4, dtype=torch.int32, device=dev)
pred = torch.zeros((), dtype=torch.bool, device=dev)
a = torch.randn(4096, device=dev)
b = torch.empty_like(a)
c = torch.empty_like(a)
w = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
x = torch.randn(5, 1024, device=dev, dtype=torch.bfloat16)
y = torch.empty(5, 1024, device=dev, dtype=torch.bfloat16)

def body():
    add1[(4,)](a, b, N=4096, B=1024, launch_pdl=pdl)
    pf = l2pf is not None and l2pf.prefetch(w)
    torch.mm(x, w, out=y)
    if pf:
        l2pf.join(dev)
    add1[(4,)](b, c, N=4096, B=1024, launch_pdl=pdl)

body()
torch.cuda.synchronize()
ref_c, ref_y = c.clone(), y.clone()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    torch.eq(gate[0], 0, out=pred)
    g.begin_capture_to_if_node(pred)
    try:
        body()
    finally:
        g.end_capture_to_conditional_node()
c.zero_()
y.zero_()
g.replay()
torch.cuda.synchronize()
assert torch.equal(c, ref_c) and torch.equal(y, ref_y)
gate[0] = 1
c.fill_(3)
y.fill_(3)
g.replay()
torch.cuda.synchronize()
assert bool((c == 3).all()) and bool((y == 3).all())
print("probe-ok")
"""
_if_ok = None


def _if_probe():
    """Build and replay, in a child process, a graph of the same shape: the gate
    compare, then an IF node whose body holds PDL Triton launches, an l2pf
    side-stream fork/join and a cuBLAS GEMM. A refusal or a device fault there
    kills only the child; the parent then keeps the record's graph."""
    global _if_ok
    if _if_ok is None:
        from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
        l2pf = getattr(kernels._l2pf, '__file__', None) or '-'
        try:
            out = subprocess.run([sys.executable, '-c', _IF_PROBE, l2pf],
                                 capture_output=True, text=True, timeout=240)
            _if_ok = out.returncode == 0 and 'probe-ok' in out.stdout
            if not _if_ok:
                logger.warning('Native MTP draft skip: IF-node probe failed (rc %s): %s',
                               out.returncode, out.stderr[-400:])
        except Exception as exc:
            logger.warning('Native MTP draft skip: IF-node probe did not run (%r)', exc)
            _if_ok = False
    return _if_ok


def _skip_graph(p, graph, kwargs, metadata, slots, gate, padded, used_rows,
                original_mode):
    """The same step graph with the model forward inside a CUDA graph IF node on
    gate[0] == 0. Once the depth gate stopped the draft loop, a pass's forward is
    never read: the screen's gated kernels write the +0 scores and fixed candidate
    ids without reading h (nemotron_h_mtp_screen.py _nvfp4_rows_dot, _chunk_topk),
    so the stopped draft token does not depend on it, and the stopped pass's KV
    slot lies past every live pass of the cycle and is rewritten by the next
    cycle's first pass before any read. A skipped pass leaves its outputs and the
    gate untouched. Admitted only when it matches the record's graph bit for bit
    with the gate open, skips with it stopped, and the timing pays; else None."""
    from vllm.utils.torch_utils import current_stream
    outer = current_stream()
    pool = captured = None
    try:
        if not _if_probe():
            return None
        pred = torch.zeros((), dtype=torch.bool, device=gate.device)
        skip = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        gate.zero_()
        with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                 cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                 slot_mapping=slots):
            from vllm.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id)
            set_graph_pool_id(pool)
            with torch.cuda.graph(skip, pool=pool, stream=outer):
                torch.eq(gate[0], 0, out=pred)
                skip.begin_capture_to_if_node(pred)
                try:
                    # The body captures on a stream torch switched to in C++;
                    # vLLM's tracked current stream must name it too.
                    torch.cuda.set_stream(torch.cuda.current_stream())
                    out = p.model(**kwargs)
                finally:
                    try:
                        skip.end_capture_to_conditional_node()
                    finally:
                        torch.cuda.set_stream(outer)
        captured = True
        torch.cuda.synchronize(p.device)

        def leaves(value):
            if isinstance(value, torch.Tensor):
                return [value]
            return [t for v in value for t in leaves(v)]

        def rec():
            with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                     cudagraph_runtime_mode=CUDAGraphMode.FULL,
                                     slot_mapping=slots):
                return graph(**kwargs)

        # Stopped: the body must not run (sentinel outputs, gate as set).
        gate.zero_()
        gate[0].fill_(1)
        stopped = gate.clone()
        for t in leaves(out):
            t.fill_(3)
        skip.replay()
        torch.cuda.synchronize(p.device)
        if (not all(torch.equal(t, torch.full_like(t, 3)) for t in leaves(out))
                or not torch.equal(gate, stopped)):
            logger.warning('Native MTP draft skip: stopped pass ran; original graph kept')
            return None
        # Open: bitwise the record's graph and the eager model, after a skipped pass.
        for case in (1, 0):
            gate.zero_()
            p.input_ids[:padded].fill_(case)
            p.hidden_states[:padded].fill_(0.125 * case)
            skip.replay()
            actual = used_rows(out)
            reference = used_rows(rec())
            with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                                     cudagraph_runtime_mode=original_mode,
                                     slot_mapping=slots):
                eager = used_rows(p.model(**kwargs))
            if len(actual) != len(reference) or not all(
                    torch.equal(a, b) and torch.equal(a, c)
                    for a, b, c in zip(actual, reference, eager)):
                logger.warning('Native MTP draft skip: rows differ; original graph kept')
                return None
        # Admission: the IF node's cost on live passes against the stopped-pass saving
        # (about 3 live passes per half a stopped pass per cycle).
        # The record's raw graph, so neither side pays the wrapper's host work.
        raw = graph.concrete_cudagraph_entries[BatchDescriptor(num_tokens=padded)].cudagraph
        gate.zero_()
        rec_live = _replay_us(raw.replay)
        skip_live = _replay_us(skip.replay)
        gate[0].fill_(1)
        rec_stop = _replay_us(raw.replay)
        skip_stop = _replay_us(skip.replay)
        gate.zero_()
        cost = max(0.0, skip_live - rec_live)
        ok = skip_live <= rec_live + 2.0 and rec_stop - skip_stop >= 6.0 * cost
        logger.info('Native MTP draft skip: live %.1f/%.1f us, stopped %.1f/%.1f us '
                    '(record/skip), %s', rec_live, skip_live, rec_stop, skip_stop,
                    'admitted' if ok else 'original graph kept')
        return (skip, out, pred, pool) if ok else None
    except Exception as exc:
        logger.warning('Native MTP draft skip unavailable: %r', exc)
        if pool is not None and not captured:
            # A capture that died before capture_end leaves the pool registered
            # as capturing in the allocator.
            try:
                torch._C._cuda_endAllocateToPool(gate.device.index, pool)
            except Exception:
                pass
        return None
    finally:
        torch.cuda.set_stream(outer)
        gate.zero_()
        p.input_ids[:padded].zero_()
        p.hidden_states[:padded].zero_()


def maybe_run(p, kwargs, metadata, padded, batch_size):
    state = getattr(p, '_nemo_full', None)
    if state is None or batch_size != 1 or padded != state[1]:
        return None
    graph, _, signature, inputs, _outputs, skip = state
    if (BatchDescriptor(num_tokens=padded) not in graph.concrete_cudagraph_entries
            or _inputs(kwargs) != inputs or _metadata(p, metadata) != signature):
        return None
    if skip is not None:
        skip[0].replay()
        return skip[1]
    with set_forward_context(metadata, p.vllm_config, num_tokens=padded,
                             cudagraph_runtime_mode=CUDAGraphMode.FULL,
                             slot_mapping=p._get_slot_mapping(padded)):
        return graph(**kwargs)
