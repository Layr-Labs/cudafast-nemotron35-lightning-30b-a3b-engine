# SPDX-License-Identifier: Apache-2.0
"""Optional unconditional parent graph over the ORIGINAL full-step and screen.

Authoring only: all qualification/capture/timing executes on the official box.
Three token destinations prevent later draft passes from overwriting earlier ids.
There is no IF node, arm/clear kernel, changed native plan or token cache.
"""
import os
import statistics
import time
from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import NoopOffloader, get_offloader
from vllm.v1.spec_decode.nemotron_full_step import _inputs, _metadata, _tensor
from vllm.v1.spec_decode.nemotron_full_step import maybe_run as full_run

logger = init_logger(__name__)
ENABLED = os.environ.get("NEMO_MTP_PAIRED_STEP", "1") != "0"
_SCREEN_WORK = ("_h", "_out", "scores", "refine_scores", "cand_logits", "_cand",
                "_coarse_cand", "_pre_cand", "mid_scores", "_hc", "_hs", "_hr",
                "_hp", "_hm", "_full_part")


def _layout(wrapper):
    from vllm.v1.spec_decode.nemotron_decode_plan import _SCALARS
    return (tuple(wrapper._plan_info),
            tuple((n, getattr(wrapper, n)) for n in _SCALARS if hasattr(wrapper, n)))


def _ready(metadata):
    seen = set()
    for md in metadata.values():
        event = getattr(md, "plan_ready_event", None)
        if event is not None and id(event) not in seen:
            torch.cuda.current_stream().wait_event(event)
            seen.add(id(event))


def maybe_run(p, kwargs, metadata, padded, batch_size, token_index):
    s = getattr(p, "_nemo_paired_step", None)
    return _replay(s, p, kwargs, metadata, padded, batch_size, token_index)


def _replay(s, p, kwargs, metadata, padded, batch_size, token_index):
    # Shared by production and qualification: do not omit guards from timing.
    if (s is None or batch_size != 1 or padded != s.padded
            or p._nemo_full is not s.full or p._draft_head_screen is not s.screen
            or get_offloader() is not s.offloader
            or not 0 <= token_index < len(s.graphs)
            or torch.cuda.is_current_stream_capturing()
            or _inputs(kwargs) != s.inputs or _metadata(p, metadata) != s.signature
            or _layout(s.wrapper) != s.layout):
        return None
    _ready(metadata)
    s.graphs[token_index].replay()
    return s.hidden, s.tokens[token_index]


def _bytes(t):
    return t.contiguous().view(torch.uint8).clone()


@torch.no_grad()
def prepare(p):
    if (not ENABLED or getattr(p, "_nemo_paired_attempted", False)
            or getattr(p, "_nemo_step", None) is not None
            or torch.cuda.is_current_stream_capturing()
            or p.num_speculative_tokens != 4 or p.model_returns_tuple()):
        return
    full = getattr(p, "_nemo_full", None)
    screen = getattr(p, "_draft_head_screen", None)
    native = getattr(p, "_nemo_full_native_layout", None)
    if full is None or screen is None or screen.graph is None or native is None:
        return
    if type(get_offloader()) is not NoopOffloader:
        return
    p._nemo_paired_attempted = True
    p._nemo_paired_step = None
    saved = []
    try:
        _prepare(p, full, screen, native, saved)
    except Exception as exc:
        logger.warning("Native MTP paired step refused; original kept: %r", exc)
        p._nemo_paired_step = None
    finally:
        # A restore failure must not publish a path with damaged shared state.
        try:
            torch.cuda.synchronize(p.device)
            for target, value in saved:
                target.copy_(value)
            torch.cuda.synchronize(p.device)
        except Exception:
            p._nemo_paired_step = None
            raise


def _prepare(p, full, screen, native, saved):
    from vllm.model_executor.models import nemotron_h_mtp_kernels as kernels
    _, padded, signature, inputs, (_, private, metadata, zero_row) = full
    wrapper = native[0]
    if padded not in (1, 5) or _layout(wrapper) != (native[1], native[2]):
        return
    # Keep this qualification's original one-token private native plan valid.
    cpu_seq = private._seq_lens_cpu
    if (not isinstance(cpu_seq, torch.Tensor) or cpu_seq.device.type != "cpu"
            or cpu_seq.numel() != 1 or int(cpu_seq[0]) != 1):
        return
    gate = kernels.depth_gate_init(p.device)
    if gate is None:
        return
    kwargs = dict(input_ids=p.input_ids[:padded], positions=p._get_positions(padded),
                  inputs_embeds=None, hidden_states=p.hidden_states[:padded])
    if _inputs(kwargs) != inputs or _metadata(p, metadata) != signature:
        return
    block = int(p.block_size)
    caches = [p.compilation_config.static_forward_context[n].kv_cache
              for n in sorted(p._draft_attn_layer_names)]
    if (block <= 0 or any(not isinstance(t, torch.Tensor) or not t.is_cuda
                         or t.ndim != 4 or t.shape[0] < 4 or t.shape[2] != block
                         for t in caches)):
        return
    values = [gate, p.input_ids[:padded], p.positions[:padded],
              p.hidden_states[:padded], p._slot_mapping_buffer[:padded]]
    values += [t[:4] for t in caches]
    values += list(p.model._moe_row_src_buffers())
    values += [getattr(screen, n) for n in _SCREEN_WORK
               if isinstance(getattr(screen, n, None), torch.Tensor)]
    # Metadata is observed but never rebuilt or changed by this graph wrapper.
    builder = p.draft_attn_groups[0].get_metadata_builder()
    for name in ("paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        values += [getattr(builder, name).cpu, getattr(builder, name).gpu]
    values += [wrapper._int_workspace_buffer, wrapper._float_workspace_buffer]
    seen = set()
    torch.cuda.synchronize(p.device)
    for target in values:
        key = _tensor(target)
        if key not in seen:
            saved.append((target, target.clone()))
            seen.add(key)

    def restore():
        for target, original in saved:
            target.copy_(original)
        p.model.set_moe_row_src(padded, zero_row)

    def original():
        hidden = full_run(p, kwargs, metadata, padded, 1)
        if hidden is None:
            raise ValueError("original full-step declined exact private owners")
        return hidden, screen.greedy(hidden[:1])

    restore()
    gate.zero_()
    hidden, _ = original()
    tokens = tuple(torch.empty_like(screen._out) for _ in range(3))
    if len({t.data_ptr() for t in tokens}) != 3:
        raise ValueError("paired token owners alias")
    graphs = []
    # Capture actual old child executables, their existing hidden-copy kernel,
    # and one token-copy kernel. No new model/screen lowering or conditional arm.
    for token in tokens:
        restore()
        gate.zero_()
        graph = torch.cuda.CUDAGraph()
        _ready(metadata)
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            captured_hidden = full_run(p, kwargs, metadata, padded, 1)
            if captured_hidden is None:
                raise ValueError("full-step declined during parent capture")
            screen._h.copy_(captured_hidden[0])
            screen.graph.replay()
            token.copy_(screen._out)
        graphs.append(graph)
        if captured_hidden.data_ptr() != hidden.data_ptr():
            raise ValueError("original hidden output owner changed during capture")
    torch.cuda.synchronize(p.device)

    def setup(case, stopped):
        restore()
        gen = torch.Generator(device=p.device).manual_seed(2026102800 + case)
        for cache in caches:
            cache[:4].view(torch.uint8).copy_(torch.randint(
                0, 120, cache[:4].view(torch.uint8).shape,
                device=p.device, dtype=torch.uint8, generator=gen))
        p.input_ids[:padded].copy_(torch.randint(
            0, 131072, (padded,), device=p.device, dtype=p.input_ids.dtype,
            generator=gen))
        p.hidden_states[:padded].copy_((torch.randn(
            p.hidden_states[:padded].shape, device=p.device,
            dtype=torch.float32, generator=gen) * (0.25 + case)).to(p.dtype))
        if case == 5:
            p.hidden_states[:padded].zero_()
            p.hidden_states[:padded, ::2] = -0.0
        elif case == 6:
            words = torch.tensor([1, -32767, 128, -32640], dtype=torch.int16,
                                 device=p.device)
            p.hidden_states[:padded].view(torch.int16)[:, :4] = words
        elif case == 7:
            p.hidden_states[:padded, :3] = torch.tensor(
                [float("inf"), -float("inf"), float("nan")],
                dtype=p.dtype, device=p.device)
        p._slot_mapping_buffer[:padded].fill_(-1)
        p._slot_mapping_buffer[0] = 0
        p.positions[:padded].zero_()
        gate.copy_(torch.tensor([int(stopped), case % 3, 1, kernels.GATE_PROD_ONE],
                                dtype=torch.int32, device=p.device))
        for i, token in enumerate(tokens):
            token.fill_(-10 - i)

    def observe(h, token):
        return [_bytes(h), _bytes(token)] + [_bytes(t) for t in values]

    state = SimpleNamespace(
        full=full, screen=screen, wrapper=wrapper, layout=_layout(wrapper),
        inputs=inputs, signature=signature, padded=padded, graphs=tuple(graphs),
        hidden=hidden, tokens=tokens, owners=(kwargs, metadata, zero_row),
        offloader=get_offloader(), timing=None)

    def paired(index):
        result = _replay(state, p, kwargs, metadata, padded, 1, index)
        if result is None:
            raise ValueError("complete production paired guards declined private owner")
        return result

    # Match every padded hidden word and every private modified buffer, including
    # stopped KV writes (no conditional-step normalization of observed caches).
    for case in range(8):
        for stopped in (False, True):
            for index in range(3):
                setup(case, stopped)
                expected = observe(*original())
                setup(case, stopped)
                got = observe(*paired(index))
                if not all(torch.equal(a, b) for a, b in zip(expected, got)):
                    raise ValueError(f"paired state mismatch case={case} stop={stopped}")
            # All three token owners must remain unchanged after later passes.
            setup(case, stopped)
            retained = []
            for index in range(3):
                _, token = paired(index)
                retained.append(token.clone())
                if not all(torch.equal(a, b) for a, b in zip(tokens, retained)):
                    raise ValueError("later graph overwrote an earlier token")
            # An original continuation observes the exact same post-pair state.
            setup(case, stopped)
            original()
            expected = observe(*original())
            setup(case, stopped)
            paired(case % 3)
            got = observe(*original())
            if not all(torch.equal(a, b) for a, b in zip(expected, got)):
                raise ValueError("original continuation after paired step differs")

    pressure = torch.empty(64 << 20, dtype=torch.uint8, device=p.device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    event_ratios, host_ratios = [], []
    for case in (0, 2, 4):
        for stopped in (False, True):
            for cold in (False, True):
                samples = [[], []]
                host = [[], []]
                for trial in range(7):
                    for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        setup(case, stopped)
                        if cold:
                            pressure.fill_(trial + case)
                        torch.cuda.synchronize(p.device)
                        begin = time.perf_counter()
                        start.record()
                        (original if arm == 0 else lambda: paired(trial % 3))()
                        end.record()
                        end.synchronize()
                        elapsed = time.perf_counter() - begin
                        device_ms = start.elapsed_time(end)
                        if not elapsed > 0 or not device_ms > 0:
                            raise ValueError("paired nonpositive full-work timing")
                        samples[arm].append(device_ms)
                        host[arm].append(elapsed)
                a, b = (statistics.median(x) for x in samples)
                c, d = (statistics.median(x) for x in host)
                event_ratios.append(b / a)
                host_ratios.append(d / c)
    metrics = (statistics.median(event_ratios), max(event_ratios),
               statistics.median(host_ratios), max(host_ratios))
    if not (metrics[0] < 0.99 and metrics[1] < 1.01
            and metrics[2] < 0.995 and metrics[3] < 1.01):
        raise ValueError(f"paired full-work warm/pressure refused {metrics}")
    state.timing = metrics
    p._nemo_paired_step = state
    logger.info("Native MTP unconditional paired step admitted metrics=%s", metrics)
