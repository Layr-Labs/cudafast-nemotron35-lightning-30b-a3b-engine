# SPDX-License-Identifier: Apache-2.0
"""Optional native MTP model and original greedy-screen graph, remote startup only."""
import os

import torch

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


class _ModelAndScreen:
    def __init__(self, model, screen):
        self.model = model
        self.screen = screen
        # Use the exact original copy/conversion, dtype and screen body. These
        # buffers are owned by this graph, rather than the standalone screen.
        self.h = torch.empty_like(screen._h)
        self.token = torch.empty_like(screen._out)

    def __call__(self, **kwargs):
        result = self.model(**kwargs)
        last_hidden = result[0] if isinstance(result, tuple) else result
        self.h.copy_(last_hidden[0])
        self.screen._greedy_row(self.h, gated=True, out=self.token)
        return result, self.token


def prepare(p, kwargs, metadata, slots, padded, used_rows, original_mode, gate):
    """Retain the admitted original split path if any new slot is unavailable."""
    p._nemo_full_head = None
    screen = getattr(p, '_draft_head_screen', None)
    if (os.environ.get('NEMO_MTP_FULL_HEAD', '1') == '0'
            or screen is None or getattr(screen, 'graph', None) is None
            or not screen.cascade or not screen.fold or not screen.fused_topk):
        return
    from vllm.v1.spec_decode.nemotron_full_step import (
        _inputs, _launch_ready_slot, _metadata,
    )
    signature = _metadata(p, metadata)
    if signature is None:
        return
    unique = tuple({id(m): m for m in metadata.values()}.values())
    saved_events = [(m, hasattr(m, 'plan_ready_event'),
                     getattr(m, 'plan_ready_event', None)) for m in unique]
    saved_gate = None if gate is None else gate.clone()
    pool = []
    try:
        relay = torch.cuda.Stream(device=p.device)
        source = torch.cuda.Event()
        for _slot in range(3):
            if gate is not None:
                gate.zero_()
            ready = torch.cuda.Event(external=True)
            ready.record()
            ready.synchronize()
            for m in unique:
                m.plan_ready_event = ready
            body = _ModelAndScreen(p.model, screen)
            with set_forward_context(
                    metadata, p.vllm_config, num_tokens=padded,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    slot_mapping=slots):
                for _warm in range(2):
                    body(**kwargs)
                torch.cuda.synchronize(p.device)
                graph = CUDAGraphWrapper(
                    body, p.vllm_config, CUDAGraphMode.FULL,
                    CUDAGraphOptions(weak_ref_output=False))
                graph.graph_pool = torch.cuda.graph_pool_handle()
                outputs = graph(**kwargs)
            torch.cuda.synchronize(p.device)
            key = BatchDescriptor(num_tokens=padded)
            if key not in graph.concrete_cudagraph_entries:
                raise RuntimeError('model/screen graph was not captured')
            slot = dict(graph=graph, ready=ready, retired=torch.cuda.Event(),
                        busy=False, outputs=outputs, body=body)
            states = ((0, 0, 0), (0, 0, 1), (0, 2, 1), (1, 2, 1))
            for case, state in enumerate(states):
                if slot['busy'] and not slot['retired'].query():
                    raise RuntimeError('private model/screen slot is still busy')
                p.input_ids[:padded].fill_(case % 2)
                p.hidden_states[:padded].fill_((0.0, 0.125, -0.25, 0.0)[case])
                initial = None
                if gate is not None:
                    initial = torch.tensor(state, dtype=gate.dtype, device=gate.device)
                    gate.copy_(initial)
                source.record()
                actual_model, actual_token = _launch_ready_slot(
                    p, slot, relay, kwargs, metadata, padded, (source,))
                actual = used_rows(actual_model) + [actual_token.clone()]
                if gate is not None:
                    actual.append(gate.clone())
                    gate.copy_(initial)
                # The standalone screen runs exactly once on the original
                # model's row, so active/stopped/kept counts are compared too.
                with set_forward_context(
                        metadata, p.vllm_config, num_tokens=padded,
                        cudagraph_runtime_mode=original_mode,
                        slot_mapping=slots):
                    expected_model = p.model(**kwargs)
                last = (expected_model[0] if isinstance(expected_model, tuple)
                        else expected_model)
                expected_token = screen.greedy(last[:1])
                expected = used_rows(expected_model) + [expected_token.clone()]
                if gate is not None:
                    expected.append(gate.clone())
                if len(actual) != len(expected) or not all(
                        torch.equal(a, b) for a, b in zip(actual, expected)):
                    raise RuntimeError('model/screen private outputs or gate differ')
            pool.append(slot)
        p._nemo_full_head = (tuple(pool), relay, padded, signature, _inputs(kwargs),
                             screen, screen.graph)
        logger.info('Native MTP model+screen: %d admitted slots', len(pool))
    except Exception as exc:
        logger.warning('Native MTP model+screen unavailable; split path retained: %r', exc)
    finally:
        if saved_gate is not None:
            gate.copy_(saved_gate)
        for m, existed, event in saved_events:
            if existed:
                m.plan_ready_event = event
            elif hasattr(m, 'plan_ready_event'):
                delattr(m, 'plan_ready_event')


def maybe_run(p, kwargs, metadata, padded, batch_size, greedy):
    """Return explicit (model_output, independent_token), never a token cache."""
    state = getattr(p, '_nemo_full_head', None)
    if (state is None or not greedy or batch_size != 1 or padded != state[2]
            or torch.cuda.is_current_stream_capturing()):
        return None
    from vllm.v1.spec_decode.nemotron_full_step import (
        _inputs, _launch_ready_slot, _metadata,
    )
    pool, relay, _, signature, inputs, screen, screen_graph = state
    if (getattr(p, '_draft_head_screen', None) is not screen
            or screen.graph is not screen_graph
            or p.model is not pool[0]['body'].model
            or _inputs(kwargs) != inputs or _metadata(p, metadata) != signature):
        return None
    events = {id(event): event for m in metadata.values()
              if (event := getattr(m, 'plan_ready_event', None)) is not None}
    for slot in pool:
        if (BatchDescriptor(num_tokens=padded) not in
                slot['graph'].concrete_cudagraph_entries
                or (slot['busy'] and not slot['retired'].query())):
            continue
        if events:
            result, token = _launch_ready_slot(
                p, slot, relay, kwargs, metadata, padded, events.values())
        else:
            # Current plans are already queued on main. The previous external
            # event generation completed before this slot could be reused.
            with set_forward_context(
                    metadata, p.vllm_config, num_tokens=padded,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    slot_mapping=p._get_slot_mapping(padded)):
                result, token = slot['graph'](**kwargs)
            slot['retired'].record()
            slot['busy'] = True
        # A graph owns static buffers. A fresh clone per draft is necessary:
        # earlier draft IDs remain live until the caller's final stack.
        return result, token.clone()
    return None
