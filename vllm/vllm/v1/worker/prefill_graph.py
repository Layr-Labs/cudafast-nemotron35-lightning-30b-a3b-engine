# SPDX-License-Identifier: Apache-2.0
"""Boot-only whole-forward graph for one fresh, exact 496-token prefill."""

import dataclasses

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context


def _static(value, memo):
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = torch.empty_strided(value.shape, value.stride(),
                                     dtype=value.dtype, device=value.device)
        result.copy_(value)
    elif dataclasses.is_dataclass(value):
        result = dataclasses.replace(value, **{
            field.name: _static(getattr(value, field.name), memo)
            for field in dataclasses.fields(value)
        })
    elif isinstance(value, dict):
        result = {key: _static(item, memo) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = type(value)(_static(item, memo) for item in value)
    else:
        # FlashInfer wrappers own persistent plan/index/workspace buffers.
        result = value
    memo[id(value)] = result
    return result


def _refresh(dst, src, seen):
    if isinstance(dst, (torch.Tensor, dict, list, tuple)) or dataclasses.is_dataclass(dst):
        if id(dst) in seen:
            return
        seen.add(id(dst))
    if isinstance(dst, torch.Tensor):
        assert (dst.shape, dst.dtype, dst.stride()) == (
            src.shape, src.dtype, src.stride())
        dst.copy_(src, non_blocking=True)
    elif dataclasses.is_dataclass(dst):
        assert type(src) is type(dst)
        for field in dataclasses.fields(dst):
            _refresh(getattr(dst, field.name), getattr(src, field.name), seen)
    elif isinstance(dst, dict):
        assert dst.keys() == src.keys()
        for key in dst:
            _refresh(dst[key], src[key], seen)
    elif isinstance(dst, (list, tuple)):
        assert len(dst) == len(src)
        for left, right in zip(dst, src):
            _refresh(left, right, seen)
    else:
        assert dst is src or dst == src, (type(dst), dst, src)


def stabilize_native_plan(wrapper):
    """Keep native non-graph planning and its exact kernel/reduction dispatch.

    Native plan uploads fresh index tensors on each request. Copy those
    values into fixed-address buffers before capture/replay. The native
    plan_info must stay identical for this one shape, not be silently frozen.
    """
    names = ("_qo_indptr_buf", "_paged_kv_indptr_buf", "_paged_kv_indices_buf",
             "_paged_kv_last_page_len_buf", "_kv_lens_buffer")
    info = tuple(wrapper._plan_info)
    saved = getattr(wrapper, "_fresh_prefill_native_buffers", None)
    if saved is None:
        saved = {name: getattr(wrapper, name).clone() for name in names
                 if isinstance(getattr(wrapper, name, None), torch.Tensor)}
        wrapper._fresh_prefill_native_buffers = saved
        wrapper._fresh_prefill_native_plan_info = info
    else:
        assert info == wrapper._fresh_prefill_native_plan_info
    for name, dst in saved.items():
        src = getattr(wrapper, name)
        assert (src.shape, src.dtype, src.stride()) == (
            dst.shape, dst.dtype, dst.stride())
        dst.copy_(src, non_blocking=True)
        setattr(wrapper, name, dst)


class FreshPrefillGraph:
    def __init__(self, runner, metadata, slots, input_ids, positions):
        self.runner = runner
        memo = {}
        self.metadata = _static(metadata, memo)
        self.slots = _static(slots, memo)
        self.input_ids = input_ids.clone()
        self.positions = positions.clone()
        self.logits_indices = torch.tensor([495], device=input_ids.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graph = torch.cuda.CUDAGraph()
        self.replays = 0
        # Compile, tune and allocate all kernels before capture, at M=496.
        for _ in range(2):
            self._forward()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, pool=self.pool):
            self.hidden, self.logits = self._forward()
        torch.cuda.synchronize()

    def _forward(self):
        # Only row 495 of the final hidden state is read (:94, :115, and the
        # same index in replay). While this call runs, the last MoE layer
        # routes every row through that row's experts, so it reads 6 experts'
        # weights instead of nearly all 128 and the other rows' MoE outputs,
        # which reach only norm_f's per-row norm and are never read, come from
        # the same experts. M, valid_shape_m and the kernel config do not
        # change, and row 495's own input, experts, weights and K order do
        # not change, so its logits stay bit-identical. The two warm-ups above
        # and the capture below all run this way, so the captured graph holds
        # the switch; replay (:118-128) runs only the recorded kernels and
        # never re-decides it, and the clear keeps every later forward (the
        # decode graphs are captured earlier, :7036 against :7024) untouched.
        model = self.runner.get_model()
        model.set_last_moe_row_src(self.logits_indices)
        try:
            with set_forward_context(
                self.metadata, self.runner.vllm_config, num_tokens=496,
                cudagraph_runtime_mode=CUDAGraphMode.NONE, slot_mapping=self.slots,
            ):
                hidden = self.runner._model_forward(
                    input_ids=self.input_ids, positions=self.positions,
                    intermediate_tensors=None, inputs_embeds=None,
                )
                logits = self.runner.model.compute_logits(hidden[self.logits_indices])
        finally:
            model.set_last_moe_row_src(None)
        return hidden, logits

    def replay(self, metadata, slots, input_ids, positions):
        # Rewrite every captured request tensor, even shape-derived constants.
        seen = set()
        _refresh(self.metadata, metadata, seen)
        _refresh(self.slots, slots, seen)
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.logits_indices.fill_(495)
        self.graph.replay()
        self.replays += 1
        return self.hidden, self.logits
