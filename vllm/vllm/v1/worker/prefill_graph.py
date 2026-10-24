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
    """Two captured variants of the same fresh, exact 496-token prefill.

    STOCK is the whole forward exactly as this engine ran it before: each row
    of the last MoE layer takes its own experts. LAST_ROW is the same forward
    with the last MoE layer routing every row through row 495 (see
    NemotronHForCausalLM.set_last_moe_row_src): six experts' weights are read
    instead of nearly all 128, and the other rows' MoE outputs, which reach
    only norm_f's per-row norm, come from the same experts.

    Each variant owns its graph pool, so its captured activations and outputs
    are its own memory; the one static set of request tensors is written before
    either replays and is read by both, and only one graph runs per step. The
    caller picks LAST_ROW only for a prefill whose other rows nothing reads
    (GPUModelRunner._prefill_reads_one_row); every other prefill replays STOCK
    under the condition it always had.
    """

    STOCK = 0
    LAST_ROW = 1

    def __init__(self, runner, metadata, slots, input_ids, positions):
        self.runner = runner
        memo = {}
        self.metadata = _static(metadata, memo)
        self.slots = _static(slots, memo)
        self.input_ids = input_ids.clone()
        self.positions = positions.clone()
        self.logits_indices = torch.tensor([495], device=input_ids.device)
        self.replays = 0
        self.pools = [torch.cuda.graph_pool_handle(),
                      torch.cuda.graph_pool_handle()]
        self.graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        self.outputs = [None, None]
        # STOCK first, so the variant every other prefill replays is captured
        # before anything else can fail.
        self._capture(self.STOCK, False)
        self._capture(self.LAST_ROW, True)

    def _capture(self, variant: int, last_row_only: bool) -> None:
        """Warm up and capture one variant; M, the kernel config and the
        static request tensors are the same for both."""
        model = self.runner.get_model()
        model.set_last_moe_row_src(self.logits_indices if last_row_only else None)
        try:
            # Compile, tune and allocate all kernels before capture, at M=496.
            for _ in range(2):
                self._forward()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graphs[variant],
                                  pool=self.pools[variant]):
                hidden, logits = self._forward()
            torch.cuda.synchronize()
        finally:
            # The switch lives only in the variant captured above: replay runs
            # the recorded kernels and never re-decides it, and the decode
            # graphs were captured before this method ran.
            model.set_last_moe_row_src(None)
        self.outputs[variant] = (hidden, logits)

    def _forward(self):
        with set_forward_context(
            self.metadata, self.runner.vllm_config, num_tokens=496,
            cudagraph_runtime_mode=CUDAGraphMode.NONE, slot_mapping=self.slots,
        ):
            hidden = self.runner._model_forward(
                input_ids=self.input_ids, positions=self.positions,
                intermediate_tensors=None, inputs_embeds=None,
            )
            logits = self.runner.model.compute_logits(hidden[self.logits_indices])
        return hidden, logits

    def replay(self, metadata, slots, input_ids, positions, last_row_only=False):
        # Rewrite every captured request tensor, even shape-derived constants.
        # Both variants read these same buffers, so one rewrite serves either.
        seen = set()
        _refresh(self.metadata, metadata, seen)
        _refresh(self.slots, slots, seen)
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.logits_indices.fill_(495)
        variant = self.LAST_ROW if last_row_only else self.STOCK
        self.graphs[variant].replay()
        self.replays += 1
        hidden, logits = self.outputs[variant]
        return hidden, logits
