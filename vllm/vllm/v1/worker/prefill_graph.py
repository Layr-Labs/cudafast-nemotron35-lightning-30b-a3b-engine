# SPDX-License-Identifier: Apache-2.0
"""Boot-only whole-forward graph for one fresh, exact 496-token prefill."""

import dataclasses
import os

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
        # LOCAL PATCH (cudafast-nemotron35): a tensor its builder marked constant (built
        # once for a batch shape, never written after) already sits in dst if dst's last
        # copy came from that same object (mamba2_attn.py prefill metadata cache).
        if getattr(src, "_vp_const", False):
            if getattr(dst, "_vp_last_src", None) is src:
                return
            dst.copy_(src, non_blocking=True)
            dst._vp_last_src = src
            return
        dst._vp_last_src = None
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
        # LOCAL PATCH (cudafast-nemotron35): a second graph for a prefill that samples
        # one token and drafts nothing (gpu_model_runner._drafts_can_be_verified() is
        # False): its last MoE layer computes row 495 alone (moe_runner keep_m_lite).
        # Row 495's logits are the ones the graph above gives. Its own pool, so the
        # outputs above are never overwritten by it. PREFILL_LITE=0 disables.
        self.graph_lite = None
        if os.environ.get("PREFILL_LITE", "1") != "0":
            try:
                self.pool_lite = torch.cuda.graph_pool_handle()
                for _ in range(2):
                    self._forward(lite=True)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool_lite):
                    self.hidden_lite, self.logits_lite = self._forward(lite=True)
                torch.cuda.synchronize()
                self.graph_lite = graph
            except Exception:  # noqa: BLE001
                self.graph_lite = None
        # LOCAL PATCH (cudafast-nemotron35): early launch (gpu_model_runner
        # _fresh_prefill_early). The request tensors above are written only by
        # replay(); early_sig names what the last replay wrote them for: the
        # request's block ids plus the plan state of every FlashInfer wrapper
        # the graph reads by reference.
        self.early_sig = None
        self.wrappers = _wrappers(self.metadata)
        for wrapper in self.wrappers:
            _count_plans(wrapper)

    def _forward(self, lite: bool = False):
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
        model.set_last_moe_row_src(self.logits_indices, lite=lite)
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

    def plan_state(self):
        return tuple((id(w), getattr(w, "_vp_plan_calls", 0)) for w in self.wrappers)

    def replay(self, metadata, slots, input_ids, positions, lite: bool = False,
               sig=None):
        # Rewrite every captured request tensor, even shape-derived constants.
        seen = set()
        _refresh(self.metadata, metadata, seen)
        _refresh(self.slots, slots, seen)
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.logits_indices.fill_(495)
        self.early_sig = None if sig is None else (sig, self.plan_state())
        self.replays += 1
        if lite and self.graph_lite is not None:
            self.graph_lite.replay()
            return self.hidden_lite, self.logits_lite
        self.graph.replay()
        return self.hidden, self.logits

    def replay_early(self, input_ids, lite: bool):
        """Replay for a request whose block ids and wrapper plan state equal
        early_sig: every request tensor already holds what replay() would write
        (positions 0..495, logits row 495 and the block-derived metadata), so
        only the prompt ids are new."""
        self.input_ids.copy_(input_ids, non_blocking=True)
        self.replays += 1
        if lite and self.graph_lite is not None:
            self.graph_lite.replay()
            return self.hidden_lite, self.logits_lite
        self.graph.replay()
        return self.hidden, self.logits

    def request_tensors(self):
        """Every tensor a replay reads that replay() writes, plus the wrappers'
        fixed plan buffers (for the early-launch check)."""
        out, seen = [], set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                out.append(value)
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    walk(getattr(value, field.name))
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(self.metadata)
        walk(self.slots)
        walk(self.input_ids)
        walk(self.positions)
        walk(self.logits_indices)
        for wrapper in self.wrappers:
            for name in sorted(vars(wrapper)):
                value = getattr(wrapper, name, None)
                if (isinstance(value, torch.Tensor) and value.is_cuda
                        and name.endswith(("_buf", "_buffer"))
                        and name != "_float_workspace_buffer"):
                    walk(value)
        return out


def _wrappers(metadata):
    """FlashInfer prefill wrappers the captured metadata references."""
    found, seen = [], set()

    def walk(value):
        if id(value) in seen or isinstance(value, torch.Tensor):
            return
        seen.add(id(value))
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                if field.name == "wrapper" and hasattr(item, "plan"):
                    if all(item is not w for w in found):
                        found.append(item)
                else:
                    walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(metadata)
    return found


def _count_plans(wrapper):
    """Count plan() calls on a wrapper: a plan may rewrite the buffers the
    captured attention reads, so an early launch must see none since the
    last replay()."""
    if getattr(wrapper, "_vp_plan_counted", False):
        return
    plan = wrapper.plan

    def counted(*args, **kwargs):
        wrapper._vp_plan_calls = getattr(wrapper, "_vp_plan_calls", 0) + 1
        return plan(*args, **kwargs)

    wrapper._vp_plan_calls = 0
    wrapper.plan = counted
    wrapper._vp_plan_counted = True
