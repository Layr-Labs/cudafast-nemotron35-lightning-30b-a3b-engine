# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)

# LOCAL PATCH (cudafast-nemotron35): shco (harness/kernels/shco). When the whole shared
# expert of a prefill call runs on shx (one small CTA per SM), queue it on the aux stream
# beside the routed experts even above VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD.
try:
    import shco as _shco
except ImportError:
    _shco = None


def _shx_overlap(layer: torch.nn.Module, x: torch.Tensor) -> bool:
    if _shco is None:
        return False
    return _shco.overlap_ok(getattr(layer, "up_proj", None), getattr(layer, "down_proj", None), x)


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


# LOCAL PATCH (cudafast-nemotron35): deferred shared-expert down projection for
# small calls (see SharedExperts._launch_aux). VLLM_SHARED_DOWN_DEFER=0 disables.
_DEFER_DOWN = os.environ.get("VLLM_SHARED_DOWN_DEFER", "1") != "0"
_DEFER_MAX_TOKENS = 8


def launch_pending_shared_down() -> None:
    """Queue the pending deferred shared-expert down projection, if any, behind
    the main stream's work queued so far (called by the routed path after its
    up GEMM)."""
    se = SharedExperts._pending
    if se is not None:
        se._launch_down()


def deferred_shared_lease():
    """Join current shared work, but retain its output until fold acceptance.

    Only the single-output opaque op opens fold_add. A refusal leaves the
    original buffer/runner add intact; no global success flag is used.
    """
    se = SharedExperts._defer_live
    if se is None or not se.fold_add or se.enable_dbo:
        return None
    se._join_aux()
    idx = se._output_idx
    out = se._output[idx]
    return None if out is None else (se, idx, out)


def commit_deferred_shared(lease):
    se, idx, out = lease
    if (SharedExperts._defer_live is not se or se._output[idx] is not out
            or not se.fold_add):
        raise RuntimeError('shared fold lease changed before commit')
    se._output[idx] = None
    se._folded = True
    SharedExperts._defer_live = None


class SharedExperts(torch.nn.Module):
    _early_logged = False
    _pending: "SharedExperts | None" = None
    _pending_h: torch.Tensor | None = None
    _defer_live: "SharedExperts | None" = None

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts
        self._aux_open = False
        self.fold_add = False
        self._folded = False

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
                or _shx_overlap(self._layer, hidden_states)
            )
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())
            self._launch_aux(shared_experts_input)

    def _launch_aux(self, shared_experts_input: torch.Tensor) -> None:
        """Queue shared work without immediately blocking the routed path."""
        assert self._stream is not None
        idx = self._output_idx
        assert self._output[idx] is None
        # LOCAL PATCH (cudafast-nemotron35): for a small call, queue only the
        # up projection (+ ReLU^2) now; the down projection is queued on the same
        # aux stream once the routed experts' up GEMM has been queued
        # (launch_pending_shared_down, called by the routed path), so the aux
        # stream's down runs beside the latency-bound routed down projection
        # instead of streaming against the routed up GEMM. Same modules, same
        # inputs, same stream; only when the down starts changes.
        defer = (
            _DEFER_DOWN
            and not self.enable_dbo
            and shared_experts_input.shape[0] <= _DEFER_MAX_TOKENS
            and callable(getattr(self._layer, "forward_up", None))
            and callable(getattr(self._layer, "forward_down", None))
        )
        if defer:
            if SharedExperts._pending is not None:
                SharedExperts._pending._launch_down()
            with torch.cuda.stream(self._stream):
                self._pending_h = self._layer.forward_up(shared_experts_input)
            SharedExperts._pending = self
            SharedExperts._defer_live = self
        else:
            with torch.cuda.stream(self._stream):
                self._output[idx] = self._layer(shared_experts_input)
        self._aux_open = True
        if not SharedExperts._early_logged:
            SharedExperts._early_logged = True
            logger.info(
                "shared experts: early aux-stream launch active (M=%d, deferred down=%s)",
                shared_experts_input.shape[0], defer,
            )

    def _launch_down(self) -> None:
        h = self._pending_h
        if h is None:
            return
        self._pending_h = None
        if SharedExperts._pending is self:
            SharedExperts._pending = None
        assert self._stream is not None
        idx = self._output_idx
        # The aux stream starts the down once the main stream's queued work
        # (through the routed up GEMM) is done.
        self._stream.wait_stream(current_stream())
        with torch.cuda.stream(self._stream):
            self._output[idx] = self._layer.forward_down(h)

    def _join_aux(self) -> None:
        if not self._aux_open:
            return
        assert self._stream is not None
        if self._pending_h is not None:
            self._launch_down()
        current_stream().wait_stream(self._stream)
        self._aux_open = False

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)
        self._aux_open = False

        return output

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor | None:
        if self._folded:
            self._folded = False
            return None
        if SharedExperts._defer_live is self:
            SharedExperts._defer_live = None
        self._join_aux()
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        if self._folded:
            return None
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        # If another routed backend has no midpoint hook, finish the one
        # pending shared up/down exactly once instead of launching full MLP
        # again. The same aux/main join remains the output visibility boundary.
        if (order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
                and self._pending_h is not None):
            self._launch_down()
            return None

        if (
            order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
            and self._output[self._output_idx] is not None
        ):
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
