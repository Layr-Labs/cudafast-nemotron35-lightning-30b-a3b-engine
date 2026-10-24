# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A proposal lifetime for one installed FA2 schedule, never model results."""

import sys
from dataclasses import dataclass
from functools import wraps
from threading import get_ident
from typing import Any

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper


@dataclass
class InstalledPlan:
    owner: Any
    key: tuple
    # Keep every object represented by an identity/address in key alive.
    keepalive: tuple


@dataclass
class ProposalPlanScope:
    epoch: object
    thread: int
    active: bool = True
    next_stage: int = 0
    installed: InstalledPlan | None = None

    def invalidate(self) -> None:
        self.active = False
        self.installed = None


def _native_builder(proposer, depth):
    """Resolve only one native MTP builder, once per ordinary proposal."""
    if (
        depth != 3
        or proposer.method != "mtp"
        or proposer.parallel_drafting
        or proposer.constant_draft_positions
        or proposer.supports_mm_inputs
        or proposer.use_heterogeneous_vocab
        or proposer.eplb_state is not None
        or proposer.speculative_config.enforce_eager
        or getattr(proposer.draft_model_config.hf_config, "model_type", None)
        != "nemotron_h_mtp"
    ):
        return None
    parallel = proposer.vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.use_ubatching
        or len(proposer.draft_attn_groups) != 1
    ):
        return None
    model = proposer.model
    if isinstance(model, BreakableCUDAGraphWrapper):
        model = model.unwrap()
    native_module = sys.modules.get("vllm.model_executor.models.nemotron_h_mtp")
    if native_module is None or type(model) is not getattr(
        native_module, "NemotronHMTP", None
    ):
        return None
    group = proposer.draft_attn_groups[0]
    backend_module = sys.modules.get("vllm.v1.attention.backends.flashinfer")
    if backend_module is None:
        return None
    builder = group.get_metadata_builder()
    if type(builder) is not getattr(backend_module, "FlashInferMetadataBuilder", None):
        return None
    return builder


def proposal_plan_scope(propose):
    """End the explicit scope on every return/exception of the original method."""

    @wraps(propose)
    def scoped(proposer, *args, **kwargs):
        depth = kwargs.get("num_speculative_tokens", args[0] if args else None)
        try:
            builder = _native_builder(proposer, depth)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            builder = None
        epoch = object()
        if builder is None or not builder.begin_proposal_plan_scope(epoch):
            return propose(proposer, *args, **kwargs)
        try:
            return propose(proposer, *args, **kwargs)
        finally:
            builder.end_proposal_plan_scope(epoch)

    return scoped


def new_scope(epoch: object) -> ProposalPlanScope:
    return ProposalPlanScope(epoch=epoch, thread=get_ident())
