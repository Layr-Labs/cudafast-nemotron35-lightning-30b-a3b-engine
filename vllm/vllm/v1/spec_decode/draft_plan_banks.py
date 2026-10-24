# SPDX-License-Identifier: Apache-2.0
"""Give each native draft stage its own pinned FlashInfer plan storage."""

import torch


class DraftPlanBanks:
    @classmethod
    def create(cls, groups, config, device, depth):
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

        if not 2 <= depth <= 6 or not groups:
            return None
        originals = [group.get_metadata_builder() for group in groups]
        if any(
            type(builder) is not FlashInferMetadataBuilder
            or builder.use_dcp
            or builder.is_kvcache_nvfp4
            or builder.flashinfer_trtllm_api_decode_kernel is not None
            for builder in originals
        ):
            return None
        stages = [originals]
        for _ in range(1, depth):
            stage = []
            for group, original in zip(groups, originals):
                builder = FlashInferMetadataBuilder(
                    original.kv_cache_spec,
                    group.layer_names,
                    config,
                    device,
                )
                # GPU scratch is sequentially consumed on the same stream.
                # Only plan/index storage, including the wrapper's pinned
                # integer plan workspace, must be private to each stage.
                builder.set_workspace_buffer(original._get_workspace_buffer())
                stage.append(builder)
            stages.append(stage)
        return cls(groups, stages, device)

    def __init__(self, groups, stages, device):
        self._groups = groups
        self._stages = stages
        self._device = device
        self._events = [None] * len(stages)
        self._owner = None

    def build(self, common_attn_metadata, draft_index):
        stream = torch.cuda.current_stream(self._device)
        if self._owner is not None and stream != self._owner:
            # Sharing GPU scratch requires stream order. Do not silently
            # introduce cross-stream races if a future caller violates it.
            raise RuntimeError("DraftPlanBanks requires one owning stream")
        self._owner = stream
        # Dynamic depths outside the allocated bank use bank zero with its
        # normal ownership fence. No additional memory is allocated here.
        index = draft_index if 0 <= draft_index < len(self._stages) else 0
        event = self._events[index]
        if event is not None and not event.query():
            event.synchronize()
        if event is None:
            event = torch.cuda.Event()
            self._events[index] = event
        per_group = []
        per_layer = {}
        try:
            for group, builder in zip(self._groups, self._stages[index]):
                metadata = builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=draft_index,
                )
                per_group.append(metadata)
                for name in group.layer_names:
                    per_layer[name] = metadata
        finally:
            # Cover queued H2D work even if a later builder raises. This
            # event protects reuse on the next proposal/request as well.
            event.record()
        return per_group, per_layer
