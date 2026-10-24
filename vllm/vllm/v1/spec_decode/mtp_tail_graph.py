# SPDX-License-Identifier: Apache-2.0
"""Bounded full-forward graphs for native Nemotron's one-row draft tail.

Only execution storage is retained. The caller refreshes IDs, hidden states,
positions, KV slots and the native FlashInfer plan before every invocation.
"""

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context


def tail_graph_enabled(proposer) -> bool:
    parallel = proposer.vllm_config.parallel_config
    return (
        proposer.method == "mtp"
        and proposer.draft_model_config.hf_config.model_type == "nemotron_h_mtp"
        and proposer.speculative_config.draft_sample_method == "greedy"
        and not proposer.speculative_config.enforce_eager
        and not proposer.vllm_config.model_config.enforce_eager
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and not proposer.parallel_drafting
        and not proposer.supports_mm_inputs
        and not proposer.use_heterogeneous_vocab
        and not proposer.use_local_argmax_reduction
        and not proposer.uses_mrope
        and proposer.draft_uses_xdrope_dim == 0
        and proposer.eplb_state is None
        and proposer.pass_hidden_states_to_model
        and not proposer.model_returns_tuple()
        and proposer.vllm_config.offload_config.uva.cpu_offload_gb == 0
        and proposer.vllm_config.offload_config.prefetch.offload_group_size == 0
    )


def _tensor_key(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class MTPTailGraph:
    """At most one graph per tail index (1..5), on one owning CUDA stream."""

    def __init__(self):
        self._entries = {}

    def run(self, model, config, draft_index, kwargs, metadata, slot_mapping):
        if not 1 <= draft_index <= 5 or torch.is_grad_enabled():
            return None
        ids = kwargs.get("input_ids")
        hidden = kwargs.get("hidden_states")
        positions = kwargs.get("positions")
        if (
            kwargs.get("inputs_embeds") is not None
            or any(not isinstance(t, torch.Tensor) for t in (ids, hidden, positions))
            or ids.shape != (1,)
            or ids.dtype != torch.int32
            or hidden.ndim != 2
            or hidden.shape[0] != 1
            or hidden.dtype != torch.bfloat16
            or positions.shape != (1,)
            or any(
                not t.is_cuda or not t.is_contiguous() or t.device != hidden.device
                for t in (ids, hidden, positions)
            )
            or not metadata
            or set(metadata) != set(slot_mapping)
        ):
            return None
        # Keep the optional backend import out of CPU/other-device paths.
        from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata

        metadata_key = []
        for name, md in sorted(metadata.items()):
            if (
                type(md) is not FlashInferMetadata
                or md.num_actual_tokens != 1
                or md.num_decodes != 1
                or md.num_decode_tokens != 1
                or md.num_prefills != 0
                or md.num_prefill_tokens != 0
                or md.prefill is not None
                or md.use_cascade
                or not md.causal
                or type(md.decode) is not FIDecode
                or not md.decode.wrapper.is_cuda_graph_enabled
            ):
                return None
            slot = slot_mapping[name]
            if (
                not slot.is_cuda
                or slot.shape != (1,)
                or slot.device != hidden.device
                or not slot.is_contiguous()
            ):
                return None
            metadata_key.append(
                (
                    name,
                    id(md.decode.wrapper),
                    md.q_data_type_decode,
                    md.q_data_type_prefill,
                    _tensor_key(md.slot_mapping),
                    _tensor_key(slot),
                )
            )

        with torch.cuda.device(hidden.device):
            if torch.cuda.is_current_stream_capturing():
                return None
            stream = torch.cuda.current_stream(hidden.device)
            signature = (
                id(model),
                id(config),
                _tensor_key(ids),
                _tensor_key(hidden),
                _tensor_key(positions),
                tuple(metadata_key),
            )
            entry = self._entries.get(draft_index)
            if entry is not None:
                graph, output, tokens, old_signature, owner, references = entry
                if signature != old_signature or stream != owner:
                    return None
            else:
                # NONE bypasses the nested PIECEWISE wrappers but executes the
                # same compiled kernels. Native graph-mode FlashInfer plans
                # have already been refreshed outside this capture.
                def execute():
                    with set_forward_context(
                        metadata,
                        config,
                        num_tokens=1,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        slot_mapping=slot_mapping,
                    ):
                        output = model(**kwargs)
                        tokens = model.compute_logits(output).argmax(dim=-1)
                        return output, tokens

                capture_stream = torch.cuda.Stream(device=hidden.device)
                capture_stream.wait_stream(stream)
                # Nemotron MTP contains attention/MoE, not recurrent Mamba
                # state. Repeating the same slot write during warm-up is
                # idempotent; positions never advance inside execute().
                with torch.cuda.stream(capture_stream):
                    for _ in range(3):
                        execute()
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    output, tokens = execute()
                stream.wait_stream(capture_stream)
                # Hold every captured input/metadata object alive; no entry
                # is published on a failed capture and failures propagate.
                references = (kwargs, metadata, slot_mapping)
                self._entries[draft_index] = (
                    graph,
                    output,
                    tokens,
                    signature,
                    stream,
                    references,
                )
            graph.replay()
            # Hidden output is consumed/copied before this index is replayed.
            # Draft IDs live until stack() and must have independent storage.
            return output, tokens.clone()
