# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import os
import types
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

from .interfaces import SupportsPP
from .nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMoEDecoderLayer,
)
from .nemotron_h_mtp_kernels import (
    install_mtp_small_linear,
    prepare_mtp_relu2,
    mtp_fast_enabled as _mtp_fast_enabled,
    register_mtp_moe_layer,
)

logger = init_logger(__name__)

# Drafter vocabulary restriction (affects draft proposals only).
# The MTP drafter shares the target's lm_head. Draft argmax is taken over the
# leading MTP_DRAFT_VOCAB token ids only: the drafter gets its own copy of rows
# [0, N) of that same head (same quantized values and scales, sliced at load
# time, no re-quantization). Target logits, verification and sampling still use
# the full shared head, so the emitted tokens are unchanged; a token id >= N
# is never proposed, only produced by the target. The subset is fixed by id
# order (prompt-independent). Read at import; a value >= vocab_size turns the
# restriction off. Module-level so in-process screening can switch it.
MTP_DRAFT_VOCAB: int = int(os.environ.get("VLLM_MTP_DRAFT_VOCAB", "81920"))


def _truncate_lm_head(lm_head: nn.Module, n: int) -> object | None:
    """Return a head computing rows [0, n) of lm_head, or None if the head's
    weight format is not supported. The returned object reuses lm_head's
    quant_method; its tensors are copies of the leading n output rows."""
    qm = getattr(lm_head, "quant_method", None)
    weight = getattr(lm_head, "weight", None)
    if qm is None or not isinstance(weight, torch.Tensor):
        return None
    if getattr(lm_head, "tp_size", 1) != 1 or getattr(lm_head, "bias", None) is not None:
        return None
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    if isinstance(qm, UnquantizedEmbeddingMethod):
        # weight: [vocab, hidden]
        return types.SimpleNamespace(
            quant_method=qm, tp_size=1, bias=None, weight=weight[:n].contiguous()
        )
    kernel = getattr(qm, "kernel", None)
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        MarlinNvFp4LinearKernel,
    )

    if not isinstance(kernel, MarlinNvFp4LinearKernel):
        return None
    # Marlin NVFP4 (W4A16) layout after repack, size_n = vocab (no padding):
    #   weight       int32 [size_k / 16, size_n * 2]: one row per 16-deep k
    #                tile, 64-wide n tiles laid out in n order (128 int32 each)
    #   weight_scale fp8   [size_k / 16, size_n]: permuted within 64-col groups
    # so the leading n output rows (n % 64 == 0) are the leading columns of
    # each packed row. Global scale and workspace are shared unchanged.
    size_n = lm_head.output_size_per_partition
    size_k = lm_head.input_size_per_partition
    ws = lm_head.weight_scale
    if (
        n % 64 != 0
        or weight.dim() != 2
        or weight.dtype != torch.int32
        or tuple(weight.shape) != (size_k // 16, size_n * 2)
        or tuple(ws.shape) != (size_k // 16, size_n)
    ):
        return None
    return types.SimpleNamespace(
        quant_method=qm,
        tp_size=1,
        bias=None,
        weight=weight[:, : n * 2].contiguous(),
        weight_scale=ws[:, :n].contiguous(),
        weight_global_scale=lm_head.weight_global_scale,
        workspace=lm_head.workspace,
        output_size_per_partition=n,
        input_size_per_partition=size_k,
    )



class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            assert inputs_embeds is not None
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (Attention)
        # Parent forward expects: hidden_states, residual
        hidden_states, residual = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

        # Row source for the MoE input. The drafter only samples one row per
        # request, so the proposer points every row of a request at that row.
        # All rows then route to the same experts and the MoE reads only those
        # experts' weights. The kernels are row-independent, so the sampled
        # row's output is bit-identical. Identity by default.
        max_rows = (
            model_config.max_model_len if model_config is not None else 8192
        )
        from vllm.config import get_current_vllm_config

        try:
            max_rows = max(
                max_rows,
                get_current_vllm_config().scheduler_config.max_num_batched_tokens,
            )
        except Exception:
            pass
        self.register_buffer(
            "moe_row_src", torch.arange(max_rows, dtype=torch.long), persistent=False
        )

        # Single-row MoE: with one request in flight every MoE input row is
        # the same row (see above), so the MoE runs once on that row and the
        # result is written to every output row. Drafter-only numerics.
        self._fast_moe = False
        self._fast_moe_weights = None
        self._fast_moe_name = prefix
        try:
            max_num_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
        except Exception:
            max_num_seqs = None
        if _mtp_fast_enabled("MOE") and max_num_seqs == 1:
            register_mtp_moe_layer(prefix, self)
            self._fast_moe = True

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            assert inputs_embeds is not None
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # MoE (parent forward inlined to gather the routed rows)
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        if self._fast_moe:
            hidden_states = torch.ops.vllm.mtp_moe_single_row(
                hidden_states,
                self.moe_row_src[: hidden_states.shape[0]],
                self._fast_moe_name,
            )
        else:
            hidden_states = hidden_states.index_select(
                0, self.moe_row_src[: hidden_states.shape[0]]
            )
            hidden_states = self.mixer(hidden_states)

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


@support_torch_compile
class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with NemotronH layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        assert self.num_mtp_layers == 1, (
            "Only one MTP layer is supported for NemotronH-MTP"
        )

        self.pattern_str = config.mtp_hybrid_override_pattern
        self.pattern_len = len(self.pattern_str)
        assert self.pattern_len > 0

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Build flat list of layers
        self.layers = torch.nn.ModuleDict()

        # Total number of physical layers = num_steps * pattern_len
        total_layers = self.num_mtp_layers * self.pattern_len
        for i in range(total_layers):
            step_rel_idx = i % self.pattern_len

            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            layer_prefix = f"{prefix}.layers.{i}"

            # TODO smor- remove double layers formation
            common_kwargs = dict(
                config=config,
                layer_idx=self.mtp_start_layer_idx + i,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                parallel_config=vllm_config.parallel_config,
                prefix=layer_prefix,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
            )

            if char == "*":
                self.layers[str(i)] = NemotronHMTPAttentionDecoderLayer(**common_kwargs)
            elif char == "E":
                self.layers[str(i)] = NemotronHMTPMoEDecoderLayer(**common_kwargs)
            else:
                raise NotImplementedError(
                    f"Pattern char '{char}' in {self.pattern_str} not implemented"
                )

        self.make_empty_intermediate_tensors: Callable[..., IntermediateTensors] = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        residual = None

        for i in range(self.pattern_len):
            hidden_states, residual = self.layers[str(i)](
                inputs_embeds=inputs_embeds,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        return hidden_states


class NemotronHMTP(nn.Module, SupportsPP):
    """NemotronH MTP model."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        # Needed for load_weights mapping
        self.mtp_start_layer_idx = config.num_hidden_layers

        # EPLB config for experts
        self.num_redundant_experts = 0
        if vllm_config.parallel_config and vllm_config.parallel_config.eplb_config:
            self.num_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )

        # MTP predictor
        self.model = NemotronHMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )
        # Small-M BF16 GEMMs for the head's dense layers (drafter only).
        if _mtp_fast_enabled("DENSE"):
            install_mtp_small_linear(self.model)

        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def process_weights_after_loading(self) -> None:
        prepare_mtp_relu2(self.model)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def _moe_row_src_buffers(self) -> list[torch.Tensor]:
        bufs = getattr(self, "_moe_row_src_cache", None)
        if bufs is None:
            bufs = [
                m.moe_row_src
                for m in self.model.layers.values()
                if isinstance(m, NemotronHMTPMoEDecoderLayer)
            ]
            self._moe_row_src_cache = bufs
        return bufs

    def set_moe_row_src(self, num_rows: int, src_row: torch.Tensor | None) -> None:
        """Point MoE rows [0, num_rows) at src_row (a 1-element GPU tensor),
        or at the identity when src_row is None."""
        for buf in self._moe_row_src_buffers():
            if src_row is None:
                buf[:num_rows].copy_(self._arange_like(buf, num_rows))
            else:
                buf[:num_rows].copy_(src_row.reshape(1).expand(num_rows))

    def _arange_like(self, buf: torch.Tensor, n: int) -> torch.Tensor:
        ar = getattr(self, "_moe_row_arange", None)
        if ar is None or ar.numel() < buf.numel():
            ar = torch.arange(buf.numel(), dtype=buf.dtype, device=buf.device)
            self._moe_row_arange = ar
        return ar[:n]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation (draft ids [0, N) only,
        N = MTP_DRAFT_VOCAB; see restrict_draft_vocab)."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        return self.logits_processor(self._draft_head(), hidden_states)

    def _draft_head(self):
        n = MTP_DRAFT_VOCAB
        if n >= self.config.vocab_size:
            return self.lm_head
        heads = self.__dict__.setdefault("_draft_heads", {})
        head = heads.get(n)
        if head is None:
            head = self.restrict_draft_vocab(n)
        return self.lm_head if head is False else head

    @torch.inference_mode()
    def restrict_draft_vocab(self, n: int | None = None):
        """Build the drafter-only head for the leading n token ids: a copy of
        rows [0, n) of the shared lm_head (same quantized weights and scales,
        no re-quantization). Called once after the target lm_head is shared;
        the target keeps using the full head. Returns the head, or False when
        the head format is not supported (the full head is then used)."""
        n = MTP_DRAFT_VOCAB if n is None else n
        heads = self.__dict__.setdefault("_draft_heads", {})
        if n >= self.config.vocab_size:
            return False
        if n in heads:
            return heads[n]
        head = _truncate_lm_head(self.lm_head, n) or False
        if head is False:
            logger.warning(
                "MTP draft vocab restriction to %d ids not applied: "
                "unsupported lm_head format.",
                n,
            )
        else:
            # Check the slice against the full head on fixed random inputs:
            # the leading n logits must agree (layout sanity check).
            w = self.lm_head.weight
            g = torch.Generator(device=w.device).manual_seed(0)
            x = torch.randn(
                4,
                self.config.hidden_size,
                generator=g,
                device=w.device,
                dtype=self.vllm_config.model_config.dtype,
            )

            def logits(h) -> torch.Tensor:
                # one M=4 call plus four M=1 calls (the drafter samples 1 row)
                out = [self.logits_processor(h, x)]
                out += [self.logits_processor(h, x[i : i + 1]) for i in range(4)]
                return torch.cat(out).float()

            full = logits(self.lm_head)[:, :n]
            part = logits(head)
            err = (full - part).abs().max().item()
            scale = full.abs().max().item()
            ndiff = int((full != part).sum().item())
            same_argmax = bool((full.argmax(-1) == part.argmax(-1)).all())
            if not (err <= 1e-2 * max(scale, 1e-6) and same_argmax):
                logger.warning(
                    "MTP draft vocab restriction to %d ids disabled: sliced "
                    "head disagrees with the full head (max err %.4g, scale "
                    "%.4g).",
                    n,
                    err,
                    scale,
                )
                head = False
            else:
                logger.info(
                    "MTP drafter proposes token ids [0, %d) only (sliced copy "
                    "of the shared lm_head; target head unchanged). Check "
                    "(M=4 and M=1): %d of %d leading logits differ from the "
                    "full head, max abs err %.3g.",
                    n,
                    ndiff,
                    full.numel(),
                    err,
                )
        heads[n] = head
        return head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = []
        num_experts = getattr(self.config, "n_routed_experts", None)
        if getattr(self.config, "model_type", None) == "nemotron_h_puzzle":
            num_experts = self.config.mtp_n_routed_experts
        if num_experts is not None:
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",  # Empty - non-gated MoE
                num_experts=num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Only process MTP weights - skip all non-MTP weights
            if not name.startswith("mtp.") and "embeddings" not in name:
                continue
            # Skip rotary embeddings (computed, not loaded)
            if "rotary_emb.inv_freq" in name:
                continue

            name = name.replace("mtp.layers.", "model.layers.")

            if "embeddings" in name:
                name = name.replace("embeddings", "embed_tokens")
                if name.startswith("backbone."):
                    name = name.replace("backbone.", "model.")

            # Handle stacked parameters (qkv_proj) for attention layers
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Must be in a mixer (attention layer)
                if ".mixer." not in name:
                    continue

                is_stacked = True
                stacked_name = name.replace(weight_name, param_name)

                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue

                if stacked_name not in params_dict:
                    # Might be that mapping failed or param doesn't exist
                    continue

                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(stacked_name)
                break

            if is_stacked:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                # weight_name is like "experts.0.up_proj."
                if weight_name not in name:
                    continue

                is_expert_weight = True

                # Replace the expert-specific weight name with fused parameter name
                # e.g., "experts.0.up_proj." -> "experts.w13_"
                name_mapped = name.replace(weight_name, param_name)

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                break

            if is_expert_weight:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
