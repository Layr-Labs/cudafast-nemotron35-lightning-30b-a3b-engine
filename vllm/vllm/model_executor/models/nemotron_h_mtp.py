# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import copy
import os
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.parallel import ParallelConfig
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
from vllm.logger import init_logger
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


def _start_projection(
    layer: nn.Module,
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    embeds_projected: bool,
) -> torch.Tensor:
    """eh_proj(cat(enorm(e), hnorm(h))) for the MTP start projections.

    With ``embeds_projected`` the caller passes the token's embedding half
    already projected (``NemotronHMultiTokenPredictor.eh_table``, fp32), so
    only the hidden-state half of eh_proj is read per draft pass. Drafter-only
    numerics: the two halves are summed in fp32 instead of in one chain.
    """
    previous_hidden_states_normed = layer.hnorm(hidden_states)
    if embeds_projected:
        # W_h (14.5 MB) is read by every draft pass of a cycle; ordinary loads
        # let it stay in L2 between passes while the rest stream evict-first.
        if hasattr(layer, "_mtp_eh_pk_sm"):
            h_part = torch.ops.vllm.mtp_packed_linear_keep(
                previous_hidden_states_normed,
                layer._mtp_eh_pk_sm, layer._mtp_eh_pk_ex,
                layer._mtp_eh_pk_base, layer._mtp_eh_pk_ri, layer._mtp_eh_pk_raw,
                layer.eh_w_h.shape[0], layer.eh_w_h.shape[1], layer.eh_w_h,
            )
        else:
            h_part = torch.ops.vllm.mtp_small_linear_keep(
                previous_hidden_states_normed, layer.eh_w_h
            )
        return (h_part.float() + inputs_embeds).to(h_part.dtype)
    inputs_embeds_normed = layer.enorm(inputs_embeds)
    # Fuse via concatenation and linear projection
    fused = torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)
    hidden_states, _ = layer.eh_proj(fused)
    return hidden_states


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
        embeds_projected: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            assert inputs_embeds is not None
            hidden_states = _start_projection(
                self, inputs_embeds, hidden_states, embeds_projected
            )

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
        self._mtp_fold_residual_ok = False
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
        embeds_projected: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            assert inputs_embeds is not None
            hidden_states = _start_projection(
                self, inputs_embeds, hidden_states, embeds_projected
            )

        # MoE (parent forward inlined to gather the routed rows)
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        if self._fast_moe:
            if self.has_end_norm and self._mtp_fold_residual_ok and residual is not None:
                hidden_states = torch.ops.vllm.mtp_moe_single_row_add_residual(
                    hidden_states, self.moe_row_src[: hidden_states.shape[0]],
                    residual, self._fast_moe_name)
                residual = None
            else:
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

        # Drafter-only: the embedding half of the start layer's eh_proj,
        # projected once per vocabulary row (fp32, [vocab, hidden]); set by
        # build_eh_table at load. None keeps the stock path.
        self.register_buffer("eh_table", None, persistent=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def build_eh_table(self) -> bool:
        """Precompute W_e @ enorm(E[t]) for every token t, where W_e is the
        embedding half of eh_proj. The table depends only on checkpoint
        weights (input-independent, built once at load); each draft pass then
        reads one table row per input row and the hidden-state half of eh_proj
        only, instead of the whole [hidden, 2 * hidden] matrix."""
        layer = self.layers["0"]
        emb = getattr(self.embed_tokens, "weight", None)
        w = getattr(getattr(layer, "eh_proj", None), "weight", None)
        if not getattr(layer, "has_start_projections", False) or emb is None or w is None:
            return False
        hidden = emb.shape[1]
        if (
            emb.dim() != 2
            or w.dim() != 2
            or tuple(w.shape) != (hidden, 2 * hidden)
            or w.dtype != torch.bfloat16
            or emb.dtype != torch.bfloat16
            or not w.is_cuda
            or emb.device != w.device
        ):
            return False
        w_e = w[:, :hidden].float().t().contiguous()  # [hidden_in, hidden_out]
        table = torch.empty(
            (emb.shape[0], w.shape[0]), dtype=torch.float32, device=w.device
        )
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            chunk = 8192
            for start in range(0, emb.shape[0], chunk):
                rows = layer.enorm(emb[start : start + chunk])
                torch.mm(rows.float(), w_e, out=table[start : start + chunk])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        del w_e
        layer.register_buffer("eh_w_h", w[:, hidden:].contiguous(), persistent=False)
        from .nemotron_h_mtp_packed import prepare_eh_hidden_packed

        prepare_eh_hidden_packed(layer)
        self.register_buffer("eh_table", table, persistent=False)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        embeds_projected = False
        if inputs_embeds is None:
            if self.eh_table is not None:
                inputs_embeds = self.eh_table[input_ids]
                embeds_projected = True
            else:
                inputs_embeds = self.get_input_embeddings(input_ids)

        residual = None

        for i in range(self.pattern_len):
            hidden_states, residual = self.layers[str(i)](
                inputs_embeds=inputs_embeds,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                embeds_projected=embeds_projected,
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

        # MTP-only: timing-based combo-kernel selection changes RMSNorm's
        # reduction order between boots. Keep the target compilation settings
        # intact and compile the drafter with ONE fixed choice: the combined
        # kernels, never benchmarked against the separate ones. VLLM_MTP_COMBO
        # =off pins the separate kernels instead; =bench restores the timed pick.
        combo = os.environ.get("VLLM_MTP_COMBO", "pin")
        predictor_config = copy.copy(vllm_config)
        predictor_config.compilation_config = copy.copy(vllm_config.compilation_config)
        if combo != "bench":
            predictor_config.compilation_config.inductor_compile_config = {
                **vllm_config.compilation_config.inductor_compile_config,
                "combo_kernels": combo != "off",
                "benchmark_combo_kernel": False,
            }
        # The compile wrapper takes its config from the current context.
        with set_current_vllm_config(predictor_config):
            self.model = NemotronHMultiTokenPredictor(
                vllm_config=predictor_config, prefix=maybe_prefix(prefix, "mtp")
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
        from .nemotron_h_mtp_packed import prepare_mtp_packed

        prepare_mtp_packed(self.model)

    def prepare_draft_head_screen(self):
        """The drafter's greedy head over the shared NVFP4 lm_head (see
        nemotron_h_mtp_screen.py), or None when it does not apply."""
        from .nemotron_h_mtp_screen import DraftHeadScreen, screen_enabled

        if not screen_enabled():
            return None
        spec = getattr(self.vllm_config, "speculative_config", None)
        depth = getattr(spec, "num_speculative_tokens", None)
        try:
            return DraftHeadScreen(
                self.vllm_config.model_config.model, self.lm_head.weight.device,
                depth=depth,
            )
        except Exception as e:  # noqa: BLE001 - any mismatch keeps the stock head
            logger.warning("MTP draft-head screen disabled: %r", e)
            return None

    def prepare_eh_table(self) -> bool:
        """Build the drafter's projected-embedding table once the embeddings
        are shared (see NemotronHMultiTokenPredictor.build_eh_table).
        VLLM_MTP_EH_TABLE=0 keeps the stock eh_proj."""
        if os.environ.get("VLLM_MTP_EH_TABLE", "1") == "0":
            return False
        try:
            built = self.model.build_eh_table()
        except Exception as e:  # noqa: BLE001 - any failure keeps the stock eh_proj
            logger.warning("MTP eh_proj table disabled: %r", e)
            self.model.eh_table = None
            return False
        if built:
            logger.info(
                "MTP eh_proj: projected-embedding table %s built",
                tuple(self.model.eh_table.shape),
            )
        return built

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
        """Compute logits for DRAFT token generation."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        return self.logits_processor(self.lm_head, hidden_states)

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
