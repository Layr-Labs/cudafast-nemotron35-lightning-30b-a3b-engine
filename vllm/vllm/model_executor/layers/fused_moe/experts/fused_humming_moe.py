# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for Humming."""

import copy
import json
import math
from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.humming_tuned_moe import (
    relu2_bf16,
    tuned_moe_configs,
)
from vllm.model_executor.layers.fused_moe.experts.humming_zskip import (
    maybe_down as maybe_zskip_down,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

# l2pf (harness/kernels/l2pf); absent, nothing changes.
try:
    import l2pf as _l2pf
except ImportError:
    _l2pf = None


def _l2pf_next_layer(layer, num_tokens: int) -> bool:
    """LOCAL PATCH (cudafast-nemotron35): small calls only: queue an L2 prefetch
    of the head of the next Mamba layer's in_proj weight (its fp8c buffer or fp8small pack when
    present) on a side stream while the latency-bound routed down projection
    runs. Moves no value; the caller rejoins after the down projection."""
    if _l2pf is None or num_tokens > 8 or _l2pf.MOE_BYTES <= 0:
        return False
    nxt = _l2pf.AHEAD_MOE.get(str(getattr(layer, "layer_name", "")))
    if nxt is None:
        return False
    w = getattr(nxt, "_fp8c_weight", None)  # fp8c buffer: tile-major, read order
    if w is None:
        w = getattr(nxt, "_fp8small_weight", None)
    if w is None:
        w = getattr(nxt, "weight", None)
    queued = _l2pf.prefetch(w, max_bytes=_l2pf.MOE_BYTES)
    return _l2pf_next_ssm_ring(layer) or queued


def _l2pf_next_ssm_ring(layer) -> bool:
    """LOCAL PATCH (cudafast-nemotron35): with the lazy anchor (mamba_ssm_spec_replay), also
    prefetch the following Mamba layer's spec-decode input ring (slot 1, the entries its
    replay reads) and the prompt-length word after its slot-0 state into L2, in the same
    window. The slots are read on the device from the decode metadata. Moves no value."""
    if not getattr(_l2pf, "SSM_RING", False) or not hasattr(_l2pf, "prefetch_slots2"):
        return False
    mixer = _l2pf.AHEAD_SSM.get(str(getattr(layer, "layer_name", "")))
    if mixer is None or not getattr(mixer, "_spec_replay_lazy", False):
        return False
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return False
    meta = attn_metadata.get(getattr(mixer, "prefix", None))
    slots = getattr(meta, "state_indices_tensor_d", None)
    kv = getattr(mixer, "kv_cache", None)
    if slots is None or slots.dim() != 2 or slots.shape[0] != 1 or slots.shape[1] < 2 or not kv:
        return False
    state = kv[1]
    if state.dim() != 4:
        return False
    _, nheads, dim, dstate = state.shape
    ngroups = getattr(mixer, "n_groups", 0) // max(getattr(mixer, "tp_size", 1), 1)
    ring_bytes = mixer.spec_replay_ring_len * (nheads * dim + ngroups * dstate + nheads) * 2
    tag_off = nheads * dim * dstate * state.element_size()
    queued = _l2pf.prefetch_slots2(state, slots[:, 1], 0, ring_bytes, slots[:, 0], tag_off, 16)
    if getattr(_l2pf, "SSM_ANCHOR", False) and hasattr(_l2pf, "prefetch_anchor") and slots.shape[1] >= 3:
        from vllm.model_executor.layers.mamba.ops.mamba_ssm_spec_replay import LAZY_PERIOD

        seq_lens = getattr(meta, "seq_lens", None)
        qsl = getattr(meta, "query_start_loc_d", None)
        if seq_lens is not None and qsl is not None and qsl.numel() >= 2:
            queued = _l2pf.prefetch_anchor(
                state, slots[0], seq_lens, qsl, tag_off, LAZY_PERIOD, tag_off
            ) or queued
    return queued
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.utils.humming import GemmType as HummingGemmType


logger = init_logger(__name__)


# LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"): humming
# estimates GPU memory bandwidth from NVML's memory clock to pick its kernel
# tuning configs. GB10's unified LPDDR5X exposes no memory clock, so NVML raises
# NotSupported and the backend fails to build. Fall back to GB10's published
# 273 GB/s. This only steers the tuning heuristic, never the numerics.
def _patch_humming_bandwidth_for_gb10() -> None:
    try:
        import humming.utils.device as _hd
    except Exception:
        return
    if getattr(_hd.calculate_gpu_bandwidth, "_gb10_fallback", False):
        return
    _orig = _hd.calculate_gpu_bandwidth

    def calculate_gpu_bandwidth(gpu_index=0):
        try:
            return _orig(gpu_index)
        except Exception:
            name = torch.cuda.get_device_name(gpu_index)
            if "GB10" in name:
                return 273.0
            raise

    calculate_gpu_bandwidth._gb10_fallback = True
    _hd.calculate_gpu_bandwidth = calculate_gpu_bandwidth


_patch_humming_bandwidth_for_gb10()


# LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"): let the
# humming MoE run its batch-invariant (deterministic) kernels on their own. The
# upstream switch is the global VLLM_BATCH_INVARIANT, which also forces every
# NVFP4 linear onto CUTLASS W4A4 -- a kernel this checkpoint's W4A16 layers
# cannot use. VLLM_HUMMING_BATCH_INVARIANT=1 turns on humming's flag alone.
def _humming_batch_invariant() -> bool:
    import os

    return bool(envs.VLLM_BATCH_INVARIANT) or os.environ.get(
        "VLLM_HUMMING_BATCH_INVARIANT", "0"
    ) == "1"


# LOCAL PATCH (cudafast-nemotron35, vllm/VENDOR.json "local_patches"): humming's
# heuristic picks num_ctas_per_sm before its batch-invariant step shrinks block K
# to 512 // a_bits, so the small-M entries (decode, verify) keep one 4-warp CTA
# per SM. Give them the second resident CTA the heuristic grants to small CTAs
# whose shared memory fits twice. This only doubles the persistent grid: without
# stream-K every output tile is still computed by one CTA over all K blocks in
# the same order, so the numerics are unchanged.
def _two_ctas_per_sm(tuning_configs: list, meta: Any, gemm_type: Any) -> list:
    from humming.tune import get_heuristics_class
    from humming.utils.smem import estimate_smem_size_layer

    max_smem_size = get_heuristics_class().max_smem_size
    tuning_configs = copy.deepcopy(tuning_configs)
    for _, _, config in tuning_configs:
        block_shape = tuple(config["block_shape"])
        num_warps = math.prod(block_shape) // math.prod(config["warp_shape"])
        smem_size = estimate_smem_size_layer(
            meta, block_shape, gemm_type, config["num_stages"]
        )
        if (
            not config["use_stream_k"]
            and config.get("num_ctas_per_sm", 1) == 1
            and num_warps < 8
            and smem_size * 2 <= max_smem_size
        ):
            config["num_ctas_per_sm"] = 2
    return tuning_configs


def get_humming_moe_gemm_type() -> str:
    env_gemm_type: str | None = envs.VLLM_HUMMING_MOE_GEMM_TYPE
    gemm_type = "indexed"
    if env_gemm_type is not None:
        env_gemm_type = env_gemm_type.lower()
        if env_gemm_type == "indexed":
            gemm_type = env_gemm_type
        elif env_gemm_type in ["grouped_contiguous", "grouped"]:
            gemm_type = "grouped_contiguous"
        else:
            gemm_type = "indexed"

    logger.info_once(f"Using {gemm_type} gemm for humming moe")  # noqa
    return gemm_type


class HummingExpertsBase(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        layer: "RoutedExperts",
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        self.layer = layer
        self.num_experts = self.layer.local_num_experts
        self.global_num_experts = self.layer.global_num_experts
        self.init_humming_moe()

        if self.is_batched():
            assert max_num_tokens is not None and num_dispatchers is not None

        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._permute_scratch: MoEPermuteScratch | None = None

    def init_humming_moe(self):
        from vllm.utils.humming import HummingMethod

        self.compute_config = {
            "use_batch_invariant": _humming_batch_invariant(),
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": self.humming_gemm_type().value,
        }
        # LOCAL PATCH (cudafast-nemotron35): explicit batch-invariant configs
        # for unpadded experts of the tuned shapes (humming_tuned_moe.py).
        tuned = None
        if self.humming_gemm_type().value == "indexed":
            tuned = tuned_moe_configs(self.layer)
        # The same patch fuses relu^2 into one bit-identical kernel.
        self._fused_relu2 = tuned is not None
        if tuned is not None:
            self.w13_tuning_config, self.w2_tuning_config = tuned
        else:
            self.w13_tuning_config = HummingMethod.get_default_tuning_configs(
                layer=self.layer,
                use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
                use_batch_invariant=_humming_batch_invariant(),
                gemm_type=self.humming_gemm_type(),
                sublayer_name="w13",
            )
            self.w2_tuning_config = HummingMethod.get_default_tuning_configs(
                layer=self.layer,
                use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
                use_batch_invariant=_humming_batch_invariant(),
                gemm_type=self.humming_gemm_type(),
                sublayer_name="w2",
            )
            if _humming_batch_invariant():
                metas = self.layer.humming_metas
                gemm_type = self.humming_gemm_type()
                self.w13_tuning_config = _two_ctas_per_sm(
                    self.w13_tuning_config, metas["w13"], gemm_type
                )
                self.w2_tuning_config = _two_ctas_per_sm(
                    self.w2_tuning_config, metas["w2"], gemm_type
                )
        self.compute_config_str = json.dumps(self.compute_config)
        self.w13_tuning_config_str = json.dumps(self.w13_tuning_config)
        self.w2_tuning_config_str = json.dumps(self.w2_tuning_config)

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=self.moe_config.max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
                hidden_size=self.moe_config.hidden_dim,
                hidden_dtype=self.moe_config.in_dtype,
            )
        return self._permute_scratch

    def get_global_valid_shape_m(self, topk_ids: torch.Tensor):
        num_tokens = topk_ids.size(0)
        ctx = get_forward_context()
        if ctx.dp_metadata is not None:
            num_tokens = ctx.dp_metadata.num_tokens_across_dp_cpu.sum().item()

        return num_tokens * topk_ids.size(1)

    def estimate_local_valid_shape_m(self, topk_ids: torch.Tensor):
        # estimate shape_m for kernel tuning
        global_valid_shape_m = self.get_global_valid_shape_m(topk_ids)
        num_experts = self.num_experts
        global_num_experts = self.global_num_experts
        return math.ceil(global_valid_shape_m * num_experts / global_num_experts)

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        raise NotImplementedError

    @classmethod
    def is_batched(cls) -> bool:
        return cls.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, kFp8DynamicTokenSym),
            (kNvfp4Static, None),
            (kNvfp4Static, kFp8DynamicTokenSym),
            (kMxfp8Static, None),
            (kMxfp8Static, kFp8DynamicTokenSym),
            (kFp8StaticChannelSym, None),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8Static128BlockSym, None),
            (kFp8Static128BlockSym, kFp8DynamicTokenSym),
            (kInt4Static, None),
            (kInt4Static, kFp8DynamicTokenSym),
            (kInt8Static, None),
            (kInt8Static, kFp8DynamicTokenSym),
            # Checkpoint-driven (weight, activation) pairs the dense/MoE oracles
            # pass. Humming defers input quant (see expects_unquantized_inputs),
            # so the activation key does not constrain support.
            # fp8 (compressed-tensors / native / modelopt)
            (kFp8StaticChannelSym, kFp8StaticTensorSym),
            (kFp8StaticChannelSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8Dynamic128Sym),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            # int8 (compressed-tensors w8a8 / experts_int8)
            (kInt8StaticChannelSym, None),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
            # nvfp4 (compressed-tensors / modelopt / quark)
            (kNvfp4Static, kNvfp4Dynamic),
            # mxfp8 (compressed-tensors / modelopt / online)
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Humming kernels handle input quantization internally via
        HummingMethod.may_quant_input() in the apply() method.

        This property tells the prepare/finalize step to skip input
        quantization (by setting defer_input_quant=True) and pass
        unquantized inputs to the experts. This prevents double
        quantization: once in prepare and once in Humming's apply().

        Returns:
            True to indicate that this expert expects unquantized inputs
            and will handle quantization internally.
        """
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        platform = current_platform
        return (
            has_humming()
            and platform.is_cuda()
            and platform.has_device_capability((7, 5))
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Humming uses apply_moe_activation() callback for activation,
        # so any activation supported there can be used here.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        from vllm.utils.humming import HummingLayerMeta

        meta1: HummingLayerMeta = self.layer.humming_metas["w13"]
        meta2: HummingLayerMeta = self.layer.humming_metas["w2"]

        assert meta1.num_experts == meta2.num_experts

        num_experts = meta1.num_experts
        top_k = topk_ids.size(1)
        assert w1.size(0) == num_experts
        assert w2.size(0) == num_experts

        if not self.is_batched():
            num_tokens = a1.size(0)
            assert topk_ids.size(0) == num_tokens
        else:
            assert a1.dim() == 3
            assert a1.size(0) == num_experts
            num_tokens = a1.size(1)

        return (
            meta1.num_experts,
            num_tokens,
            # Logical intermediate width for both gated and non-gated activations
            self.layer.intermediate_size_per_partition,
            meta1.shape_k,
            top_k,
        )

    def get_buffer_metas(self, M: int, topk: int, activation: MoEActivation):
        from vllm.utils.humming import GemmType as HummingGemmType
        from vllm.utils.humming import dtypes

        num_experts = self.num_experts
        N = self.layer.intermediate_size_per_partition
        K = self.layer.hidden_size
        assert isinstance(num_experts, int)
        assert isinstance(N, int)
        assert isinstance(K, int)

        # hidden_states
        # (-> quanted_gate_up_input) (if not BF16/FP16 activation)
        # -> gate_up_output
        # -> activation_output
        # (-> quanted_down_input) (if not BF16/FP16 activation)
        # -> down_output
        # (-> output) (if not is_batched)
        # Neighboring nodes are required to utilize distinct workspaces.
        # The output must be derived from workspace1.

        output_shape: tuple[int, ...]
        if self.is_batched():
            max_num_tokens = self.max_num_tokens
            num_dispatchers = self.num_dispatchers
            assert max_num_tokens is not None and num_dispatchers is not None
            input_shape_m = num_experts * max_num_tokens
            real_shape_m = num_experts * max_num_tokens * num_dispatchers
            output_shape = (num_experts, max_num_tokens * num_dispatchers, K)
        else:
            input_shape_m = M
            if self.humming_gemm_type() != HummingGemmType.INDEXED:
                input_shape_m = M * topk
            real_shape_m = M * topk
            output_shape = (M, K)

        gate_up_size = N * (2 if activation.is_gated else 1)
        down_input_size = N
        a_dtype = self.layer.humming_metas["w13"].a_dtype
        c_dtype = self.layer.humming_metas["w13"].c_dtype
        num_bits = a_dtype.num_bits
        torch_dtype_map = {
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.float32: torch.float32,
            dtypes.float8e4m3: torch.float8_e4m3fn,
            dtypes.float8e5m2: torch.float8_e5m2,
            dtypes.int8: torch.int8,
            dtypes.int4: torch.uint8,
        }

        buffer_metas = {
            "quanted_gate_up_input": {
                "shape": (input_shape_m, K),
                "dtype": torch_dtype_map[a_dtype],
            },
            "gate_up_output": {
                "shape": (real_shape_m, gate_up_size),
                "dtype": torch_dtype_map[c_dtype],
            },
            "activation_output": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[c_dtype],
            },
            "quanted_down_input": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[a_dtype],
            },
            "down_output": {
                "shape": output_shape if self.is_batched() else (real_shape_m, K),
                "dtype": torch_dtype_map[c_dtype],
            },
            "output": {
                "shape": output_shape,
                "dtype": torch_dtype_map[c_dtype],
            },
        }

        for key in buffer_metas:
            meta = buffer_metas[key]
            if "quanted" in key and a_dtype.num_bits == 4:
                last_dim = meta["shape"][-1]
                if last_dim % 2 != 0:
                    raise ValueError(
                        f"Int4 packing requires last dimension to be even, "
                        f"got {last_dim} for buffer '{key}'"
                    )
                meta["shape"] = meta["shape"][:-1] + (last_dim // 2,)

        if num_bits == 16:
            required_buffers = ["gate_up_output", "activation_output", "down_output"]
        else:
            required_buffers = [
                "quanted_gate_up_input",
                "gate_up_output",
                "activation_output",
                "quanted_down_input",
                "down_output",
            ]

        # batched moe use down_output as output
        if not self.is_batched():
            required_buffers.append("output")

        return buffer_metas, required_buffers

    def _workspace_shapes(self, M: int, topk: int, activation: MoEActivation):
        buffer_metas, required_buffers = self.get_buffer_metas(M, topk, activation)

        workspace1_nbytes = 0
        workspace2_nbytes = 0

        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            nelement = math.prod(buffer_meta["shape"])
            nbytes = nelement * buffer_meta["dtype"].itemsize
            if index % 2 == 0:
                workspace1_nbytes = max(workspace1_nbytes, nbytes)
            else:
                workspace2_nbytes = max(workspace2_nbytes, nbytes)

        output_key = "down_output" if self.is_batched() else "output"
        output_shape = buffer_metas[output_key]["shape"]
        elem_size = self.layer.params_dtype.itemsize

        return (
            (workspace1_nbytes // elem_size,),
            (workspace2_nbytes // elem_size,),
            output_shape,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self._workspace_shapes(M, topk, activation)

    def make_workspaces(self, M: int, topk: int, activation: MoEActivation):
        shapes = self._workspace_shapes(M, topk, activation)
        workspace1_shape, workspace2_shape, output_shape = shapes
        torch_dtype = self.layer.params_dtype
        workspace1, workspace2 = current_workspace_manager().get_simultaneous(
            (workspace1_shape, torch_dtype),
            (workspace2_shape, torch_dtype),
        )
        output = _resize_cache(workspace1, output_shape)
        return workspace1, workspace2, output

    def prepare_buffers(
        self,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        M: int,
        topk: int,
        activation: MoEActivation,
    ) -> dict[str, torch.Tensor]:
        buffer_metas, required_buffers = self.get_buffer_metas(M, topk, activation)
        buffers = {}
        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            workspace = workspace1 if index % 2 == 0 else workspace2
            workspace = workspace.view(buffer_meta["dtype"])
            buffers[name] = _resize_cache(workspace, buffer_meta["shape"])

        return buffers

    # Note: apply method is implemented by subclasses following the
    # standard FusedMoEExpertsModular.apply signature

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExpertsModular.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

        if supported:
            assert hasattr(cls, "humming_gemm_type")
            gemm_type = cls.humming_gemm_type().value.lower()
            preferred_gemm_type = get_humming_moe_gemm_type()
            supported = preferred_gemm_type.lower() == gemm_type
            if not supported:
                reason = (
                    f"preferred gemm type {preferred_gemm_type} != "
                    f"supported gemm type {gemm_type}"
                )

        return supported, reason

    def apply_activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> None:
        swiglu_limit = self.quant_config.gemm1_clamp_limit
        if (
            activation == MoEActivation.RELU2_NO_MUL
            and getattr(self, "_fused_relu2", False)
            and relu2_bf16(input, output)
        ):
            return
        if activation == MoEActivation.SILU and swiglu_limit is not None:
            swiglu_limit_func(output=output, input=input, swiglu_limit=swiglu_limit)
        else:
            self.activation(activation=activation, input=input, output=output)


class HummingIndexedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.INDEXED

    def prepare_humming_moe_kwargs(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        moe_block_size = None
        for min_shape_m, max_shape_m, config in self.w13_tuning_config:
            if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
                moe_block_size = config["block_shape"][0]
                break

        if moe_block_size is None:
            logger.warning_once(
                "No tuning config found for shape %s, using default block_size=64",
                valid_shape_m,
            )
            moe_block_size = 64

        self._moe_block_size = moe_block_size
        sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=moe_block_size,
            num_experts=self.global_num_experts,
            expert_map=expert_map,
            ignore_invalid_experts=True,
        )

        moe_common_kwargs = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "compute_config": self.compute_config_str,
            "valid_shape_m": valid_shape_m,
        }

        top_k = topk_ids.size(1)
        moe_kwargs1 = {"top_k": top_k, "tuning_config": self.w13_tuning_config_str}
        moe_kwargs2 = {"top_k": 1, "tuning_config": self.w2_tuning_config_str}
        moe_kwargs1.update(moe_common_kwargs)
        moe_kwargs2.update(moe_common_kwargs)

        return moe_kwargs1, moe_kwargs2

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Standard apply implementation for Humming indexed experts.

        Note: Humming kernels handle weights and quantization internally through
        the layer object, so w1, w2, a1q_scale, a2_scale parameters are not used.
        The output is written into workspace13 via the buffer management.
        """
        from vllm.utils.humming import HummingMethod

        assert not apply_router_weight_on_input

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        moe_kwargs1, moe_kwargs2 = self.prepare_humming_moe_kwargs(
            topk_ids=topk_ids,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        up_masks = None
        # The tuned NVFP4/BF16 path retains the original GEMM output rounding
        # inside its epilogue, then writes relu(output)**2 directly to the
        # activation buffer. Other shapes, formats and activations keep the
        # ordinary projection and activation calls.
        if (
            activation == MoEActivation.RELU2_NO_MUL
            and getattr(self, "_fused_relu2", False)
            and inputs.dtype == torch.bfloat16
            and input_scale is None
        ):
            from vllm.model_executor.layers.fused_moe.experts.humming_relu2_epilogue import (  # noqa: E501
                forward_relu2,
            )

            if expert_map is None and topk_ids.size(0) == 5 and topk_ids.size(1) == 6:
                from vllm.model_executor.layers.fused_moe.experts.humming_produced_masks import try_launch

                up_masks = try_launch(self.layer, inputs, buffers["activation_output"], moe_kwargs1)
            if up_masks is None:
                forward_relu2(
                    layer=self.layer,
                    inputs=inputs,
                    outputs=buffers["activation_output"],
                    **moe_kwargs1,
                )
        else:
            HummingMethod.forward_layer(
                layer=self.layer,
                inputs=inputs,
                input_scale=input_scale,
                outputs=buffers["gate_up_output"],
                sublayer_name="w13",
                **moe_kwargs1,
            )

            self.apply_activation(
                activation=activation,
                input=buffers["gate_up_output"],
                output=buffers["activation_output"],
            )

        # LOCAL PATCH (cudafast-nemotron35): the routed up GEMM is queued; let
        # the shared expert's deferred down projection follow it on the aux
        # stream (runner/shared_experts.py; nothing pending: no-op).
        if topk_ids.size(0) <= 8:
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                launch_pending_shared_down,
            )

            launch_pending_shared_down()

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        down_output = buffers["down_output"].view(-1, hidden_states.size(-1))
        l2pf_pending = expert_map is None and _l2pf_next_layer(
            self.layer, topk_ids.size(0)
        )
        # LOCAL PATCH (cudafast-nemotron35): zskip computes humming's w2 output
        # for <= 8 tokens, and for prefill on the sparse layers, without loading
        # the weight rows whose activations are zero (humming_zskip.py);
        # otherwise routed_sparse or humming runs.
        sparse_done = expert_map is None and maybe_zskip_down(
            self.layer,
            inputs,
            input_scale,
            down_output,
            topk_ids.size(0),
            topk_ids.size(1),
            getattr(self, "_moe_block_size", None),
            moe_kwargs2,
            topk_ids=topk_ids,
            producer_masks=(up_masks if input_scale is None and inputs is buffers["activation_output"] else None),
        )
        sparse_weights = getattr(self.layer, "routed_sparse_down", None)
        if (
            not sparse_done
            and sparse_weights is not None
            and input_scale is None
            and expert_map is None
            and _humming_batch_invariant()
        ):
            from routed_sparse import forward

            sparse_done = forward(inputs, topk_ids, down_output, sparse_weights)
        if not sparse_done:
            HummingMethod.forward_layer(
                layer=self.layer,
                inputs=inputs,
                input_scale=input_scale,
                outputs=down_output,
                sublayer_name="w2",
                **moe_kwargs2,
            )
        if l2pf_pending:
            _l2pf.join(down_output.device)

        moe_fused_mul_sum(
            inputs=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            outputs=output,
        )


class HummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.GROUPED_CONTIGUOUS

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Standard apply implementation for Humming grouped experts.

        Note: Humming kernels handle weights and quantization internally through
        the layer object, so w1, w2, a1q_scale, a2_scale parameters are not used.
        The output is written into workspace13 via the buffer management.
        """
        from vllm.utils.humming import HummingMethod

        assert not apply_router_weight_on_input

        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        hidden_states, _, expert_first_token_offset, inv_perm, _ = moe_permute(
            hidden_states=hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=global_num_experts,
            n_local_expert=self.num_experts,
            expert_map=expert_map,
            scratch=self._get_permute_scratch(),
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
            sublayer_name="w13",
        )

        self.apply_activation(
            activation=activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
            sublayer_name="w2",
        )

        moe_unpermute(
            out=output,
            permuted_hidden_states=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_first_token_offset,
        )


class BatchedHummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.GROUPED_MASKED

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Standard apply implementation for Humming batched grouped experts.

        Note: Humming kernels handle weights and quantization internally through
        the layer object, so w1, w2, a1q_scale, a2_scale parameters are not used.
        The output is written into workspace13 via the buffer management.
        """
        from vllm.utils.humming import HummingMethod

        assert not apply_router_weight_on_input
        assert expert_tokens_meta is not None

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
            sublayer_name="w13",
        )

        self.apply_activation(
            activation=activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=output.view(-1, hidden_states.size(-1)),
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
            sublayer_name="w2",
        )
