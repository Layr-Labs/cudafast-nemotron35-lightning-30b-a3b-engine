# SPDX-License-Identifier: Apache-2.0
"""Optional post-CMP2, actual-weight-qualified FP8C mixed-six/seven packets.

Production prefetch and GEMV call the same metadata-only selector. No package
is published before its private remote admission and stream drain complete.
Missing library, rejected shape/owner/configuration, or no winning mixed tile
coverage leaves the original seven-bit implementation intact. Production
launch errors are visible and never retried after a possibly written output.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import weakref

import torch

from .codec import encode

LOG = logging.getLogger(__name__)
ENABLED = os.environ.get("FP8MIX", "1") != "0"
SHAPES = {(10304, 2688), (2688, 4096)}
_lib = None
_load_attempted = False
_entries = {}
stats = {"admitted": 0, "refused": 0, "c6": 0, "c7": 0}


def _load():
    global _lib, _load_attempted
    if _load_attempted:
        return _lib
    _load_attempted = True
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libfp8mix.so")))
        lib.fp8mix_has_pdl.argtypes = []
        lib.fp8mix_has_pdl.restype = ctypes.c_int
        lib.fp8mix_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.fp8mix_run.restype = ctypes.c_int
        lib.fp8mix_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.fp8mix_decode.restype = ctypes.c_int
        lib.fp8mix_publish_run.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.fp8mix_publish_run.restype = ctypes.c_int
        _lib = lib
    except (OSError, AttributeError):
        _lib = None
    return _lib


def _key(t):
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        return None
    try:
        # Inference tensors cannot provide the mutation-version proof used by
        # this optional immutable-weight cache; refusing them keeps old FP8C.
        version = t._version
    except RuntimeError:
        return None
    return (id(t), t.data_ptr(), t.untyped_storage().data_ptr(), t.storage_offset(),
            tuple(t.shape), tuple(t.stride()), t.dtype, t.device, version)


def _config_key(config):
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    model = config.model_config
    return (parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
            getattr(parallel, "data_parallel_size", 1),
            getattr(parallel, "decode_context_parallel_size", 1),
            getattr(parallel, "prefill_context_parallel_size", 1),
            getattr(parallel, "enable_expert_parallel", False),
            getattr(parallel, "enable_eplb", False),
            getattr(parallel, "enable_dbo", False), scheduler.max_num_seqs,
            model.hf_config.model_type, model.dtype, config.lora_config is None,
            getattr(config, "weight_transfer_config", None) is None,
            config.offload_config.uva.cpu_offload_gb,
            config.offload_config.prefetch.offload_group_size,
            getattr(parallel, "use_ubatching", False))


def _eligible_config(config):
    key = _config_key(config)
    return (key[:5] == (1, 1, 1, 1, 1) and key[5:8] == (False, False, False)
            and key[8] == 1 and key[9] == "nemotron_h"
            and key[10] == torch.bfloat16 and key[11] and key[12]
            and key[13:] == (0, 0, False))


class _Entry:
    pass


def _ring(fp8c, shape):
    return fp8c.RING or fp8c._RING_DEFAULT[shape]


def _selection(comp):
    ref = _entries.get(id(comp))
    entry = ref() if ref is not None else None
    if entry is None:
        return None
    layer, model, kernel = entry.layer(), entry.model(), entry.kernel()
    packet = entry.packet()
    if (layer is None or model is None or kernel is None or packet is None
            or entry.comp() is not comp or getattr(layer, "_fp8mix_entry", None) is not entry
            or getattr(layer, "_fp8mix_packet", None) is not packet
            or getattr(layer, "_fp8c_weight", None) is not comp
            or kernel._fp8c_weight is not comp
            or getattr(getattr(layer, "quant_method", None), "fp8_linear", None) is not kernel
            or _config_key(entry.config) != entry.config_key):
        return None
    fp8c, l2pf = entry.fp8c, entry.l2pf
    if (fp8c._lib is not entry.reference_lib or _lib is not entry.native_lib
            or _ring(fp8c, entry.shape) != entry.ring
            or bool(fp8c._pdl) not in entry.pdl_modes
            or entry.prefetch_key_fn(l2pf) != entry.prefetch_key):
        return None
    params = kernel._get_layer_params(layer)
    current = (comp, packet, params[0], params[1], params[2])
    if any(a is not b() or _key(a) != expected
           for a, b, expected in zip(current, entry.owners, entry.keys)):
        return None
    return entry


def select(comp):
    """Exactly the same selection for every original prefetch and consumer.

    The result is an offset-zero, dedicated allocation, because old l2pf reads
    the complete untyped storage rather than a tensor view's element range.
    """
    entry = _selection(comp)
    return comp if entry is None else entry.packet()


def _run(a, packet, sa, sb, out, n, k, ring, pdl, attrs=None):
    rc = _lib.fp8mix_run(
        a.data_ptr(), packet.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(),
        a.shape[0], n, k, ring, int(pdl),
        torch.cuda.current_stream(a.device).cuda_stream, attrs)
    if rc:
        raise RuntimeError(f"FP8MIX native refusal/error {rc}; no production retry")


def _decode(packet, n, k):
    out = torch.empty((n, k), dtype=torch.uint8, device=packet.device)
    rc = _lib.fp8mix_decode(packet.data_ptr(), out.data_ptr(), n, k,
                            torch.cuda.current_stream(packet.device).cuda_stream)
    if rc:
        raise RuntimeError(f"FP8MIX decoder launch failed {rc}")
    return out


def maybe_gemm(a, comp, sa, sb, n, k):
    # Called only inside the complete original fp8c_scaled_mm guard. Do not add
    # a consumer-only M/scale/input-pointer gate that would disagree with l2pf.
    entry = _selection(comp)
    if entry is None:
        return None
    out = torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)
    _run(a, entry.packet(), sa, sb, out, n, k, entry.ring, bool(entry.fp8c._pdl))
    return out


@torch.no_grad()
def prepare_model(model, config):
    """Only after all target/drafter packing and CMP2, before compile/capture.

    The new persistent packet is deliberately NOT assumed to inherit CMP2's
    physical-memory compression. Old original weights/FP8C remain available.
    """
    if (not ENABLED or _load() is None or not _eligible_config(config)
            or type(model).__name__ != "NemotronHForCausalLM"
            or torch.cuda.is_current_stream_capturing()):
        return
    import fp8c
    import l2pf
    from .admission import prepare, prefetch_key

    for name, layer in model.named_modules():
        if not name.endswith((".in_proj", ".out_proj")):
            continue
        if getattr(layer, "_fp8mix_entry", None) is not None:
            continue
        kernel = getattr(getattr(layer, "quant_method", None), "fp8_linear", None)
        if type(kernel).__name__ != "CutlassFP8ScaledMMLinearKernel":
            continue
        comp = getattr(layer, "_fp8c_weight", None)
        if comp is None or kernel._fp8c_weight is not comp:
            continue
        b, sb, sa, _ = kernel._get_layer_params(layer)
        if (any(_key(x) is None for x in (comp, b, sa, sb))
                or b.dtype != torch.float8_e4m3fn or b.ndim != 2
                or (b.shape[1], b.shape[0]) not in SHAPES
                or tuple(b.stride()) != (1, b.shape[0])
                or sa.dtype != torch.float32 or sb.dtype != torch.float32
                or sa.numel() != 1 or sb.numel() != 1
                or not sa.is_contiguous() or not sb.is_contiguous()
                or comp.dtype != torch.uint8 or not comp.is_contiguous()
                or comp.storage_offset() != 0 or comp.data_ptr() % 256
                or any(x.device != comp.device for x in (b, sa, sb))
                or torch.cuda.get_device_capability(comp.device) != (12, 1)):
            continue
        shape = (b.shape[1], b.shape[0])
        ring = _ring(fp8c, shape)
        if ring not in (3, 4, 6) or (fp8c._pdl and not _lib.fp8mix_has_pdl()):
            continue
        before = tuple(_key(x) for x in (comp, b, sb, sa))
        result = prepare(comp, b, sa, sb, ring, name)
        if result is None or before != tuple(_key(x) for x in (comp, b, sb, sa)):
            stats["refused"] += 1
            continue
        packet, info, modes = result
        # Publish only after the admission has drained every probe/side stream.
        # All request-time references are weak, with the model owning tensors.
        layer.register_buffer("_fp8mix_packet", packet, persistent=False)
        entry = _Entry()
        entry.layer, entry.model, entry.kernel = weakref.ref(layer), weakref.ref(model), weakref.ref(kernel)
        entry.comp, entry.packet = weakref.ref(comp), weakref.ref(packet)
        entry.config, entry.config_key = config, _config_key(config)
        entry.shape, entry.ring, entry.pdl_modes = shape, ring, modes
        entry.reference_lib, entry.native_lib = fp8c._lib, _lib
        entry.fp8c, entry.l2pf, entry.prefetch_key_fn = fp8c, l2pf, prefetch_key
        entry.prefetch_key = prefetch_key(l2pf)
        owners = (comp, packet, b, sb, sa)
        entry.owners = tuple(weakref.ref(x) for x in owners)
        entry.keys = tuple(_key(x) for x in owners)
        if any(x is None for x in entry.keys):
            del layer._fp8mix_packet
            stats["refused"] += 1
            continue
        layer._fp8mix_entry = entry
        _entries[id(comp)] = weakref.ref(entry)
        stats["admitted"] += 1
        stats["c6"] += info["c6"]
        stats["c7"] += info["c7"]
        LOG.info("FP8MIX admitted %s evidence=%s; private gate, whole result unknown", name, info)
