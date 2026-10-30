"""Optional real-head admission; executed only by official remote model load.

The production path never probes, changes provider, caches activations/results,
or retries a launched operation. Refusal keeps the original MarlinX SU path.
All scratch/graph probes below use private activations and outputs, real read-only
checkpoint B/S/GS and the already-created SU. They do not run the model or KV.
"""
from __future__ import annotations

import ctypes
import os
import statistics
import weakref
from dataclasses import dataclass

import torch

ENABLED = os.environ.get("MARLINX_HEAD_PAIR", "1") != "0"
N, K = 131072, 2688


def _config_key(cfg):
    p = cfg.parallel_config
    o = cfg.offload_config
    return (
        p.world_size, p.data_parallel_size, p.tensor_parallel_size,
        p.pipeline_parallel_size, p.prefill_context_parallel_size,
        p.decode_context_parallel_size, p.enable_expert_parallel,
        p.enable_eplb, p.use_ubatching,
        cfg.scheduler_config.max_num_seqs,
        cfg.model_config.hf_config.model_type, cfg.model_config.dtype,
        cfg.lora_config is not None, cfg.weight_transfer_config is not None,
        o.uva.cpu_offload_gb, o.prefetch.offload_group_size,
    )


def _supported_config(key):
    return key == (1, 1, 1, 1, 1, 1, False, False, False, 1,
                   "nemotron_h", torch.bfloat16, False, False, 0, 0)


def _tensor_key(t):
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.is_inference():
        return None
    # Versioned immutable owners only; no pointer-only/ABA cache and no attempt
    # to read _version on inference tensors that have no version counter.
    try:
        return (id(t), t.data_ptr(), t.untyped_storage().data_ptr(),
                t.untyped_storage().nbytes(), t.storage_offset(),
                tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)
    except RuntimeError:
        return None


def _owners(layer):
    return (layer.weight, layer.weight_scale, layer.weight_global_scale,
            getattr(layer, "_marlinx_su", None))


def _owner_key(layer):
    tensors = _owners(layer)
    keys = tuple(_tensor_key(t) for t in tensors)
    if any(key is None for key in keys):
        return None
    w, s, gs, su = tensors
    if (w.dtype != torch.int32 or tuple(w.shape) != (K // 16, 2 * N)
            or s.dtype != torch.float8_e4m3fn or tuple(s.shape) != (K // 16, N)
            or gs.dtype != torch.float32 or gs.numel() != 1
            or su.dtype != torch.uint8 or tuple(su.shape) != (N // 64, K // 16, 64)
            or any(not t.is_contiguous() or t.storage_offset() != 0 for t in tensors)
            or any(t.device != w.device for t in tensors)
            or any(t.data_ptr() % 16 for t in (w, s, su))
            or gs.data_ptr() % 4):
        return None
    return keys


@dataclass(frozen=True)
class _Entry:
    refs: tuple
    key: tuple
    config_ref: object
    config_key: tuple
    lib: object
    pdl: bool
    sms: int
    device: torch.device


def _abi(lib):
    if not all(hasattr(lib, name) for name in
               ("marlinx_head_pair", "marlinx_head_pair_info", "marlinx_gemm_su")):
        return False
    # Original 13-argument prefix, then SU,pdl. The reference is the unchanged
    # old ABI, with Z=None, the same SU, relu2=0, and explicit PDL mode.
    lib.marlinx_head_pair.argtypes = lib.marlinx_gemm.argtypes + [
        ctypes.c_void_p, ctypes.c_int]
    lib.marlinx_head_pair.restype = ctypes.c_int
    lib.marlinx_head_pair_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.marlinx_head_pair_info.restype = ctypes.c_int
    return True


def _call(entry, x, out, paired, pdl):
    w, s, gs, su = (ref() for ref in entry.refs)
    args = (x.data_ptr(), x.stride(0), x.shape[0], w.data_ptr(), s.data_ptr(),
            gs.data_ptr(), out.data_ptr(), out.stride(0), N, K, entry.sms, 8,
            torch.cuda.current_stream(x.device).cuda_stream)
    if paired:
        rc = entry.lib.marlinx_head_pair(*args, su.data_ptr(), int(pdl))
    else:
        rc = entry.lib.marlinx_gemm_su(*args, None, su.data_ptr(), 0, int(pdl))
    if rc != 0:
        # Production never retries. Private admission may decline only after
        # the same-frame finally has successfully drained ALL private work.
        raise RuntimeError(f"head-pair {'new' if paired else 'old'} launch failed: {rc}")


def _raw_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _guarded_output(m, device, poison):
    raw = torch.full((m * N + 32,), poison, dtype=torch.int16, device=device)
    out = raw[16:-16].view(torch.bfloat16).view(m, N)
    return raw, out


def _check_outputs(raw0, out0, raw1, out1):
    if not _raw_equal(out0, out1):
        return False
    return (bool(torch.all(raw0[:16] == 0x55AA))
            and bool(torch.all(raw0[-16:] == 0x55AA))
            and bool(torch.all(raw1[:16] == 0x2AAA))
            and bool(torch.all(raw1[-16:] == 0x2AAA)))


def _case(m, device, case, generator):
    # All dtypes explicit: the loader's default dtype may be BF16.
    x = torch.randn((m, K), dtype=torch.float32, device=device,
                    generator=generator).to(torch.bfloat16)
    if case == 0:
        x.mul_(0.001)
    elif case == 1:
        pass
    elif case == 2:
        x.mul_(30.0)
    elif case == 3:
        x.zero_()
        bits = torch.tensor([0, -32768, 1, -32767, 127, -32641, 128, -32640],
                            dtype=torch.int16, device=device)
        x.view(torch.int16)[:, :8] = bits
    elif case == 4:
        # Distinct NaN payloads, +/-inf, largest finite, and signed zeros.
        bits = torch.tensor([32640, -128, 32704, -64, 32641, -127, 32639, -129],
                            dtype=torch.int16, device=device)
        x.view(torch.int16)[:, :8] = bits
    elif case == 5:
        x[1::2].neg_()
        x[:, -8:].zero_()
    return x.contiguous()


def _elapsed(entry, x, out, paired, pdl, pressure=None):
    if pressure is not None:
        pairs = []
        for _ in range(8):
            pressure.add_(1)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            _call(entry, x, out, paired, pdl)
            end.record()
            pairs.append((begin, end))
        pairs[-1][1].synchronize()
        return sum(begin.elapsed_time(end) for begin, end in pairs) / 8.0
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(8):
        _call(entry, x, out, paired, pdl)
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 8.0


def _speed(entry, x, old, new, pressure):
    results = {}
    # Warm and cache-pressure ABBA are both required. These are private kernel
    # timings, not a claim about the official complete model or its score.
    for cold in (False, True):
        times = {False: [], True: []}
        for _ in range(3):
            for paired in (False, True, True, False):
                times[paired].append(_elapsed(entry, x, new if paired else old,
                                              paired, entry.pdl,
                                              pressure if cold else None))
        a, b = statistics.median(times[False]), statistics.median(times[True])
        if not (a > 0.0 and b > 0.0):
            return False, results
        results["pressure" if cold else "warm"] = (a, b)
        if b > a * (1.0 if cold else 0.995):
            return False, results
    return True, results


def _probe_inner(entry, stream0, stream1, drained, evidence):
    # All owners, including graph objects, remain in this frame until finally
    # has drained both streams. No probe reads/writes any model activation/KV.
    graphs = []
    buffers = []
    try:
        with torch.cuda.stream(stream0):
            _, s, _, su = (ref() for ref in entry.refs)
            # Verify the existing byte-permutation relationship without making
            # a second full weight/scale copy. Both native paths consume SU.
            if not torch.equal(su.transpose(0, 1),
                               s.view(torch.uint8).view(K // 16, N // 64, 64)):
                return False
            gen = torch.Generator(device=entry.device).manual_seed(721316)
            modes = (False, True) if entry.pdl else (False,)
            for pdl in modes:
                for m in (5, 6):
                    raw0, out0 = _guarded_output(m, entry.device, 0x55AA)
                    raw1, out1 = _guarded_output(m, entry.device, 0x2AAA)
                    buffers.extend((raw0, raw1, out0, out1))
                    for case in range(6):
                        x = _case(m, entry.device, case, gen)
                        snapshot = x.view(torch.int16).clone()
                        buffers.extend((x, snapshot))
                        _call(entry, x, out0, False, pdl)
                        _call(entry, x, out1, True, pdl)
                        if (not _check_outputs(raw0, out0, raw1, out1)
                                or not torch.equal(x.view(torch.int16), snapshot)):
                            return False

            # Two independently captured streams/inputs. Replays consume new
            # activation contents every time. Native PDL waits remain in each
            # kernel; the copy is an actual preceding producer in each graph.
            for stream in (stream0, stream1):
                stream.wait_stream(stream0)
                with torch.cuda.stream(stream):
                    for m in (5, 6):
                        source = torch.empty((m, K), dtype=torch.bfloat16,
                                             device=entry.device)
                        x = torch.empty_like(source)
                        raw0, out0 = _guarded_output(m, entry.device, 0x55AA)
                        raw1, out1 = _guarded_output(m, entry.device, 0x2AAA)
                        buffers.extend((source, x, raw0, out0, raw1, out1))
                        source.copy_(_case(m, entry.device, 1, gen))
                        x.copy_(source)
                        _call(entry, x, out0, False, entry.pdl)
                        _call(entry, x, out1, True, entry.pdl)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        graphs.append(graph)
                        with torch.cuda.graph(graph, stream=stream):
                            x.copy_(source)
                            _call(entry, x, out1, True, entry.pdl)
                        for case in (1, 5, 3, 4, 2, 0, 1):
                            source.copy_(_case(m, entry.device, case, gen))
                            graph.replay()
                            # Reference after replay, rather than before the
                            # paired grid: it must not accidentally serialize
                            # away the candidate's activation producer edge.
                            _call(entry, x, out0, False, entry.pdl)
                            if (not _check_outputs(raw0, out0, raw1, out1)
                                    or not _raw_equal(source, x)):
                                return False

            stream0.wait_stream(stream1)
            with torch.cuda.stream(stream0):
                pressure = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8,
                                       device=entry.device)
                buffers.append(pressure)
                for m in (5, 6):
                    x = _case(m, entry.device, 1, gen)
                    old = torch.empty((m, N), dtype=torch.bfloat16, device=entry.device)
                    new = torch.empty_like(old)
                    buffers.extend((x, old, new))
                    for _ in range(3):
                        _call(entry, x, old, False, entry.pdl)
                        _call(entry, x, new, True, entry.pdl)
                    ok, times = _speed(entry, x, old, new, pressure)
                    evidence[f"M{m}"] = times
                    if not ok or not _raw_equal(old, new):
                        return False
            return True
    finally:
        # Do NOT swallow failure to drain: that is unsafe for fallback. This
        # frame still owns every graph and allocation when synchronizing.
        try:
            stream0.synchronize()
        finally:
            stream1.synchronize()
        drained[0] = True


@torch.no_grad()
def prepare(layer, cfg):
    """Final post-load only, actual lm_head and fixed immutable single worker."""
    # Refuse later post-load calls rather than silently requalifying live graphs.
    seen = getattr(layer, "_marlinx_head_pair_prepared", False)
    layer._marlinx_head_pair = None
    layer._marlinx_head_pair_prepared = True
    if seen or not ENABLED or torch.cuda.is_current_stream_capturing():
        return False
    from . import DEPTH, _load, _pdl_active, _sm_count
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    kernel = getattr(getattr(layer, "quant_method", None), "kernel", None)
    if (type(kernel) is not MarlinNvFp4LinearKernel or DEPTH != 8
            or not getattr(layer, "_marlinx_ok", False)
            or getattr(layer, "_marlinx_zs", None) is not None
            or getattr(layer, "_marlinx_lean", False)
            or getattr(layer, "bias", None) is not None
            or (layer.output_size_per_partition, layer.input_size_per_partition) != (N, K)):
        return False
    try:
        config_key = _config_key(cfg)
        key = _owner_key(layer)
    except (AttributeError, RuntimeError):
        return False
    if key is None or not _supported_config(config_key):
        return False
    device = layer.weight.device
    if (torch.cuda.current_device() != device.index
            or torch.cuda.get_device_capability(device) != (12, 1)
            or _sm_count(device) != 48):
        return False
    lib = _load()
    if lib is None or not _abi(lib):
        return False
    attrs = (ctypes.c_int * 4)()
    resources = {}
    for mode in ((False, True) if _pdl_active else (False,)):
        if lib.marlinx_head_pair_info(int(mode), attrs) != 0:
            return False
        resources[str(mode)] = tuple(attrs)
        if not (0 < attrs[0] <= 255 and attrs[1] == 0
                and attrs[2] + 21504 <= 48 * 1024 and attrs[3] >= 1):
            return False
    tensors = _owners(layer)
    entry = _Entry(tuple(weakref.ref(t) for t in tensors), key, weakref.ref(cfg),
                   config_key, lib, bool(_pdl_active), 48, device)
    caller = torch.cuda.current_stream(device)
    stream0 = torch.cuda.Stream(device=device)
    stream1 = torch.cuda.Stream(device=device)
    stream0.wait_stream(caller)
    stream1.wait_stream(caller)
    drained = [False]
    evidence = {"resource": resources}
    try:
        ok = _probe_inner(entry, stream0, stream1, drained, evidence)
    except Exception as exc:  # private refusal only AFTER a successful drain
        if not drained[0]:
            raise
        layer._marlinx_head_pair_evidence = {"status": "rejected", "error": type(exc).__name__}
        return False
    finally:
        caller.wait_stream(stream0)
        caller.wait_stream(stream1)
    layer._marlinx_head_pair_evidence = evidence
    if not ok or _owner_key(layer) != key or _config_key(cfg) != config_key:
        return False
    layer._marlinx_head_pair = entry
    return True


def maybe_apply(layer, x):
    """Only immutable metadata lookup, then one fresh full output allocation."""
    from . import DEPTH, _lib, _pdl_active
    entry = getattr(layer, "_marlinx_head_pair", None)
    if (not ENABLED or entry is None or DEPTH != 8 or torch.compiler.is_compiling()
            or _lib is not entry.lib or bool(_pdl_active) != entry.pdl
            or x.dtype != torch.bfloat16 or x.device != entry.device
            or tuple(x.shape) not in ((5, K), (6, K))
            or not x.is_contiguous() or x.storage_offset() != 0
            or x.data_ptr() % 16
            or torch.cuda.current_device() != entry.device.index):
        return None
    cfg = entry.config_ref()
    if cfg is None or any(ref() is not t for ref, t in zip(entry.refs, _owners(layer))):
        return None
    if _owner_key(layer) != entry.key or _config_key(cfg) != entry.config_key:
        return None
    out = torch.empty((x.shape[0], N), dtype=torch.bfloat16, device=x.device)
    _call(entry, x, out, True, entry.pdl)
    return out
