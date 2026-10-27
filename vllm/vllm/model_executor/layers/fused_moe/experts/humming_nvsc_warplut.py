# SPDX-License-Identifier: Apache-2.0
"""Warp-distributed NVSC lookup, qualified on the actual selected w13 owner.

All compilation, private calls and graph timing run at official remote boot.
Original scale encoding, scheduler, K16 arithmetic and epilogue are retained.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import statistics
from pathlib import Path

import humming
import torch
from humming import ops
from humming.kernel.humming import HummingKernel

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("HUMMING_NVSC_WARP_LUT", "1") != "0"
_CLASSES = {}
_KERNELS = {}
_SENTINEL = 0x2BAD
_HEADERS_OK = None
_HEADER_SHA256 = {
    "include/humming/kernel/humming.cuh": "7d2f335332c3aafa9267d63cf20550dfbb5ab7072e8b9defb0d7a2be375ede30",
    "include/humming/memory/s2r_pipeline.cuh": "7c3cca562b9c1718b3be73126183efe74802d213937f7f7ca23961104ab6a4bd",
    "include/humming/memory/g2s_pipeline.cuh": "8e2b6a6f8ca9dd0700e21c54c6c4e021c4c6d76cd536ec4bd17540dec353e46b",
}

_OLD_LOOKUP = r"""  CUDA_INLINE static uint32_t lookup(uint32_t c) {
    const uint32_t b0 = nvsc_table[c & 0xFFu];
    const uint32_t b1 = nvsc_table[(c >> 8) & 0xFFu];
    const uint32_t b2 = nvsc_table[(c >> 16) & 0xFFu];
    const uint32_t b3 = nvsc_table[c >> 24];
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }"""

_NEW_LOOKUP = r"""  // All math lanes execute each even-group load. The first lookup follows
  // the original stage-0 cp.async wait + full CTA fence, never construction.
  static_assert(!TuningConfig::kUseWarpSpec);
  static_assert(TuningConfig::kNumThreads % 32 == 0);
  static_assert(TuningConfig::kNumMathThreads == TuningConfig::kNumThreads);
  uint32_t nvsc_lane_word = 0;
  bool nvsc_lut_ready = false;

  CUDA_INLINE uint32_t lookup(uint32_t c) {
    if (!nvsc_lut_ready) {
      // Volatile read stays at this post-wait point. Sixteen unique words
      // cover all64 bytes; lanes16..31 duplicate0..15. The table is immutable
      // for this CTA and the state dies with this launch.
      nvsc_lane_word = reinterpret_cast<const volatile uint32_t *>(nvsc_table)[threadIdx.x & 15u];
      nvsc_lut_ready = true;
    }
    const uint32_t w0 = __shfl_sync(0xffffffffu, nvsc_lane_word, (c >> 2) & 15u);
    const uint32_t w1 = __shfl_sync(0xffffffffu, nvsc_lane_word, (c >> 10) & 15u);
    const uint32_t w2 = __shfl_sync(0xffffffffu, nvsc_lane_word, (c >> 18) & 15u);
    const uint32_t w3 = __shfl_sync(0xffffffffu, nvsc_lane_word, (c >> 26) & 15u);
    const uint32_t lo = __byte_perm(w0, w1, (c & 3u) | ((4u + ((c >> 8) & 3u)) << 4));
    const uint32_t hi = __byte_perm(w2, w3, ((c >> 16) & 3u) | ((4u + ((c >> 24) & 3u)) << 4));
    return __byte_perm(lo, hi, 0x5410);
  }"""


def _header_guard():
    global _HEADERS_OK
    if _HEADERS_OK is None:
        root = Path(humming.__file__).parent
        _HEADERS_OK = all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h
                          for p, h in _HEADER_SHA256.items())
    return _HEADERS_OK


def _lookup_text(text):
    if text.count(_OLD_LOOKUP) != 1:
        raise ValueError("actual selected NVSC lookup changed")
    return text.replace(_OLD_LOOKUP, _NEW_LOOKUP, 1)


def _twin(original):
    key = original.kernel_id
    if key in _KERNELS:
        return _KERNELS[key]
    if (original.use_warp_spec or original.use_stream_k
            or not original.use_cp_async or original.use_mbarrier
            or getattr(original, "num_experts", 0) != 128
            or tuple(original.problem_shape[1:]) != (1856, 2688)
            or "nvsc_table" not in original.code):
        raise ValueError("not the selected non-warp-specialized NVSC w13 kernel")
    if not _header_guard():
        raise ValueError("pinned all-lane pipeline identity changed")
    expected = _lookup_text(original.code)
    digest = hashlib.sha256(expected.encode()).hexdigest()[:16]
    class_key = (type(original), digest)
    if class_key not in _CLASSES:
        name = f"WarpNvscLut{digest}{type(original).__name__}"
        _CLASSES[class_key] = type(name, (type(original),), {
            "nvsc_lut_transform": staticmethod(_lookup_text),
        })
    params = {}
    for f in dataclasses.fields(original):
        if f.init:
            value = getattr(original, f.name)
            params[f.name] = copy.copy(value) if isinstance(value, (list, dict)) else value
    new = _CLASSES[class_key](**params)
    new.load_cubin()
    if new.kernel_id == original.kernel_id or new.kernel_filename == original.kernel_filename:
        raise ValueError("warp LUT reused the reference cubin")
    if any(getattr(new, f.name) != getattr(original, f.name)
           for f in dataclasses.fields(original) if f.init):
        raise ValueError("warp LUT changed original kernel parameters")
    if new.code != expected:
        raise ValueError("warp LUT changed source beyond the selected lookup")
    _KERNELS[key] = (original, new)
    return original, new


def _guarded(rows, cols, device):
    b = torch.empty(rows * cols + 256, device=device, dtype=torch.bfloat16)
    b.view(torch.int16).fill_(_SENTINEL)
    return b, b[128:-128].view(rows, cols)


def _bits(t):
    return t.reshape(-1).view(torch.uint8)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def _guards(b):
    v = b.view(torch.int16)
    return bool(torch.all(v[:128] == _SENTINEL)) and bool(torch.all(v[-128:] == _SENTINEL))


@torch.no_grad()
def _qualify(configs, twins, base):
    """Private full calls, actual weight/scale owners and original config list.

    Repeated changed-route runs/graphs exercise termination/reset; there is no
    claim of reading the device counters or of measuring runtime occupancy.
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    dev = base["inputs"].device
    prefill = base["inputs"].shape[0] == 496
    ms = (496,) if prefill else tuple(range(1, 9))
    immutable_names = ("weight", "weight_scale", "zero_point", "bias", "global_scale")
    immutable = [(base[k], base[k].clone()) for k in immutable_names
                 if isinstance(base.get(k), torch.Tensor)]
    gen = torch.Generator(device=dev).manual_seed(23027)
    pressure = torch.empty(64 << 20, device=dev, dtype=torch.uint8)
    timings = []
    ratios = []
    try:
        for m in ms:
            valid = m * 6
            values = configs.view(-1, 4).tolist()
            row = next(x for x in values if x[0] < valid <= x[1])
            original = HummingKernel._id2kernel[row[2]]
            block = original.block_shape[0]
            ib, x = _guarded(m, 2688, dev)
            rb, y0 = _guarded(valid, 1856, dev)
            nb, y1 = _guarded(valid, 1856, dev)
            if any(t.data_ptr() % 256 for t in (x, y0, y1)):
                return False
            locks = torch.zeros_like(base["locks"])
            metadata = None

            def route(shift, crowded, inplace=False):
                nonlocal metadata
                ids = torch.arange(6 if crowded else valid, device=dev, dtype=torch.int32)
                ids = ((ids + shift) % 128)
                ids = ids.view(1, 6).expand(m, 6).contiguous() if crowded else ids.view(m, 6)
                new = moe_align_block_size(ids, block, 128, ignore_invalid_experts=True)
                if inplace:
                    if any(a.shape != b.shape for a, b in zip(metadata, new)):
                        raise ValueError("route metadata capacity changed")
                    for a, b in zip(metadata, new):
                        a.copy_(b)
                else:
                    metadata = new

            def fill(edge=False):
                if edge:
                    values = torch.tensor([0, -32768, 1, -32767, 16256, -16512,
                                           32640, -128, 32705, -63], device=dev, dtype=torch.int16)
                    x.view(torch.int16).copy_(values.repeat((2688 + 9) // 10)[:2688]
                                              .view(1, 2688).expand(m, 2688))
                else:
                    x.copy_(torch.randn((m, 2688), generator=gen, device=dev))

            def run(which):
                kw = dict(base, configs=configs if which == 0 else twins,
                          inputs=x, outputs=y0 if which == 0 else y1,
                          sorted_ids=metadata[0], expert_ids=metadata[1],
                          num_tokens_padded=metadata[2], locks=locks,
                          top_k=6, valid_shape_m=valid)
                return ops.launch_kernel(**kw)

            def compare(graphs=None):
                before_x = ib.clone()
                before_meta = tuple(t.clone() for t in metadata)
                before_locks = locks.clone()
                rb.view(torch.int16).fill_(_SENTINEL)
                nb.view(torch.int16).fill_(_SENTINEL)
                for i in (0, 1):
                    graphs[i].replay() if graphs else run(i)
                if (not _same(rb, nb) or not _same(ib, before_x)
                        or not _guards(ib) or not _guards(rb) or not _guards(nb)
                        or any(not _same(a, b) for a, b in zip(metadata, before_meta))
                        or not _same(locks, before_locks)
                        or any(not _same(a, b) for a, b in immutable)):
                    raise ValueError("warp LUT changed full private output/guards/read-only state")

            # Every M/config family covers all128 experts, not only M8's
            # evict-first class. Separate crowded and edge cases follow.
            shifts = tuple(range(0, 128, valid))
            for shift in shifts:
                route(shift, False); fill(); compare()
            for crowded in (False, True):
                route(29, crowded)
                for edge in (False, True):
                    fill(edge); compare(); compare()
            if m not in (1, 5, 8, 496):
                continue
            for crowded in (False, True):
                route(37, crowded); fill(); compare()
                graphs = []
                for which in (0, 1):
                    for _ in range(2):
                        run(which)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                        run(which)
                    graphs.append(graph)
                for shift, new_crowded in ((91, not crowded), (3, crowded), (122, False)):
                    route(shift, new_crowded, inplace=True); fill(); compare(graphs)
                route(37, crowded, inplace=True); fill(); compare(graphs)
                for cold in (False, True):
                    samples = [[], []]
                    for trial in range(5):
                        for which in ((0, 1) if trial % 2 == 0 else (1, 0)):
                            if cold:
                                pressure.fill_(trial + 1)
                            else:
                                for _ in range(16):
                                    graphs[which].replay()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            repeats = 1 if cold else 32
                            start.record()
                            for _ in range(repeats):
                                graphs[which].replay()
                            end.record(); end.synchronize()
                            value = start.elapsed_time(end) / repeats
                            if not value > 0:
                                return False
                            samples[which].append(value)
                    old, new = (statistics.median(s) for s in samples)
                    timings.append((old, new)); ratios.append(new / old)
                    compare(graphs)
        return (bool(ratios) and statistics.median(ratios) < .995
                and sum(y for x, y in timings) / sum(x for x, y in timings) < .995
                and max(ratios) < 1.01)
    finally:
        # Private locks/metadata never alias layer state. On a detected
        # read-only violation restore original static owners before fallback.
        for actual, saved in immutable:
            actual.copy_(saved)


def launch(layer, **kwargs):
    """Launch with the selected original unless this exact owner passed.

    Qualification/compilation happen only in eager boot. Captures without a
    prepared owner use the original; optional failures never alter its caches.
    """
    configs = kwargs["configs"]
    if (not ENABLED or kwargs.get("weight_scale") is not getattr(layer, "nvsc_w13_bs", None)
            or kwargs["inputs"].shape[0] not in (*range(1, 9), 496)
            or kwargs["inputs"].dtype != torch.bfloat16
            or kwargs["outputs"].dtype != torch.bfloat16
            or not kwargs["inputs"].is_contiguous() or not kwargs["outputs"].is_contiguous()
            or kwargs["top_k"] != 6 or kwargs.get("input_scale") is not None):
        return ops.launch_kernel(**kwargs)
    cache = getattr(layer, "_nvsc_warp_lut", None)
    if cache is None:
        cache = layer._nvsc_warp_lut = {}
    weight = kwargs["weight"]
    scale = kwargs.get("weight_scale")
    key = (id(configs), weight.data_ptr(), scale.data_ptr() if scale is not None else 0,
           kwargs["inputs"].shape[0] == 496)
    if key not in cache and not torch.cuda.is_current_stream_capturing():
        # Retain strong config ownership even after failure; id cannot recycle.
        cache[key] = (configs, None)
        try:
            if (torch.cuda.get_device_capability() == (12, 1)
                    and kwargs["inputs"].shape[0] in (*range(1, 9), 496)
                    and kwargs["inputs"].dtype == kwargs["outputs"].dtype == torch.bfloat16
                    and kwargs["top_k"] == 6 and kwargs.get("input_scale") is None):
                twins = configs.clone()
                for row in twins.view(-1, 4):
                    old, new = _twin(HummingKernel._id2kernel[int(row[2])])
                    row[2] = new.kernel_id
                if _qualify(configs, twins, kwargs):
                    cache[key] = (configs, twins)
                    logger.info("humming warp LUT admitted for actual weight owner %s", key)
        except Exception as exc:  # optional; retain the actual selected original
            logger.warning_once("humming warp LUT unavailable: %r", exc)
    selected = cache.get(key, (configs, None))[1]
    return ops.launch_kernel(**dict(kwargs, configs=selected if selected is not None else configs))
