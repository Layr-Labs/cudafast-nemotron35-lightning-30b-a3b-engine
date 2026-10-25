"""Production SHX shared-expert prefill; actual-weight admission and joint grid enablement."""

from __future__ import annotations

import ctypes
import os

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libshco.so")

_lib = None
_load_error: str | None = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(LIB_PATH)
        vp, ci, cll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.shx_attrs.argtypes = [ci, vp]
        lib.shx_attrs.restype = ci
        lib.shx_scratch_floats.argtypes = [ci]
        lib.shx_scratch_floats.restype = ci
        lib.shx_gemm.argtypes = [vp, cll, ci, vp, vp, vp, vp, cll, ci, ci, vp, ci, vp, ci, ci, vp]
        lib.shx_gemm.restype = ci
        lib.shx2_attrs.argtypes = [ci, vp]
        lib.shx2_attrs.restype = ci
        lib.shx2_gemm.argtypes = [vp, cll, ci, vp, vp, vp, vp, cll, ci, ci, vp, ci, vp, ci, ci, vp, ci, vp]
        lib.shx2_gemm.restype = ci
        _lib = lib
    except (OSError, AttributeError) as e:
        _load_error = repr(e)
    return _lib


def available() -> bool:
    return _load() is not None


def load_error() -> str | None:
    _load()
    return _load_error


# ---------------------------------------------------------------------------------------- shx
# Marlin's per-element summation tree for an NVFP4 W4A16 GEMM with 16-bit activations and
# M > 16 (vLLM v0.27.1 marlin_mm + marlin_template.h), for the thread config Marlin picks for
# the shared expert shapes: {thread_k 64, thread_n 128, 128 threads} (the {64, 256, 256} config
# needs N % 256 == 0, false for N = 3712 and 2688), grid = #SMs.

THREAD_K, THREAD_N = 64, 128


def _launches(m: int, n: int):
    """marlin_mm's row split: (row0, prob_m_split, thread_m_blocks) per kernel launch."""
    max_par = 16 * 8 if n <= 4096 else 16
    out, row, rest = [], 0, m
    while rest:
        par = min(rest // 64, max_par)
        split = par * 64 if par > 0 else rest
        out.append((row, split, min(-(-split // 16), 4)))
        row += split
        rest -= split
    return out


def marlin_units(m: int, n: int, k: int, sms: int = 48, unit_features: int = 64):
    """Work units for shx: (row0, nrows, f0, [(lo, hi) k-tile segments in fold order]) per
    64-row sub-problem x 64-feature block, feature-major. A segment is the k-tile range one
    Marlin CTA reduces; fold order is Marlin's lock order (slice_idx 0 = highest K first)."""
    assert n % THREAD_N == 0 and k % THREAD_K == 0
    k_tiles, n_tiles = k // THREAD_K, n // THREAD_N
    cols = []  # per launch: {global column g: (rows0, nrows, segments)}
    for row0, prob_m, tmb in _launches(m, n):
        assert tmb > 1, "M <= 16 runs other Marlin configs"
        mbs = 16 * tmb
        parallel = prob_m // mbs if prob_m > mbs else 1
        gmn = parallel * n_tiles
        part2 = gmn
        if gmn > sms:
            part2 = gmn % sms
            if part2 * 3 <= sms:
                part2 += sms
        iters = -(-k_tiles * part2 // sms)  # group_blocks = 1 < thread_k_blocks: no rounding
        launch = {}
        for g in range(gmn):
            par_id, slice_col = divmod(g, n_tiles)
            rows0 = row0 + par_id * mbs
            nrows = min(mbs, prob_m - par_id * mbs)
            if g < gmn - part2:
                segs = [(0, k_tiles)]
            else:
                cp = g - (gmn - part2)
                lo, hi = cp * k_tiles, (cp + 1) * k_tiles
                segs = []
                while lo < hi:
                    end = min(hi, (lo // iters + 1) * iters)
                    segs.append((lo - cp * k_tiles, end - cp * k_tiles))
                    lo = end
                segs.sort(key=lambda s: -s[0])
            launch[g] = (slice_col, rows0, nrows, segs)
        cols.append(launch)
    assert THREAD_N % unit_features == 0
    units = []
    for fb in range(n // unit_features):
        col = fb * unit_features // THREAD_N
        for launch in cols:
            for g, (slice_col, rows0, nrows, segs) in launch.items():
                if slice_col != col:
                    continue
                for r in range(0, nrows, 64):
                    assert len(segs) <= 4, segs
                    units.append((rows0 + r, min(64, nrows - r), unit_features * fb, segs))
    return units


def units_tensor(units, device) -> torch.Tensor:
    rows = []
    for row0, nrows, f0, segs in units:
        lo = [s[0] for s in segs] + [0] * (4 - len(segs))
        hi = [s[1] for s in segs] + [0] * (4 - len(segs))
        rows.append([row0, nrows, f0, len(segs)] + lo + hi)
    return torch.tensor(rows, dtype=torch.int32, device=device).contiguous()


def shx_attrs(relu2: bool) -> dict:
    lib = _load()
    assert lib is not None, _load_error
    out = (ctypes.c_int * 4)()
    rc = lib.shx_attrs(int(relu2), out)
    if rc != 0:
        raise RuntimeError(f"shx_attrs failed: {rc}")
    return {"regs": out[0], "static_smem": out[1], "max_threads": out[2], "local_bytes": out[3]}


def shx2_attrs(relu2: bool) -> dict:
    lib = _load()
    assert lib is not None, _load_error
    out = (ctypes.c_int * 4)()
    rc = lib.shx2_attrs(int(relu2), out)
    if rc != 0:
        raise RuntimeError(f"shx2_attrs failed: {rc}")
    return {"regs": out[0], "static_smem": out[1], "max_threads": out[2], "local_bytes": out[3]}


def shx_gemm(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, global_scale: torch.Tensor,
             out: torch.Tensor, units: torch.Tensor, scratch: torch.Tensor, relu2: bool = False,
             grid: int = 48, version: int = 1, work: torch.Tensor | None = None, hint: int = 0) -> None:
    lib = _load()
    assert lib is not None, _load_error
    m, k = x.shape
    n = out.shape[1]
    assert x.dtype == out.dtype == torch.bfloat16 and x.stride(1) == 1 and out.stride(1) == 1
    assert weight.dtype == torch.int32 and tuple(weight.shape) == (k // 16, 2 * n) and weight.is_contiguous()
    assert weight_scale.dtype == torch.float8_e4m3fn and tuple(weight_scale.shape) == (k // 16, n)
    assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
    assert scratch.dtype == torch.float32 and scratch.numel() >= lib.shx_scratch_floats(grid)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    if version == 2:
        assert work is not None and work.dtype == torch.int32 and work.numel() >= 2
        rc = lib.shx2_gemm(x.data_ptr(), x.stride(0), m, weight.data_ptr(), weight_scale.data_ptr(),
                           global_scale.data_ptr(), out.data_ptr(), out.stride(0), n, k, units.data_ptr(),
                           units.shape[0], scratch.data_ptr(), grid, int(relu2), work.data_ptr(), int(hint), stream)
    else:
        rc = lib.shx_gemm(x.data_ptr(), x.stride(0), m, weight.data_ptr(), weight_scale.data_ptr(),
                          global_scale.data_ptr(), out.data_ptr(), out.stride(0), n, k, units.data_ptr(),
                          units.shape[0], scratch.data_ptr(), grid, int(relu2), stream)
    if rc != 0:
        raise RuntimeError(f"shx_gemm failed: {rc} (M={m} N={n} K={k})")


VERSION = int(os.environ.get("SHX_VERSION", "2"))
HINT = int(os.environ.get("SHX_HINT", "1"))


class Shx:
    """The shared expert (up -> relu^2 -> down) at one M with Marlin's exact trees."""

    def __init__(self, m: int, device, sms: int = 48, grid: int = 48, version: int | None = None,
                 hint: int | None = None):
        self.m, self.grid = m, grid
        self.version = VERSION if version is None else version
        self.hint = HINT if hint is None else hint
        uf = 128 if self.version == 2 else 64
        self.units_up = units_tensor(marlin_units(m, 3712, 2688, sms, uf), device)
        self.units_dn = units_tensor(marlin_units(m, 2688, 3712, sms, uf), device)
        lib = _load()
        assert lib is not None, _load_error
        self.scratch = torch.empty(lib.shx_scratch_floats(grid), dtype=torch.float32, device=device)
        self.work = torch.zeros(2, dtype=torch.int32, device=device)

    def up(self, x, layer, out, relu2=True):
        shx_gemm(x, layer.weight, layer.weight_scale, layer.weight_global_scale, out, self.units_up,
                 self.scratch, relu2=relu2, grid=self.grid, version=self.version, work=self.work, hint=self.hint)

    def down(self, a, layer, out):
        shx_gemm(a, layer.weight, layer.weight_scale, layer.weight_global_scale, out, self.units_dn,
                 self.scratch, grid=self.grid, version=self.version, work=self.work, hint=self.hint)


# ------------------------------------------------------------------------- engine entry points
# SHCO=0 disables everything below (the engine then runs stock Marlin on the main stream).
ENABLED = os.environ.get("SHCO", "1") != "0"
SHX_M = 496  # the prefill row count whose Marlin trees shx reproduces
SHAPES = frozenset({(3712, 2688), (2688, 3712)})  # (N, K): shared expert up, down
_ctx: dict[int, Shx] = {}
_logged = False
_ready = False
_experts = []


def prepare(layer) -> bool:
    """Marks a Marlin NVFP4 linear (after prepare_fp4_layer_for_marlin) of a shared-expert shape as
    eligible and builds the per-device unit tables and scratch (load time, outside graph capture)."""
    layer._shx_ok = False
    if not ENABLED or _load() is None or torch.cuda.is_current_stream_capturing():
        return False
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    w, s, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
    ok = (
        (n, k) in SHAPES
        and getattr(layer, "bias", None) is None
        and gs is not None and gs.dtype == torch.float32 and gs.numel() == 1
        and w.dtype == torch.int32 and tuple(w.shape) == (k // 16, 2 * n) and w.is_contiguous() and w.is_cuda
        and s.dtype == torch.float8_e4m3fn and tuple(s.shape) == (k // 16, n) and s.is_contiguous()
    )
    if not ok:
        return False
    idx = w.device.index if w.device.index is not None else torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(idx)
    if (prop.major, prop.minor, prop.multi_processor_count) != (12, 1, 48):
        return False
    if idx not in _ctx:
        sms = prop.multi_processor_count
        _ctx[idx] = Shx(SHX_M, w.device, sms=sms, grid=sms)
    layer._shx_ok = True
    return True


def _eligible(layer, x: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling()
        and ENABLED and _ready
        and getattr(layer, "_shx_ok", False)
        and x.dim() == 2
        and x.shape[0] == SHX_M
        and x.shape[1] == layer.input_size_per_partition
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def overlap_ok(up_layer, down_layer, x: torch.Tensor) -> bool:
    """True when the whole shared expert of this call runs on shx (then it may share the GPU
    with the routed experts on the aux stream)."""
    return up_layer is not None and down_layer is not None and _eligible(up_layer, x) and getattr(
        down_layer, "_shx_ok", False)


def _run(layer, x: torch.Tensor, relu2: bool) -> torch.Tensor | None:
    global _logged
    if not _eligible(layer, x):
        return None
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    ctx = _ctx.get(idx)
    if ctx is None:
        return None
    out = torch.empty((x.shape[0], layer.output_size_per_partition), dtype=torch.bfloat16, device=x.device)
    if relu2:
        ctx.up(x, layer, out, relu2=True)
    else:
        ctx.down(x, layer, out)
    if not _logged:
        _logged = True
        import logging

        logging.getLogger("vllm").info("shx: Marlin-exact shared expert at M=%d (grid %d)", SHX_M, ctx.grid)
    return out


def maybe_up_relu2(layer, x: torch.Tensor) -> torch.Tensor | None:
    """bf16(relu(up(x))^2) with Marlin's exact up output, or None (caller runs stock)."""
    return _run(layer, x, True)


def maybe_down(layer, a: torch.Tensor) -> torch.Tensor | None:
    """Marlin-exact down(a), or None (caller runs stock)."""
    return _run(layer, a, False)


# All execution here is part of official remote model loading, never a local preflight.
def register_expert(expert):
    import weakref
    _experts.append(weakref.ref(expert))


def _admit(up, down):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear

    def stock(layer, x):
        return apply_fp4_marlin_linear(
            input=x, weight=layer.weight, weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,
            size_n=layer.output_size_per_partition, size_k=layer.input_size_per_partition, bias=None)

    dev = up.weight.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    ctx = _ctx[idx]
    ids = torch.arange(SHX_M * 2688, device=dev, dtype=torch.int32).reshape(SHX_M, 2688)
    pattern = ((ids % 127).float() - 63) / 32
    # Dense, low amplitude, high amplitude, sparse and signed-zero cases, at actual M496.
    for case in range(5):
        if case < 3:
            x = (pattern * (1.0, 0.0001, 64.0)[case]).to(torch.bfloat16)
        elif case == 3:
            x = torch.where(ids % 17 == 0, pattern, 0.0).to(torch.bfloat16)
        else:
            x = torch.where(ids % 2 == 0, -0.0, 0.0).to(torch.bfloat16)
        ref = stock(up, x)
        ref = ref.float().clamp_min_(0).square_().to(torch.bfloat16)
        act = torch.empty_like(ref)
        ctx.up(x, up, act)
        if not torch.equal(act.view(torch.int16), ref.view(torch.int16)):
            return False
        expected = stock(down, ref)
        out = torch.empty_like(expected)
        ctx.down(act, down, out)
        if not torch.equal(out.view(torch.int16), expected.view(torch.int16)):
            return False
    return True


def finalize_cohort(model):
    """Commit the 23 shared experts and their already-built humming grids together.

    No persistent-grid reduction is left behind if any actual-weight admission fails.
    Original main/aux stream joins continue to serialize the per-device SHX scratch.
    """
    import json
    import logging

    global _ready
    _ready = False
    modules = list(model.modules())
    ids = {id(m) for m in modules}
    pairs = [m for m in modules if type(m).__name__ == "NemotronHMLP"]
    experts = [e for r in _experts if (e := r()) is not None and id(e.layer) in ids]
    ok = ENABLED and len(pairs) == len(experts) == 23
    ok = ok and all(getattr(m.up_proj, "_shx_ok", False)
                    and getattr(m.down_proj, "_shx_ok", False) for m in pairs)
    ok = ok and all(getattr(e, "_fused_relu2", False)
                    and e.w13_tuning_config[-1][1] == 1 << 30 for e in experts)
    try:
        sms = int(os.environ.get("SHCO_HUMMING_SMS", "36"))
        ok = ok and sms in (36, 48)
        if ok:
            # Attributes constrain co-residency, not the numerical admission.
            attrs = shx2_attrs(False) if VERSION == 2 else shx_attrs(False)
            ok = attrs["regs"] <= 128 and attrs["static_smem"] <= 32768
        if ok:
            with torch.inference_mode():
                ok = all(_admit(m.up_proj, m.down_proj) for m in pairs)
    except Exception as exc:
        logging.getLogger("vllm").warning("SHX cohort admission unavailable: %s", exc)
        ok = False
    for e in experts:
        # init_humming_moe cached both a list and its JSON before the model hook.
        e.w13_tuning_config[-1][2]["num_sms"] = sms if ok else 48
        e.w13_tuning_config_str = json.dumps(e.w13_tuning_config)
    if not ok:
        for m in pairs:
            m.up_proj._shx_ok = m.down_proj._shx_ok = False
    _ready = bool(ok)
    logging.getLogger("vllm").info("SHX cohort: ready=%s shared=%d routed=%d sms=%d",
                                   _ready, len(pairs), len(experts), sms if ok else 48)
    return _ready
