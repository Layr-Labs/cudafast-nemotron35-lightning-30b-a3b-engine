"""Optional native-MTP combine/residual/final-RMS fusion, remote-admitted only.

The old model branch remains visible for refused/unsupported rows. No local
candidate execution is part of authoring this source. Native IR semantics only.
"""
import math
import os
import statistics
from types import SimpleNamespace

import torch
from vllm import envs, ir
from vllm.triton_utils import tl, triton

@triton.jit
def combine_end_norm(
    part, weights, y, gamma,
    M: tl.constexpr, H: tl.constexpr, ROUTED_SCALE: tl.constexpr,
    TOPK: tl.constexpr, S: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr,
    launch_pdl: tl.constexpr = False,
    residual_ptr=None, ADD_RES: tl.constexpr = False,
):
    """Original combine/residual rounds plus native-provider final RMS, one CTA per row."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    m = tl.program_id(0)
    n = tl.arange(0, BN)
    routed = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        routed += tl.load(weights + j).to(tl.float32) * tl.load(part + j * H + n, mask=n < H, other=0.0)
    shared = tl.zeros((BN,), tl.float32)
    for s in range(S):
        shared += tl.load(part + (TOPK + s) * H + n, mask=n < H, other=0.0)
    out = (routed * ROUTED_SCALE + shared).to(y.dtype.element_ty)
    value = out
    if ADD_RES:
        residual = tl.load(residual_ptr + m * H + n, mask=n < H, other=0.0)
        value = (out.to(tl.float32) + residual.to(tl.float32)).to(y.dtype.element_ty)
    # Native ir.ops.rms_norm has TWO dtype boundaries: after normalize to
    # weight.dtype, then after the weighted result back to original dtype.
    # This is not asserted to match a different C/oink/batch-invariant provider.
    x = tl.where(n < H, value.to(tl.float32), 0.0)
    variance = tl.sum(x * x, axis=0) / H
    inv = tl.rsqrt(variance + EPS)
    weight = tl.load(gamma + n, mask=n < H, other=1.0)
    normalized = (x * inv).to(gamma.dtype.element_ty)
    result = (normalized.to(tl.float32) * weight.to(tl.float32)).to(y.dtype.element_ty)
    tl.store(y + m * H + n, result, mask=n < H)


_ENABLED = os.environ.get("MTP_DOWN_COMBINE_NORM", "1") != "0"
_ENTRIES = {}


def _words(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _guarded(m, device):
    owner = torch.empty(m * 2688 + 128, dtype=torch.bfloat16, device=device)
    return owner, owner[64:-64].view(m, 2688)


def _poison(pair, byte):
    pair[0].view(torch.uint8).fill_(byte)


def _guard(pair, byte):
    return bool(torch.all(pair[0][:64].view(torch.uint8) == byte).item()
                and torch.all(pair[0][-64:].view(torch.uint8) == byte).item())


def can_combine(spec, y, residual, scale, topk):
    return (_ENABLED and spec.gamma is spec.norm.weight
            and float(spec.norm.variance_epsilon) == spec.eps
            and y.dtype == torch.bfloat16 and y.is_cuda and y.is_contiguous()
            and y.ndim == 2 and y.shape[1] == 2688 and y.shape[0] in spec.rows
            and residual is not None and residual.dtype == y.dtype
            and residual.shape == y.shape and residual.device == y.device
            and residual.is_contiguous() and spec.gamma.device == y.device
            and topk in (3, 6) and float(scale) == spec.scale)


def launch(spec, part, weights, residual, y, pdl):
    combine_end_norm[(y.shape[0],)](
        part, weights, y, spec.gamma, M=y.shape[0], H=2688,
        ROUTED_SCALE=spec.scale, TOPK=weights.numel(), S=2,
        BN=4096, EPS=spec.eps, launch_pdl=pdl,
        residual_ptr=residual, ADD_RES=True, num_warps=8, num_stages=1)


def entry_for(layer):
    spec = _ENTRIES.get(layer._fast_moe_name)
    if (spec is None or spec.norm is not layer.final_layernorm
            or spec.gamma is not layer.final_layernorm.weight
            or spec.eps != float(layer.final_layernorm.variance_epsilon)
            or spec.scale != float(layer.mixer.routed_scaling_factor)):
        raise RuntimeError("admitted MTP combine+norm immutable owner changed")
    return spec


@torch.no_grad()
def _actual_part(layer, topk, gen):
    from .nemotron_h_mtp_kernels import relu2_prepare, mtp_pdl
    from .nemotron_h_mtp_compact import _compact, _down_slots, _BF16_BN
    from .nemotron_h_mtp_down_packed import maybe_slots
    _, w2, _, sd, inter, ns = layer._fast_moe_weights
    packed = relu2_prepare(w2, sd, inter, ns)
    if packed is False or ns != 2 or inter != 1856:
        raise RuntimeError("actual MTP compact-down owner unsupported")
    device = w2.device
    slots = topk + 2
    h = torch.randn((slots, 1856), device=device, generator=gen).relu().square().to(torch.bfloat16)
    h *= (torch.rand(h.shape, device=device, generator=gen) < 0.5)
    indices = torch.empty(h.shape, dtype=torch.int32, device=device)
    values = torch.empty_like(h)
    counts = torch.empty(slots, dtype=torch.int32, device=device)
    ids = ((torch.arange(topk, device=device, dtype=torch.int32) + 65) % 128).contiguous()
    _compact[(slots,)](h, indices, values, counts, I=1856, BLOCK=2048, num_warps=4,
                      launch_pdl=mtp_pdl())
    part = torch.empty((slots, 2688), dtype=torch.float32, device=device)
    if not maybe_slots(packed, indices, values, counts, ids, part, mtp_pdl()):
        _down_slots[(triton.cdiv(2688, _BF16_BN), slots)](
            indices, values, counts, *packed, ids, part,
            H=2688, I=1856, TOPK=topk, BN=_BF16_BN, BK=64,
            num_warps=4, num_stages=3, launch_pdl=mtp_pdl())
    return part


@torch.no_grad()
def _qualify_row(layer, spec, m, topk, pdl):
    from .nemotron_h_mtp_compact import _down_combine
    device = spec.gamma.device
    gen = torch.Generator(device=device).manual_seed(2026102711 + m + topk)
    part = _actual_part(layer, topk, gen)
    actual = part.clone()
    weights = torch.rand(topk, dtype=torch.float32, device=device, generator=gen)
    weights /= weights.sum()
    residual = torch.empty((m, 2688), dtype=torch.bfloat16, device=device)
    oldtmp, new = _guarded(m, device), _guarded(m, device)
    args = dict(M=m, H=2688, ROUTED_SCALE=spec.scale, TOPK=topk, S=2,
                BN=256, num_warps=4, launch_pdl=pdl,
                residual_ptr=residual, ADD_RES=True)

    def reference(tmp):
        _down_combine[(triton.cdiv(2688, 256),)](part, weights, tmp, **args)
        return spec.norm_fn(tmp)

    def change(case):
        scale = (0.0, 0.001, 1.0, 32.0)[case % 4]
        if case == 0:
            part.copy_(actual)
        else:
            part.copy_(torch.randn(part.shape, device=device, generator=gen) * scale)
        weights.copy_(torch.rand(topk, device=device, generator=gen))
        weights.div_(weights.sum())
        residual.copy_((torch.randn(residual.shape, device=device, generator=gen)
                        * scale).to(torch.bfloat16))
        if case == 4:
            residual[:, ::2] = -0.0
            residual.view(torch.int16)[:, 1:5] = torch.tensor(
                [1, -32767, 128, -32640], dtype=torch.int16, device=device)
        if case == 5:
            # Cancellation is against the actual old BF16 combined result.
            _down_combine[(triton.cdiv(2688, 256),)](
                part, weights, oldtmp[1], M=m, H=2688, ROUTED_SCALE=spec.scale,
                TOPK=topk, S=2, BN=256, num_warps=4, launch_pdl=pdl)
            residual.copy_(-oldtmp[1])
            residual[:, ::3] += 0.00390625
        if case == 6:
            part[0, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0],
                                       dtype=torch.float32, device=device)
        if case == 7:
            part.mul_(1e20)

    for case in range(8):
        change(case)
        saved = tuple(t.clone() for t in (part, weights, residual, spec.gamma))
        _poison(oldtmp, 0x7F)
        _poison(new, 0xA5)
        ref = reference(oldtmp[1])
        launch(spec, part, weights, residual, new[1], pdl)
        if not (_words(ref, new[1]) and _guard(oldtmp, 0x7F) and _guard(new, 0xA5)
                and all(_words(a, b) for a, b in zip((part, weights, residual, spec.gamma), saved))):
            raise RuntimeError(f"combine+norm owner words/guards/read-only differ case={case}")

    # The old path is the actual combine plus compiled actual layer norm.
    # No eager norm surrogate or guarded-destination output copy is timed.
    oldtemps = [_guarded(m, device) for _ in range(8)]
    newouts = [_guarded(m, device) for _ in range(8)]
    oldouts = []
    change(2)
    for pair in oldtemps:
        _poison(pair, 0x7F)
        reference(pair[1])
    for pair in newouts:
        _poison(pair, 0xA5)
        launch(spec, part, weights, residual, pair[1], pdl)
    torch.cuda.synchronize(device)
    oldgraph, newgraph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(oldgraph, pool=torch.cuda.graph_pool_handle()):
        for pair in oldtemps:
            oldouts.append(reference(pair[1]))
    if len({t.data_ptr() for t in oldouts}) != 8:
        raise RuntimeError("old norm graph did not retain eight distinct full outputs")
    with torch.cuda.graph(newgraph, pool=torch.cuda.graph_pool_handle()):
        for pair in newouts:
            launch(spec, part, weights, residual, pair[1], pdl)
    for case in range(8):
        change(case)
        saved = tuple(t.clone() for t in (part, weights, residual, spec.gamma))
        for _ in range(2):
            oldgraph.replay()
            newgraph.replay()
            if not (all(_words(a, b[1]) for a, b in zip(oldouts, newouts))
                    and all(_guard(x, 0x7F) for x in oldtemps)
                    and all(_guard(x, 0xA5) for x in newouts)
                    and all(_words(a, b) for a, b in zip((part, weights, residual, spec.gamma), saved))):
                raise RuntimeError("combine+norm changed-input graph words/guards/read-only differ")

    pressure = torch.empty(64 * (1 << 20), dtype=torch.uint8, device=device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graphs = (oldgraph, newgraph)
    ratios, old_total, new_total = [], 0.0, 0.0
    for case in (0, 1, 2, 3, 4, 5):
        change(case)
        for cold in (False, True):
            times = [[], []]
            for trial in range(5):
                for arm in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    graphs[arm].replay()
                    elapsed = 0.0
                    for repeat in range(8):
                        if cold:
                            pressure.fill_(case + trial + repeat)
                        begin.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        elapsed += begin.elapsed_time(end) / 64.0
                    if not elapsed > 0.0:
                        raise RuntimeError("combine+norm nonpositive event interval")
                    times[arm].append(elapsed)
            a, b = (statistics.median(t) for t in times)
            ratios.append(b / a)
            old_total += a
            new_total += b
    metrics = statistics.median(ratios), new_total / old_total, max(ratios)
    if not (metrics[0] < 0.99 and metrics[1] < 0.99 and metrics[2] < 1.01):
        raise RuntimeError(f"combine+norm full-work warm/pressure refused {metrics}")
    return metrics


@torch.no_grad()
def prepare(layer):
    layer._mtp_combine_norm_rows = ()
    if not (_ENABLED and getattr(layer, "has_end_norm", False)
            and getattr(layer, "_mtp_fold_residual_ok", False)):
        return
    from .nemotron_h_mtp_kernels import mtp_pdl, _RELU2_MODE
    from .nemotron_h_mtp_compact import _SLOTS
    from vllm.model_executor.layers.layernorm import RMSNorm
    norm = layer.final_layernorm
    gamma = norm.weight
    method = getattr(getattr(norm, "_forward_method", None), "__func__", None)
    if (_RELU2_MODE != 2 or not _SLOTS or torch.cuda.is_current_stream_capturing()
            or envs.VLLM_BATCH_INVARIANT or not isinstance(norm, RMSNorm)
            or method not in (RMSNorm.forward_native, RMSNorm.forward_cuda)
            or norm.variance_size_override is not None or not norm.pass_weight
            or gamma.dtype != torch.bfloat16 or not gamma.is_cuda
            or gamma.shape != (2688,) or not gamma.is_contiguous()
            or torch.cuda.get_device_capability(gamma.device) != (12, 1)):
        print("MTP_DOWN_COMBINE_NORM unsupported original norm/owner/mode; original kept", flush=True)
        return
    eps = float(norm.variance_epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        return
    probe = torch.empty((1, 2688), dtype=torch.bfloat16, device=gamma.device)
    provider = ir.ops.rms_norm.dispatch(probe, gamma, eps, None).provider
    if provider != "native":
        print(f"MTP_DOWN_COMBINE_NORM original provider={provider}; original kept", flush=True)
        return

    def actual_norm(x):
        return norm(x)

    norm_fn = torch.compile(actual_norm, fullgraph=True, dynamic=True,
                            options={"triton.cudagraphs": False})
    spec = SimpleNamespace(norm=norm, gamma=gamma, eps=eps,
                           scale=float(layer.mixer.routed_scaling_factor),
                           rows=(), norm_fn=norm_fn)
    gamma_saved = gamma.clone()
    safe, metrics = [], {}
    for m in (1, 5, 8):
        try:
            # Warm the ordinary offset0 fallback shape before any private
            # capture; the guarded reference also warms its offset64 view.
            spec.norm_fn(torch.zeros((m, 2688), dtype=torch.bfloat16, device=gamma.device))
            row_metrics = [_qualify_row(layer, spec, m, topk, mtp_pdl()) for topk in (3, 6)]
            safe.append(m)
            metrics[m] = row_metrics
        except Exception as exc:
            print(f"MTP_DOWN_COMBINE_NORM row={m} refused: {exc!r}; original kept", flush=True)
    if not _words(gamma, gamma_saved):
        raise RuntimeError("combine+norm immutable actual gamma modified")
    if safe:
        spec.rows = tuple(safe)
        _ENTRIES[layer._fast_moe_name] = spec
        layer._mtp_combine_norm_rows = spec.rows
    print(f"MTP_DOWN_COMBINE_NORM admitted rows={safe} provider={provider} ratios={metrics}", flush=True)

