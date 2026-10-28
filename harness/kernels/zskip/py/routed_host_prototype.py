"""Experimental host wrapper; do not import, build or run locally.

The caller must complete the existing side-stream join before finish() and
compare both output and down-workspace to the original at remote startup.
"""
from __future__ import annotations

import ctypes

import torch


def _api(lib):
    P, LL, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
    size_fn = getattr(lib, "zskip_routed_sizes_prototype", None)
    start_fn = getattr(lib, "zskip_routed_start_prototype", None)
    finish_fn = getattr(lib, "zskip_routed_finish_prototype", None)
    if size_fn is None or start_fn is None or finish_fn is None:
        return None
    size_fn.argtypes = [I, I, P, P, P, P]
    size_fn.restype = I
    start_fn.argtypes = [P, LL, I, P, P, P, I, P, P, P, P, LL,
                         LL, LL, LL, LL, LL, LL, P, P, P, P, P, P, P, I]
    start_fn.restype = I
    finish_fn.argtypes = [I, P, LL, P, P, P, P, P]
    finish_fn.restype = I
    return size_fn, start_fn, finish_fn


def _span(t):
    """Conservative byte range for a nonnegative-stride, nonempty tensor."""
    if t.numel() == 0 or any(s < 0 for s in t.stride()):
        return None
    last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    begin = t.data_ptr()
    return begin, begin + (last + 1) * t.element_size()


def _overlaps(a, b):
    x, y = _span(a), _span(b)
    return x is None or y is None or (x[0] < y[1] and y[0] < x[1])


def start(lib, activations, sorted_ids, expert_ids, padded, packed, config, down,
          weights, output, verify_rows=None, force_fallback=False):
    """Return a handle for finish, or None before any launch on guard failure.

    This prototype is only defined for checked config 663, six float32 weights
    per token and 30/36 activation rows. A live handle owns all scratch until
    finish returns; graph capture must retain its allocations for each replay.
    """
    api = _api(lib)
    if api is None or activations.ndim != 2 or down.ndim != 2:
        return None
    rows = activations.shape[0]
    if (rows not in (30, 36) or activations.shape[1] != 1856 or
        down.shape != (rows, 2688) or activations.dtype != torch.bfloat16 or
        down.dtype != torch.bfloat16 or weights.dtype != torch.float32 or
        weights.shape != (rows // 6, 6) or not weights.is_contiguous() or
        activations.stride(1) != 1 or activations.stride(0) % 8 or
        down.stride(1) != 1 or down.stride(0) % 2 or
        activations.data_ptr() % 16 or packed.layout != "swap2" or config != 663 or
        _overlaps(activations, down)):
        return None
    tensors = (activations, down, weights, sorted_ids, expert_ids, padded,
               packed.w, packed.s, packed.gs)
    if any(not t.is_cuda or t.device != activations.device for t in tensors):
        return None
    if (output.dtype != torch.bfloat16 or output.shape != (rows // 6, 2688) or
        not output.is_contiguous() or output.device != down.device or
        _overlaps(output, down) or _overlaps(output, weights) or
        any(_overlaps(down, t) for t in tensors[2:])):
        return None
    if (sorted_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32 or
        padded.dtype != torch.int32 or not sorted_ids.is_contiguous() or
        not expert_ids.is_contiguous() or not padded.is_contiguous() or
        padded.numel() != 1):
        return None
    if verify_rows is not None and (verify_rows.dtype != torch.int32 or
        verify_rows.device != activations.device or verify_rows.numel() != 1 or
        not verify_rows.is_contiguous() or _overlaps(down, verify_rows)):
        return None
    max_blocks = min(rows, expert_ids.numel())
    if sorted_ids.numel() < 16 * max_blocks:
        return None
    sizes = [ctypes.c_size_t() for _ in range(4)]
    rc = api[0](rows, max_blocks, *(ctypes.byref(x) for x in sizes))
    if rc != 0:
        return None
    # Fresh allocations are deliberately retained in the returned handle.
    buffers = [torch.empty(x.value, dtype=torch.uint8, device=activations.device)
               for x in sizes]
    rc = api[1](activations.data_ptr(), activations.stride(0), rows,
        sorted_ids.data_ptr(), expert_ids.data_ptr(), padded.data_ptr(), max_blocks,
        packed.w.data_ptr(), packed.s.data_ptr(), packed.gs.data_ptr(),
        down.data_ptr(), down.stride(0), packed.w_expert, packed.w_tile, packed.w_k,
        packed.s_expert, packed.s_tile, packed.s_g,
        verify_rows.data_ptr() if verify_rows is not None else None,
        weights.data_ptr(), *(t.data_ptr() for t in buffers),
        torch.cuda.current_stream(activations.device).cuda_stream,
        int(force_fallback))
    if rc != 0:
        raise RuntimeError(f"routed start prototype launch failed: {rc}")
    return api, buffers, down, weights, rows


def finish(handle, output):
    api, buffers, down, weights, rows = handle
    if (output.dtype != torch.bfloat16 or output.shape != (rows // 6, 2688) or
        not output.is_contiguous() or output.device != down.device or
        _overlaps(output, down) or _overlaps(output, weights)):
        # A launched start cannot be retried as a safe ordinary path here.
        raise ValueError("routed output must satisfy guards before start")
    rc = api[2](rows, down.data_ptr(), down.stride(0), buffers[2].data_ptr(),
                weights.data_ptr(), buffers[3].data_ptr(), output.data_ptr(),
                torch.cuda.current_stream(down.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"routed finish prototype launch failed: {rc}")
