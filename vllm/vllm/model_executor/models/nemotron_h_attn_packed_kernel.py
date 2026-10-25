"""Triton GEMM for the lossless 12-bit attention projections (see nemotron_h_attn_packed.py).

Imports only torch and triton so a child process can compile and run it in isolation.
"""

import torch
import triton
import triton.language as tl

ROWS = 5
# (columns per program, warps), tried in this order; BN is a multiple of 16.
CONFIGS = ((32, 4), (64, 4), (16, 4))


@triton.jit
def _packed_gemm(
    x_ptr, sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr, y_ptr,
    M, N, KB, stride_xm, stride_ym,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    offs_m = tl.arange(0, BM)
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    row = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, KB):
        blk = offs_n * KB + ki
        rid = tl.load(ri_ptr + blk)
        bs = tl.load(base_ptr + blk).to(tl.int32)
        flat = blk[None, :] * BK + offs_k[:, None]
        sm = tl.load(sm_ptr + flat).to(tl.int32)
        nib = tl.load(ex_ptr + (flat >> 1)).to(tl.int32)
        code = (nib >> ((flat & 1) * 4)) & 15
        e = tl.where(code == 0, 0, bs[None, :] + code - 1)
        bits = ((sm & 128) << 8) | (e << 7) | (sm & 127)
        esc = rid >= 0
        rw = tl.load(
            raw_ptr + rid[None, :].to(tl.int64) * BK + offs_k[:, None],
            mask=esc[None, :],
            other=0,
        ).to(tl.int32) & 65535
        bits = tl.where(esc[None, :], rw, bits)
        b = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (ki * BK + offs_k)[None, :],
            mask=row[:, None],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=row[:, None],
    )


def run(x, sm, ex, base, ri, raw, n: int, k: int, cfg: tuple[int, int]) -> torch.Tensor:
    """y = x @ W.T for W stored in the planes (raw is the int16 view of the uint16 plane)."""
    bn, warps = cfg
    y = x.new_empty((x.shape[0], n))
    _packed_gemm[(n // bn,)](
        x, sm, ex, base, ri, raw, y,
        x.shape[0], n, k // 128, x.stride(0), y.stride(0),
        BM=16, BN=bn, BK=128, num_warps=warps, num_stages=1,
    )
    return y
