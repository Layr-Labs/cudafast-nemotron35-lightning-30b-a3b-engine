"""Triton GEMM for the lossless 12-bit attention projections (see nemotron_h_attn_packed.py).

Base imports only torch/triton; optional Gluon adapter imports the existing native MTP helper lazily.
"""

import torch
import triton
import triton.language as tl

ROWS = 5
# (columns per program, warps), tried in this order; BN is a multiple of 16.
CONFIGS = ((32, 4), (64, 4), (16, 4),
           (32, 4, True, True), (64, 4, True, True), (16, 4, True, True),
           (64, 4, "gluon"))


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


@triton.jit
def _decode_word_ahead(sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr,
                       cols, kk, ki, valid, KB: tl.constexpr, BK: tl.constexpr,
                       BN: tl.constexpr):
    blk = cols * KB + ki
    rid = tl.load(ri_ptr + blk, mask=valid, other=-1)
    coded = valid & (rid < 0)
    bs = tl.load(base_ptr + blk, mask=coded, other=0).to(tl.int32)
    flat = blk[None, :] * BK + kk[:, None]
    sm = tl.load(sm_ptr + flat, mask=coded[None, :], other=0).to(tl.int32)
    # Four bytes carry eight consecutive K nibbles. Each group loads one
    # aligned u32 and expands its eight nibbles locally, in original K order.
    group = tl.arange(0, BK // 8)
    nibble = tl.arange(0, 8)
    words = tl.load(ex_ptr.to(tl.pointer_type(tl.uint32))
                    + blk[None, :] * (BK // 8) + group[:, None],
                    mask=coded[None, :], other=0)
    code = ((words[:, None, :] >> (4 * nibble[None, :, None])) & 15)
    code = tl.reshape(code, (BK, BN)).to(tl.int32)
    e = tl.where(code == 0, 0, bs[None, :] + code - 1)
    bits = ((sm & 128) << 8) | (e << 7) | (sm & 127)
    esc = valid & (rid >= 0)
    rw = tl.load(raw_ptr + rid[None, :].to(tl.int64) * BK + kk[:, None],
                 mask=esc[None, :], other=0).to(tl.int32) & 65535
    return tl.where(esc[None, :], rw, bits).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _packed_gemm_wordlookahead(
    x_ptr, sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr, y_ptr,
    M, N, KB: tl.constexpr, stride_xm, stride_ym,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    mm = tl.arange(0, BM)
    nn = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    row = mm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    b = _decode_word_ahead(sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr,
                      nn, kk, 0, True, KB, BK, BN)
    for ki in range(0, KB):
        a = tl.load(x_ptr + mm[:, None] * stride_xm + (ki * BK + kk)[None, :],
                    mask=row[:, None], other=0.0)
        # Independent next-tile plane reads/bit reconstruction precede the
        # current dot. The current accumulator still sees K0,K1,... exactly once.
        b_next = _decode_word_ahead(sm_ptr, ex_ptr, base_ptr, ri_ptr, raw_ptr,
                               nn, kk, ki + 1, ki + 1 < KB, KB, BK, BN)
        acc = tl.dot(a, b, acc)
        b = b_next
    tl.store(y_ptr + mm[:, None] * stride_ym + nn[None, :],
             acc.to(y_ptr.dtype.element_ty), mask=row[:, None])


def run(x, sm, ex, base, ri, raw, n: int, k: int, cfg: tuple) -> torch.Tensor:
    """y = x @ W.T for W stored in the planes (raw is the int16 view of the uint16 plane)."""
    bn, warps = cfg[:2]
    if len(cfg) == 3 and cfg[2] == "gluon":
        from vllm.model_executor.models.nemotron_h_mtp_packed import packed_small

        y = x.new_empty((x.shape[0], n))
        # GATED=False removes every gate read. Reuse x as the unused pointer;
        # unlike the source adapter, allocate/fill no dummy device gate tensor.
        packed_small[(triton.cdiv(n, bn),)](
            x, sm, ex, base, ri, raw, y,
            x.shape[0], n, k, x.stride(0), y.stride(0), x,
            BM=16, BN=bn, BK=128, GATED=False, WORD=False,
            num_warps=warps, num_stages=4, launch_pdl=False,
        )
        return y
    kernel = _packed_gemm_wordlookahead if len(cfg) == 4 else _packed_gemm
    y = x.new_empty((x.shape[0], n))
    kernel[(n // bn,)](
        x, sm, ex, base, ri, raw, y,
        x.shape[0], n, k // 128, x.stride(0), y.stride(0),
        BM=16, BN=bn, BK=128, num_warps=warps, num_stages=1,
    )
    return y
