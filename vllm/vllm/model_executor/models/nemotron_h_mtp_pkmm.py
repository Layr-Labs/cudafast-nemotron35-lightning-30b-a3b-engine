"""packed_mm: one fp32 chain of m16n8k16 MMAs over K in ascending order, rounded once to
BF16, for 2..16 rows against a weight held in the drafter's lossless 12-bit storage
(nemotron_h_mtp_packed). It is the chain packed_small runs (mma_v2 over k chunks,
ascending), and that of _mtp_small_gemm_kernel's tl.dot, on narrower N tiles so the
projections of the drafter's first pass run on more programs. _CONFIGS lists the
shapes it takes (BN, warps along N, BK); the k-chunk width does not change the chain.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from .nemotron_h_mtp_packed import load_packed, pdl_prologue

_CONFIGS = {(4608, 2688): (32, 4, 256), (2688, 4096): (16, 2, 256)}


@gluon.jit
def packed_mm(x, SM, EX, BASE, RI, RAW, y, M, N, K, sxm, sym,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WN: tl.constexpr,
              launch_pdl: tl.constexpr = False, gate_ptr=None, GATED: tl.constexpr = False,
              WORD: tl.constexpr = False):
    """y[m, n] = bf16(sum over k16 steps ascending of mma(x[m, k], W[n, k])) for m < M <= BM.
    GATED: once the depth gate's stop flag is set, the zeros its masked loads would give."""
    if launch_pdl:
        pdl_prologue()
    ld: tl.constexpr = BlockedLayout([1, 8], [2, 16], [WN, 1], [1, 0])
    mma: tl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, WN], instr_shape=[16, 8])
    pid = tl.program_id(0)
    nn = pid * BN + tl.arange(0, BN, layout=SliceLayout(1, ld))
    kk0 = tl.arange(0, BK, layout=SliceLayout(0, ld))
    mm = tl.arange(0, BM, layout=SliceLayout(1, ld))
    acc = tl.full((BM, BN), 0, tl.float32, mma)
    kend = K
    if GATED:
        kend = tl.where(tl.load(gate_ptr) != 0, 0, K)
    for k0 in range(0, kend, BK):
        kk = k0 + kk0
        wmask = (nn[:, None] < N) & (kk[None, :] < K)
        w = load_packed(SM, EX, BASE, RI, RAW, nn[:, None] * K + kk[None, :], wmask, WORD=WORD)
        a = tl.load(x + mm[:, None] * sxm + kk[None, :],
                    mask=(mm[:, None] < M) & (kk[None, :] < K), other=0.)
        a = tl.convert_layout(a, DotOperandLayout(0, mma, 2))
        b = tl.convert_layout(tl.permute(w, (1, 0)), DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
    out: tl.constexpr = BlockedLayout([1, 4], [4, 8], [WN, 1], [1, 0])
    acc = tl.convert_layout(acc, out)
    om = tl.arange(0, BM, layout=SliceLayout(1, out))
    on = pid * BN + tl.arange(0, BN, layout=SliceLayout(0, out))
    tl.store(y + om[:, None] * sym + on[None, :], acc.to(y.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))
