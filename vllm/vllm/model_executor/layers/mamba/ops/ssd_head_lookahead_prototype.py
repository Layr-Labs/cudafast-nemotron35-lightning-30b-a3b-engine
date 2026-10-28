# SPDX-License-Identifier: Apache-2.0
"""Unwired prefill scan experiment: issue each next-head state read before
current-head math. No local compilation, execution, or measured claim.
"""
from vllm.triton_utils import tl, triton
from .triton_helpers import fast_exp

@triton.jit
def _cb_scan_head_lookahead_kernel(
    X,
    B,
    C,
    DT,
    DA,
    CU,
    SEQ,
    STATES,
    INIT,
    D,
    Z,
    OUT,
    H: tl.constexpr,
    N: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK: tl.constexpr,
    XS: tl.constexpr,
    BS: tl.constexpr,
    CS: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    SS: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    ZS: tl.constexpr,
    DS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_D: tl.constexpr,
    D_VECTOR: tl.constexpr,
    HAS_Z: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    DN: tl.constexpr,
    HSPLIT: tl.constexpr = 1,
):
    tile = tl.program_id(0)
    chunk = tl.program_id(1)
    group = tl.program_id(2) // HSPLIT
    part = tl.program_id(2) % HSPLIT
    begin = tl.load(CU + chunk)
    limit = tl.load(CU + chunk + 1) - begin
    rows = tile * BM + tl.arange(0, BM)
    columns = tl.arange(0, CHUNK)
    inner = tl.arange(0, BC)
    cp = C + begin * CS[0] + group * CS[1]
    bp = B + begin * BS[0] + group * BS[1]

    # One CB computation for the whole group, outside the eight-head loop.
    # Preserve the pinned BMM's ascending 32-element K groups and FP32 sum.
    cb_full = tl.zeros((BM, CHUNK), tl.float32)
    for k in range(0, N, BC):
        c = tl.load(
            cp + rows[:, None] * CS[0] + (k + inner[None, :]) * CS[2],
            mask=(rows[:, None] < limit) & (inner[None, :] < N - k),
            other=0.0,
        )
        b = tl.load(
            bp + (k + inner[:, None]) * BS[2] + columns[None, :] * BS[0],
            mask=(inner[:, None] < N - k) & (columns[None, :] < limit),
            other=0.0,
        )
        cb_full += tl.dot(c, b)
    seq = tl.load(SEQ + chunk)
    previous_seq = tl.load(SEQ + chunk - 1, mask=chunk > 0, other=-1)
    ns = tl.arange(0, BN)
    ds = tl.arange(0, DN)
    ks = tl.arange(0, BK)
    c_state = tl.load(
        cp + rows[:, None] * CS[0] + ds[None, :] * CS[2],
        mask=(rows[:, None] < limit) & (ds[None, :] < N),
        other=0.0,
    )

    # Prime the first head; the next head's state is read before this head's
    # dot/scan, retaining the original order and dtype for every head.
    first_head = group * RATIO + part * (RATIO // HSPLIT)
    if HAS_INIT and seq != previous_seq:
        prev = tl.load(INIT + seq * IS[0] + first_head * IS[1]
                       + ns[None, :] * IS[2] + ds[:, None] * IS[3],
                       mask=(ns[None, :] < H) & (ds[:, None] < N), other=0.0).to(C.dtype.element_ty)
    elif not HAS_INIT and seq != previous_seq:
        prev = tl.zeros((DN, BN), C.dtype.element_ty)
    else:
        prev = tl.load(STATES + (chunk - 1) * SS[0] + first_head * SS[1]
                       + ns[None, :] * SS[2] + ds[:, None] * SS[3],
                       mask=(ns[None, :] < H) & (ds[:, None] < N), other=0.0).to(C.dtype.element_ty)
    for part_head in range(RATIO // HSPLIT):
        local_head = part * (RATIO // HSPLIT) + part_head
        head = group * RATIO + local_head
        xp = X + begin * XS[0] + head * XS[1]
        tp = DT + head * TS[0] + chunk * TS[1]
        ap = DA + head * AS[0] + chunk * AS[1]
        next_prev = prev
        if part_head + 1 < RATIO // HSPLIT:
            next_head = head + 1
            if HAS_INIT and seq != previous_seq:
                next_prev = tl.load(INIT + seq * IS[0] + next_head * IS[1]
                                    + ns[None, :] * IS[2] + ds[:, None] * IS[3],
                                    mask=(ns[None, :] < H) & (ds[:, None] < N), other=0.0).to(C.dtype.element_ty)
            elif not HAS_INIT and seq != previous_seq:
                next_prev = tl.zeros((DN, BN), C.dtype.element_ty)
            else:
                next_prev = tl.load(STATES + (chunk - 1) * SS[0] + next_head * SS[1]
                                    + ns[None, :] * SS[2] + ds[:, None] * SS[3],
                                    mask=(ns[None, :] < H) & (ds[:, None] < N), other=0.0).to(C.dtype.element_ty)
        dA_cs_m = tl.load(ap + rows * AS[2], mask=rows < CHUNK, other=0.0).to(
            tl.float32
        )
        acc = tl.dot(c_state, prev) * fast_exp(dA_cs_m)[:, None]
        kmax = tl.minimum((tile + 1) * BM, limit)
        for k in range(0, kmax, BK):
            # Gather only register tensor values: no global CB scratch.
            indices = tl.broadcast_to((k + ks)[None, :], (BM, BK))
            cb = tl.gather(cb_full, indices, axis=1)
            dA_cs_k = tl.load(ap + (k + ks) * AS[2], mask=ks < CHUNK - k, other=0.0).to(
                tl.float32
            )
            cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
            dt_k = tl.load(tp + (k + ks) * TS[2], mask=ks < CHUNK - k, other=0.0).to(
                tl.float32
            )
            cb *= dt_k
            cb = tl.where(rows[:, None] >= k + ks[None, :], cb, 0.0)
            cb = cb.to(X.dtype.element_ty)
            x = tl.load(
                xp + (k + ks[:, None]) * XS[0] + ns[None, :] * XS[2],
                mask=(ks[:, None] < limit - k) & (ns[None, :] < H),
                other=0.0,
            )
            acc += tl.dot(cb, x)
        if HAS_D:
            if D_VECTOR:
                d = tl.load(D + head * DS + ns, mask=ns < H, other=0.0).to(tl.float32)
            else:
                d = tl.load(D + head * DS).to(tl.float32)
            residual = tl.load(
                xp + rows[:, None] * XS[0] + ns[None, :] * XS[2],
                mask=(rows[:, None] < limit) & (ns[None, :] < H),
                other=0.0,
            ).to(tl.float32)
            acc += residual * d
        if HAS_Z:
            z = tl.load(
                Z
                + (begin + rows[:, None]) * ZS[0]
                + head * ZS[1]
                + ns[None, :] * ZS[2],
                mask=(rows[:, None] < limit) & (ns[None, :] < H),
                other=0.0,
            ).to(tl.float32)
            acc *= z * tl.sigmoid(z)
        tl.store(
            OUT + (begin + rows[:, None]) * OS[0] + head * OS[1] + ns[None, :] * OS[2],
            acc,
            mask=(rows[:, None] < limit) & (ns[None, :] < H),
        )
        prev = next_prev

