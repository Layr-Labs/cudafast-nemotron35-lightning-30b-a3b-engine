# SPDX-License-Identifier: Apache-2.0
"""Two-stage GB10 prefill SSD: state+passing, then group-shared CB+scan.

The existing cumsum remains outside this module. An optional single-sequence
terminal cache sink stores the unchanged final FP32 recurrence in registers.
Raw chunk states and CB never become global tensors. All chunk boundaries
are still returned in FP32, including boundaries used by prefix-cache callers.
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.logger import init_logger

logger = init_logger(__name__)
_SSD_STATE_NSPLIT = int(os.environ.get("VLLM_SSD_STATE_NSPLIT", "1"))
_SSD_SCAN_HSPLIT = int(os.environ.get("VLLM_SSD_SCAN_HSPLIT", "2"))
_final_logged = False
from vllm.triton_utils import tl, triton

from .triton_helpers import fast_exp


@triton.jit
def _state_pass_kernel(
    X,
    B,
    DT,
    DA,
    CU,
    LAST,
    INIT,
    OUT,
    FINAL_CACHE,
    FINAL_INDEX,
    H: tl.constexpr,
    N: tl.constexpr,
    RATIO: tl.constexpr,
    CHUNK: tl.constexpr,
    XS: tl.constexpr,
    BS: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
    FS: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(LAST + batch - 1, mask=batch > 0, other=-1) + 1
    end = tl.load(LAST + batch) + 1
    m = tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    state_offsets = m[:, None] * OS[2] + n[None, :] * OS[3]
    state_mask = (m[:, None] < H) & (n[None, :] < N)
    if HAS_INIT:
        states = tl.load(
            INIT
            + batch * IS[0]
            + head * IS[1]
            + m[:, None] * IS[2]
            + n[None, :] * IS[3],
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        states = tl.zeros((BM, BN), tl.float32)
    for chunk in range(start, end):
        begin = tl.load(CU + chunk)
        limit = tl.load(CU + chunk + 1) - begin
        xp = X + begin * XS[0] + head * XS[1]
        bp = B + begin * BS[0] + (head // RATIO) * BS[1]
        tp = DT + head * TS[0] + chunk * TS[1]
        ap = DA + head * AS[0] + chunk * AS[1]
        dA_cs = tl.load(ap + (CHUNK - 1) * AS[2]).to(tl.float32)
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(0, limit, BK):
            x = tl.load(
                xp + m[:, None] * XS[2] + (k + kk[None, :]) * XS[0],
                mask=(m[:, None] < H) & (kk[None, :] < limit - k),
                other=0.0,
            )
            b = tl.load(
                bp + (k + kk[:, None]) * BS[0] + n[None, :] * BS[2],
                mask=(kk[:, None] < limit - k) & (n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            dA_cs_k = tl.load(ap + (k + kk) * AS[2], mask=kk < limit - k, other=0.0).to(
                tl.float32
            )
            dt_k = tl.load(tp + (k + kk) * TS[2], mask=kk < limit - k, other=0.0).to(
                tl.float32
            )
            scale = fast_exp(tl.minimum(dA_cs - dA_cs_k, 0.0)) * dt_k
            b *= scale[:, None]
            b = b.to(X.dtype.element_ty)
            acc += tl.dot(x, b)
        # Same FP32 recurrence as _state_passing_fwd_kernel, in chunk order.
        new_states = acc.to(tl.float32)
        states = fast_exp(dA_cs) * states + new_states
        tl.store(
            OUT + chunk * OS[0] + head * OS[1] + state_offsets, states, mask=state_mask
        )

    # Same terminal FP32 registers as OUT[end - 1]. No new reduction or rounding.
    # The host guard restricts this sink to one nonempty prefill sequence.
    if WRITE_FINAL:
        cache_slot = tl.load(FINAL_INDEX).to(tl.int64)
        cache_slot = tl.where(cache_slot < 0, cache_slot + CACHE_SLOTS, cache_slot)
        tl.store(
            FINAL_CACHE + cache_slot * FS[0] + head * FS[1]
            + m[:, None] * FS[2] + n[None, :] * FS[3],
            states,
            mask=state_mask,
        )


@triton.jit
def _cb_scan_kernel(
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

    for part_head in range(RATIO // HSPLIT):
        local_head = part * (RATIO // HSPLIT) + part_head
        head = group * RATIO + local_head
        xp = X + begin * XS[0] + head * XS[1]
        tp = DT + head * TS[0] + chunk * TS[1]
        ap = DA + head * AS[0] + chunk * AS[1]
        dA_cs_m = tl.load(ap + rows * AS[2], mask=rows < CHUNK, other=0.0).to(
            tl.float32
        )
        if HAS_INIT and seq != previous_seq:
            prev = tl.load(
                INIT
                + seq * IS[0]
                + head * IS[1]
                + ns[None, :] * IS[2]
                + ds[:, None] * IS[3],
                mask=(ns[None, :] < H) & (ds[:, None] < N),
                other=0.0,
            ).to(C.dtype.element_ty)
        elif not HAS_INIT and seq != previous_seq:
            prev = tl.zeros((DN, BN), C.dtype.element_ty)
        else:
            prev = tl.load(
                STATES
                + (chunk - 1) * SS[0]
                + head * SS[1]
                + ns[None, :] * SS[2]
                + ds[:, None] * SS[3],
                mask=(ns[None, :] < H) & (ds[:, None] < N),
                other=0.0,
            ).to(C.dtype.element_ty)
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


def can_fuse_prefill(x, B, C, dt, dA, cu, last, seq, initial, state_dtype):
    """Shape/layout dispatch only; never read token values or device metadata."""
    if not (
        x.is_cuda
        and current_platform.is_device_capability(121)
        and x.ndim == 3
        and x.shape[1:] == (64, 64)
        and 0 < x.shape[0] <= 2048
        and B.shape == C.shape == (x.shape[0], 8, 128)
        and dt.ndim == 3
        and dt.shape == dA.shape
        and dt.shape[0] == 64
        and dt.shape[2] in (128, 256)
        and 0 < dt.shape[1] <= 16
        and state_dtype == torch.float32
        and last.ndim == seq.ndim == cu.ndim == 1
        and 0 < last.numel() <= dt.shape[1]
        and seq.numel() == dt.shape[1]
        and cu.numel() == dt.shape[1] + 1
    ):
        return False
    for tensor, dtype in (
        (x, torch.bfloat16),
        (B, torch.bfloat16),
        (C, torch.bfloat16),
        (dt, torch.float32),
        (dA, torch.float32),
    ):
        if tensor.dtype != dtype or tensor.device != x.device or tensor.stride(-1) != 1:
            return False
    for tensor in (cu, last, seq):
        if (
            tensor.device != x.device
            or tensor.dtype not in (torch.int32, torch.int64)
            or not tensor.is_contiguous()
        ):
            return False
    return initial is None or (
        initial.shape == (last.numel(), 64, 64, 128)
        and initial.dtype == torch.float32
        and initial.device == x.device
        and initial.stride(-1) == 1
    )


def fused_prefill(
    x, B, C, dt, dA, cu, last, seq, out, D=None, z=None, initial=None,
    final_cache=None, final_index=None,
):
    """Return all boundaries, or None when the terminal cache was already written.

    Every boundary is still materialized for the unchanged scan stage. Only a
    redundant final-state gather and cache scatter can be omitted.
    """
    _, heads, hdim = x.shape
    _, groups, dstate = B.shape
    _, chunks, chunk = dt.shape
    states = torch.empty(
        (chunks, heads, hdim, dstate), dtype=torch.float32, device=x.device
    )
    initial_strides = initial.stride() if initial is not None else (0, 0, 0, 0)
    write_final = (
        os.environ.get("VLLM_PREFILL_FINAL_CACHE", "1") != "0"
        and final_cache is not None
        and final_index is not None
        and last.numel() == 1
        and final_cache.ndim == 4
        and final_cache.shape[0] > 0
        and final_cache.shape[1:] == (heads, hdim, dstate)
        and final_cache.dtype == torch.float32
        and final_cache.device == x.device
        and final_cache.stride(-1) == 1
        and final_cache.stride(-2) == dstate
        and final_cache.stride(-3) == hdim * dstate
        and final_cache.stride(0) >= heads * hdim * dstate
        and final_index.ndim == 1
        and final_index.numel() == 1
        and final_index.is_contiguous()
        and final_index.dtype in (torch.int32, torch.int64)
        and final_index.device == x.device
    )
    final_strides = final_cache.stride() if write_final else (0, 0, 0, 0)
    global _final_logged
    if write_final and not _final_logged:
        _final_logged = True
        logger.info("ssd prefill: terminal state stored by the state kernel (cache sink active)")
    def launch_ssd(
        use_prefix, px, pb, pc, pdt, pda, pcu, plast, pseq,
        pinit, pstates, pout, pfinal, pindex,
    ):
        with torch.accelerator.device_index(px.device.index):
            nsplit = _SSD_STATE_NSPLIT if dstate % (16 * _SSD_STATE_NSPLIT) == 0 else 1
            hsplit = _SSD_SCAN_HSPLIT if (heads // groups) % _SSD_SCAN_HSPLIT == 0 else 1
            state_kernel = _state_pass_kernel
            state_grid = (plast.numel(), heads, nsplit)
            if use_prefix:
                from .ssd_prefix_owners import _state_prefix_owner_kernel

                state_kernel = _state_prefix_owner_kernel
                state_grid = (chunks, heads, nsplit)
            state_kernel[state_grid](
                px,
                pb,
                pdt,
                pda,
                pcu,
                plast,
                pinit,
                pstates,
                pfinal if write_final else None,
                pindex if write_final else None,
                H=hdim,
                N=dstate,
                RATIO=heads // groups,
                CHUNK=chunk,
                XS=px.stride(),
                BS=pb.stride(),
                TS=pdt.stride(),
                AS=pda.stride(),
                IS=initial_strides,
                OS=pstates.stride(),
                HAS_INIT=pinit is not None,
                WRITE_FINAL=write_final,
                FS=final_strides,
                CACHE_SLOTS=pfinal.shape[0] if write_final else 0,
                BM=64,
                BN=128 // nsplit,
                BK=32,
                num_warps=4,
                num_stages=4,
            )
            _cb_scan_kernel[(triton.cdiv(chunk, 64), chunks, groups * hsplit)](
                px,
                pb,
                pc,
                pdt,
                pda,
                pcu,
                pseq,
                pstates,
                pinit,
                D,
                z,
                pout,
                H=hdim,
                N=dstate,
                RATIO=heads // groups,
                CHUNK=chunk,
                XS=px.stride(),
                BS=pb.stride(),
                CS=pc.stride(),
                TS=pdt.stride(),
                AS=pda.stride(),
                SS=pstates.stride(),
                IS=initial_strides,
                OS=pout.stride(),
                ZS=z.stride() if z is not None else (0, 0, 0),
                DS=D.stride(0) if D is not None else 0,
                HAS_INIT=pinit is not None,
                HAS_D=D is not None,
                D_VECTOR=D.dim() == 2 if D is not None else False,
                HAS_Z=z is not None,
                BM=64,
                BN=64,
                BK=64,
                BC=32,
                DN=128,
                HSPLIT=hsplit,
                num_warps=8,
                num_stages=1,
            )

    use_prefix = False
    if last.numel() == 1 and 1 < chunks <= 4 and _SSD_STATE_NSPLIT == 1 and z is None:
        try:
            from .ssd_prefix_owners import admit_prefix_owners

            use_prefix = admit_prefix_owners(
                launch_ssd, x, B, C, dt, dA, cu, last, seq, initial, states, out,
                final_cache if write_final else None,
                final_index if write_final else None, D,
            )
        except Exception as prefix_error:
            logger.warning_once(
                "SSD prefix ownership optional path refused: %s",
                type(prefix_error).__name__,
            )
    launch_ssd(
        use_prefix, x, B, C, dt, dA, cu, last, seq, initial, states, out,
        final_cache, final_index,
    )
    return None if write_final else states
