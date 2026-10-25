# SPDX-License-Identifier: Apache-2.0
"""Boot-only qualification of the optional historical FP32 dt/decay ring.

The cache is request-local SSM history, not a weight/cache format change.
Its two FP32 words are the existing recurrence's intermediate scalars. Raw
16-bit history remains in its original layout. No current-step producer,
cross-CTA wait, or extra decode launch is introduced.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def _offset(state, dtype, ring_len):
    """Metadata/storage proof before any history can be produced."""
    if (not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4
            or dtype not in (torch.float16, torch.bfloat16)
            or tuple(state.shape[1:]) != (64, 64, 128)
            or state.shape[0] < 6 or ring_len != 17
            or state.stride()[1:] != (64 * 128, 128, 1)
            or state.stride(0) < 64 * 64 * 128 + 1
            or state.data_ptr() % 16 or state.stride(0) % 4):
        return None
    last = state.storage_offset() + (state.shape[0] - 1) * state.stride(0)
    if (last + 64 * 64 * 128 + 1) * 4 > state.untyped_storage().nbytes():
        return None
    raw = ring_len * (64 * 64 + 8 * 128 + 64)
    off = ((raw + 7) // 8) * 8  # original 16-bit ring units, 16-byte alignment
    if off * 2 + 2 * ring_len * 64 * 4 > 64 * 64 * 128 * 4:
        return None
    return off


@triton.jit
def _scalar_bits(raw, A, bias, valid, bad,
                 STRIDE: tl.constexpr, OFF: tl.constexpr,
                 AS: tl.constexpr, BS: tl.constexpr,
                 H: tl.constexpr, M: tl.constexpr, G: tl.constexpr,
                 D: tl.constexpr, RL: tl.constexpr, R: tl.constexpr):
    """Compare each actually written FP32 word with raw-history recomputation.

    This independent expression check supplements the original full recurrence
    below; neither algebra nor an unrelated vector kernel is used as its sole
    correctness argument.
    """
    h = tl.program_id(0)
    r = tl.arange(0, R)
    live = (r < RL) & (tl.load(valid + r, r < RL, other=0) != 0)
    base = raw + 2 * STRIDE  # private request's ring is physical slot 2
    dt = tl.load(base + RL * (H * M + G * D) + r * H + h,
                 mask=live, other=0).to(tl.float32)
    dt += tl.load(bias + h * BS).to(tl.float32)
    dt = softplus(dt)
    decay = fast_exp(tl.load(A + h * AS).to(tl.float32) * dt)
    scalars = (base + OFF).to(tl.pointer_type(tl.float32))
    got_dt = tl.load(scalars + r * H + h, mask=r < RL, other=0)
    got_decay = tl.load(scalars + RL * H + r * H + h, mask=r < RL, other=0)
    dt_bits = got_dt.to(tl.int32, bitcast=True)
    decay_bits = got_decay.to(tl.int32, bitcast=True)
    fail = (live & ((dt.to(tl.int32, bitcast=True) != dt_bits)
                    | (decay.to(tl.int32, bitcast=True) != decay_bits)))
    # Never-written cache words must still carry private backing's exact
    # 0.03125f sentinel; checking just the used values could miss a stray store.
    fail |= (~live) & ((dt_bits != 0x3D000000) | (decay_bits != 0x3D000000))
    tl.store(bad + h * RL + r, fail.to(tl.int32), mask=r < RL)


def _same(a, b):
    # Byte identity includes signed zero and NaN payloads in untouched guards.
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@torch.no_grad()
def prepare(state, dtype, A, bias, D, ring_len, is_blackwell):
    """Return a fixed scalar offset after remote boot qualification, else None.

    Called immediately after real cache binding, before compile/graph capture,
    with the layer's already loaded immutable A/dt_bias/D. Never read or mutate
    live state values or live confidence buffers. An unsuccessful admission
    never starts a scalar history; there is no late activation during requests.
    """
    if os.environ.get("NEMO_MAMBA_SCALAR_RING", "1") == "0":
        return None
    try:
        off = _offset(state, dtype, ring_len)
        if (off is None or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(state.device) != (12, 1)
                or A.dtype != torch.float32 or A.shape != (64, 64, 128)
                or A.stride()[1:] != (0, 0) or bias.shape != (64, 64)
                or bias.stride(1) != 0 or D.shape != (64, 64)
                or any(t.device != state.device for t in (A, bias, D))):
            return None
        from . import mamba_ssm_spec_replay as rp

        stride = state.stride(0)
        dev = state.device
        # Full physical stride, prefix/suffix guards and padding are allocated
        # privately. Slots0/6 are sentinels, slot1 is the prefill anchor, slot2
        # the raw+scalar ring, slot3 the alternate checkpoint, slots4/5 unused.
        length = 7 * stride + 16
        shape = (7, 64, 64, 128)
        strides = (stride, 64 * 128, 128, 1)
        scalar_lo = 2 * stride * 4 + off * 2
        scalar_hi = scalar_lo + 2 * ring_len * 64 * 4
        advances = (1, 1, 3, 0, 2, 5, 1, 4, 2, 0, 5, 3)
        pdl_modes = (False, True) if rp._pdl_enabled() else (False,)
        # No global RNG state and no prompt/expected-output input is involved.
        for input_dtype in (torch.bfloat16, torch.float16):
            for pdl in pdl_modes:
                for observe in (False, True):
                    reference = torch.full((length,), 0.03125, device=dev,
                                           dtype=torch.float32)
                    candidate = reference.clone()
                    states = [torch.as_strided(t, shape, strides) for t in (reference, candidate)]
                    table = torch.tensor([[1, 2, 3, 4, 5]], device=dev, dtype=torch.int32)
                    accepted = torch.ones(1, device=dev, dtype=torch.int32)
                    cu = torch.tensor([0, 5], device=dev, dtype=torch.int32)
                    seq = torch.empty(1, device=dev, dtype=torch.int32)
                    rows = torch.empty(1, device=dev, dtype=torch.int32)
                    valid = torch.zeros(ring_len, device=dev, dtype=torch.int32)
                    bad = torch.zeros((64, ring_len), device=dev, dtype=torch.int32)
                    outputs = [torch.empty((7, 64, 64), device=dev, dtype=input_dtype)
                               for _ in range(2)]
                    replayed = [torch.empty((3, 64, 64, 128), device=dev,
                                            dtype=torch.float32) for _ in range(2)]
                    prompt, pos, previous_advance = 37, 37, 1
                    for st in states:
                        rp.prefill_tag_view(st)[1] = prompt
                    written = set()
                    # Multiple wraps/parities, speculative overwrite/rejection,
                    # repeated positions, every prefix1..5 and accepted0..4drafts.
                    for cycle in range(48):
                        prefix = 5 if cycle % 6 == 0 else 1 + cycle % 5
                        projection = (((torch.arange(5 * (64 * 64 + 2 * 8 * 128 + 64),
                                                      device=dev, dtype=torch.int32)
                                        + cycle * 19) % 257).float() / 128 - 1)
                        projection = projection.to(input_dtype).view(5, -1)
                        x = projection[:, :4096].view(5, 64, 64)
                        B = projection[:, 4096:5120].view(5, 8, 128)
                        C = projection[:, 5120:6144].view(5, 8, 128)
                        raw_dt = projection[:, 6144:]
                        regimes = torch.tensor([-40., -25., -8., -1., -0., 0.,
                                                .001, 1., 8., 20., 25., 40.],
                                               device=dev, dtype=input_dtype)
                        index = (torch.arange(320, device=dev).view(5, 64)
                                 + cycle * 3) % regimes.numel()
                        raw_dt.copy_(regimes[index])
                        dt = raw_dt[:, :, None].expand(-1, -1, 64)
                        rows.fill_(prefix)
                        seq.fill_(pos + 5)
                        accepted.fill_(previous_advance)
                        for t in outputs:
                            t.fill_(0.0625)
                        for t in replayed:
                            t.fill_(0.09375)
                        for which in range(2):
                            rp.selective_state_update_spec_replay(
                                states[which], x, dt, A, B, C, D, bias,
                                dt_softplus=True, state_batch_indices=table,
                                out=outputs[which][1:6], num_accepted_tokens=accepted,
                                cu_seqlens=cu, seq_lens=seq, ring_len=ring_len,
                                is_blackwell=is_blackwell, replayed_out=(replayed[which][1:2] if observe else None),
                                lazy=True, _scalar_ring_offset=off if which else None,
                                _private_verify_rows=rows, _private_launch_pdl=pdl,
                                _private_no_prefix=(cycle % 6 == 0),
                            )
                        torch.cuda.synchronize(dev)
                        # Includes entire guarded output, replayed pre-verify state,
                        # both checkpoint slots, raw ring, page tags and all unused
                        # state/padding. Only the new cache's exact byte range differs.
                        if not (_same(outputs[0], outputs[1])
                                and _same(replayed[0], replayed[1])
                                and _same(reference.view(torch.uint8)[:scalar_lo],
                                          candidate.view(torch.uint8)[:scalar_lo])
                                and _same(reference.view(torch.uint8)[scalar_hi:],
                                          candidate.view(torch.uint8)[scalar_hi:])):
                            return None
                        written.update((pos + t) % ring_len for t in range(prefix))
                        valid.zero_()
                        valid[list(sorted(written))] = 1
                        _scalar_bits[(64,)](
                            states[1].view(input_dtype), A, bias, valid, bad,
                            STRIDE=states[1].view(input_dtype).stride(0), OFF=off,
                            AS=A.stride(0), BS=bias.stride(0), H=64, M=64, G=8,
                            D=128, RL=ring_len, R=triton.next_power_of_2(ring_len),
                        )
                        if bool(bad.any().item()):
                            return None
                        advance = min(prefix, advances[cycle % len(advances)])
                        pos += advance
                        previous_advance = advance

                    # Original no-work and all-null-slot behavior must also survive;
                    # no cache load is allowed for a null ring. Both calls use the
                    # same private guards and compare all data outside the cache.
                    for empty in (True, False):
                        cu[1] = 0 if empty else 5
                        table.zero_()
                        accepted.zero_()
                        for which in range(2):
                            outputs[which].fill_(0.0625)
                            replayed[which].fill_(0.09375)
                            rp.selective_state_update_spec_replay(
                                states[which], x, dt, A, B, C, D, bias,
                                dt_softplus=True, state_batch_indices=table,
                                out=outputs[which][1:6], num_accepted_tokens=accepted,
                                cu_seqlens=cu, seq_lens=seq, ring_len=ring_len,
                                is_blackwell=is_blackwell, replayed_out=(replayed[which][1:2] if observe else None),
                                lazy=True, _scalar_ring_offset=off if which else None,
                                _private_verify_rows=rows, _private_launch_pdl=pdl,
                                _private_no_prefix=(cycle % 6 == 0),
                            )
                        torch.cuda.synchronize(dev)
                        if not (_same(outputs[0], outputs[1])
                                and _same(replayed[0], replayed[1])
                                and _same(reference.view(torch.uint8)[:scalar_lo],
                                          candidate.view(torch.uint8)[:scalar_lo])
                                and _same(reference.view(torch.uint8)[scalar_hi:],
                                          candidate.view(torch.uint8)[scalar_hi:])):
                            return None
        logger.info("Mamba scalar-ring boot history bits admitted; whole speed unmeasured")
        return off
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mamba scalar-ring qualification kept original path: %r", exc)
        return None
