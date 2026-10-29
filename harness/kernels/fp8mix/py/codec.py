# SPDX-License-Identifier: Apache-2.0
"""Startup-only exact per-original-N32-tile packet builder; never request data.

Six-bit code math is adapted from 0fde1e7f-cced-4bea-bfe7-cc2eece540be.
The original registered seven-bit tensor is neither modified nor discarded.
All tensor operations here run only in official remote post-load admission.
"""

import torch


def round_up(x, n):
    return (x + n - 1) // n * n


def header_bytes(k):
    return round_up((4 * (k // 128) + 1) * 2 + 64, 128)


def _requests(counts, main, c):
    # Four-byte exception loads include a possible unaligned zero-length
    # record. Count the source requests actually expressed by old load_x,
    # rather than the fixed padding allocated by the old encoder.
    result = []
    for row in counts:
        offset, exc = 0, 0
        for count in row:
            exc += round_up((offset & 3) + count, 4)
            offset += count
        result.append(main * c + exc + 16 * c + 256)
    return result


def _six(packed):
    n, k = packed.shape
    t, c = n // 32, k // 128
    dev = packed.device
    r = (packed.view(t, c, 4, 4, 32, 2, 4).permute(0, 1, 3, 4, 2, 5, 6)
         .reshape(t, c, 4, 32, 8, 4))
    rc = r.view(t, c, 4, 8, 4, 8, 4)
    exp = (rc >> 3) & 15
    hc = torch.stack([(exp == x).sum((1, 4, 5, 6)) for x in range(16)], -1)
    cs = torch.stack([hc[..., x:x + 4].sum(-1) for x in range(13)], -1)
    e0 = (cs * 16 + torch.arange(13, device=dev)).argmax(-1).to(torch.uint8)
    e = e0.view(t, 1, 4, 8, 1, 1, 1)
    win = (exp >= e) & (exp <= e + 3)
    code = ((rc & 0x80) >> 2) | (((exp - e) & 3) << 3) | (rc & 7)
    col = torch.arange(n, device=dev, dtype=torch.int32).view(t, 1, 4, 8, 1, 1, 1)
    ix = torch.where(win, col * 64 + code.to(torch.int32), torch.full_like(col, n * 64))
    hist = torch.bincount(ix.reshape(-1), minlength=n * 64 + 1)[:n * 64]
    esc_code = hist.view(t, 4, 8, 64).argmin(-1).to(torch.uint8)
    eb = esc_code.view(t, 1, 4, 8, 1, 1, 1)
    esc = (~win | (code == eb)).reshape(t, c, 4, 32, 8, 4)
    code = torch.where(esc.view_as(code), eb.expand_as(code), code).reshape_as(esc)
    shift = torch.arange(6, device=dev, dtype=torch.uint8).view(6, 1)
    cw = (code[..., :6, :] | (((code[..., 6:7, :] >> shift) & 1) << 6)
          | (((code[..., 7:8, :] >> shift) & 1) << 7))
    main = torch.cat([cw[..., :4, :].reshape(t, c, 2048),
                      cw[..., 4:6, :].reshape(t, c, 1024)], 2).reshape(t, c * 3072)
    counts = esc.reshape(t, c * 4, 1024).sum(-1, dtype=torch.int32)
    # Keep all arithmetic wide until after per-tile eligibility is known.
    offsets = torch.zeros((t, c * 4 + 1), device=dev, dtype=torch.int32)
    offsets[:, 1:] = counts.cumsum(-1)
    params = (e0.to(torch.int32) | (esc_code.to(torch.int32) << 8)).view(t, 32)
    return r, esc, main, offsets, params, counts.cpu().tolist()


@torch.no_grad()
def encode(packed, original, force=None):
    """Return (one allocation, evidence) or None.

    ``force`` is private decoder-admission only (6, 7 or alternating); production
    selection requires both C6>0 and C7>0 and a measured source-request bound.
    """
    n, k = packed.shape
    t, c, hdr = n // 32, k // 128, header_bytes(k)
    if original.ndim != 2 or original.shape[0] != t:
        return None
    old = original[:, :(c * 4 + 1) * 2].contiguous().view(torch.int16)
    old = (old.to(torch.int32) & 65535).cpu().tolist()
    count7 = [[b - a for a, b in zip(row, row[1:])] for row in old]
    if any(row[0] != 0 or any(x < 0 or x > 124 for x in cnt)
           or row[-1] >= 32768 for row, cnt in zip(old, count7)):
        return None
    r, esc, main6, offsets6, params6, count6 = _six(packed)
    req7, req6 = _requests(count7, 3584, c), _requests(count6, 3072, c)
    valid6 = [max(row) <= 124 and sum(row) < 32768 for row in count6]
    use6 = [ok and x + 16 < y for ok, x, y in zip(valid6, req6, req7)]
    if force is not None:
        use6 = [force == 6 or (force == "alternating" and i % 2 == 0) for i in range(t)]
        if any(take and not valid6[i] for i, take in enumerate(use6)):
            return None
    c6 = sum(use6)
    old_request = sum(req7)
    new_request = sum((req6[i] if use6[i] else req7[i]) + 16 for i in range(t))
    if force is None and (not 0 < c6 < t or new_request >= .99 * old_request):
        return None
    totals = [sum(count6[i]) if use6[i] else old[i][-1] for i in range(t)]
    sizes = [round_up(hdr + c * (3072 if take else 3584)
                      + max(round_up(total, 16), 16), 256)
             for take, total in zip(use6, totals)]
    starts, cursor = [], round_up(t * 4, 256)
    for size in sizes:
        starts.append(cursor)
        cursor += size
    if cursor >= (1 << 31) or (force is None and cursor >= original.numel()):
        return None
    packet = torch.zeros(cursor, device=packed.device, dtype=torch.uint8)
    if packet.data_ptr() % 256 or packet.untyped_storage().data_ptr() != packet.data_ptr():
        return None
    desc = torch.tensor([p | int(b) for p, b in zip(starts, use6)],
                        device=packed.device, dtype=torch.int32)
    packet[:t * 4].copy_(desc.view(torch.uint8))
    # Work only with selected safe six-bit records when narrowing offsets to
    # uint16. A failed tile cannot wrap offsets into an otherwise valid packet.
    for i, (start, size, take, total) in enumerate(zip(starts, sizes, use6, totals)):
        dst = packet[start:start + size]
        if take:
            dst[:(c * 4 + 1) * 2].copy_(offsets6[i].to(torch.int16).view(torch.uint8))
            dst[hdr - 64:hdr].copy_(params6[i].to(torch.int16).view(torch.uint8))
            dst[hdr:hdr + c * 3072].copy_(main6[i])
            values = r[i].reshape(-1)[esc[i].reshape(-1)]
            dst[hdr + c * 3072:hdr + c * 3072 + total].copy_(values)
        else:
            used = hdr + c * 3584 + total
            if used > original.shape[1]:
                return None
            dst[:used].copy_(original[i, :used])
    return packet, dict(tiles=t, c6=c6, c7=t - c6, bytes=cursor,
                        old_bytes=original.numel(), requests=old_request,
                        new_requests=new_request,
                        six_window_refused=sum(not x for x in valid6),
                        six_request_refused=sum(ok and not take
                            for ok, take in zip(valid6, use6)),
                        escape6=sum(sum(row) for row in count6),
                        selected_escape6=sum(sum(count6[i]) for i in range(t) if use6[i]))
