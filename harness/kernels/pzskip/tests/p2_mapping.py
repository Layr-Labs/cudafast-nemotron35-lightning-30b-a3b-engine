"""CPU-only index model of pzskip's P2 configs (336-338) against config 333 (smem4_kernel, BMT 32).

Ports the index arithmetic of csrc/pzskip.cu (p2_kernel / smem4_kernel issue loops, the cp.async
commit/wait pipeline, the compute loop's shared-memory reads and the epilogue) and checks, for
random union masks and row counts, sparse and dense passes:
  * nth_set_bit16 for every 16-bit mask and rank;
  * each stage copies exactly the active weight rows (each 16-byte piece once), all scale pieces
    and 2 pieces per valid row, from the "swap" addresses, into in-bounds slots; P2's union of
    weight/scale bytes per block equals 333's;
  * every shared-memory word a lane reads for an active k row was copied in the same stage (wait
    accounting; no buffer is refilled before it is consumed) from the right global bytes, and those
    bytes hold exactly the output columns the lane's epilogue stores (swap: n = 64T+16c+8h+g ->
    byte 32T+4g+c, nibble h);
  * every (row, column) of the block is stored exactly once, and its k16 groups are visited in
    ascending order: the active-group list (sparse) or 0..115 (dense), the same chain as 333.
  python3 harness/kernels/pzskip/tests/p2_mapping.py
"""
import random
import sys

K, N = 1856, 2688
G = K // 16
W_K, S_G = N // 2, N  # swap strides inside one expert
BMT = 32
CONFIGS = {336: (2, 4, 14), 337: (2, 4, 7), 338: (2, 3, 6), 333: (2, 3, 6)}  # (KG, NS, NTL)


def popc(m):
    return bin(m).count("1")


def nth_set_bit16(m, a):  # line-for-line port of the device function
    r = 0
    c = popc(m & 0xFF)
    if a >= c:
        a -= c; m >>= 8; r += 8
    c = popc(m & 0xF)
    if a >= c:
        a -= c; m >>= 4; r += 4
    c = popc(m & 0x3)
    if a >= c:
        a -= c; m >>= 2; r += 2
    c = m & 1
    if a >= c:
        r += 1
    return r


def swap_columns(byte):
    """Output columns held by byte `byte` of a swap k row (low nibble h=0, high h=1)."""
    T, rem = divmod(byte, 32)
    g, c = divmod(rem, 4)
    return {64 * T + 16 * c + 8 * h + g for h in (0, 1)}


def issue(cfg, P, st, nsteps, dense, gact, gmask, nrows):
    """One issue(st): list of (dst, src) 16-byte copies; dst = (kind, gi, row, byte)."""
    KG, NS, NTL = CONFIGS[cfg]
    WP, SP, NTH = 2 * NTL, 4 * NTL, 32 * NTL
    wrow = 32 * NTL + (16 if cfg != 333 else 0)
    out = []
    for gi in range(KG):
        j = st * KG + gi
        if j >= nsteps:
            break
        grp = j if dense else gact[j]
        m = 0xFFFF if dense else gmask[grp]
        nw = popc(m) * WP if cfg != 333 else 16 * WP
        total = nw + SP + 2 * nrows
        seen = set()
        for tid in range(NTH):
            for q in range(tid, total, NTH):
                if q < nw:
                    if cfg == 333:
                        r, piece = divmod(q, WP)
                        if not (m >> r) & 1:
                            continue
                    else:
                        a = q // WP
                        piece = q - a * WP
                        r = nth_set_bit16(m, a)
                    assert 16 * piece + 16 <= 32 * NTL <= wrow
                    dst, src = ("w", gi, r, 16 * piece), ("W", (16 * grp + r) * W_K + NTL * P * 32 + 16 * piece)
                elif q < nw + SP:
                    piece = q - nw
                    assert 16 * piece + 16 <= 64 * NTL
                    dst, src = ("s", gi, 0, 16 * piece), ("S", grp * S_G + NTL * P * 64 + 16 * piece)
                else:
                    r, piece = (q - nw - SP) >> 1, (q - nw - SP) & 1
                    assert r < nrows and 16 * piece + 16 <= 48
                    dst, src = ("x", gi, r, 16 * piece), ("A", r, 16 * grp + 8 * piece)
                assert dst not in seen, ("copied twice", cfg, dst)
                seen.add(dst)
                out.append((dst, src))
        want = {("w", gi, r, 16 * pc) for r in range(16) if (m >> r) & 1 for pc in range(WP)}
        want |= {("s", gi, 0, 16 * pc) for pc in range(SP)}
        want |= {("x", gi, r, 16 * pc) for r in range(nrows) for pc in range(2)}
        assert seen == want, ("copy set", cfg, st, gi)
    return out


def run_cta(cfg, P, nrows, gmask, dense, owner, gbytes, chains):
    KG, NS, NTL = CONFIGS[cfg]
    gact = [g for g in range(G) if gmask[g]]
    nsteps = G if dense else len(gact)
    nst = (nsteps + KG - 1) // KG
    ntt = (nrows + 7) >> 3
    groups = []  # commit groups in order: (stage, copies)
    for st in range(NS - 1):
        groups.append((st, issue(cfg, P, st, nsteps, dense, gact, gmask, nrows) if st < nst else []))
    landed, n_landed, visited = {}, 0, []
    for st in range(nst):
        while n_landed < len(groups) - (NS - 2):  # cp.async.wait_group NS-2
            s, copies = groups[n_landed]
            for dst, src in copies:
                landed[(s % NS, dst)] = (src, s)
            n_landed += 1
        # __syncthreads, then issue(st + NS - 1) into a buffer no unconsumed stage occupies
        if st + NS - 1 < nst:
            b = (st + NS - 1) % NS
            assert all(s2 % NS != b for s2 in range(st, st + NS - 1)), "buffer reused early"
            groups.append((st + NS - 1, issue(cfg, P, st + NS - 1, nsteps, dense, gact, gmask, nrows)))
        else:
            groups.append((None, []))
        buf = st % NS
        for gi in range(KG):
            j = st * KG + gi
            if j >= nsteps:
                break
            grp = j if dense else gact[j]
            m = 0xFFFF if dense else gmask[grp]
            visited.append(grp)
            for Tl in range(NTL):
                T = NTL * P + Tl
                for g8 in range(8):
                    lane_cols = {T * 64 + 16 * ci + g8 + 8 * h for ci in range(4) for h in (0, 1)}
                    so = Tl * 64 + g8 * 8
                    src, s = landed[(buf, ("s", gi, 0, so - so % 16))]
                    assert s == st and src == ("S", grp * S_G + NTL * P * 64 + so - so % 16)
                    for t in range(4):
                        for kr in (2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9):
                            if not (m >> kr) & 1:
                                continue  # stale, masked to +0 by keep01/keep23
                            off = Tl * 32 + g8 * 4
                            src, s = landed[(buf, ("w", gi, kr, off - off % 16))]
                            assert s == st, "weight word not from this stage"
                            byte = src[1] + off % 16 - (16 * grp + kr) * W_K  # 32 T + 4 g8
                            assert src[0] == "W" and src[1] - (16 * grp + kr) * W_K == NTL * P * 32 + off - off % 16
                            cols = set().union(*(swap_columns(byte + c) for c in range(4)))
                            assert cols == lane_cols, "weight bytes hold other columns"
                        for tt in range(ntt):
                            r = 8 * tt + g8
                            if r >= nrows:
                                continue  # token column never stored
                            for boff, elem in ((4 * t, 2 * t), (16 + 4 * t, 8 + 2 * t)):
                                src, s = landed[(buf, ("x", gi, r, boff - boff % 16))]
                                assert s == st and src == ("A", r, 16 * grp + 8 * (boff // 16))
                                assert src[2] + (boff % 16) // 2 == 16 * grp + elem
    for (dst, src) in [c for _, cs in groups for c in cs]:
        if src[0] in ("W", "S"):
            gbytes.add((src[0], src[1]))
    want = list(range(G)) if dense else [g for g in range(G) if gmask[g]]
    assert visited == want and all(a < b for a, b in zip(visited, visited[1:])), "group order"
    for Tl in range(NTL):
        for lane in range(32):
            g8, t = lane >> 2, lane & 3
            for tt in range(4):
                for ci in range(4):
                    col = (NTL * P + Tl) * 64 + 16 * ci + g8
                    for q in range(4):
                        r = 8 * tt + 2 * t + (q & 1)
                        cc = col + (8 if q >> 1 else 0)
                        if r < nrows:
                            assert 0 <= cc < N
                            owner[(r, cc)] = owner.get((r, cc), 0) + 1
                            chains.setdefault((r, cc), []).append(tuple(visited))


def run_block(cfg, nrows, gmask, dense):
    KG, NS, NTL = CONFIGS[cfg]
    assert N % (64 * NTL) == 0
    owner, gbytes, chains = {}, set(), {}
    for P in range(N // (64 * NTL)):
        run_cta(cfg, P, nrows, gmask, dense, owner, gbytes, chains)
    assert len(owner) == nrows * N and set(owner.values()) == {1}, "ownership"
    return {k: v[0] for k, v in chains.items()}, gbytes


def random_masks(rng):
    masks = []
    for _ in range(G):
        u = rng.random()
        if u < 0.08:
            masks.append(0)
        elif u < 0.3:
            masks.append(0xFFFF)
        elif u < 0.4:
            masks.append(1 << rng.randrange(16))
        else:
            p = rng.choice((0.1, 0.3, 0.5, 0.8))
            masks.append(sum(1 << b for b in range(16) if rng.random() < p))
    return masks


def main():
    for m in range(1 << 16):
        bits = [b for b in range(16) if (m >> b) & 1]
        for a, b in enumerate(bits):
            assert nth_set_bit16(m, a) == b
    print("nth_set_bit16: all 65536 masks, every rank: ok")
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 336)
    cases = [(n, random_masks(rng)) for n in (1, 7, 8, 9, 17, 24, 31, 32)]
    cases.append((32, [0] * G))                              # no active group
    cases.append((5, [1 << (g % 16) for g in range(G)]))     # single rows
    cases.append((32, [0xFFFF] * G))                         # fully dense union
    nblk = 0
    for nrows, masks in cases:
        for dense in (False, True):
            ref_chain, ref_bytes = run_block(333, nrows, masks, dense)
            for cfg in (336, 337, 338):
                chain, gb = run_block(cfg, nrows, masks, dense)
                assert chain == ref_chain, ("chain differs from 333", cfg)
                assert gb == ref_bytes, ("weight/scale bytes differ from 333", cfg)
            nblk += 1
    print(f"configs 336/337/338 vs 333: {nblk} block passes (sparse + dense), every (row, column) "
          "stored once, ascending k16 chain identical to 333, copies/reads consistent: ok")


if __name__ == "__main__":
    main()
