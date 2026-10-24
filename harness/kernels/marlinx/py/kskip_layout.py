"""k-row layout of the marlinx KSkip path (pure Python: integers only, no torch).

Crown layout (Marlin, read by gemm_kernel): for k16 block kb and 64-feature half
column unit = 2*col + h, lane l of a warp reads the int4 B[kb][32*unit + l], i.e.
the four u32 words W[kb][128*unit + 4*l + j], j = 0..3. With g = l >> 2 and
t = l & 3, Marlin's dequant (dequant_fp4(q << 8) -> frag_b0, dequant_fp4(q) ->
frag_b1) and mma_trans's A operand {b0[0], b1[0], b0[1], b1[1]} (PTX m16n8k16 A
fragment: reg0 row g k 2t+{0,1}, reg1 row g+8 k 2t+{0,1}, reg2 row g k 2t+8+{0,1},
reg3 row g+8 k 2t+8+{0,1}, low half = lower k) place nibble p (bits 4p..4p+3) of
word j at feature 16*j + g + 8*((p >> 1) & 1) of the unit and k row (inside the k16
block) K_OF_P[p] + 2*t with K_OF_P = (0, 8, 0, 8, 1, 9, 1, 9).

KSkip layout BK (same int32 shape [K/16, 2N] as the crown tensor, every nibble
kept): u32 index ((kb*(N/64) + unit)*16 + r)*8 + g holds k row r for the eight
lanes 4*g + t of one t, so the 8 lanes sharing t read one 32-byte sector per k
row. Byte j of that u32 = (nibble of feature 16j+g) | (nibble of feature
16j+g+8) << 4. The kernel rebuilds the crown word j of lane (g, t) from the four
u32 of k rows 2t, 2t+1, 2t+8, 2t+9 with kernel_word() (csrc/marlinx.cu
kskip_word, the same expression); the repack is defined as its exact inverse.
"""

from __future__ import annotations

UNIT_WORDS = 128  # u32 per (kb, unit) in both layouts
K_OF_P = (0, 8, 0, 8, 1, 9, 1, 9)  # k row offset (before + 2t) of crown nibble p
# u32 offsets (inside one kb row of BK, from 128*unit + 16*t + g) of k rows 2t, 2t+1, 2t+8,
# 2t+9: the kernel's loads q[0..3] (csrc/marlinx.cu kskip_load).
LOAD_OFFSETS = (0, 8, 64, 72)
LOAD_ROWS = (0, 1, 8, 9)  # k row of q[i] minus 2t


def crown_nibble_meaning(lane: int, j: int, p: int) -> tuple[int, int]:
    """(feature inside the 64-feature unit, k row inside the k16 block) of crown nibble p of
    word j of lane `lane`."""
    g, t = lane >> 2, lane & 3
    return 16 * j + g + 8 * ((p >> 1) & 1), K_OF_P[p] + 2 * t


def bk_nibble_meaning(r: int, g: int, j: int, half: int) -> tuple[int, int]:
    """(feature, k row) of nibble `half` of byte j of BK u32 (r, g)."""
    return 16 * j + g + 8 * half, r


def src_of_dst() -> list[int]:
    """Gather table: for BK nibble index d = (r*8 + g)*8 + 2*j + half of one (kb, unit),
    the crown nibble index s = (4*l + j)*8 + p it is copied from."""
    out = []
    for r in range(16):
        t = (r & 7) >> 1
        p_lo = 4 * (r & 1) + (r >> 3)  # crown nibble of feature 16j+g at k row r
        for g in range(8):
            lane = 4 * g + t
            for j in range(4):
                for half in range(2):
                    out.append((4 * lane + j) * 8 + p_lo + 2 * half)
    return out


def dst_of_src() -> list[int]:
    """Inverse of src_of_dst()."""
    src = src_of_dst()
    inv = [-1] * len(src)
    for d, s in enumerate(src):
        inv[s] = d
    return inv


def kernel_word(r0: int, r1: int, r8: int, r9: int, j: int) -> int:
    """Crown word j of lane (g, t) from the BK u32 of k rows 2t, 2t+1, 2t+8, 2t+9 (lane g).
    Mirrors kskip_word in csrc/marlinx.cu operator for operator."""
    s = 8 * j
    return (
        ((r0 >> s) & 0xF)
        | (((r8 >> s) & 0xF) << 4)
        | (((r0 >> (s + 4)) & 0xF) << 8)
        | (((r8 >> (s + 4)) & 0xF) << 12)
        | (((r1 >> s) & 0xF) << 16)
        | (((r9 >> s) & 0xF) << 20)
        | (((r1 >> (s + 4)) & 0xF) << 24)
        | (((r9 >> (s + 4)) & 0xF) << 28)
    ) & 0xFFFFFFFF


def repack_unit(words: list[int]) -> list[int]:
    """128 crown u32 of one (kb, unit) (index 4*l + j) -> 128 BK u32 (index r*8 + g)."""
    assert len(words) == UNIT_WORDS
    src = src_of_dst()
    out = []
    for u in range(UNIT_WORDS):
        v = 0
        for q in range(8):
            s = src[8 * u + q]
            v |= ((words[s >> 3] >> (4 * (s & 7))) & 0xF) << (4 * q)
        out.append(v)
    return out


def unpack_unit(bk: list[int]) -> list[int]:
    """128 BK u32 of one (kb, unit) -> 128 crown u32, through kernel_word (what the kernel
    rebuilds)."""
    assert len(bk) == UNIT_WORDS
    out = [0] * UNIT_WORDS
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        q = [bk[(2 * t + d) * 8 + g] for d in LOAD_ROWS]
        for j in range(4):
            out[4 * lane + j] = kernel_word(q[0], q[1], q[2], q[3], j)
    return out
