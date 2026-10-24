"""CPU checks of the marlinx KSkip k-row layout (harness/kernels/marlinx/py/kskip_layout.py).

Pure Python: no torch, no GPU. The module is loaded from its file path.
"""

from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "kernels" / "marlinx" / "py" / "kskip_layout.py"
_spec = importlib.util.spec_from_file_location("marlinx_kskip_layout", _PATH)
kl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kl)


def dequant_fp4(q: int) -> tuple[int, int]:
    """csrc/marlinx.cu dequant_fp4: (out[0], out[1])."""
    q &= 0xFFFFFFFF
    o1 = (q & 0x80008000) | ((q & 0x70007000) >> 6)
    q = (q << 4) & 0xFFFFFFFF
    o2 = (q & 0x80008000) | ((q & 0x70007000) >> 6)
    return o2, o1


def a_fragment(word: int) -> list[int]:
    """block_mma for one word j: b0 = dequant_fp4(q << 8), b1 = dequant_fp4(q); mma_trans A
    operand {b0[0], b1[0], b0[1], b1[1]} (scales multiply each half, they do not move it)."""
    b0 = dequant_fp4((word << 8) & 0xFFFFFFFF)
    b1 = dequant_fp4(word)
    return [b0[0], b1[0], b0[1], b1[1]]


# PTX mma.m16n8k16 A fragment (row-major 16x16, bf16): (row, k) of reg, half for lane (g, t)
def a_slot(reg: int, half: int, g: int, t: int) -> tuple[int, int]:
    row = g + (8 if reg & 1 else 0)
    k = 2 * t + half + (8 if reg & 2 else 0)
    return row, k


def nibble_slot(lane: int, j: int, p: int) -> tuple[int, int]:
    """(feature in unit, k row) of crown nibble p of word j of lane, by running the dequant."""
    g, t = lane >> 2, lane & 3
    hits = []
    word = 0x7 << (4 * p)  # magnitude bits only: lands as a nonzero bf16 exponent field
    regs = a_fragment(word)
    for reg in range(4):
        for half in range(2):
            v = (regs[reg] >> (16 * half)) & 0xFFFF
            if v:
                hits.append(a_slot(reg, half, g, t))
    assert len(hits) == 1, (lane, j, p, hits)
    row, k = hits[0]
    return 16 * j + row, k


def activation_slots(t: int) -> list[int]:
    """k rows of ra[0] low/high and ra[1] low/high for lane t: Ap = A + tok*lda (u32) + t,
    ra[0] = Ap[8kb], ra[1] = Ap[8kb + 4] -> bf16 elements 16kb + 2t + {0, 1}, 16kb + 8 + 2t + {0, 1}.
    The kernel's predicates nz0, nz1, nz8, nz9 test these halves in this order."""
    out = []
    for word_off in (0, 4):
        for half in range(2):
            out.append(2 * (t + word_off) + half)
    return out


class KskipLayoutTests(unittest.TestCase):
    def test_z1_1_bijection(self):
        crown = {}
        for lane in range(32):
            for j in range(4):
                for p in range(8):
                    crown[(lane, j, p)] = kl.crown_nibble_meaning(lane, j, p)
        self.assertEqual(len(crown), 1024)
        self.assertEqual(sorted(crown.values()), sorted((f, r) for f in range(64) for r in range(16)))
        src = kl.src_of_dst()
        self.assertEqual(sorted(src), list(range(1024)))
        dst = kl.dst_of_src()
        for d, s in enumerate(src):
            self.assertEqual(dst[s], d)
        # every BK nibble holds the crown nibble with the same (feature, k row)
        for r in range(16):
            for g in range(8):
                for j in range(4):
                    for half in range(2):
                        d = (r * 8 + g) * 8 + 2 * j + half
                        s = src[d]
                        lane, jj, p = (s >> 3) >> 2, (s >> 3) & 3, s & 7
                        self.assertEqual(kl.crown_nibble_meaning(lane, jj, p),
                                         kl.bk_nibble_meaning(r, g, j, half))

    def test_z1_2_round_trip(self):
        rng = random.Random(20261024)
        cases = [[0] * 128, [0xFFFFFFFF] * 128, [rng.getrandbits(32) for _ in range(128)]]
        cases += [[rng.getrandbits(32) for _ in range(128)] for _ in range(1000)]
        for words in cases:
            bk = kl.repack_unit(words)
            self.assertEqual(kl.unpack_unit(bk), words)
            self.assertEqual(kl.repack_unit(kl.unpack_unit(bk)), bk)

    def test_z1_2_single_nibble(self):
        # each crown nibble moves to exactly one BK nibble, in k row K_OF_P[p] + 2t
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for j in range(4):
                for p in range(8):
                    words = [0] * 128
                    words[4 * lane + j] = 0x5 << (4 * p)
                    bk = kl.repack_unit(words)
                    nz = [(i, v) for i, v in enumerate(bk) if v]
                    self.assertEqual(len(nz), 1)
                    i, v = nz[0]
                    r, gg = i >> 3, i & 7
                    self.assertEqual(gg, g)
                    self.assertEqual(r, kl.K_OF_P[p] + 2 * t)
                    self.assertEqual(kl.unpack_unit(bk), words)

    def test_z1_3_dequant_and_mma_placement(self):
        for lane in range(32):
            for j in range(4):
                for p in range(8):
                    self.assertEqual(nibble_slot(lane, j, p), kl.crown_nibble_meaning(lane, j, p))

    def test_z1_3_loads_match_activation_predicates(self):
        # lane (g, t) loads BK k rows 2t + LOAD_ROWS[i] at u32 offset 16t + LOAD_OFFSETS[i] (+ g);
        # the predicate guarding load i tests the activation of exactly that k row
        for t in range(4):
            acts = activation_slots(t)
            for i in range(4):
                r = 2 * t + kl.LOAD_ROWS[i]
                self.assertEqual(acts[i], r)
                self.assertEqual(16 * t + kl.LOAD_OFFSETS[i], 8 * r)
        # and the four rows hold every crown nibble the lane reads
        for lane in range(32):
            t = lane & 3
            rows = {2 * t + d for d in kl.LOAD_ROWS}
            for j in range(4):
                for p in range(8):
                    self.assertIn(kl.crown_nibble_meaning(lane, j, p)[1], rows)

    def test_ballot_masks(self):
        # live(row 2t + d) = ballot & (0x11111111 << t): the lanes 4g + t, g = 0..7, hold the
        # activations of k row 2t + d for tokens g
        for t in range(4):
            mask = 0x11111111 << t
            self.assertEqual([b for b in range(32) if mask >> b & 1], [4 * g + t for g in range(8)])

    def test_kernel_source_matches_python(self):
        src = (Path(__file__).resolve().parents[1] / "kernels" / "marlinx" / "csrc" / "marlinx.cu").read_text()
        self.assertIn("((r0 >> s) & 0xFu) | (((r8 >> s) & 0xFu) << 4) | (((r0 >> (s + 4)) & 0xFu) << 8) |", src)
        self.assertIn("(((r8 >> (s + 4)) & 0xFu) << 12) | (((r1 >> s) & 0xFu) << 16) | (((r9 >> s) & 0xFu) << 20) |", src)
        self.assertIn("(((r1 >> (s + 4)) & 0xFu) << 24) | (((r9 >> (s + 4)) & 0xFu) << 28);", src)
        self.assertIn("q[0] = (full || (b0 & tm)) ? ld_nc_u32(kp) : 0u;", src)
        self.assertIn("q[1] = (full || (b1 & tm)) ? ld_nc_u32(kp + 8) : 0u;", src)
        self.assertIn("q[2] = (full || (b8 & tm)) ? ld_nc_u32(kp + 64) : 0u;", src)
        self.assertIn("q[3] = (full || (b9 & tm)) ? ld_nc_u32(kp + 72) : 0u;", src)


if __name__ == "__main__":
    unittest.main()
