"""humming_zskip.widen_blocks (PZ): 8-row moe_align layout -> zskip's 16-slot blocks.

Checks on CPU, against a reference 8-row alignment written here (same layout as vLLM's
moe_align_block_size: experts ascending, each expert's rows padded to a multiple of 8 with the
id numel, num_tokens_padded the padded total), that every 16-slot block holds at most 8 ids
< shape_m, that every row appears exactly once among the covered blocks, and that block b keeps
expert_ids[b]. Needs torch and an importable vLLM overlay; skipped elsewhere.
"""

from __future__ import annotations

import unittest

try:
    import torch

    from vllm.model_executor.layers.fused_moe.experts import humming_zskip
except ImportError:  # no torch / vLLM on this host
    humming_zskip = None

E, TOP_K, B = 128, 6, 8


def ref_align(topk_ids):
    flat = topk_ids.reshape(-1).tolist()
    numel = len(flat)
    cap = numel + E * (B - 1)
    if numel < E:
        cap = min(numel * B, cap)
    sorted_ids, expert_ids = [], []
    for e in range(E):
        rows = [i for i, x in enumerate(flat) if x == e]
        if not rows:
            continue
        pad = (-len(rows)) % B
        rows = rows + [numel] * pad
        sorted_ids += rows
        expert_ids += [e] * (len(rows) // B)
    ntp = len(sorted_ids)
    sorted_ids += [-7] * (cap - ntp)  # uninitialised tail (torch.empty in vLLM)
    expert_ids += [-9] * (-(-cap // B) - len(expert_ids))
    return torch.tensor(sorted_ids, dtype=torch.int32), expert_ids, ntp


@unittest.skipIf(humming_zskip is None, "torch / vLLM is not importable here")
class WidenBlocks(unittest.TestCase):
    def check(self, topk_ids):
        m = topk_ids.shape[0]
        rows = m * TOP_K
        sorted_ids, expert_ids, ntp = ref_align(topk_ids)
        ids16, nblocks = humming_zskip.widen_blocks(sorted_ids, rows)
        self.assertEqual(ids16.numel(), 16 * nblocks)
        self.assertLessEqual(2 * ntp, ids16.numel())
        self.assertLessEqual(nblocks, len(expert_ids))
        seen = []
        flat = topk_ids.reshape(-1).tolist()
        for b in range(2 * ntp // 16):
            blk = ids16[16 * b : 16 * b + 16].tolist()
            valid = [i for i in blk if 0 <= i < rows]
            self.assertLessEqual(len(valid), 8)
            self.assertTrue(all(i == rows for i in blk[8:]))
            for i in valid:
                self.assertEqual(flat[i], expert_ids[b])
            seen += valid
        self.assertEqual(sorted(seen), list(range(rows)))

    def test_random_routing(self):
        gen = torch.Generator().manual_seed(5)
        for i in range(20):
            m = [9, 13, 64, 200, 496, 512][i % 6]
            ids = torch.rand(m, E, generator=gen).argsort(dim=1)[:, :TOP_K]
            self.check(ids.to(torch.int32))

    def test_one_expert_set(self):
        for m in (9, 496, 512):
            ids = torch.arange(TOP_K, dtype=torch.int32).expand(m, TOP_K).contiguous()
            self.check(ids)

    def test_block_boundaries(self):
        # experts 0..5 carry 8, 9, 16, 17 rows (tokens m of them on each)
        for m in (8, 9, 16, 17):
            ids = torch.arange(TOP_K, dtype=torch.int32).expand(m, TOP_K).contiguous()
            if m >= 9:
                self.check(ids)
            ids = (torch.arange(TOP_K, dtype=torch.int32) * 7).expand(m, TOP_K).contiguous()
            if m >= 9:
                self.check(ids)

    def test_max_blocks_bound(self):
        rows = 512 * TOP_K
        cap = rows + E * (B - 1)
        _, nblocks = humming_zskip.widen_blocks(torch.zeros(cap, dtype=torch.int32), rows)
        self.assertEqual(nblocks, 496)
        self.assertLessEqual(nblocks, 512)


if __name__ == "__main__":
    unittest.main()
