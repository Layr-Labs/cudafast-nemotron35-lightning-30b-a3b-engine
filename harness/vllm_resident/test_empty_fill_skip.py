"""GPU-free tests of the empty padding-fill skip in gpu_model_runner.py.

Each guarded fill runs when VLLM_EMPTY_FILL_SKIP=0 (the record's call) and is
skipped only when its slice bounds make the slice empty, so a skipped call is
one that writes nothing. The source checks need no torch; the tensor checks
are skipped where torch is missing.

Run: python3 -m unittest discover -s harness/vllm_resident -v
"""

from __future__ import annotations

import ast
import os
import random
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "..", "vllm", "vllm", "v1", "worker", "gpu_model_runner.py")
FLAG = "_EMPTY_FILL_SKIP"

try:
    import torch
except ImportError:  # no torch on this host
    torch = None


def _source_tree() -> ast.Module:
    with open(PATH, encoding="utf-8") as f:
        return ast.parse(f.read())


def _guarded_ifs(tree: ast.Module) -> list[ast.If]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
            continue
        first = node.test.values[0]
        if (
            isinstance(first, ast.UnaryOp)
            and isinstance(first.op, ast.Not)
            and isinstance(first.operand, ast.Name)
            and first.operand.id == FLAG
        ):
            found.append(node)
    return found


def _sliced(stmt: ast.stmt) -> ast.Subscript:
    """The subscript a guarded statement writes: `x[a:b].fill_(v)` or `x[a:] = v`."""
    if isinstance(stmt, ast.Expr):
        call = stmt.value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        assert call.func.attr == "fill_", ast.dump(call)
        target = call.func.value
    else:
        assert isinstance(stmt, ast.Assign) and len(stmt.targets) == 1, ast.dump(stmt)
        target = stmt.targets[0]
    assert isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Slice)
    return target


class SourceGuards(unittest.TestCase):
    def test_flag_defaults_on_and_zero_turns_it_off(self):
        tree = _source_tree()
        assigns = [
            n for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == FLAG for t in n.targets)
        ]
        self.assertEqual(len(assigns), 1)
        expr = ast.Expression(assigns[0].value)
        code = compile(expr, PATH, "eval")

        class Env:
            def __init__(self, env):
                self.environ = env

        self.assertTrue(eval(code, {"os": Env({})}))
        self.assertTrue(eval(code, {"os": Env({"VLLM_EMPTY_FILL_SKIP": "1"})}))
        self.assertFalse(eval(code, {"os": Env({"VLLM_EMPTY_FILL_SKIP": "0"})}))

    def test_each_guard_skips_only_an_empty_slice(self):
        guards = _guarded_ifs(_source_tree())
        self.assertEqual(len(guards), 4)
        for node in guards:
            with self.subTest(line=node.lineno):
                self.assertIsInstance(node.test.op, ast.Or)
                self.assertEqual(len(node.test.values), 2)
                self.assertEqual(node.orelse, [])
                self.assertEqual(len(node.body), 1)
                cmp = node.test.values[1]
                self.assertIsInstance(cmp, ast.Compare)
                self.assertEqual([type(o) for o in cmp.ops], [ast.Lt])
                lo, hi = cmp.left, cmp.comparators[0]
                target = _sliced(node.body[0])
                self.assertIsNone(target.slice.step)
                self.assertEqual(ast.dump(target.slice.lower), ast.dump(lo))
                if target.slice.upper is not None:
                    # x[a:b] with guard a < b: skipped only when a >= b.
                    self.assertEqual(ast.dump(target.slice.upper), ast.dump(hi))
                else:
                    # x[a:] with guard a < x.shape[0]: skipped only when a >= len(x).
                    self.assertEqual(
                        ast.unparse(hi), f"{ast.unparse(target.value)}.shape[0]"
                    )


@unittest.skipIf(torch is None, "torch is not installed")
class EmptySliceWritesNothing(unittest.TestCase):
    def test_skipped_bounds_leave_the_tensor_bitwise_equal(self):
        rng = random.Random(0)
        for _ in range(400):
            n = rng.randint(0, 9)
            lo = rng.randint(0, 12)
            hi = rng.randint(0, lo)
            for dtype, fill in ((torch.int32, -1), (torch.int64, -1), (torch.int32, 0)):
                t = torch.randint(-50, 50, (n, 3), dtype=dtype)
                ref = t.clone()
                part = t[lo:hi]
                self.assertEqual(part.numel(), 0)
                part.fill_(fill)
                self.assertTrue(torch.equal(t, ref))
            flags = torch.rand(n) < 0.5
            ref_flags = flags.clone()
            if lo >= flags.shape[0]:
                flags[lo:] = False
                self.assertTrue(torch.equal(flags, ref_flags))
            seq = torch.randint(0, 9000, (n,), dtype=torch.int32)
            ref_seq = seq.clone()
            if lo >= seq.shape[0]:
                seq[lo:].fill_(0)
                self.assertTrue(torch.equal(seq, ref_seq))

    def test_unskipped_bounds_match_the_record(self):
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randint(1, 9)
            lo = rng.randint(0, n - 1)
            hi = rng.randint(lo + 1, n + 3)
            t = torch.randint(-50, 50, (n,), dtype=torch.int64)
            record, guarded = t.clone(), t.clone()
            record[lo:hi].fill_(-1)
            if lo < hi:
                guarded[lo:hi].fill_(-1)
            self.assertTrue(torch.equal(record, guarded))


if __name__ == "__main__":
    unittest.main()
