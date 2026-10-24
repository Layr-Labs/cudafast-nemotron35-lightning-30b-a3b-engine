"""CPU integration of our fused SSD and terminal sink; no GPU claims."""

import ast
from contextlib import nullcontext
import inspect
from types import SimpleNamespace as NS
import unittest

try:
    from . import test_ssd_prefill_fused as sim
except ImportError:
    import test_ssd_prefill_fused as sim

np = sim.np


class FusedSinkTests(unittest.TestCase):
    def compare(self, length, chunk, initial, index, full=False):
        q = sim.case(
            [length],
            chunk,
            initial=initial,
            heads=8 if full else 2,
            groups=1,
            h=64 if full else 4,
            n=128 if full else 8,
        )
        tl = sim.TL()
        baseline = sim.base_states(q, tl)
        expected_out = sim.base_output(q, baseline, tl).copy()
        initial_copy = q.initial.data.copy() if initial else None
        # Deliberately leave padding after every head, including the last one.
        span = q.h * q.n + 11
        cache = sim.Tensor(np.full((4, q.heads, span), -917.0, np.float32))
        indices = sim.Tensor([index, 99, 99, 99], np.int32)
        states, out = sim.fused_result(
            q,
            tl,
            cache,
            indices,
            final_strides=(q.heads * span, span, q.n, 1),
            index_stride=4,
        )
        np.testing.assert_array_equal(out.view(np.uint32), expected_out.view(np.uint32))
        np.testing.assert_array_equal(
            states[:-1].view(np.uint32), baseline.data[:-1].view(np.uint32)
        )
        self.assertTrue(np.isnan(states[-1]).all(), "terminal scratch was written")
        target = index % 4
        np.testing.assert_array_equal(
            cache.data[target, :, : q.h * q.n]
            .reshape(q.heads, q.h, q.n)
            .view(np.uint32),
            baseline.data[-1].view(np.uint32),
        )
        untouched = np.ones(cache.shape, bool)
        untouched[target, :, : q.h * q.n] = False
        self.assertTrue((cache.data[untouched] == -917.0).all())
        if initial:
            np.testing.assert_array_equal(
                q.initial.data.view(np.uint32), initial_copy.view(np.uint32)
            )

    def test_one_chunk_and_boundary_tails(self):
        for chunk in (128, 256):
            for length in (1, chunk - 1, chunk, chunk + 1):
                with self.subTest(chunk=chunk, length=length):
                    self.compare(length, chunk, False, 0)

    def test_initial_state_negative_index_and_multiple_chunks(self):
        for chunk in (128, 256):
            for length in (496, 513):
                with self.subTest(chunk=chunk, length=length):
                    self.compare(length, chunk, True, -1)

    def test_full_inner_state_and_eight_shared_heads(self):
        self.compare(129, 128, True, 2, full=True)

    def test_combined_routes_sink_and_never_gathers_terminal_scratch(self):
        q = sim.case([1], 128)
        cache, indices = object(), object()
        trace = []

        def stage(name, result=None):
            def call(*args, **kwargs):
                trace.append((name, kwargs))
                return result

            return call

        for fused in (False, True):
            trace.clear()
            ns = dict(
                torch=sim.torch_double(),
                is_int_pow_2=lambda n: n == 128,
                rearrange=lambda t, *a, **kw: t,
                _chunk_cumsum_fwd=stage("cumsum", (q.da, q.dt)),
                can_fuse_prefill=lambda *a: fused,
                # A non-indexable result proves that the sink path does not gather.
                fused_prefill=stage("fused", object()),
                _chunk_state_fwd=stage("chunk", object()),
                _state_passing_fwd=stage("pass", object()),
                _bmm_chunk_fwd=stage("cb"),
                _chunk_scan_fwd=stage("scan"),
            )
            fn = sim.function(
                sim.OPS / "ssd_combined.py", "_mamba_chunk_scan_combined_fwd", ns
            )
            args = (
                q.x,
                sim.Tensor(np.zeros((1, q.heads))),
                sim.Tensor(np.zeros(q.heads)),
                q.b,
                q.c,
                128,
                q.out,
            )
            kw = dict(
                seq_idx=q.seq,
                cu_seqlens=sim.Tensor([0, 1], np.int32),
                cu_chunk_seqlens=q.cu,
                last_chunk_indices=q.last,
                state_dtype=np.float32,
                final_states_out=cache,
                final_states_indices=indices,
            )
            self.assertIsNone(fn(*args, **kw))
            routed = dict(trace)["fused" if fused else "pass"]
            self.assertIs(routed["final_states_out"], cache)
            self.assertIs(routed["final_states_indices"], indices)
            self.assertEqual(
                [name for name, _ in trace],
                ["cumsum", "fused"]
                if fused
                else ["cumsum", "chunk", "pass", "cb", "scan"],
            )
            with self.assertRaises(AssertionError):
                fn(*args, **kw, return_intermediate_states=True)
            kw["final_states_indices"] = None
            with self.assertRaises(AssertionError):
                fn(*args, **kw)

    def test_fused_wrapper_binds_sink_strides_and_validates_shape(self):
        q = sim.case([1], 128)
        cache = sim.Tensor(np.zeros((4, q.heads, q.h, q.n), np.float32))
        indices = sim.Tensor([2], np.int32)
        device = NS(index=0)
        for t in (q.x, cache, indices):
            t.device = device
        torch = sim.torch_double()
        torch.empty = lambda shape, **kw: sim.Tensor(np.empty(shape, np.float32))
        torch.accelerator = NS(device_index=lambda i: nullcontext())
        ns = dict(torch=torch, triton=NS(cdiv=lambda a, b: (a + b - 1) // b))
        calls = {}

        class Launch:
            def __init__(self, name):
                self.name = name
                self.sig = inspect.signature(
                    sim.function(sim.FUSED, name, dict(tl=sim.TL(), fast_exp=sim.exp))
                )

            def __getitem__(self, grid):
                def run(*args, **kw):
                    kw.pop("num_warps")
                    kw.pop("num_stages")
                    calls[self.name] = self.sig.bind(*args, **kw).arguments

                return run

        for name in ("_state_pass_kernel", "_cb_scan_kernel"):
            ns[name] = Launch(name)
        fn = sim.function(sim.FUSED, "fused_prefill", ns)
        args = (q.x, q.b, q.c, q.dt, q.da, q.cu, q.last, q.seq, q.out)
        fn(*args, final_states_out=cache, final_states_indices=indices)
        bound = calls["_state_pass_kernel"]
        self.assertIs(bound["FINAL"], cache)
        self.assertIs(bound["INDICES"], indices)
        self.assertTrue(bound["STORE_FINAL"])
        self.assertEqual(bound["FS"], cache.stride())
        self.assertEqual(bound["CACHE_SIZE"], 4)
        with self.assertRaises(AssertionError):
            fn(*args, final_states_out=cache)
        indices.dtype = np.float32
        with self.assertRaises(AssertionError):
            fn(*args, final_states_out=cache, final_states_indices=indices)

    def test_mixer_initial_state_is_materialized_before_cache_write(self):
        tree = ast.parse((sim.OPS.parent / "mamba_mixer2.py").read_text())
        initial = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "initial_states" for t in n.targets
            )
            and isinstance(n.value, ast.Call)
        )
        self.assertEqual(ast.unparse(initial.value.func), "torch.where")
        self.assertIsInstance(initial.value.args[1], ast.Subscript)


if __name__ == "__main__":
    unittest.main()
