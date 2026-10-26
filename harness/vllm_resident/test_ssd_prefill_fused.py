"""CPU execution of production kernel ASTs, not Triton/GPU numerical proof."""

import ast
from contextlib import nullcontext
import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

# Keep this small correctness emulator from creating a BLAS thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm/vllm/model_executor/layers/mamba/ops"
FUSED = OPS / "ssd_prefill_fused.py"


def function(path, name, namespace):
    node = copy.deepcopy(
        next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def bf16(x):
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    return (
        ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    )


class Value(np.ndarray):
    def to(self, dtype):
        return value(bf16(self) if dtype == "bf16" else self.astype(dtype))


def value(x):
    return np.asarray(x).view(Value)


class Tensor:
    def __init__(self, data, dtype=None, device="cuda:0"):
        self.kind = dtype or np.asarray(data).dtype
        self.data = np.ascontiguousarray(
            bf16(data) if dtype == "bf16" else np.asarray(data, dtype=dtype)
        )
        self.dtype, self.device = self.kind, device
        self.shape, self.ndim = self.data.shape, self.data.ndim
        self.is_cuda, self.contiguous = device.startswith("cuda"), True

    def stride(self, dim=None):
        strides = tuple(s // self.data.itemsize for s in self.data.strides)
        return strides if dim is None else strides[dim]

    def numel(self):
        return self.data.size

    def dim(self):
        return self.ndim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return Tensor(
            self.data[index.data if isinstance(index, Tensor) else index], self.kind
        )

    def is_contiguous(self):
        return self.contiguous


class Ptr:
    def __init__(self, tensor, offset=0):
        self.tensor = tensor
        self.data = tensor.data.reshape(-1)
        self.offset = np.asarray(offset)
        self.dtype = NS(element_ty=tensor.kind)

    def __add__(self, offset):
        return Ptr(self.tensor, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class TL:
    float32, int64, int32 = np.float32, np.int64, np.int32
    pid = (0, 0, 0)

    def program_id(self, axis):
        return value(np.int32(self.pid[axis]))

    def arange(self, a, b):
        return value(np.arange(a, b, dtype=np.int32))

    def zeros(self, shape, dtype):
        return value(np.zeros(shape, np.float32 if dtype == "bf16" else dtype))

    def load(self, ptr, mask=True, other=0.0):
        offsets, masks = np.broadcast_arrays(ptr.offset, mask)
        active = offsets[masks]
        assert np.all(active >= 0) and np.all(active < ptr.data.size), "OOB load"
        result = np.full(offsets.shape, other, ptr.data.dtype)
        result[masks] = ptr.data[active]
        return value(result)

    def store(self, ptr, data, mask=True):
        offsets, data, masks = np.broadcast_arrays(ptr.offset, data, mask)
        active = offsets[masks]
        assert np.all(active >= 0) and np.all(active < ptr.data.size), "OOB store"
        values = bf16(data[masks]) if ptr.tensor.kind == "bf16" else data[masks]
        ptr.data[active] = values

    def dot(self, a, b):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        # Deterministic small CPU contraction; no BLAS kernel/timing assumptions.
        result = np.einsum(
            "ik,kj->ij",
            np.asarray(a, np.float32),
            np.asarray(b, np.float32),
            optimize=False,
        )
        assert np.isfinite(result).all()
        return value(result)

    def minimum(self, a, b):
        return value(np.minimum(a, b))

    def where(self, mask, a, b):
        return value(np.where(mask, a, b))

    def sigmoid(self, x):
        return value(1 / (1 + np.exp(-x)))

    def cdiv(self, a, b):
        return (a + b - 1) // b

    def broadcast_to(self, x, shape):
        return value(np.broadcast_to(x, shape))

    def gather(self, x, indices, axis):
        assert np.all(indices >= 0) and np.all(indices < x.shape[axis])
        return value(np.take_along_axis(x, indices, axis))


def exp(x):
    return value(np.exp2(np.float32(1.4426950408889634) * x).astype(np.float32))


def call(fn, tl, grid, **kwargs):
    inspect.signature(fn).bind(**kwargs)
    args = {k: Ptr(v) if isinstance(v, Tensor) else v for k, v in kwargs.items()}
    for pid in np.ndindex(grid):
        tl.pid = tuple(pid) + (0,) * (3 - len(pid))
        fn(**args)


def case(lengths, chunk, initial=True, heads=2, groups=1, h=4, n=8):
    rng = np.random.default_rng(182)
    cu, last, seq = [0], [], []
    for batch, length in enumerate(lengths):
        for start in range(0, length, chunk):
            cu.append(cu[-1] + min(chunk, length - start))
            seq.append(batch)
        last.append(len(seq) - 1)
    t, nc = sum(lengths), len(seq)

    def random(shape):
        return Tensor(rng.integers(-4, 5, size=shape).astype(np.float32) / 16, "bf16")

    x, b, c = random((t, heads, h)), random((t, groups, n)), random((t, groups, n))
    dt = np.zeros((heads, nc, chunk), np.float32)
    da = np.zeros_like(dt)
    for k in range(nc):
        length = cu[k + 1] - cu[k]
        dt[:, k, :length] = 1 / 16
        da[:, k] = np.cumsum(-dt[:, k] / 8, axis=-1)
    init = (
        Tensor(
            rng.integers(-8, 9, size=(len(lengths), heads, h, n)).astype(np.float32)
            / 64
        )
        if initial
        else None
    )
    return NS(
        x=x,
        b=b,
        c=c,
        dt=Tensor(dt),
        da=Tensor(da),
        cu=Tensor(cu, np.int32),
        last=Tensor(last, np.int32),
        seq=Tensor(seq, np.int32),
        initial=init,
        out=Tensor(np.full((t, heads, h), np.nan), "bf16"),
        d=Tensor(np.arange(heads * h, dtype=np.float32).reshape(heads, h) / 64),
        z=random((t, heads, h)),
        chunk=chunk,
        nc=nc,
        heads=heads,
        groups=groups,
        h=h,
        n=n,
    )


def base_states(q, tl):
    ns = dict(tl=tl, fast_exp=exp)
    raw = Tensor(np.full((q.nc, q.heads, q.h, q.n), np.nan, np.float32))
    fn = function(OPS / "ssd_chunk_state.py", "_chunk_state_fwd_kernel", ns)
    call(
        fn,
        tl,
        (1, q.nc, q.heads),
        x_ptr=q.x,
        b_ptr=q.b,
        states_ptr=raw,
        dt_ptr=q.dt,
        dA_cumsum_ptr=q.da,
        cu_chunk_seqlens_ptr=q.cu,
        hdim=q.h,
        dstate=q.n,
        chunk_size=q.chunk,
        nheads_ngroups_ratio=q.heads // q.groups,
        **dict(
            zip(("stride_x_seqlen", "stride_x_head", "stride_x_hdim"), q.x.stride())
        ),
        **dict(
            zip(("stride_b_seqlen", "stride_b_head", "stride_b_dstate"), q.b.stride())
        ),
        **dict(
            zip(
                (
                    "stride_states_chunk",
                    "stride_states_head",
                    "stride_states_hdim",
                    "stride_states_dstate",
                ),
                raw.stride(),
            )
        ),
        **dict(
            zip(("stride_dt_head", "stride_dt_chunk", "stride_dt_csize"), q.dt.stride())
        ),
        **dict(
            zip(
                ("stride_dA_cs_head", "stride_dA_cs_chunk", "stride_dA_cs_csize"),
                q.da.stride(),
            )
        ),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    boundary = Tensor(np.full_like(raw.data, np.nan))
    fn = function(OPS / "ssd_state_passing.py", "_state_passing_fwd_kernel", ns)
    call(
        fn,
        tl,
        ((q.h * q.n + 1023) // 1024, q.last.numel(), q.heads),
        states_ptr=raw,
        out_ptr=boundary,
        dA_cs_ptr=q.da,
        initstates_ptr=q.initial,
        last_chunk_indices_ptr=q.last,
        dim=q.h * q.n,
        chunk_size=q.chunk,
        stride_states_chunk=raw.stride(0),
        stride_states_head=raw.stride(1),
        stride_states_dim=1,
        stride_out_chunk=boundary.stride(0),
        stride_out_head=boundary.stride(1),
        stride_out_dim=1,
        **dict(
            zip(
                ("stride_dA_cs_head", "stride_dA_cs_chunk", "stride_dA_cs_csize"),
                q.da.stride(),
            )
        ),
        stride_initstates_batch=q.initial.stride(0) if q.initial is not None else 0,
        stride_initstates_head=q.initial.stride(1) if q.initial is not None else 0,
        stride_initstates_dim=1,
        HAS_INITSTATES=q.initial is not None,
        BLOCK_SIZE=1024,
    )
    return boundary


def base_output(q, boundary, tl):
    ns = dict(tl=tl, fast_exp=exp)
    cb = Tensor(np.full((q.nc, q.groups, q.chunk, q.chunk), np.nan, np.float32))
    fn = function(OPS / "ssd_bmm.py", "_bmm_chunk_fwd_kernel", ns)
    call(
        fn,
        tl,
        ((q.chunk + 127) // 128 * ((q.chunk + 31) // 32), q.nc * q.groups),
        a_ptr=q.c,
        b_ptr=q.b,
        out_ptr=cb,
        cu_chunk_seqlens_ptr=q.cu,
        chunk_size=q.chunk,
        K=q.n,
        ngroups=q.groups,
        stride_a_seqlen=q.c.stride(0),
        stride_a_head=q.c.stride(1),
        stride_ak=q.c.stride(2),
        stride_b_seqlen=q.b.stride(0),
        stride_b_head=q.b.stride(1),
        stride_bk=q.b.stride(2),
        stride_out_chunk=cb.stride(0),
        stride_out_head=cb.stride(1),
        stride_outm=cb.stride(2),
        stride_outn=cb.stride(3),
        IS_CAUSAL=False,
        dot_dtype="bf16",
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    fn = function(OPS / "ssd_chunk_scan.py", "_chunk_scan_fwd_kernel", ns)
    args = dict(
        cb_ptr=cb,
        x_ptr=q.x,
        z_ptr=q.z,
        out_ptr=q.out,
        dt_ptr=q.dt,
        dA_cumsum_ptr=q.da,
        seq_idx_ptr=q.seq,
        C_ptr=q.c,
        states_ptr=boundary,
        D_ptr=q.d,
        initstates_ptr=q.initial,
        cu_chunk_seqlens_ptr=q.cu,
        chunk_size=q.chunk,
        hdim=q.h,
        dstate=q.n,
        nheads_ngroups_ratio=q.heads // q.groups,
        stride_seq_idx_chunk=1,
        stride_D_head=q.d.stride(0) if q.d is not None else 0,
        IS_CAUSAL=True,
        HAS_D=q.d is not None,
        D_HAS_HDIM=q.d.ndim == 2 if q.d is not None else False,
        HAS_Z=q.z is not None,
        HAS_INITSTATES=q.initial is not None,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_DSTATE=128,
        IS_TRITON_22=True,
    )
    names = [
        ("cb", cb, ("chunk", "head", "csize_m", "csize_k")),
        ("x", q.x, ("seqlen", "head", "hdim")),
        ("z", q.z, ("seqlen", "head", "hdim")),
        ("out", q.out, ("seqlen", "head", "hdim")),
        ("C", q.c, ("seqlen", "head", "dstate")),
        ("states", boundary, ("chunk", "head", "hdim", "dstate")),
        ("init_states", q.initial, ("batch", "head", "hdim", "dstate")),
    ]
    for name, tensor, dims in names:
        strides = tensor.stride() if tensor is not None else (0,) * len(dims)
        args.update({f"stride_{name}_{d}": s for d, s in zip(dims, strides)})
    for name, tensor in [("dt", q.dt), ("dA_cs", q.da)]:
        args.update(
            {
                f"stride_{name}_{d}": s
                for d, s in zip(("head", "chunk", "csize"), tensor.stride())
            }
        )
    call(fn, tl, (q.chunk // 64, q.nc, q.heads), **args)
    return q.out.data.copy()


def fused_result(q, tl):
    state = Tensor(np.full((q.nc, q.heads, q.h, q.n), np.nan, np.float32))
    final_cache = Tensor(
        np.full((1, q.heads, q.h, q.n), np.nan, np.float32)
    )
    final_index = Tensor(np.array([-1], np.int32))
    write_final = q.last.numel() == 1
    ns = dict(tl=tl, fast_exp=exp)
    fn = function(FUSED, "_state_pass_kernel", ns)
    istrides = q.initial.stride() if q.initial is not None else (0, 0, 0, 0)
    call(
        fn,
        tl,
        (q.last.numel(), q.heads),
        X=q.x,
        B=q.b,
        DT=q.dt,
        DA=q.da,
        CU=q.cu,
        LAST=q.last,
        INIT=q.initial,
        OUT=state,
        FINAL_CACHE=final_cache,
        FINAL_INDEX=final_index,
        H=q.h,
        N=q.n,
        RATIO=q.heads // q.groups,
        CHUNK=q.chunk,
        XS=q.x.stride(),
        BS=q.b.stride(),
        TS=q.dt.stride(),
        AS=q.da.stride(),
        IS=istrides,
        OS=state.stride(),
        HAS_INIT=q.initial is not None,
        WRITE_FINAL=write_final,
        FS=final_cache.stride(),
        CACHE_SLOTS=1 if write_final else 0,
        BM=64,
        BN=128,
        BK=32,
    )
    fn = function(FUSED, "_cb_scan_kernel", ns)
    call(
        fn,
        tl,
        (q.chunk // 64, q.nc, q.groups),
        X=q.x,
        B=q.b,
        C=q.c,
        DT=q.dt,
        DA=q.da,
        CU=q.cu,
        SEQ=q.seq,
        STATES=state,
        INIT=q.initial,
        D=q.d,
        Z=q.z,
        OUT=q.out,
        H=q.h,
        N=q.n,
        RATIO=q.heads // q.groups,
        CHUNK=q.chunk,
        XS=q.x.stride(),
        BS=q.b.stride(),
        CS=q.c.stride(),
        TS=q.dt.stride(),
        AS=q.da.stride(),
        SS=state.stride(),
        IS=istrides,
        OS=q.out.stride(),
        ZS=q.z.stride() if q.z is not None else (0, 0, 0),
        DS=q.d.stride(0) if q.d is not None else 0,
        HAS_INIT=q.initial is not None,
        HAS_D=q.d is not None,
        D_VECTOR=q.d.ndim == 2 if q.d is not None else False,
        HAS_Z=q.z is not None,
        BM=64,
        BN=64,
        BK=64,
        BC=32,
        DN=128,
    )
    return state.data.copy(), q.out.data.copy(), final_cache.data.copy()


class KernelTests(unittest.TestCase):
    def compare(self, q):
        tl = TL()
        init = q.initial.data.copy() if q.initial is not None else None
        state = base_states(q, tl)
        output = base_output(q, state, tl)
        q.out.data.fill(np.nan)
        actual, y, final_cache = fused_result(q, tl)
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(y).all())
        np.testing.assert_array_equal(
            actual.view(np.uint32), state.data.view(np.uint32)
        )
        np.testing.assert_array_equal(y.view(np.uint32), output.view(np.uint32))
        if q.last.numel() == 1:
            self.assertTrue(np.isfinite(final_cache).all())
            np.testing.assert_array_equal(
                final_cache[0].view(np.uint32), state.data[-1].view(np.uint32)
            )
        if init is not None:
            np.testing.assert_array_equal(q.initial.data, init)

    def test_one_token_no_initial(self):
        self.compare(case([1], 128, initial=False))

    def test_chunk_and_tail_boundaries(self):
        for n in (127, 128, 129, 496, 513):
            with self.subTest(n=n):
                self.compare(case([n], 128))

    def test_256_chunk(self):
        self.compare(case([496], 256))

    def test_multiple_sequences_pseudo_chunks(self):
        self.compare(case([3, 129, 17], 128))

    def test_no_initial_multiple_sequences(self):
        self.compare(case([1, 129], 256, initial=False))

    def test_all_eight_heads_share_cb(self):
        self.compare(case([129], 128, heads=8))

    def test_full_state_inner_shape(self):
        self.compare(case([129], 128, h=64, n=128))

    def test_no_residual_or_gate(self):
        q = case([129], 128)
        q.d = q.z = None
        self.compare(q)

    def test_scalar_D(self):
        q = case([33], 256)
        q.d = Tensor(np.array([0.25, -0.5], np.float32))
        self.compare(q)


def torch_double():
    return NS(
        float32=np.dtype("float32"),
        bfloat16="bf16",
        int32=np.dtype("int32"),
        int64=np.dtype("int64"),
    )


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(
            torch=torch_double(),
            current_platform=NS(is_device_capability=lambda cap: cap == 121),
        )
        self.guard = function(FUSED, "can_fuse_prefill", self.ns)
        # Full metadata shape, small enough for a CPU guard-only check.
        self.q = case([1], 128, heads=64, groups=8, h=64, n=128)

    def check(self, q=None, **kw):
        q = q or self.q
        a = dict(
            x=q.x,
            B=q.b,
            C=q.c,
            dt=q.dt,
            dA=q.da,
            cu=q.cu,
            last=q.last,
            seq=q.seq,
            initial=q.initial,
            state_dtype=np.dtype("float32"),
        )
        a.update(kw)
        return self.guard(**a)

    def test_supported(self):
        self.assertTrue(self.check())

    def test_dtype_device_and_layout_fallback(self):
        self.assertFalse(self.check(state_dtype=None))
        self.assertFalse(self.check(state_dtype="bf16"))
        for attr, bad in [
            ("dtype", np.dtype("float32")),
            ("device", "cpu"),
            ("is_cuda", False),
        ]:
            old = getattr(self.q.x, attr)
            setattr(self.q.x, attr, bad)
            self.assertFalse(self.check())
            setattr(self.q.x, attr, old)
        self.q.seq.contiguous = False
        self.assertFalse(self.check())

    def test_other_chunk_shape_arch_and_empty_fallback(self):
        self.assertFalse(self.check(case([1], 512, heads=64, groups=8, h=64, n=128)))
        self.assertFalse(self.check(case([1], 128, heads=8)))
        self.assertFalse(self.check(last=Tensor([], np.int32)))
        self.ns["current_platform"].is_device_capability = lambda cap: False
        self.assertFalse(self.check())

    def test_recurrence_and_cb_reuse_structure(self):
        old = ast.parse((OPS / "ssd_state_passing.py").read_text())
        new = ast.parse(FUSED.read_text())

        def recurrence(tree):
            return [
                ast.dump(n.value)
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "states" for t in n.targets)
                and isinstance(n.value, ast.BinOp)
            ]

        self.assertEqual(recurrence(old), recurrence(new))
        scan = next(
            n
            for n in new.body
            if isinstance(n, ast.FunctionDef) and n.name == "_cb_scan_kernel"
        )
        head_loop = next(
            n
            for n in scan.body
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "part_head"
        )
        self.assertFalse(
            any(
                isinstance(n, ast.AugAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == "cb_full"
                for n in ast.walk(head_loop)
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Attribute)
                and n.attr in ("cpu", "item", "numpy", "synchronize")
                for n in ast.walk(new)
            )
        )

    def test_wrapper_launch_binding(self):
        ns = {}
        calls = []
        for name in ("_state_pass_kernel", "_cb_scan_kernel"):
            f = function(FUSED, name, dict(tl=TL(), fast_exp=exp))
            signature = inspect.signature(f)

            class Launch:
                def __init__(self, sig, which):
                    self.sig, self.which = sig, which

                def __getitem__(self, grid):
                    def run(*args, **kw):
                        warps = kw.pop("num_warps")
                        stages = kw.pop("num_stages")
                        self.sig.bind(*args, **kw)
                        calls.append((self.which, grid, warps, stages, kw))

                    return run

            ns[name] = Launch(signature, name)
        device = NS(index=0)
        q = self.q
        for t in (
            q.x,
            q.b,
            q.c,
            q.dt,
            q.da,
            q.cu,
            q.last,
            q.seq,
            q.out,
            q.initial,
            q.d,
            q.z,
        ):
            t.device = device
        torch = torch_double()
        torch.empty = lambda shape, dtype, device: Tensor(np.empty(shape, np.float32))
        torch.accelerator = NS(device_index=lambda index: nullcontext())
        ns.update(torch=torch, triton=NS(cdiv=lambda a, b: (a + b - 1) // b), os=os,
                  _SSD_STATE_NSPLIT=1, _SSD_SCAN_HSPLIT=2)
        function(FUSED, "fused_prefill", ns)(
            q.x,
            q.b,
            q.c,
            q.dt,
            q.da,
            q.cu,
            q.last,
            q.seq,
            q.out,
            D=q.d,
            z=q.z,
            initial=q.initial,
        )
        self.assertEqual(
            [x[:4] for x in calls],
            [
                ("_state_pass_kernel", (1, 64, 1), 4, 4),
                ("_cb_scan_kernel", (2, 1, 16), 8, 1),
            ],
        )
        self.assertEqual(
            (calls[0][4]["BM"], calls[0][4]["BN"], calls[0][4]["BK"]), (64, 128, 32)
        )
        self.assertEqual(
            (
                calls[1][4]["BM"],
                calls[1][4]["BN"],
                calls[1][4]["BK"],
                calls[1][4]["BC"],
            ),
            (64, 64, 64, 32),
        )

    def test_combined_selects_two_stages_or_original_four(self):
        q = self.q
        states = Tensor(np.zeros((q.nc, 64, 64, 128), np.float32))
        for enabled in (True, False):
            for intermediate in (True, False):
                trace = []

                def stage(name, result):
                    def run(*args, **kwargs):
                        trace.append(name)
                        return result

                    return run

                ns = dict(
                    torch=torch_double(),
                    is_int_pow_2=lambda n: n > 0 and n & (n - 1) == 0,
                    rearrange=lambda x, *a, **kw: x,
                    _chunk_cumsum_fwd=stage("cumsum", (q.da, q.dt)),
                    can_fuse_prefill=lambda *a: enabled,
                    fused_prefill=stage("fused", states),
                    _chunk_state_fwd=stage("chunk", states),
                    _state_passing_fwd=stage("pass", states),
                    _bmm_chunk_fwd=stage("cb", None),
                    _chunk_scan_fwd=stage("scan", None),
                )
                run = function(
                    OPS / "ssd_combined.py", "_mamba_chunk_scan_combined_fwd", ns
                )
                answer = run(
                    q.x,
                    Tensor(np.zeros((1, 64), np.float32)),
                    Tensor(np.zeros(64, np.float32)),
                    q.b,
                    q.c,
                    128,
                    q.out,
                    initial_states=q.initial,
                    return_intermediate_states=intermediate,
                    seq_idx=q.seq,
                    cu_seqlens=Tensor([0, 1], np.int32),
                    cu_chunk_seqlens=q.cu,
                    last_chunk_indices=q.last,
                    state_dtype=np.dtype("float32"),
                )
                self.assertEqual(
                    trace,
                    ["cumsum", "fused"]
                    if enabled
                    else ["cumsum", "chunk", "pass", "cb", "scan"],
                )
                self.assertEqual(
                    answer.shape,
                    (q.nc if intermediate else q.last.numel(), 64, 64, 128),
                )


if __name__ == "__main__":
    unittest.main()
