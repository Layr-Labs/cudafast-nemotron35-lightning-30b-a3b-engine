"""GPU check of the Mamba2 spec-decode anchor+ring SSU kernel against the reference.

Run inside the pinned vLLM image with this tree's overlay on PYTHONPATH (for
example `bin/atom nemo exec SHA harness/tests/mamba_spec_replay_exact.py`).

1. Bit identity. For depths 1..6 it simulates multi-request verify sequences:
   a "prefill" writes a random anchor, then every cycle runs the reference
   kernel (`mamba_ssm.selective_state_update`, spec mode, d + 1 state slots)
   and the replay kernel on the same inputs, draws an acceptance count
   1..query_len per request, and advances positions. Each cycle compares with
   torch.equal:
     - every output row;
     - the state the replay kernel starts from (test hook) against the
       reference's slot num_accepted - 1 before the reference runs;
     - slot 0 after the verify (both kernels leave the state after token 0);
   and at the end, that the replay kernel wrote nothing outside slot 0 and the
   ring part of slot 1. Inputs use the production views (strided x/B/C/dt out
   of one projected buffer, stride-0 A/dt/dt_bias/D, the state inside a
   conv+ssm page) and, when the snapshot is mounted, the checkpoint's A_log,
   D and dt_bias of real Mamba layers.
2. Timing. 23 layers' worth of state pages (so states do not sit in L2), one
   verify per layer per cycle at N=1, reference vs replay, per acceptance count.

Writes $NEMO_OUT/mamba_spec_replay.json (and TTGIR dumps); exits 1 on any
mismatch.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import torch

from vllm.model_executor.layers.mamba.ops import mamba_ssm
from vllm.model_executor.layers.mamba.ops import mamba_ssm_spec_replay as rp

NHEADS, HEADDIM, DSTATE, NGROUPS, CONV_K = 64, 64, 128, 8, 4
INTER = NHEADS * HEADDIM
CONV_DIM = INTER + 2 * NGROUPS * DSTATE
PROJ = INTER + CONV_DIM + NHEADS  # [z | xBC | dt]
DEV = torch.device("cuda")
OUT_DIR = os.environ.get("NEMO_OUT", "/tmp/mamba_spec_replay")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# Parameters (checkpoint when available, else synthetic)


def checkpoint_params(max_layers: int = 4):
    snap = os.environ.get("NEMO_SNAPSHOT")
    if not snap:
        return []
    index = os.path.join(snap, "model.safetensors.index.json")
    if not os.path.exists(index):
        return []
    from safetensors import safe_open

    weight_map = json.load(open(index))["weight_map"]
    layers = sorted(
        {k[: -len(".A_log")] for k in weight_map if k.endswith("mixer.A_log")},
        key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1)),
    )
    out = []
    for prefix in layers[:: max(1, len(layers) // max_layers)][:max_layers]:
        t = {}
        for name in ("A_log", "D", "dt_bias"):
            key = f"{prefix}.{name}"
            with safe_open(os.path.join(snap, weight_map[key]), "pt") as f:
                t[name] = f.get_tensor(key)
        out.append(
            dict(
                name=prefix,
                A=-torch.exp(t["A_log"].float()).to(DEV),
                D=t["D"].to(torch.bfloat16).to(DEV),
                dt_bias=t["dt_bias"].to(torch.bfloat16).to(DEV),
            )
        )
    return out


def synthetic_params(gen: torch.Generator):
    A_log = torch.log(torch.rand(NHEADS, generator=gen) * 15 + 1)
    dt = torch.exp(torch.rand(NHEADS, generator=gen) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001))) + torch.log(torch.tensor(0.001)))
    dt_bias = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
    D = torch.randn(NHEADS, generator=gen) * 0.5 + 1.0
    return dict(
        name="synthetic",
        A=(-torch.exp(A_log)).to(DEV),
        D=D.to(torch.bfloat16).to(DEV),
        dt_bias=dt_bias.to(torch.bfloat16).to(DEV),
    )


# --------------------------------------------------------------------------
# Production-shaped views


SSM_BYTES = NHEADS * HEADDIM * DSTATE * 4


def conv_bytes(d: int) -> int:
    return CONV_DIM * (CONV_K - 1 + d) * 2


def ssm_view(pages: torch.Tensor, d: int) -> torch.Tensor:
    """The ssm_state view bind_kv_cache cuts out of a [conv | ssm | pad] page."""
    ssm = pages[:, conv_bytes(d) : conv_bytes(d) + SSM_BYTES].view(torch.float32)
    return ssm.view(-1, NHEADS, HEADDIM, DSTATE)


def make_pages(num_blocks: int, d: int, pad: int = 4096):
    pages = torch.empty(num_blocks, conv_bytes(d) + SSM_BYTES + pad,
                        dtype=torch.int8, device=DEV)
    return pages, ssm_view(pages, d)


def make_inputs(T: int, gen: torch.Generator, scale_dt: float = 1.0):
    proj = torch.randn(T, PROJ, generator=gen, dtype=torch.float32)
    proj[:, INTER + CONV_DIM :] *= scale_dt
    # A few tokens with dt > 20 (softplus pass-through branch) and large x.
    if T > 2:
        proj[0, INTER + CONV_DIM :] += 25.0
        proj[1, INTER : INTER + INTER] *= 8.0
    proj = proj.to(torch.bfloat16).to(DEV)
    hbc = proj[:, INTER : INTER + CONV_DIM]
    dt = proj[:, INTER + CONV_DIM :]
    x, B, C = torch.split(hbc, [INTER, NGROUPS * DSTATE, NGROUPS * DSTATE], dim=-1)
    return (
        x.view(-1, NHEADS, HEADDIM),
        dt[:, :, None].expand(-1, -1, HEADDIM),
        B.view(-1, NGROUPS, DSTATE),
        C.view(-1, NGROUPS, DSTATE),
    )


def expand_params(p):
    A_d = p["A"][:, None, None].expand(-1, HEADDIM, DSTATE).to(torch.float32)
    dt_bias = p["dt_bias"][:, None].expand(-1, HEADDIM)
    D_d = p["D"][:, None].expand(-1, HEADDIM)
    return A_d, dt_bias, D_d


def run_ref(ssm, x, dt, A_d, B, C, D_d, dt_bias, tbl, na, cu):
    out = torch.empty_like(x)
    mamba_ssm.selective_state_update(
        ssm, x, dt, A_d, B, C, D_d, dt_bias,
        dt_softplus=True,
        state_batch_indices=tbl,
        dst_state_batch_indices=tbl,
        out=out,
        num_accepted_tokens=na,
        cu_seqlens=cu,
        is_blackwell=False,
    )
    return out


def run_replay(ssm, x, dt, A_d, B, C, D_d, dt_bias, tbl, na, cu, seq_lens, ring_len, replayed=None):
    out = torch.empty_like(x)
    rp.selective_state_update_spec_replay(
        ssm, x, dt, A_d, B, C, D_d, dt_bias,
        dt_softplus=True,
        state_batch_indices=tbl,
        out=out,
        num_accepted_tokens=na,
        cu_seqlens=cu,
        seq_lens=seq_lens,
        ring_len=ring_len,
        is_blackwell=False,
        replayed_out=replayed,
    )
    return out


# --------------------------------------------------------------------------
# 1. Bit identity


def identity_case(d, params, seed, n_req, cycles, var_len, report):
    gen = torch.Generator().manual_seed(seed)
    ring_len = rp.spec_replay_ring_len(d)
    num_blocks = 1 + n_req * (1 + d) + 3  # block 0 is the null block; 3 spare
    perm = (torch.randperm(num_blocks - 1, generator=gen) + 1)[: n_req * (1 + d)]
    tbl = perm.view(n_req, 1 + d).to(torch.int32).to(DEV)
    pages_ref, ssm_ref = make_pages(num_blocks, d)
    pages_rep, ssm_rep = make_pages(num_blocks, d)
    garbage = torch.randint(-128, 128, pages_ref.shape, generator=gen, dtype=torch.int8)
    pages_ref.copy_(garbage.to(DEV))
    pages_rep.copy_(garbage.to(DEV))
    A_d, dt_bias, D_d = expand_params(params)
    x0, dt0, B0, _ = make_inputs(d + 1, gen)
    assert rp.spec_replay_supported(ssm_rep, x0, dt0, A_d, B0, dt_bias, tbl, ring_len)

    pos = torch.randint(5, 700, (n_req,), generator=gen)
    na = torch.ones(n_req, dtype=torch.int32)

    def prefill(r):
        st = (torch.randn(NHEADS, HEADDIM, DSTATE, generator=gen) * 2.0).to(DEV)
        ssm_ref[int(tbl[r, 0])] = st
        ssm_rep[int(tbl[r, 0])] = st
        na[r] = 1
        pos[r] = int(torch.randint(5, 700, (1,), generator=gen))

    for r in range(n_req):
        prefill(r)

    bad = []
    for cyc in range(cycles):
        if var_len:
            qlen = torch.randint(1, d + 2, (n_req,), generator=gen)
        else:
            qlen = torch.full((n_req,), d + 1)
        cu = torch.zeros(n_req + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(qlen, 0)
        T = int(cu[-1])
        seq_lens = (pos + qlen).to(torch.int32)
        x, dt, B, C = make_inputs(T, gen, scale_dt=float(torch.rand(1, generator=gen) * 3))
        na_d, cu_d, sl_d = na.to(DEV), cu.to(DEV), seq_lens.to(DEV)

        expect_init = torch.stack(
            [ssm_ref[int(tbl[r, int(na[r]) - 1])].clone() for r in range(n_req)]
        )
        # Probe on a copy: the state the replay kernel starts from.
        probe_ssm = ssm_view(pages_rep.clone(), d)
        replayed = torch.empty(n_req, NHEADS, HEADDIM, DSTATE, dtype=torch.float32, device=DEV)
        run_replay(probe_ssm, x, dt, A_d, B, C, D_d, dt_bias, tbl, na_d, cu_d, sl_d, ring_len, replayed)

        out_ref = run_ref(ssm_ref, x, dt, A_d, B, C, D_d, dt_bias, tbl, na_d, cu_d)
        out_rep = run_replay(ssm_rep, x, dt, A_d, B, C, D_d, dt_bias, tbl, na_d, cu_d, sl_d, ring_len)
        torch.cuda.synchronize()

        checks = {
            "out": torch.equal(out_ref, out_rep),
            "init_state": torch.equal(expect_init, replayed),
            "anchor": torch.equal(ssm_ref[tbl[:, 0].long()], ssm_rep[tbl[:, 0].long()]),
        }
        for r in range(n_req):
            report["coverage"].setdefault(f"d{d}", {}).setdefault(str(int(na[r]) - 1), 0)
            report["coverage"][f"d{d}"][str(int(na[r]) - 1)] += 1
        if not all(checks.values()):
            bad.append({"cycle": cyc, **{k: bool(v) for k, v in checks.items()}})
            if len(bad) > 5:
                break
        report["comparisons"] += 1

        # Acceptance: 1..qlen per request, uniform; sometimes a new request.
        for r in range(n_req):
            if torch.rand(1, generator=gen) < 0.04:
                prefill(r)
                continue
            n = int(torch.randint(1, int(qlen[r]) + 1, (1,), generator=gen))
            pos[r] += n
            na[r] = n

    # Footprint: nothing outside slot 0 and the ring part of slot 1 changed.
    ring_bytes = ring_len * rp.spec_replay_ring_elems(NHEADS, HEADDIM, NGROUPS, DSTATE) * 2
    cb = conv_bytes(d)
    touched = torch.zeros(pages_rep.shape, dtype=torch.bool, device=DEV)
    for r in range(n_req):
        touched[int(tbl[r, 0]), cb : cb + SSM_BYTES] = True
        touched[int(tbl[r, 1]), cb : cb + ring_bytes] = True
    # Slot-0 rewrites by "prefill" are test writes, also inside slot 0.
    untouched_ok = torch.equal(pages_rep[~touched], garbage.to(DEV)[~touched])
    if not untouched_ok:
        bad.append({"footprint": False})
    return bad


def identity_suite(report):
    gen = torch.Generator().manual_seed(1234)
    params_list = checkpoint_params() or []
    report["params"] = [p["name"] for p in params_list]
    params_list = params_list + [synthetic_params(gen)]
    failures = []
    t0 = time.time()
    for d in range(1, 7):
        for pi, params in enumerate(params_list):
            for n_req, var_len, cycles in ((1, False, 60), (3, True, 40)):
                seed = 1000 * d + 10 * pi + n_req
                bad = identity_case(d, params, seed, n_req, cycles, var_len, report)
                tag = f"d={d} params={params['name']} n_req={n_req} var_len={var_len}"
                log(("FAIL " if bad else "ok   ") + tag + (f" {bad[:3]}" if bad else ""))
                if bad:
                    failures.append({"case": tag, "bad": bad[:5]})
    report["identity_seconds"] = round(time.time() - t0, 1)
    report["identity_failures"] = failures
    return not failures


# --------------------------------------------------------------------------
# 2. Kernel layout / code dumps


def compiled_kernels(fn):
    while hasattr(fn, "fn") and not hasattr(fn, "device_caches"):
        fn = fn.fn
    out = []
    caches = getattr(fn, "device_caches", {})
    for entry in caches.values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        out.extend(cache.values())
    return out


def dump_kernels(report):
    os.makedirs(OUT_DIR, exist_ok=True)
    info = {}
    for name, fn in (
        ("reference", mamba_ssm._selective_scan_update_kernel),
        ("replay", rp._selective_scan_update_spec_replay_kernel),
    ):
        try:
            ks = compiled_kernels(fn)
        except Exception as e:  # noqa: BLE001
            info[name] = f"unavailable: {e!r}"
            continue
        info[name] = []
        for i, k in enumerate(ks):
            asm = getattr(k, "asm", {})
            ttgir = asm.get("ttgir", "")
            for kind in ("ttgir", "ptx"):
                if kind in asm:
                    with open(os.path.join(OUT_DIR, f"{name}_{i}.{kind}"), "w") as f:
                        f.write(asm[kind])
            reduce_lines = [ln.strip() for ln in ttgir.splitlines() if "tt.reduce" in ln]
            layouts = [ln.strip() for ln in ttgir.splitlines() if ln.startswith("#")]
            info[name].append({"reduce": reduce_lines, "layouts": layouts,
                               "num_warps": getattr(getattr(k, "metadata", None), "num_warps", None)})
    report["kernels"] = info


# --------------------------------------------------------------------------
# 3. Timing


def timing_suite(report, layers=23, iters=50):
    """GPU time of one verify's 23 SSU calls (CUDA-graph replay, so launch gaps
    do not count) and the CPU time to enqueue them eagerly (the production
    mamba_mixer2 op is a PIECEWISE splitting op and runs eagerly)."""
    gen = torch.Generator().manual_seed(7)
    params = synthetic_params(gen)
    A_d, dt_bias, D_d = expand_params(params)
    res = {}
    for d in range(1, 7):
        ring_len = rp.spec_replay_ring_len(d)
        num_blocks = 2 + d
        tbl = torch.arange(1, 2 + d, dtype=torch.int32, device=DEV).view(1, 1 + d)
        layer_states = [make_pages(num_blocks, d)[1] for _ in range(layers)]
        for st in layer_states:
            st.normal_()
        x, dt, B, C = make_inputs(d + 1, gen)
        cu = torch.tensor([0, d + 1], dtype=torch.int32, device=DEV)
        sl = torch.tensor([1000 + d + 1], dtype=torch.int32, device=DEV)
        row = {}
        for n_acc in sorted({1, (d + 2) // 2, d + 1}):
            na = torch.tensor([n_acc], dtype=torch.int32, device=DEV)
            times = {}
            for kind in ("reference", "replay"):
                def cycle():
                    for st in layer_states:
                        if kind == "reference":
                            run_ref(st, x, dt, A_d, B, C, D_d, dt_bias, tbl, na, cu)
                        else:
                            run_replay(st, x, dt, A_d, B, C, D_d, dt_bias, tbl, na, cu,
                                       sl, ring_len)
                for _ in range(3):
                    cycle()
                torch.cuda.synchronize()
                # CPU enqueue time (eager), GPU kept busy by a preceding sleep.
                cpu = []
                for _ in range(10):
                    torch.cuda._sleep(50_000_000)
                    t0 = time.perf_counter()
                    cycle()
                    cpu.append((time.perf_counter() - t0) * 1e6)
                    torch.cuda.synchronize()
                cpu.sort()
                # GPU time: one graph per cycle, replayed.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    cycle()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    cycle()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                gpu = []
                for _ in range(iters):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    gpu.append(e0.elapsed_time(e1) * 1000.0)
                gpu.sort()
                del g
                times[kind] = (round(gpu[len(gpu) // 2], 1), round(cpu[len(cpu) // 2], 1))
            row[f"accept{n_acc}"] = {
                "reference_gpu_us": times["reference"][0],
                "replay_gpu_us": times["replay"][0],
                "saved_gpu_us": round(times["reference"][0] - times["replay"][0], 1),
                "reference_cpu_enqueue_us": times["reference"][1],
                "replay_cpu_enqueue_us": times["replay"][1],
            }
            log(f"timing d={d} num_accepted={n_acc} ({layers} layers): GPU reference "
                f"{times['reference'][0]} us, replay {times['replay'][0]} us; CPU enqueue "
                f"reference {times['reference'][1]} us, replay {times['replay'][1]} us")
        res[f"d{d}"] = row
        del layer_states
        torch.cuda.empty_cache()
    report["timing"] = {"layers": layers, "iters": iters, **res}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"comparisons": 0, "coverage": {}, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__}
    try:
        import triton

        report["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    log(f"device {report['device']} torch {report['torch']} triton {report.get('triton')}")
    log("ssu config (reference, N=1):", mamba_ssm.try_get_optimal_ssm_config(
        HEADDIM, DSTATE, 1, NHEADS, "float32", False))
    ok = identity_suite(report)
    dump_kernels(report)
    if "--no-timing" not in sys.argv:
        timing_suite(report)
    report["identity_ok"] = ok
    with open(os.path.join(OUT_DIR, "mamba_spec_replay.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(json.dumps({k: report[k] for k in ("identity_ok", "comparisons", "coverage")}))
    log("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
