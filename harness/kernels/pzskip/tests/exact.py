"""pzskip vs humming's prefill w2 (block 64) on real captured prefill MoE inputs, bitwise, + timing.
  PYTHONPATH=.build/vllm-overlay:.build/kernels python3 harness/kernels/pzskip/tests/exact.py <snapshot> <cap.pt>
"""
import os, sys, time, statistics as st
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests"))
os.environ.setdefault("VLLM_HUMMING_BATCH_INVARIANT", "1")
import humming_moe_exact as H  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts import humming_tuned_moe  # noqa: E402,F401
from vllm.utils.humming import HummingMethod  # noqa: E402
import zskip  # noqa: E402
import pzskip  # noqa: E402

snap = H.Snapshot(sys.argv[1])
cap = torch.load(sys.argv[2])
torch.cuda.set_device(0)
stats = torch.zeros(4, dtype=torch.int32, device="cuda")
bad = total = 0
times = {"humming": [], "pz_dense": []}
for li in [int(v) for v in os.environ.get("LAYERS", "1,13,27,40,51").split(",")]:
    raw = snap.raw_experts(li, "cuda")
    packed = zskip.repack(raw["w2_weight"].clone(), raw["w2_weight_scale"].clone(),
                          raw["w2_weight_scale_2"].clone(), layout="swap")
    ex = H.build_variant(raw, f"backbone.layers.{li}.mixer.experts", True)
    gate_w, bias, norm_w = snap.router(li, "cuda")
    for x in cap[li]:
        x = x.cuda()
        ids, w = H.route(x, gate_w, bias)
        stages = H.run_stages(ex, x, ids, w)
        act = stages["act"]
        kw1, kw2 = ex.prepare_humming_moe_kwargs(ids, None, None)
        ref = stages["w2"]
        out = torch.empty_like(ref)
        pzskip.down(act, kw2["sorted_ids"], kw2["expert_ids"], kw2["num_tokens_padded"], packed, out, stats=stats, wpc=int(os.environ.get("CHECK_WPC", "6")))
        torch.cuda.synchronize()
        d = int((ref.view(torch.int16) != out.view(torch.int16)).sum())
        bad += d; total += ref.numel()
        # adversarial: -0 instead of +0, dense random
        neg = act.clone(); neg[neg == 0] = -0.0
        out2 = torch.empty_like(ref)
        ref2 = torch.empty_like(ref)
        HummingMethod.forward_layer(layer=ex.layer, inputs=neg, input_scale=None, outputs=ref2, sublayer_name="w2", **kw2)
        pzskip.down(neg, kw2["sorted_ids"], kw2["expert_ids"], kw2["num_tokens_padded"], packed, out2, stats=stats, wpc=int(os.environ.get("CHECK_WPC", "6")))
        torch.cuda.synchronize()
        d2 = int((ref2.view(torch.int16) != out2.view(torch.int16)).sum()); bad += d2; total += ref.numel()
        if d or d2:
            print(f"  layer {li}: {d} / {d2} differ (block_m={kw2.get('block_m', '?')})", flush=True)
        def tm(f, it=20):
            for _ in range(3): f()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(it): f()
            torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
        times["humming"].append(tm(lambda: HummingMethod.forward_layer(layer=ex.layer, inputs=act, input_scale=None, outputs=ref2, sublayer_name="w2", **kw2)))
        for wpc in (2, 200, 201, 202, 203, 204):
            times.setdefault(f"pz{wpc}", []).append(tm(lambda: pzskip.down(act, kw2["sorted_ids"], kw2["expert_ids"], kw2["num_tokens_padded"], packed, out2, wpc=wpc)))
        times["pz_dense"].append(tm(lambda: pzskip.down(act, kw2["sorted_ids"], kw2["expert_ids"], kw2["num_tokens_padded"], packed, out2, dense=True)))
    del ex, packed, raw
    torch.cuda.empty_cache()
    print(f"layer {li} done", flush=True)
print(f"exactness: {bad} of {total} outputs differ; dense redo warps {int(stats[0])}")
print("timing (us, warm L2 caveat: same layer repeated): " + ", ".join(f"{k} {st.median(v):.0f}" for k, v in times.items()))
sys.exit(1 if bad else 0)
