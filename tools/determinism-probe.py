#!/usr/bin/env python3
"""determinism-probe.py -- is the engine reproducible? Run it on the box.

    tools/determinism-probe.py <resident socket> <prompt tokens.json> <reps> [<spec json>]

Drives the staged adapter against a booted resident (tools/serve-up.sh --boot)
REPS times: free_decode_begin on the prompt, then free_decode_run(128). Prints
the first position where each run's stream differs from the first run's, and
the number of distinct streams. A reproducible engine prints one stream.

WHY IT MATTERS. The serial control leg is checked against a recorded serial
tape. An engine that is not reproducible forks at the first near-tie and fails
its own golden at random, and on this checkpoint a near-tie is a few tokens
away (FP8 activation quantization amplifies a last-bit difference). Any kernel
change must keep this probe at one stream.
"""
import json, os, subprocess, sys, time
sock, prompt_file, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
prompt = json.load(open(prompt_file))
eng = os.path.expanduser("~/nemotron35/repo/.build/release/mlxfast-runtime-worker")
streams, dec = [], []
for i in range(reps):
    p = subprocess.Popen([eng], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         env=dict(os.environ, BENCH_WORKER_RESIDENT_SOCKET=sock), text=True, bufsize=1)
    json.loads(p.stdout.readline())
    req = {"id": 1, "kind": "free_decode_begin", "seed_tokens": prompt}
    if spec: req["spec"] = spec
    p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush(); r1 = json.loads(p.stdout.readline())
    t = time.perf_counter()
    p.stdin.write(json.dumps({"id": 2, "kind": "free_decode_run", "count": 128}) + "\n"); p.stdin.flush()
    r2 = json.loads(p.stdout.readline()); dec.append((time.perf_counter() - t) / 128 * 1e3)
    p.stdin.close(); p.wait()
    streams.append([r1["seed_token"]] + r2["tokens"])
def div(a, b): return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
print("first-div vs rep1:", [div(s, streams[0]) for s in streams[1:]], " distinct streams:", len({tuple(s) for s in streams}),
      " decode ms/tok:", round(sum(dec) / len(dec), 3))
out = os.environ.get("PROBE_STREAMS_OUT")
if out:
    json.dump(streams, open(out, "w"))
