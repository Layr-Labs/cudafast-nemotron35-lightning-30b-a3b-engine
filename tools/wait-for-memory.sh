#!/usr/bin/env bash
# Wait under the GPU lock before benchd starts either workspace's serve script.
#
# Usage: tools/wait-for-memory.sh   (no arguments; configured by the env below)
#
# This only observes memory; it never drops caches or changes the serve plan.
# SERVE_UP_MEMINFO matches serve-up.sh (default /proc/meminfo).
# WAIT_FOR_MEMORY_TIMEOUT_S (300) and WAIT_FOR_MEMORY_POLL_S (5) accept fractional
# seconds for tests. The extra 2 GiB absorbs small changes before resident boot.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

exec python3 - "${ROOT_DIR}" <<'PY'
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def log(message):
    print(f"wait-for-memory: {message}", file=sys.stderr, flush=True)


def seconds(name, default, allow_zero=False):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


def read_memory(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ("MemTotal:", "MemAvailable:"):
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise ValueError(f"invalid {fields[0]} in {path}")
            values[fields[0]] = int(fields[1]) * 1024
    if "MemTotal:" not in values or "MemAvailable:" not in values:
        raise ValueError(f"{path} lacks MemTotal/MemAvailable; refusing to wait blind")
    if values["MemTotal:"] <= 0:
        raise ValueError(f"invalid MemTotal in {path}")
    return values["MemTotal:"], values["MemAvailable:"]


try:
    root = Path(sys.argv[1])
    fixture = json.loads((root / "fixtures/nemotron35_lightning_30b_a3b_track.json").read_text())
    util = float(fixture["serve_configuration"]["engine_env"]["RESIDENT_GPU_MEMORY_UTILIZATION"])
    if not 0 < util <= 1:
        raise ValueError("fixture RESIDENT_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    # Host overhead is NOT a fixture field. Read the trusted literal without
    # sourcing serve-up.sh (which boots/stops residents); fail if it changes form.
    overhead = re.findall(r"^HOST_OVERHEAD_GIB=([0-9]+)$",
                          (root / "tools/serve-up.sh").read_text(), re.MULTILINE)
    if len(overhead) != 1:
        raise ValueError("cannot read HOST_OVERHEAD_GIB from tools/serve-up.sh")
    gib = 1024**3
    host_bytes = int(overhead[0]) * gib
    margin_bytes = 2 * gib
    meminfo = Path(os.environ.get("SERVE_UP_MEMINFO") or "/proc/meminfo")
    timeout = seconds("WAIT_FOR_MEMORY_TIMEOUT_S", "300", allow_zero=True)
    poll = seconds("WAIT_FOR_MEMORY_POLL_S", "5")
    started = time.monotonic()
    deadline = started + timeout
    log(f"need floor(MemTotal * {util}) + {overhead[0]} GiB host overhead "
        f"+ {margin_bytes / gib:g} GiB margin from {meminfo}; "
        f"timeout={timeout:g}s poll={poll:g}s")
    while True:
        total, available = read_memory(meminfo)
        # Same reservation truncation and host allowance as serve-up.sh.
        required = int(total * util) + host_bytes + margin_bytes
        now = time.monotonic()
        status = (f"MemAvailable={available / gib:.2f} GiB, "
                  f"required={required / gib:.2f} GiB ({required} bytes)")
        if available >= required:
            log(f"ready after {now - started:.3f}s: {status}")
            break
        if now >= deadline:
            raise ValueError(f"timed out after {now - started:.3f}s: {status}; "
                             "memory did not free; refusing to start measurement")
        log(f"waiting: {status}; elapsed={now - started:.3f}s")
        time.sleep(min(poll, deadline - now))
except (OSError, ValueError, KeyError, TypeError) as exc:
    log(f"ERROR: {exc}")
    sys.exit(1)
PY
