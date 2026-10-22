#!/usr/bin/env bash
set -euo pipefail

# Real prefill work on this engine scores near 1.0; gain above this ceiling means
# the timed passes reused earlier work (participant contract sections 5.3 and 8).
PREFILL_GAIN_CEILING=1.5

# Seed-prefill window ratios: decode work moved into free_decode_begin, or seed
# prefill reusing earlier work (participant contract sections 5.1.1 and 8).
SEED_WINDOW_SLOW_CEILING=1.5
SEED_WINDOW_GAIN_CEILING=1.5

score_path="${1:-}"
audit_path="${2:-}"
if [[ -z "${score_path}" ]]; then
  echo "score-plausibility-gate: REFUSED missing score.json path argument" >&2
  exit 1
fi

if [[ ! -f "${score_path}" ]]; then
  echo "score-plausibility-gate: no score.json; nothing to check"
  exit 0
fi

exec python3 -I - "${score_path}" "${PREFILL_GAIN_CEILING}" "${SEED_WINDOW_SLOW_CEILING}" "${SEED_WINDOW_GAIN_CEILING}" "${audit_path}" <<'PY'
import json
import math
import os
import sys

path = sys.argv[1]
ceiling = float(sys.argv[2])
seed_slow_ceiling = float(sys.argv[3])
seed_gain_ceiling = float(sys.argv[4])
audit_path = sys.argv[5] if len(sys.argv) > 5 else ""
prefix = "score-plausibility-gate:"

try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as exc:
    print(f"{prefix} REFUSED {path} is not valid JSON: {exc}")
    sys.exit(1)

if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
    print(f"{prefix} REFUSED passed is missing or not a boolean")
    sys.exit(1)
if data["passed"] is not True:
    print(f"{prefix} run did not pass; nothing to gate")
    sys.exit(0)

legs = (data.get("metrics") or {}).get("paired_legs")
if not isinstance(legs, list) or len(legs) == 0:
    print(f"{prefix} REFUSED passed run has missing or empty metrics.paired_legs")
    sys.exit(1)

max_gain = 0.0
for entry in legs:
    if not isinstance(entry, dict):
        print(f"{prefix} REFUSED paired_legs entry is not an object")
        sys.exit(1)
    pair = entry.get("pair", "?")
    ctrl = entry.get("control_prefill_seconds_per_token")
    cand = entry.get("candidate_prefill_seconds_per_token")
    try:
        if isinstance(ctrl, bool) or isinstance(cand, bool) or not isinstance(ctrl, (int, float)) or not isinstance(cand, (int, float)):
            raise TypeError("not a number")
        ctrl_f = float(ctrl)
        cand_f = float(cand)
    except (TypeError, ValueError, OverflowError):
        print(
            f"{prefix} REFUSED pair {pair}: "
            "control_prefill_seconds_per_token or candidate_prefill_seconds_per_token is missing"
        )
        sys.exit(1)
    if not (math.isfinite(ctrl_f) and math.isfinite(cand_f)):
        print(
            f"{prefix} REFUSED pair {pair}: prefill seconds-per-token values must be finite"
        )
        sys.exit(1)
    if ctrl_f <= 0.0 or cand_f <= 0.0:
        print(
            f"{prefix} REFUSED pair {pair}: prefill seconds-per-token values must be > 0"
        )
        sys.exit(1)
    gain = ctrl_f / cand_f
    if gain > ceiling:
        print(
            f"{prefix} REFUSED pair {pair}: prefill gain {gain:.4f} exceeds the "
            f"ceiling {ceiling:g} (participant contract 5.3 and 8)"
        )
        sys.exit(1)
    max_gain = max(max_gain, gain)

n = len(legs)

if not audit_path:
    print(f"{prefix} REFUSED private benchd audit missing")
    sys.exit(1)
if not os.path.isfile(audit_path):
    print(f"{prefix} REFUSED private benchd audit missing")
    sys.exit(1)

try:
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)
except json.JSONDecodeError:
    print(f"{prefix} REFUSED private benchd audit is not valid JSON")
    sys.exit(1)

if not isinstance(audit, dict):
    print(f"{prefix} REFUSED private benchd audit is not an object")
    sys.exit(1)

score_val = data.get("score")
audit_score = audit.get("score")
if (
    isinstance(score_val, bool)
    or isinstance(audit_score, bool)
    or not isinstance(score_val, (int, float))
    or not isinstance(audit_score, (int, float))
    or not math.isfinite(float(score_val))
    or not math.isfinite(float(audit_score))
    or float(score_val) != float(audit_score)
):
    print(f"{prefix} REFUSED private audit does not match score.json")
    sys.exit(1)

audit_legs = (audit.get("metrics") or {}).get("paired_legs")
if not isinstance(audit_legs, list):
    print(f"{prefix} REFUSED private audit paired_legs does not match score.json")
    sys.exit(1)

if len(audit_legs) != len(legs):
    print(f"{prefix} REFUSED private audit paired_legs does not match score.json")
    sys.exit(1)

score_pairs = []
for entry in legs:
    if not isinstance(entry, dict) or "pair" not in entry:
        print(f"{prefix} REFUSED private audit paired_legs does not match score.json")
        sys.exit(1)
    score_pairs.append(entry["pair"])

audit_pairs = []
for entry in audit_legs:
    if not isinstance(entry, dict) or "pair" not in entry:
        print(f"{prefix} REFUSED private audit paired_legs does not match score.json")
        sys.exit(1)
    audit_pairs.append(entry["pair"])

if audit_pairs != score_pairs:
    print(f"{prefix} REFUSED private audit paired_legs does not match score.json")
    sys.exit(1)

for entry, pair in zip(audit_legs, score_pairs):
    ctrl = entry.get("control_seed_prefill_window_seconds_per_token")
    cand = entry.get("candidate_seed_prefill_window_seconds_per_token")
    try:
        if (
            isinstance(ctrl, bool)
            or isinstance(cand, bool)
            or not isinstance(ctrl, (int, float))
            or not isinstance(cand, (int, float))
        ):
            raise TypeError("not a number")
        ctrl_f = float(ctrl)
        cand_f = float(cand)
    except (TypeError, ValueError, OverflowError):
        print(
            f"{prefix} REFUSED pair {pair}: seed-prefill window missing from the private audit"
        )
        sys.exit(1)
    if not (math.isfinite(ctrl_f) and math.isfinite(cand_f)):
        print(
            f"{prefix} REFUSED pair {pair}: seed-prefill window missing from the private audit"
        )
        sys.exit(1)
    if ctrl_f <= 0.0 or cand_f <= 0.0:
        print(
            f"{prefix} REFUSED pair {pair}: seed-prefill window missing from the private audit"
        )
        sys.exit(1)
    if cand_f > seed_slow_ceiling * ctrl_f:
        print(
            f"{prefix} REFUSED pair {pair}: seed-prefill window is over 1.5x the control leg's "
            "(decode work moved into free_decode_begin; participant contract 5.1.1 and 8)"
        )
        sys.exit(1)
    if ctrl_f > seed_gain_ceiling * cand_f:
        print(
            f"{prefix} REFUSED pair {pair}: seed prefill is over 1.5x faster than the control leg's "
            "(the seed prefill reused earlier work; participant contract 8)"
        )
        sys.exit(1)

print(
    f"{prefix} ok (max prefill gain {max_gain:.4f} over {n} pairs, ceiling {ceiling:g}; "
    "seed windows within limits)"
)
sys.exit(0)
PY
