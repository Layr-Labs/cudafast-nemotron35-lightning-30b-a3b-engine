#!/usr/bin/env bash
#
# benchmark.sh -- repository entry point for the local benchmark.
#
# It does two things and carries no benchmark logic of its own:
#   1. resolves the pinned benchd measurement binary with tools/fetch-benchd.sh
#      (unless the caller already set BENCHD), and
#   2. changes to the repository root and execs tools/benchmark.sh "$@".
# Flags and modes belong in tools/benchmark.sh, not here.
#
# Usage:
#   ./benchmark.sh [--local-iterate | --local-submit | --official] [args...]
#   BENCHD=/path/to/benchd ./benchmark.sh ...   # use another benchd, unverified
#
# The ranked commands in benchmark.json reach
# tools/nemotron35-measure-and-score.sh (preSubmitCommand directly,
# benchmarkCommand through tools/nemotron35-yukon-run.sh). Off a ranked box,
# `yukon run` reaches this file through that wrapper and
# tools/local-baseline.sh with --local-iterate. This file is also the developer
# entry point.
set -euo pipefail

# The repository root is the directory this script lives in, so the proxy
# behaves the same from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHD_ENTRY="${SCRIPT_DIR}/tools/benchmark.sh"

if [[ ! -f "${BENCHD_ENTRY}" ]]; then
    echo "benchmark.sh: the benchmark facade is missing" >&2
    echo "  expected at: ${BENCHD_ENTRY}" >&2
    exit 1
fi

# 1. Resolve the pinned measurement binary. fetch-benchd.sh accepts an
#    already-present benchd-bin/benchd whose sha256 and byte count match the
#    channel manifest beside it (the offline path), otherwise downloads and
#    verifies it. It never yields an unverified binary: a mismatch or a failed
#    download is a hard refusal. An explicit BENCHD= from the caller is passed
#    through untouched and is not hash-checked.
if [[ -z "${BENCHD:-}" ]]; then
    BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
    export BENCHD
fi

# 2. Run from the repository root: the facade never cd's and resolves paths
#    relative to the process CWD. Under `set -e` a failed cd aborts here.
cd "${SCRIPT_DIR}"

# 3. Hand off to the facade, forwarding every argument unchanged.
exec "${BENCHD_ENTRY}" "$@"
