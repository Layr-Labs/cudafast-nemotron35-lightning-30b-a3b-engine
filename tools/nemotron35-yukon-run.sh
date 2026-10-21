#!/usr/bin/env bash
#
# nemotron35-yukon-run.sh -- benchmark.json's benchmarkCommand.
#
# `yukon run` executes benchmarkCommand on the participant's machine. The
# ranked script, tools/nemotron35-measure-and-score.sh, needs inputs that only
# a ranked box has: the staged goldens, the reference workspace and the box
# calibration. Off a ranked box it can only refuse. This script picks the path:
#
#   RANKED  GITHUB_ACTIONS is "true", or one or more of
#           MLXFAST_NEMOTRON35_GOLDEN_DIR, MLXFAST_BASELINE_WORKSPACE and
#           MLXFAST_BASELINE_CALIBRATION is set. This script then replaces
#           itself with the ranked script and passes all arguments. The ranked
#           behavior does not change. A partial ranked environment goes to the
#           ranked script, which refuses by name. It never falls back.
#   LOCAL   none of the above. This script replaces itself with
#           tools/local-baseline.sh, the documented local path. That script
#           runs ./benchmark.sh --local-iterate over the public golden. The
#           facade boots one resident at the depth that mtp-head.manifest.json
#           declares (tools/spec-declaration.sh).
#
# The ranked workflow (.github/workflows/benchmark.yml) calls the ranked script
# directly. It does not call this script.
#
# A LOCAL RESULT HAS NO SCORE. benchd seals a local run with "score": null,
# because a local run has no serial-control leg. This script does not write a
# score of its own. Yukon reads the sealed file at scorePath
# (MLXFAST_SCORE_PATH, which benchmark.json sets to score.json).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${GITHUB_ACTIONS:-}" == "true" \
  || -n "${MLXFAST_NEMOTRON35_GOLDEN_DIR:-}" \
  || -n "${MLXFAST_BASELINE_WORKSPACE:-}" \
  || -n "${MLXFAST_BASELINE_CALIBRATION:-}" ]]; then
  exec "${ROOT}/tools/nemotron35-measure-and-score.sh" "$@"
fi

if [[ $# -ne 0 ]]; then
  echo "nemotron35-yukon-run.sh: the local path takes no arguments (got: $*)." >&2
  exit 2
fi

echo "nemotron35-yukon-run.sh: no ranked inputs. This is a LOCAL run. It has no score; the ranked runner gives the score." >&2
exec "${ROOT}/tools/local-baseline.sh"
