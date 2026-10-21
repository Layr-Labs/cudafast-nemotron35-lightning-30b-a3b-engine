#!/usr/bin/env bash
# Flatten baseline calibration JSON to the v1 field layout readers expect.
# Version 3 merges the top-level metadata with the prompts[] entry whose prompt
# matches CALIBRATION_PROMPT. Versions 1 and 2 are refused: their decode mean is
# the whole window (seed prefill plus decode, over N), not the decode window.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: baseline-calibration-v1-view.sh CALIBRATION_JSON CALIBRATION_PROMPT" >&2
  exit 2
fi

cal_file="$1"
cal_prompt="$2"

cal_version="$(jq -r '.version // empty' "${cal_file}")"

case "${cal_version}" in
  3)
    count="$(jq -r --arg p "${cal_prompt}" '[.prompts[]? | select(.prompt == $p)] | length' "${cal_file}")"
    if [[ "${count}" -eq 0 ]]; then
      echo "no prompts[] entry for prompt '${cal_prompt}'" >&2
      exit 1
    fi
    if [[ "${count}" -gt 1 ]]; then
      echo "more than one prompts[] entry for prompt '${cal_prompt}'" >&2
      exit 1
    fi
    jq -c --arg p "${cal_prompt}" '
      . as $root |
      ($root.prompts[] | select(.prompt == $p)) as $entry |
      ($root | del(.prompts, .gates)) + $entry
    ' "${cal_file}"
    ;;
  *)
    echo "unsupported calibration version '${cal_version}'" >&2
    exit 1
    ;;
esac
