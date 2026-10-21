#!/usr/bin/env bash
# shellcheck shell=bash
# synthetic-window.sh -- a stand-in target snapshot for off-box tests.
#
# Sourced, not run. It writes the FILES tools/serve-up.sh and setup.sh look for
# (config.json, the safetensors index, hf_quant_config.json and a tiny shard)
# so the real scripts can be driven with no checkpoint. The resident those
# tests boot runs RESIDENT_BACKEND=fake through SERVE_UP_LAUNCHER=host, so
# nothing reads the shard's contents.

# synthetic_window_weights DIR -- write the stand-in snapshot into DIR.
synthetic_window_weights() {
  local dir="$1"
  mkdir -p "${dir}" || return 1
  printf '{"model_type":"nemotron_h","vocab_size":131072}\n' > "${dir}/config.json"
  printf '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}\n' > "${dir}/model.safetensors.index.json"
  printf '{"quantization":{"quant_algo":"MIXED_PRECISION"}}\n' > "${dir}/hf_quant_config.json"
  head -c 4096 /dev/zero > "${dir}/model-00001-of-00001.safetensors"
}

# synthetic_window_meminfo FILE [TOTAL_KB] [AVAIL_KB] -- a meminfo the plan reads.
synthetic_window_meminfo() {
  printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' "${2:-125000000}" "${3:-120000000}" > "$1"
}

# synthetic_window_env -- the launcher seam every off-box serve uses.
synthetic_window_env() {
  export SERVE_UP_LAUNCHER=host RESIDENT_BACKEND=fake
}
