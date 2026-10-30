#!/usr/bin/env bash
# pzskip: libpzskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pzskip.
# Every csrc/*.cu is a translation unit of its own: the units compile concurrently, at most KERNELS_JOBS
# nvcc processes at a time (default: the CPU count), and are linked into the one library.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pzskip
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pzskip: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
max=${KERNELS_JOBS:-$(nproc 2>/dev/null || echo 4)}
objs=$(mktemp -d)
trap 'rm -rf "$objs"' EXIT
rm -f "$out/libpzskip.so"
# Compiles every unit with the flags $1 and links libpzskip.so from them. Each unit's output is printed
# whole, in unit order, once all have finished; if a unit or the link fails, no library is left.
build() {
  local flags=$1 src name code rc=0
  rm -f "$objs"/*
  for src in "$here"/csrc/*.cu; do
    name=$(basename "$src" .cu)
    while (( $(jobs -rp | wc -l) >= max )); do wait -n || true; done
    # shellcheck disable=SC2086
    ( set +e; "$nvcc" -O3 -std=c++17 -Xcompiler -fPIC $flags \
        -gencode "arch=compute_${sm},code=sm_${sm}" \
        -c -o "$objs/$name.o" "$src" > "$objs/$name.log" 2>&1
      echo $? > "$objs/$name.rc" ) &
  done
  wait
  for src in "$here"/csrc/*.cu; do
    name=$(basename "$src" .cu)
    grep -v '^$' "$objs/$name.log" || true
    code=$(cat "$objs/$name.rc" 2>/dev/null || echo 1)
    if [[ $code != 0 ]]; then
      echo "pzskip: csrc/$name.cu failed (exit $code)" >&2
      rc=1
    fi
  done
  (( rc == 0 )) || return 1
  # shellcheck disable=SC2086
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC $flags \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -o "$out/libpzskip.so" "$objs"/*.o > "$objs/link.log" 2>&1 || rc=1
  grep -v '^$' "$objs/link.log" || true
  if (( rc )); then
    rm -f "$out/libpzskip.so"
    return 1
  fi
}
build "" || true
if ! test -f "$out/libpzskip.so"; then
  echo "pzskip: L2 cache policy compile unavailable; rebuilding stock PZ333/PZ356" >&2
  build "-DPZSKIP_NO_CACHE_POLICY" || true
fi
test -f "$out/libpzskip.so" || { echo "pzskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pzskip: built $out/libpzskip.so for sm_${sm}"
