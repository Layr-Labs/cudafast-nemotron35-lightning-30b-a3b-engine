#!/usr/bin/env bash
# pzskip: libpzskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pzskip.
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
rm -f "$out/libpzskip.so"  # the fallbacks below test for this build's output, not a stale one
# Full build first. If it fails: without the P2 configs 336-338 (-DPZSKIP_NO_P2: the previous
# source; the Python side then keeps config 333), then without the L2 cache policy
# (-DPZSKIP_NO_CACHE_POLICY: stock PZ333), then without both.
for flags in "" "-DPZSKIP_NO_P2" "-DPZSKIP_NO_CACHE_POLICY" "-DPZSKIP_NO_P2 -DPZSKIP_NO_CACHE_POLICY"; do
  test -n "$flags" && echo "pzskip: previous build failed; rebuilding with $flags" >&2
  # shellcheck disable=SC2086
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC $flags \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -o "$out/libpzskip.so" "$here/csrc/pzskip.cu" 2>&1 | grep -v '^$' || true
  test -f "$out/libpzskip.so" && break
done
test -f "$out/libpzskip.so" || { echo "pzskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pzskip: built $out/libpzskip.so for sm_${sm}"
