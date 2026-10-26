#!/usr/bin/env bash
# smallk: libsmallk.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/smallk.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/smallk
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "smallk: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libsmallk.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
# If the bf16 gate-read variant does not build, build without it (-DSMALLK_NO_W16: no
# sk_router_gemm_w16 symbol, and the Python side keeps the fp32 read).
for flags in "" "-DSMALLK_NO_W16"; do
  test -n "$flags" && echo "smallk: previous build failed; rebuilding with $flags" >&2
  # shellcheck disable=SC2086
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v $flags \
    -o "$out/libsmallk.so" "$here/csrc/smallk.cu" 2>&1 | grep -v '^$' || true
  test -f "$out/libsmallk.so" && break
done
test -f "$out/libsmallk.so" || { echo "smallk: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "smallk: built $out/libsmallk.so for sm_${sm}"
