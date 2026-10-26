#!/usr/bin/env bash
# fp8c: libfp8c.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8c.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8c
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8c: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libfp8c.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8c.so" "$here/csrc/fp8c.cu" 2>&1 || true
# If the PDL variant does not build, build once more without it (-DFP8C_NO_PDL
# removes every PDL line: the previous source; the Python side then keeps the
# ordinary launch).
if ! test -f "$out/libfp8c.so"; then
  echo "fp8c: PDL build failed; rebuilding with -DFP8C_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DFP8C_NO_PDL \
    -o "$out/libfp8c.so" "$here/csrc/fp8c.cu" 2>&1 || true
fi
test -f "$out/libfp8c.so" || { echo "fp8c: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8c: built $out/libfp8c.so for sm_${sm}"
