#!/usr/bin/env bash
# fp8d6: libfp8d6.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8d6.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8d6
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8d6: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libfp8d6.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8d6.so" "$here/csrc/fp8d6.cu" 2>&1 || true
# If the PDL variant does not build, build once more without it (-DFP8D6_NO_PDL
# removes every PDL line: the previous source; the Python side then keeps the
# ordinary launch).
if ! test -f "$out/libfp8d6.so"; then
  echo "fp8d6: PDL build failed; rebuilding with -DFP8D6_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DFP8D6_NO_PDL \
    -o "$out/libfp8d6.so" "$here/csrc/fp8d6.cu" 2>&1 || true
fi
test -f "$out/libfp8d6.so" || { echo "fp8d6: unavailable; original fp8c retained" >&2; cp "$here/py/__init__.py" "$out/__init__.py"; exit 0; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8d6: built $out/libfp8d6.so for sm_${sm}"
