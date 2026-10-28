#!/usr/bin/env bash
# fp8six: libfp8six.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8six.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8six
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8six: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libfp8six.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8six.so" "$here/csrc/fp8six.cu" 2>&1 || true
# If the PDL variant does not build, build once more without it (-DFP8SIX_NO_PDL
# removes every PDL line: the previous source; the Python side then keeps the
# ordinary launch).
if ! test -f "$out/libfp8six.so"; then
  echo "fp8six: PDL build failed; rebuilding with -DFP8SIX_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DFP8SIX_NO_PDL \
    -o "$out/libfp8six.so" "$here/csrc/fp8six.cu" 2>&1 || true
fi
if ! test -f "$out/libfp8six.so"; then
  echo "fp8six: optional build failed; baseline remains available" >&2
else
  echo "fp8six: built $out/libfp8six.so for sm_${sm}"
fi
cp "$here/py/__init__.py" "$out/__init__.py"
