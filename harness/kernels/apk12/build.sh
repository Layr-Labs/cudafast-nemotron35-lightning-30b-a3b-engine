#!/usr/bin/env bash
# apk12: libapk12.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/apk12.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/apk12
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "apk12: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
# No --use_fast_math: the reduction must stay the MMA's fp32 accumulation and one RN rounding.
# Only the official worker executes this build. Keep the original reference ABI if
# this toolchain cannot compile the optional PDL ABI; production then keeps record Triton.
rm -f "$out/libapk12.so"
for flags in "" "-DAPK12_NO_PDL" "-DAPK12_NO_STAGED" "-DAPK12_NO_STAGED -DAPK12_NO_PDL"; do
  # shellcheck disable=SC2086
  if "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo $flags \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -o "$out/libapk12.so" "$here/csrc/apk12.cu"; then
    break
  fi
  rm -f "$out/libapk12.so"
done
test -f "$out/libapk12.so" || { echo "apk12: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "apk12: built $out/libapk12.so for sm_${sm}"
