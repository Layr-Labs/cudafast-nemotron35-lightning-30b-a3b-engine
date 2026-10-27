#!/usr/bin/env bash
# Optional early-release FP8 variant; original fp8ws builds independently.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8early
mkdir -p "$out"
cp "$here/py/__init__.py" "$out/__init__.py"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
if ! test -x "$nvcc"; then
  echo "fp8early: optional compiler unavailable; original fp8ws retained"
  exit 0
fi
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libfp8early.so"
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v \
    -o "$out/libfp8early.so" "$here/csrc/fp8early.cu"; then
  rm -f "$out/libfp8early.so"
  echo "fp8early: optional compile failed; original fp8ws retained"
fi
