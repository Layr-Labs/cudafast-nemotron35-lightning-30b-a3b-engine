#!/usr/bin/env bash
# Optional fixed-scale readiness variant; original zskip builds independently.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/scaleready
mkdir -p "$out"
cp "$here/py/__init__.py" "$out/__init__.py"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
if ! test -x "$nvcc"; then
  echo "scaleready: optional compiler unavailable; original zskip retained"
  exit 0
fi
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libscaleready.so"
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v \
    -o "$out/libscaleready.so" "$here/csrc/scaleready.cu"; then
  rm -f "$out/libscaleready.so"
  echo "scaleready: optional compile failed; original zskip retained"
fi
