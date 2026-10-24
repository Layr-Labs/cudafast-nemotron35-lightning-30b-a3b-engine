#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$KERNELS_OUT/head_fused"
mkdir -p "$out"
rm -f "$out/libhead_fused.so"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" -std=c++17 -O3 --shared -Xcompiler -fPIC -gencode "arch=compute_${sm},code=sm_${sm}" \
  "$here/csrc/head_fused.cu" -o "$out/libhead_fused.so"
cp "$here/py/__init__.py" "$out/__init__.py"
