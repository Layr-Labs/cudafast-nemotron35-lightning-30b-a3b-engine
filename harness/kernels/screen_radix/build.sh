#!/usr/bin/env bash
# Source-only on the coordinator; built by official remote setup.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/screen_radix
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "screen_radix: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libscreen_radix.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libscreen_radix.so" "$here/csrc/screen_radix.cu"
cp "$here/py/__init__.py" "$out/__init__.py"
