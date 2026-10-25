#!/usr/bin/env bash
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/vsample
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "vsample: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libvsample.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libvsample.so" "$here/csrc/vsample.cu" 2>&1 | grep -v '^$' || true
test -f "$out/libvsample.so" || { echo "vsample: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "vsample: built $out/libvsample.so for sm_${sm}"
