#!/usr/bin/env bash
# upskip: libupskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/upskip.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/upskip
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "upskip: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libupskip.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libupskip.so" "$here/csrc/upskip.cu"
test -f "$out/libupskip.so" || { echo "upskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "upskip: built $out/libupskip.so for sm_${sm}"
