#!/usr/bin/env bash
# sctopk: libsctopk.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/sctopk.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/sctopk
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "sctopk: nvcc not found" >&2; exit 1; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libsctopk.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libsctopk.so" "$here/csrc/sctopk.cu"
cp "$here/py/__init__.py" "$out/__init__.py"
echo "sctopk: built $out/libsctopk.so for sm_${sm}"
