#!/usr/bin/env bash
# routed_sparse: librouted_sparse.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/routed_sparse.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/routed_sparse
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "routed_sparse: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/librouted_sparse.so" "$here/csrc/routed_sparse.cu" 2>&1
test -f "$out/librouted_sparse.so" || { echo "routed_sparse: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "routed_sparse: built $out/librouted_sparse.so for sm_${sm}"
