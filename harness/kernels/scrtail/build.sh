#!/usr/bin/env bash
# scrtail: libscrtail.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/scrtail.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/scrtail
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "scrtail: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libscrtail.so"
# No --use_fast_math: every product and sum is an explicit IEEE fp32 rn intrinsic.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libscrtail.so" "$here/csrc/scrtail.cu" 2>&1 || true
test -f "$out/libscrtail.so" || { echo "scrtail: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "scrtail: built $out/libscrtail.so for sm_${sm}"
