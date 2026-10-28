#!/usr/bin/env bash
# bf16ws: libbf16ws.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/bf16ws.
# The prefill (M=496) attention projection BF16 GEMMs; see README.md.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/bf16ws
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "bf16ws: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libbf16ws.so"
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libbf16ws.so" "$here/csrc/bf16ws.cu" 2>&1 || true
test -f "$out/libbf16ws.so" || { echo "bf16ws: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "bf16ws: built $out/libbf16ws.so for sm_${sm}"
