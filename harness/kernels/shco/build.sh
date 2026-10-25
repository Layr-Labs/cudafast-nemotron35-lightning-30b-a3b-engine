#!/usr/bin/env bash
# shco: libshco.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/shco.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/shco
mkdir -p "$out"
# Stage the Python fallback even when the optional native build is unavailable.
cp "$here/py/__init__.py" "$out/__init__.py"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "shco: nvcc unavailable (skipped)" >&2; exit 0; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
# No --use_fast_math: the replica's reductions must stay plain IEEE fp32 adds and multiplies.
rm -f "$out/libshco.so"
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libshco.so" "$here/csrc/shco.cu" 2>&1; then
  rm -f "$out/libshco.so"
  echo "shco: optional build failed (skipped)" >&2
  exit 0
fi
# Non-fatal optional production kernel; Python retains the stock fallback.
test -f "$out/libshco.so" || { echo "shco: build failed (skipped)" >&2; exit 0; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "shco: built $out/libshco.so for sm_${sm}"
