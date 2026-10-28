#!/usr/bin/env bash
# pkfinish: libpkfinish.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pkfinish.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pkfinish
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pkfinish: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libpkfinish.so"
# No --use_fast_math: the products and sums must stay plain IEEE fp32 multiplies and adds.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libpkfinish.so" "$here/csrc/pkfinish.cu" 2>&1 || true
if ! test -f "$out/libpkfinish.so"; then
  echo "pkfinish: PDL build failed; rebuilding with -DPKFINISH_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DPKFINISH_NO_PDL \
    -o "$out/libpkfinish.so" "$here/csrc/pkfinish.cu" 2>&1 || true
fi
if ! test -f "$out/libpkfinish.so"; then
  echo "pkfinish: optional build unavailable; retaining the original path" >&2
  cp "$here/py/__init__.py" "$out/__init__.py"
  exit 0
fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pkfinish: built $out/libpkfinish.so for sm_${sm}"
