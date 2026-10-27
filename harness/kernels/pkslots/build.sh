#!/usr/bin/env bash
# pkslots: libpkslots.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pkslots.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pkslots
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pkslots: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libpkslots.so"
# No --use_fast_math: the products and sums must stay plain IEEE fp32 multiplies and adds.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libpkslots.so" "$here/csrc/pkslots.cu" 2>&1 || true
if ! test -f "$out/libpkslots.so"; then
  echo "pkslots: PDL build failed; rebuilding with -DPKSLOTS_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DPKSLOTS_NO_PDL \
    -o "$out/libpkslots.so" "$here/csrc/pkslots.cu" 2>&1 || true
fi
test -f "$out/libpkslots.so" || { echo "pkslots: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pkslots: built $out/libpkslots.so for sm_${sm}"
