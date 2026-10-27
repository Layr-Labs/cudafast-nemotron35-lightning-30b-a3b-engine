#!/usr/bin/env bash
# pksplit: libpksplit.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pksplit.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pksplit
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pksplit: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libpksplit.so"
# No --use_fast_math: the products and sums must stay plain IEEE fp32 multiplies and adds.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libpksplit.so" "$here/csrc/pksplit.cu" 2>&1 || true
if ! test -f "$out/libpksplit.so"; then
  echo "pksplit: PDL build failed; rebuilding with -DPKSPLIT_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DPKSPLIT_NO_PDL \
    -o "$out/libpksplit.so" "$here/csrc/pksplit.cu" 2>&1 || true
fi
test -f "$out/libpksplit.so" || { echo "pksplit: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pksplit: built $out/libpksplit.so for sm_${sm}"
