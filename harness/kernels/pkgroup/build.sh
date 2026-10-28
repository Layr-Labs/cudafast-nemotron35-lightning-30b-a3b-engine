#!/usr/bin/env bash
# pkgroup: libpkgroup.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pkgroup.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pkgroup
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pkgroup: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libpkgroup.so"
# No --use_fast_math: the products and sums must stay plain IEEE fp32 multiplies and adds.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libpkgroup.so" "$here/csrc/pkgroup.cu" 2>&1 || true
if ! test -f "$out/libpkgroup.so"; then
  echo "pkgroup: PDL build failed; rebuilding with -DPKGROUP_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DPKGROUP_NO_PDL \
    -o "$out/libpkgroup.so" "$here/csrc/pkgroup.cu" 2>&1 || true
fi
test -f "$out/libpkgroup.so" || { echo "pkgroup: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pkgroup: built $out/libpkgroup.so for sm_${sm}"
