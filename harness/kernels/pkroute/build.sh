#!/usr/bin/env bash
# pkroute: libpkroute.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pkroute.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pkroute
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pkroute: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libpkroute.so"
# No --use_fast_math: the products and sums must stay plain IEEE fp32 multiplies and adds.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libpkroute.so" "$here/csrc/pkroute.cu" 2>&1 || true
if ! test -f "$out/libpkroute.so"; then
  echo "pkroute: PDL build failed; rebuilding with -DPKROUTE_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DPKROUTE_NO_PDL \
    -o "$out/libpkroute.so" "$here/csrc/pkroute.cu" 2>&1 || true
fi
test -f "$out/libpkroute.so" || { echo "pkroute: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pkroute: built $out/libpkroute.so for sm_${sm}"
