#!/usr/bin/env bash
# pzskip: libpzskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pzskip.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pzskip
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pzskip: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libpzskip.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libpzskip.so" "$here/csrc/pzskip.cu" 2>&1 | grep -v '^$' || true
if ! test -f "$out/libpzskip.so"; then
  echo "pzskip: L2 cache policy compile unavailable; rebuilding stock PZ333" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -DPZSKIP_NO_CACHE_POLICY \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -o "$out/libpzskip.so" "$here/csrc/pzskip.cu" 2>&1 | grep -v '^$' || true
fi
test -f "$out/libpzskip.so" || { echo "pzskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pzskip: built $out/libpzskip.so for sm_${sm}"
