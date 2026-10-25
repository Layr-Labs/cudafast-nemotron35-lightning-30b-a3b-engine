#!/usr/bin/env bash
# fp8tile: libfp8tile.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8tile.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8tile
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8tile: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libfp8tile.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8tile.so" "$here/csrc/fp8t6.cu" "$here/csrc/fp8t7.cu" "$here/csrc/bridge.cu" 2>&1 || true
# If the PDL variant does not build, build once more without it (-DFP8TILE_NO_PDL
# removes every PDL line: the previous source; the Python side then keeps the
# ordinary launch).
if ! test -f "$out/libfp8tile.so"; then
  echo "fp8tile: PDL build failed; rebuilding with -DFP8TILE_NO_PDL" >&2
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DFP8TILE_NO_PDL \
    -o "$out/libfp8tile.so" "$here/csrc/fp8t6.cu" "$here/csrc/fp8t7.cu" "$here/csrc/bridge.cu" 2>&1 || true
fi
test -f "$out/libfp8tile.so" || { echo "fp8tile: unavailable; original fp8c retained" >&2; cp "$here/py/__init__.py" "$out/__init__.py"; exit 0; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8tile: built $out/libfp8tile.so for sm_${sm}"
