#!/usr/bin/env bash
# marlinx: libmarlinx.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/marlinx.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/marlinx
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "marlinx: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
rm -f "$out/libmarlinx.so"  # the fallback below tests for this build's output, not a stale one
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1; then
  rm -f "$out/libmarlinx.so"
fi
# The optional head-pair template must not remove the complete original PDL
# library if it fails to compile. Only official remote setup executes this.
if ! test -f "$out/libmarlinx.so"; then
  echo "marlinx: head-pair build failed; rebuilding original kernels" >&2
  if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DMARLINX_NO_HEAD_PAIR \
    -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1; then
    rm -f "$out/libmarlinx.so"
  fi
fi
# If the PDL variant does not build, build once more without it (-DMARLINX_NO_PDL
# removes every PDL line: the previous source; the Python side then keeps the
# ordinary launch).
if ! test -f "$out/libmarlinx.so"; then
  echo "marlinx: PDL build failed; rebuilding with -DMARLINX_NO_PDL" >&2
  if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DMARLINX_NO_HEAD_PAIR -DMARLINX_NO_PDL \
    -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1; then
    rm -f "$out/libmarlinx.so"
  fi
fi
test -f "$out/libmarlinx.so" || { echo "marlinx: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
cp "$here/py/head_pair.py" "$out/head_pair.py"
echo "marlinx: built $out/libmarlinx.so for sm_${sm}"
