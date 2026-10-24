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
# No --use_fast_math: the reductions must stay plain IEEE fp32 adds and multiplies.
# If the KSkip kernel does not build, build once more without it (-DMARLINX_NO_KSKIP removes
# every KSkip line: the crown kernel, and py/__init__.py then never selects KSkip).
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1 || {
  echo "marlinx: KSkip build failed; rebuilding with -DMARLINX_NO_KSKIP" >&2
  rm -f "$out/libmarlinx.so"
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v -DMARLINX_NO_KSKIP \
    -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1
}
test -f "$out/libmarlinx.so" || { echo "marlinx: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
cp "$here/py/kskip_layout.py" "$out/kskip_layout.py"
echo "marlinx: built $out/libmarlinx.so for sm_${sm}"
