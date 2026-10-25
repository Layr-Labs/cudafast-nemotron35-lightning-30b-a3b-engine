#!/usr/bin/env bash
# fp8pre: libfp8pre.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/fp8pre.
# A failed build leaves no package and exits 0: the projections then keep CUTLASS.
set -uo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8pre
mkdir -p "$out"
rm -f "$out/libfp8pre.so" "$out/__init__.py"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
if ! test -x "$nvcc"; then echo "fp8pre: nvcc not found; CUTLASS stays" >&2; exit 0; fi
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# No --use_fast_math: the epilogue must stay plain IEEE fp32 multiplies.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libfp8pre.so" "$here/csrc/fp8pre.cu" 2>&1
if ! test -f "$out/libfp8pre.so"; then echo "fp8pre: build failed; CUTLASS stays" >&2; exit 0; fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "fp8pre: built $out/libfp8pre.so for sm_${sm}"
