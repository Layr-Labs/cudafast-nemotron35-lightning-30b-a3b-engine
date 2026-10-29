#!/usr/bin/env bash
# Optional independent TU: failure leaves the existing FP8C library untouched.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8mix
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}; arch=${arch// /}; arch=${arch%+PTX}; sm=${arch/./}
rm -f "$out/libfp8mix.so"
if test -x "$nvcc"; then
  if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
      -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v \
      -o "$out/libfp8mix.so" "$here/csrc/fp8mix.cu"; then
    rm -f "$out/libfp8mix.so"
    if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
        -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v -DFP8MIX_NO_PDL \
        -o "$out/libfp8mix.so" "$here/csrc/fp8mix.cu"; then
      rm -f "$out/libfp8mix.so"
    fi
  fi
fi
cp "$here/py/"*.py "$out/"
if test -f "$out/libfp8mix.so"; then
  echo "fp8mix: optional native library built for sm_${sm}"
else
  echo "fp8mix: native build refused; original FP8C retained" >&2
fi
