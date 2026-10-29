#!/usr/bin/env bash
# Optional library; failure leaves the original smallk library untouched.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/routerw12
mkdir -p "$out"
cp "$here/py/__init__.py" "$out/__init__.py"
rm -f "$out/librouterw12.so"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "routerw12: optional compiler unavailable" >&2; exit 0; }
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# No fast-math flag; the original explicit FP32 FMA/fold operations are retained.
if "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v \
    -o "$out/librouterw12.so" "$here/csrc/routerw12.cu" 2>&1 | grep -v '^$'; then
  test -s "$out/librouterw12.so" || rm -f "$out/librouterw12.so"
else
  rm -f "$out/librouterw12.so"
  echo "routerw12: optional build refused; original smallk retained" >&2
fi
