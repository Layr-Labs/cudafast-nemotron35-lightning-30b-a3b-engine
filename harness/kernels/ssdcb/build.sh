#!/usr/bin/env bash
# Optional research kernel; a failed build leaves the existing SSD path usable.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/ssdcb
mkdir -p "$out"
cp "$here/py/__init__.py" "$out/__init__.py"
cp "$here/py/qualification.py" "$out/qualification.py"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
if ! test -x "$nvcc"; then
  echo 'ssdcb: compiler unavailable; original SSD retained' >&2
  exit 0
fi
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}; arch=${arch// /}; arch=${arch%+PTX}; sm=${arch/./}
rm -f "$out/libssdcb.so" "$out/libssdcb.tmp.so"
if "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" -Xptxas -v \
    -o "$out/libssdcb.tmp.so" "$here/csrc/ssdcb.cu" 2>&1; then
  mv "$out/libssdcb.tmp.so" "$out/libssdcb.so"
else
  rm -f "$out/libssdcb.tmp.so"
  echo 'ssdcb: optional build failed; original SSD retained' >&2
fi
