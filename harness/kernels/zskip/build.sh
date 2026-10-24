#!/usr/bin/env bash
# zskip: libzskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/zskip.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/zskip
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "zskip: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
# No --use_fast_math: the epilogue's flush-to-zero multiplies (humming is NVRTC-compiled with
# --use_fast_math) are written as explicit mul.rn.ftz.f32; nothing else in the kernel is fp32.
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -Xptxas -v \
  -o "$out/libzskip.so" "$here/csrc/zskip.cu" "$here/csrc/gather.cu" 2>&1 | grep -v '^$' || true
test -f "$out/libzskip.so" || { echo "zskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "zskip: built $out/libzskip.so for sm_${sm}"
