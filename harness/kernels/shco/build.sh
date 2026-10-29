#!/usr/bin/env bash
# shco: libshco.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/shco.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/shco
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "shco: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
"$nvcc" --version | tail -n 2
# No --use_fast_math: the replica's reductions must stay plain IEEE fp32 adds and multiplies.
build_shco() {
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v "$@" \
    -o "$out/libshco.so" "$here/csrc/shco.cu" 2>&1 | grep -v '^$'
}
rm -f "$out/libshco.so"
if ! build_shco || ! test -s "$out/libshco.so"; then
  echo "shco: paired build unavailable; compiling original SHX without pair" >&2
  rm -f "$out/libshco.so"
  if ! build_shco -DSHX_NO_PAIR=1 || ! test -s "$out/libshco.so"; then
    rm -f "$out/libshco.so"
  fi
fi
# Non-fatal: shco is a probe package; the other kernel packages must still build without it.
test -s "$out/libshco.so" || { echo "shco: build failed (skipped)" >&2; exit 0; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "shco: built $out/libshco.so for sm_${sm}"
