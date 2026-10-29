#!/usr/bin/env bash
# vsample: libvsample.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/vsample.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/vsample
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "vsample: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libvsample.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libvsample.so" "$here/csrc/vsample.cu" 2>&1 | grep -v '^$' || true
test -f "$out/libvsample.so" || { echo "vsample: build failed" >&2; exit 1; }
# The original shared object remains authoritative. An optional second build
# contains the unverified warp reduction and its remote qualification probe.
rm -f "$out/libvsample_redux.so"
if ! "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
  -DVSAMPLE_REDUX_SKETCH -gencode "arch=compute_${sm},code=sm_${sm}" \
  -o "$out/libvsample_redux.so" "$here/csrc/vsample.cu"; then
  rm -f "$out/libvsample_redux.so"
  echo "vsample: optional redux build failed; original retained" >&2
fi
cp "$here/py/__init__.py" "$out/__init__.py"
cp "$here/py/redux_qualify_draft.py" "$out/redux_qualify_draft.py"
echo "vsample: built $out/libvsample.so for sm_${sm}"
