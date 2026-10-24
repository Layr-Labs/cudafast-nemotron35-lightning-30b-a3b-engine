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
# Cover both optional features independently. Prefer retaining dm's PDL if
# segments fail to compile, then try segments without PDL, then neither.
# A failed compiler may leave an output file: remove it before EVERY attempt,
# and accept only a successful compiler exit with nonempty output.
built=0
for variant in full no-segments no-pdl legacy; do
  set --
  case "$variant" in
    no-segments) set -- -DMARLINX_NO_SEGMENTS ;;
    no-pdl) set -- -DMARLINX_NO_PDL ;;
    legacy) set -- -DMARLINX_NO_PDL -DMARLINX_NO_SEGMENTS ;;
  esac
  rm -f "$out/libmarlinx.so"
  echo "marlinx: building $variant" >&2
  if "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -lineinfo \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -Xptxas -v "$@" \
    -o "$out/libmarlinx.so" "$here/csrc/marlinx.cu" 2>&1 \
      && test -s "$out/libmarlinx.so"; then
    built=1
    break
  fi
done
if test "$built" != 1; then
  rm -f "$out/libmarlinx.so"
  echo "marlinx: all four builds failed" >&2
  exit 1
fi
cp "$here/py/__init__.py" "$out/__init__.py"
echo "marlinx: built $out/libmarlinx.so for sm_${sm}"
