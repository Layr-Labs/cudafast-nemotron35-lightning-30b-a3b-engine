#!/usr/bin/env bash
# fp8n32: a single SM120 FP8 schedule for the 4-row Mamba in_proj.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/fp8n32
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "fp8n32: nvcc not found" >&2; exit 1; }
repo=$(cd "$here/../.." && pwd)
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
# The pinned image bundles the CUTLASS SM120 headers used by this kernel.
# Only the small vLLM dispatch/epilogue shims and cuBLAS declarations are
# vendored here; no system CUTLASS or mixed CUDA runtime headers are used.
cutlass=${FP8N32_CUTLASS_DIR:-/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm}
vllm_src=${FP8N32_VLLM_SRC_DIR:-$here/third_party/vllm}
torch_dir=$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))')
cuda_dir=${CUDA_HOME:-/usr/local/cuda}
# cublas_v2.h and cublas_api.h are the only external CUDA runtime headers
# absent from the pinned toolkit. Keep them separate from cuda_runtime.h.
extra=${FP8N32_CUDA_EXTRA_DIR:-$here/third_party/cuda_extra}
"$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -shared -Xcompiler -fPIC \
  -DTORCH_TARGET_VERSION=0x020B000000000000ULL -DUSE_CUDA \
  -gencode "arch=compute_${sm},code=sm_${sm}" \
  -I"$cuda_dir/include" -I"$vllm_src/csrc" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  -I"$cutlass/include" -I"$extra" \
  "$here/csrc/fp8n32.cu" \
  -L"$torch_dir/lib" -L"$cuda_dir/lib64" \
  -lc10_cuda -lc10 -ltorch_cuda -ltorch_cpu -ltorch -lcudart \
  -o "$out/libfp8n32.so"
cp "$here/py/__init__.py" "$out/__init__.py"
test -f "$out/libfp8n32.so"
echo "fp8n32: built $out/libfp8n32.so for sm_${sm}"
