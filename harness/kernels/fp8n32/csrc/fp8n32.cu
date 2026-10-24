// fp8n32: SM120 small-M FP8 projection with a 32-column CTA tile.
//
// The pinned vLLM CUTLASS M16 schedule has TileShape 16,64,128. This changes
// only the output-column tile to 16,32,128. It uses the same SM120 FP8 MMA
// instruction, mainloop K order, accumulator dtype, and scaled epilogue. No
// activation or weight is requantized or converted. Raw pointers and strides
// describe the resident's existing FP8 A/B and BF16 output buffers.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include "libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace fp8n32 {
using namespace cute;

struct Config {
  using InType = cutlass::float_e4m3_t;
  using OutType = cutlass::bfloat16_t;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_16, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm = vllm::cutlass_3x_gemm_sm120_custom<
      InType, OutType, vllm::c3x::ScaledEpilogue, TileShape, ClusterShape,
      KernelSchedule, EpilogueSchedule, Shape<_16, _32>>;
};

}  // namespace fp8n32

extern "C" int fp8n32_run(const void* out, const void* a, const void* b,
                           const void* as, const void* bs,
                           int M, int N, int K, long long ldb,
                           long long device_index, long long stream) {
  if (M != 4 || N != 10304 || K != 2688 || ldb != 10304) return -1;
  const int64_t a_strides[2] = {K, 1};
  const int64_t b_strides[2] = {1, ldb};
  const int64_t out_strides[2] = {N, 1};
  auto device = torch::stable::Device(torch::headeronly::DeviceType::CUDA,
                                     static_cast<int32_t>(device_index));
  auto a_t = torch::stable::from_blob(const_cast<void*>(a), {M, K}, a_strides,
                                      device, torch::headeronly::ScalarType::Float8_e4m3fn);
  auto b_t = torch::stable::from_blob(const_cast<void*>(b), {K, N}, b_strides,
                                      device, torch::headeronly::ScalarType::Float8_e4m3fn);
  auto out_t = torch::stable::from_blob(const_cast<void*>(out), {M, N}, out_strides,
                                        device, torch::headeronly::ScalarType::BFloat16);
  auto as_t = torch::stable::from_blob(const_cast<void*>(as), {}, {}, device,
                                       torch::headeronly::ScalarType::Float);
  auto bs_t = torch::stable::from_blob(const_cast<void*>(bs), {}, {}, device,
                                       torch::headeronly::ScalarType::Float);
  vllm::c3x::cutlass_gemm_caller<fp8n32::Config::Cutlass3xGemm>(out_t, a_t, b_t, as_t, bs_t);
  return 0;
}
