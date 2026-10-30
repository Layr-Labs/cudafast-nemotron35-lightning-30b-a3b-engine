// pzskip C API (kernels and launchers: pzskip.cuh). pzskip_down hands a config to the unit that
// dispatches it (pzskip_<part>.cu); this unit compiles the prep_kernel instances.

#include "pzskip.cuh"

namespace pzskip {

template <int BMT>
void launch_prep(const Params& p, cudaStream_t st) {
  prep_kernel<BMT><<<p.max_blocks, 128, 0, st>>>(p);
}
// the block rows of launch_smem4
template void launch_prep<16>(const Params&, cudaStream_t);
template void launch_prep<32>(const Params&, cudaStream_t);
template void launch_prep<64>(const Params&, cudaStream_t);

}  // namespace pzskip

extern "C" {

int pzskip_down(const void* A, long long lda, int shape_m, const int* sorted_ids, const int* expert_ids,
                const int* num_tokens_padded, const void* W, const void* S, const float* GS, void* C,
                long long ldc, long long w_expert, long long w_k, long long s_expert, long long s_g,
                void* ws, int max_blocks, int dense, int* stats, int wpc, void* stream) {
  if (shape_m < 1 || max_blocks < 1 || lda % 8 || (reinterpret_cast<uintptr_t>(A) & 15)) return -1;
  pzskip::Params p;
  p.A = static_cast<const __nv_bfloat16*>(A);
  p.lda = lda;
  p.shape_m = shape_m;
  p.sorted_ids = sorted_ids;
  p.expert_ids = expert_ids;
  p.num_tokens_padded = num_tokens_padded;
  p.W = static_cast<const uint8_t*>(W);
  p.S = static_cast<const uint8_t*>(S);
  p.GS = GS;
  p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc;
  p.w_expert = w_expert;
  p.w_k = w_k;
  p.s_expert = s_expert;
  p.s_g = s_g;
  p.ws = static_cast<uint8_t*>(ws);
  p.max_blocks = max_blocks;
  p.dense = dense;
  p.stats = stats;
  p.half64 = 0;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  // configs 300-399 launch their own prep (with their block size); 3BB: BB = block rows 16/32
  // (except 356/357: 32-row blocks, see smem5_kernel)
  int rc = pzskip::dispatch_smem4(wpc, p, st);
  if (rc == pzskip::NOT_HANDLED) rc = pzskip::dispatch_smem5(wpc, p, st);
  if (rc != pzskip::NOT_HANDLED) return rc;
  pzskip::prep_kernel<<<max_blocks, 128, 0, st>>>(p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return (int)e;
  rc = pzskip::dispatch_smem(wpc, p, st);
  if (rc == pzskip::NOT_HANDLED) rc = pzskip::dispatch_down64(wpc, p, st);
  if (rc == pzskip::NOT_HANDLED) rc = pzskip::dispatch_down32(wpc, p, st);
  return rc == pzskip::NOT_HANDLED ? -5 : rc;
}

int pzskip_bi_bytes(void) { return pzskip::BI_BYTES; }

}  // extern "C"
