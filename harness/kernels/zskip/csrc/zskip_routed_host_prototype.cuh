// SPDX-License-Identifier: Apache-2.0
// Experimental host ABI for zskip_routed_sum_prototype.cuh.
// Compile, run and qualify this candidate only on the official worker.
// Ordinary same-stream launches provide full-grid ordering for this first
// design; unlike the old zskip kernels, no launch here requests PDL overlap.

#ifndef ZSKIP_NO_PAIR
extern "C" int zskip_routed_sizes_prototype(int rows, int max_blocks,
                                             size_t* map_bytes, size_t* block_bytes,
                                             size_t* flag_bytes, size_t* sum_bytes) {
  if ((rows != 30 && rows != 36) || max_blocks < 1 || max_blocks > rows)
    return -1;
  *map_bytes = static_cast<size_t>(rows) * sizeof(RoutedSlotMap);
  *block_bytes = static_cast<size_t>(max_blocks) * sizeof(RoutedBlockMeta);
  *flag_bytes = sizeof(int);
  *sum_bytes = static_cast<size_t>(rows / 6) * N * sizeof(__nv_bfloat16);
  return 0;
}

// All buffers must be graph-stable and mutually disjoint except output and A,
// which may alias because zskip_routed_finish_prototype is called only after
// the producer grids and the existing side-stream join have completed.
extern "C" int zskip_routed_start_prototype(
    const void* A, long long lda, int rows, const int* sorted_ids,
    const int* expert_ids, const int* padded, int max_blocks,
    const void* W, const void* S, const float* GS, void* C, long long ldc,
    long long w_expert, long long w_tile, long long w_k,
    long long s_expert, long long s_tile, long long s_g,
    const int* verify_rows, const float* weights, void* map_buf,
    void* block_buf, int* fallback, void* sum_buf, void* stream,
    int force_fallback) {
  if ((rows != 30 && rows != 36) || max_blocks < 1 || max_blocks > rows ||
      lda % 8 || ldc % 2 || !A || !sorted_ids || !expert_ids || !padded ||
      !W || !S || !GS || !C || !weights || !map_buf || !block_buf ||
      !fallback || !sum_buf) return -1;
  Params p = {};
  p.A = static_cast<const __nv_bfloat16*>(A); p.lda = lda; p.shape_m = rows;
  p.sorted_ids = sorted_ids; p.expert_ids = expert_ids; p.num_tokens_padded = padded;
  p.max_blocks = max_blocks; p.W = static_cast<const uint8_t*>(W);
  p.S = static_cast<const uint8_t*>(S); p.GS = GS; p.C = static_cast<__nv_bfloat16*>(C);
  p.ldc = ldc; p.w_expert = w_expert; p.w_tile = w_tile; p.w_k = w_k;
  p.s_expert = s_expert; p.s_tile = s_tile; p.s_g = s_g;
  p.verify_rows = verify_rows; p.mode = SKIP;
  auto map = static_cast<RoutedSlotMap*>(map_buf);
  auto blocks = static_cast<RoutedBlockMeta*>(block_buf);
  auto sum = static_cast<__nv_bfloat16*>(sum_buf);
  auto st = reinterpret_cast<cudaStream_t>(stream);
  prepare_routed_block_meta<<<max_blocks, 32, 0, st>>>(p, blocks);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return static_cast<int>(err);
  prepare_routed_slot_map<<<rows / 6, 192, 0, st>>>(p, map);
  err = cudaGetLastError();
  if (err != cudaSuccess) return static_cast<int>(err);
  classify_routed_operation<<<1, 32, 0, st>>>(p, map, blocks, fallback, force_fallback);
  err = cudaGetLastError();
  if (err != cudaSuccess) return static_cast<int>(err);
  guarded_routed_baseline_prototype<<<dim3(21, max_blocks), 32, 0, st>>>(p, fallback);
  err = cudaGetLastError();
  if (err != cudaSuccess) return static_cast<int>(err);
  routed_down_sum_prototype<<<dim3(21, rows / 6), 192, 0, st>>>(p, map, blocks,
                                                                fallback, weights, sum);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int zskip_routed_finish_prototype(
    int rows, void* C, long long ldc, const int* fallback,
    const float* weights, const void* sum_buf, void* output, void* stream) {
  if ((rows != 30 && rows != 36) || ldc % 2 || !C || !fallback ||
      !weights || !sum_buf || !output) return -1;
  Params p = {};
  p.shape_m = rows; p.C = static_cast<__nv_bfloat16*>(C); p.ldc = ldc;
  auto st = reinterpret_cast<cudaStream_t>(stream);
  finish_routed_sum_prototype<<<dim3(21, rows / 6), 128, 0, st>>>(
      p, fallback, weights, static_cast<const __nv_bfloat16*>(sum_buf),
      static_cast<__nv_bfloat16*>(output));
  return static_cast<int>(cudaGetLastError());
}
#endif
