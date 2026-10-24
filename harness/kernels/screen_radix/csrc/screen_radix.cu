// SPDX-License-Identifier: Apache-2.0
// Current-call draft selection only. Ordered score/low-ID key originates in
// 3e519ff4-4664-434d-b412-fc3c96227f25; block radix implementation is new.
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace screen_radix {
template<int ID_BITS>
__device__ __forceinline__ uint64_t key(float score, uint32_t id) {
  const uint32_t bits = __float_as_uint(score);
  uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  if (score == 0.0f) ordered = 0x80000000u;
  if (score != score) ordered = 0xffffffffu;
  constexpr uint32_t ID_MASK = (1u << ID_BITS) - 1u;
  return (static_cast<uint64_t>(ordered) << ID_BITS) |
         static_cast<uint64_t>(ID_MASK ^ id);
}

template<int WIDTH, int ITEMS>
__global__ void select(const float* scores, int64_t* out, int keep, int offset) {
  constexpr int THREADS = 256;
  constexpr int ID_BITS = WIDTH == 2048 ? 11 : 8;
  constexpr uint32_t ID_MASK = WIDTH - 1;
  static_assert(WIDTH == THREADS * ITEMS);
  using Sort = cub::BlockRadixSort<uint64_t, THREADS, ITEMS>;
  __shared__ typename Sort::TempStorage temp;
  uint64_t keys[ITEMS];
  const int row = blockIdx.x;
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const uint32_t id = threadIdx.x + j * THREADS;
    keys[j] = key<ID_BITS>(scores[static_cast<int64_t>(row) * WIDTH + id], id);
  }
  // Input arrangement is immaterial: the complete local ID is in each unique
  // key. Striped output gives coalesced candidate stores for KEEP<=256.
  // Sort only the 32 score bits and the 8/11 meaningful local-ID bits.
  Sort(temp).SortDescendingBlockedToStriped(keys, 0, 32 + ID_BITS);
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const int rank = threadIdx.x + j * THREADS;
    if (rank < keep) {
      const uint32_t id = ID_MASK ^ (static_cast<uint32_t>(keys[j]) & ID_MASK);
      out[static_cast<int64_t>(row) * keep + rank] =
          static_cast<int64_t>(id) + static_cast<int64_t>(row) * offset;
    }
  }
}
}

extern "C" int screen_radix_init() {
  cudaFuncAttributes attr;
  cudaError_t rc = cudaFuncGetAttributes(&attr, screen_radix::select<2048, 8>);
  if (rc != cudaSuccess) return static_cast<int>(rc);
  return static_cast<int>(cudaFuncGetAttributes(&attr, screen_radix::select<256, 1>));
}

extern "C" int screen_radix_topk(const float* scores, int64_t* out, int rows,
                                int width, int keep, int offset, void* stream) {
  if (!scores || !out || rows <= 0 || keep <= 0 || keep > 256 || keep > width
      || offset < 0 || (width != 2048 && width != 256))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  if (width == 2048)
    screen_radix::select<2048, 8><<<rows, 256, 0, st>>>(scores, out, keep, offset);
  else
    screen_radix::select<256, 1><<<rows, 256, 0, st>>>(scores, out, keep, offset);
  return static_cast<int>(cudaGetLastError());
}
