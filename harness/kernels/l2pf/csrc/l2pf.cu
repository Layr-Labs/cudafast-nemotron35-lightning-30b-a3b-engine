// l2pf: queue bulk L2 prefetches of a byte range and return. Launched on a side
// stream, the copies run while latency-bound kernels on the main stream leave
// DRAM idle, so a later bandwidth-bound kernel that reads the range finds it in
// L2. It reads memory into L2 only; it writes nothing and changes no value.
//
// A cp.async.bulk.prefetch.L2 of 64 KB is honoured; 1 MB requests are dropped
// by the hardware (measured on GB10), so the range is cut into 64 KB pieces.
// The kernel lasts about as long as the transfer, which is why it must not run
// on the stream it is meant to overlap.
#include <cuda_runtime.h>
#include <cstdint>

namespace l2pf {
__global__ void prefetch_kernel(const uint8_t* __restrict__ p, size_t bytes, uint32_t chunk,
                                const int* __restrict__ gate) {
  if (gate != nullptr && *gate != 0) return;
  for (size_t off = (size_t)threadIdx.x * chunk; off < bytes; off += (size_t)blockDim.x * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off),
                 "r"((uint32_t)(n & ~(size_t)15)));
  }
}
}  // namespace l2pf

// 0 on success; -1 for a misaligned range or chunk.
extern "C" int l2pf_prefetch_gated(const void* p, size_t bytes, int chunk, const void* gate,
                                   void* stream) {
  if (reinterpret_cast<uintptr_t>(p) % 16 || chunk < 16 || chunk % 16) return -1;
  l2pf::prefetch_kernel<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint8_t*)p, bytes, (uint32_t)chunk, (const int*)gate);
  return (int)cudaGetLastError();
}

extern "C" int l2pf_prefetch(const void* p, size_t bytes, int chunk, void* stream) {
  return l2pf_prefetch_gated(p, bytes, chunk, nullptr, stream);
}
