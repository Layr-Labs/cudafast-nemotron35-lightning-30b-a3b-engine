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
__global__ void prefetch_kernel(const uint8_t* __restrict__ p, size_t bytes, uint32_t chunk) {
  for (size_t off = (size_t)threadIdx.x * chunk; off < bytes; off += (size_t)blockDim.x * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off),
                 "r"((uint32_t)(n & ~(size_t)15)));
  }
}

// A router and one projection (or its two packed planes) share one launch,
// one side-stream fork, and the caller's existing total byte budget.
__global__ void prefetch3_kernel(const uint8_t* p0, size_t b0,
                                 const uint8_t* p1, size_t b1,
                                 const uint8_t* p2, size_t b2, uint32_t chunk) {
  const size_t c0 = (b0 + chunk - 1) / chunk;
  const size_t c1 = (b1 + chunk - 1) / chunk;
  const size_t c2 = (b2 + chunk - 1) / chunk;
  for (size_t i = threadIdx.x; i < c0 + c1 + c2; i += blockDim.x) {
    const uint8_t* p;
    size_t bytes, off;
    if (i < c0) { p = p0; bytes = b0; off = i * chunk; }
    else if (i < c0 + c1) { p = p1; bytes = b1; off = (i - c0) * chunk; }
    else { p = p2; bytes = b2; off = (i - c0 - c1) * chunk; }
    const uint32_t n = (uint32_t)((bytes - off < chunk ? bytes - off : chunk) & ~(size_t)15);
    if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p + off), "r"(n));
  }
}

#ifndef L2PF_NO_FP8C_TILES
// One fp8c output tile is consumed by one projection CTA. Distribute the
// existing budget over each tile's header/main prefix AND exception tail.
// All ranges are immutable checkpoint encodings. No value is written.
__global__ void prefetch_fp8c_out_kernel(
    const uint8_t* gate, size_t gate_bytes, const uint8_t* buffer,
    size_t tile_bytes, size_t prefix_bytes, size_t tail_offset,
    size_t tail_bytes, int tiles, uint32_t chunk) {
  const size_t gate_chunks = (gate_bytes + chunk - 1) / chunk;
  const size_t head_chunks = (prefix_bytes + chunk - 1) / chunk;
  const size_t tail_chunks = (tail_bytes + chunk - 1) / chunk;
  const size_t per_tile = head_chunks + tail_chunks;
  const size_t total = gate_chunks + (size_t)tiles * per_tile;
  for (size_t i = threadIdx.x; i < total; i += blockDim.x) {
    const uint8_t* ptr;
    size_t bytes, offset;
    if (i < gate_chunks) {
      ptr = gate;
      bytes = gate_bytes;
      offset = i * chunk;
    } else {
      const size_t item = i - gate_chunks;
      const size_t tile = item / per_tile;
      const size_t part = item % per_tile;
      ptr = buffer + tile * tile_bytes;
      if (part < head_chunks) {
        bytes = prefix_bytes;
        offset = part * chunk;
      } else {
        ptr += tail_offset;
        bytes = tail_bytes;
        offset = (part - head_chunks) * chunk;
      }
    }
    const uint32_t count = (uint32_t)(bytes - offset < chunk ? bytes - offset : chunk);
    if (count) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                            :: "l"(ptr + offset), "r"(count));
  }
}
#endif
}  // namespace l2pf

// 0 on success; -1 for a misaligned range or chunk.
extern "C" int l2pf_prefetch(const void* p, size_t bytes, int chunk, void* stream) {
  if (reinterpret_cast<uintptr_t>(p) % 16 || chunk < 16 || chunk % 16) return -1;
  l2pf::prefetch_kernel<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint8_t*)p, bytes, (uint32_t)chunk);
  return (int)cudaGetLastError();
}

extern "C" int l2pf_prefetch3(const void* p0, size_t b0,
                             const void* p1, size_t b1,
                             const void* p2, size_t b2, int chunk, void* stream) {
  if (chunk < 16 || chunk % 16 || reinterpret_cast<uintptr_t>(p0) % 16 ||
      reinterpret_cast<uintptr_t>(p1) % 16 || reinterpret_cast<uintptr_t>(p2) % 16)
    return -1;
  if ((b0 && !p0) || (b1 && !p1) || (b2 && !p2)) return -1;
  l2pf::prefetch3_kernel<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint8_t*)p0, b0, (const uint8_t*)p1, b1,
      (const uint8_t*)p2, b2, (uint32_t)chunk);
  return (int)cudaGetLastError();
}

#ifndef L2PF_NO_FP8C_TILES
// Additive ABI: old libraries or fallback builds retain contiguous prefetch.
extern "C" int l2pf_prefetch_fp8c_out(
    const void* gate, size_t gate_bytes, const void* buffer, size_t tile_bytes,
    size_t prefix_bytes, size_t tail_offset, size_t tail_bytes,
    int tiles, int chunk, void* stream) {
  if (!buffer || reinterpret_cast<uintptr_t>(buffer) % 128 ||
      (gate_bytes && !gate) || reinterpret_cast<uintptr_t>(gate) % 16 ||
      gate_bytes % 16 || chunk < 16 || chunk > 65536 || chunk % 16 ||
      tiles != 84 || tile_bytes % 128 || tile_bytes > 121856 ||
      tail_offset != 115072 || tile_bytes <= tail_offset ||
      tail_bytes != tile_bytes - tail_offset || prefix_bytes % 128 ||
      prefix_bytes < 384 + 6 * 3584 || prefix_bytes > tail_offset)
    return -1;
  l2pf::prefetch_fp8c_out_kernel<<<1, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      (const uint8_t*)gate, gate_bytes, (const uint8_t*)buffer,
      tile_bytes, prefix_bytes, tail_offset, tail_bytes, tiles, (uint32_t)chunk);
  return (int)cudaGetLastError();
}
#endif
