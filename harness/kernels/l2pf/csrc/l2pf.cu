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

#ifndef L2PF_NO_TILES
namespace l2pf {
__device__ __forceinline__ void hint_range(const uint8_t* p,size_t bytes,uint32_t chunk){
 for(size_t off=0;off<bytes;off+=chunk){
  uint32_t n=(uint32_t)((bytes-off<chunk?bytes-off:chunk)&~(size_t)15);
  if(n)asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p+off),"r"(n));
 }
}
// One thread per consumer tile. Immutable verified u16 escape offsets identify
// the exact raw prefix needed by each proposed number of whole main chunks.
__global__ void prefetch_tiles_kernel(const uint8_t* gate,size_t gate_bytes,
 const uint8_t* weight,int tiles,int stride,int header,int main_bytes,int exc,
 int tail,int allowance,uint32_t chunk){
 for(size_t off=(size_t)threadIdx.x*chunk;off<gate_bytes;off+=(size_t)blockDim.x*chunk){
  uint32_t n=(uint32_t)((gate_bytes-off<chunk?gate_bytes-off:chunk)&~(size_t)15);
  if(n)asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(gate+off),"r"(n));
 }
 int tile=threadIdx.x;
 if(tile>=tiles)return;
 const uint8_t* tb=weight+(size_t)tile*stride;
 int chunks=(allowance-header)/main_bytes;
 if(chunks>32)chunks=32;
 for(;chunks>=6;--chunks){
  int raw=((int)__ldg((const uint16_t*)tb+4*chunks)+15)&~15;
  int prefix=header+chunks*main_bytes;
  if(raw<=tail&&prefix<=exc&&prefix+raw<=allowance){
   hint_range(tb,prefix,chunk);
   hint_range(tb+exc,raw,chunk);
   return;
  }
 }
 // Producer admission establishes a fitting six-chunk prefix. If unexpected
 // metadata does not fit, issue no weight hint rather than exceed its budget.
}
}
extern "C" int l2pf_prefetch_tiles_v2(const void*gate,size_t gate_bytes,const void*weight,
 int tiles,int stride,int header,int main_bytes,int exc,int tail,int allowance,
 size_t total_budget,int chunk,void*stream){
 bool layout=(header==384&&main_bytes==3584&&exc==115072)||
             (header==2432&&main_bytes==3072&&exc==100736);
 if(!layout||!weight||tiles!=84||stride<=exc||tail<0||tail>stride-exc||
    header+6*main_bytes>allowance)return -1;
 if(chunk<16||chunk>65536||chunk%16||stride%128||exc%128||tail%128||allowance%128||
    reinterpret_cast<uintptr_t>(weight)%256||reinterpret_cast<uintptr_t>(gate)%16||
    (gate_bytes&&!gate)||gate_bytes%16)return -1;
 if(gate_bytes+(size_t)tiles*allowance>total_budget)return -1;
 l2pf::prefetch_tiles_kernel<<<1,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
  (const uint8_t*)gate,gate_bytes,(const uint8_t*)weight,tiles,stride,header,main_bytes,
  exc,tail,allowance,(uint32_t)chunk);
 return (int)cudaGetLastError();
}
#endif
