// gather: timing probe for the zskip load path. Each CTA copies the active k rows of one
// (expert, column slice) work item into shared memory and discards them:
//   mode 0: TMA bulk copies (cp.async.bulk, one per row segment) into a ring, mbarrier per stage
//   mode 1: cooperative 16-byte cp.async (consecutive threads -> consecutive 16 B of a segment)
//   mode 2: ld.global.nc.v4 into registers (consecutive lanes -> consecutive 16 B), xor-reduced
// rows of an item: k in rows[list_off .. list_off + n), byte address base + k * row_stride.

#include <cuda_runtime.h>
#include <stdint.h>

namespace gather {

struct Params {
  const uint8_t* buf;
  const long long* item_base;
  const int* item_list;
  const int* item_n;
  const uint16_t* rows;
  long long row_stride;
  int seg;      // bytes per row segment (multiple of 16)
  int ns;       // stages in flight
  int rb;       // rows per stage
  uint32_t* sink;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void tma_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t full[32];
  const int item = blockIdx.x, lane = threadIdx.x;
  const uint8_t* base = p.buf + p.item_base[item];
  const uint16_t* rows = p.rows + p.item_list[item];
  const int n = p.item_n[item];
  if (lane < p.ns) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(&full[lane])));
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncwarp();
  const int nst = (n + p.rb - 1) / p.rb;
  for (int s = 0; s < nst; ++s) {
    const int b = s % p.ns;
    if (s >= p.ns) {  // previous use of this stage must have landed
      const uint32_t parity = ((s / p.ns) - 1) & 1;
      uint32_t done = 0;
      while (!done) {
        asm volatile(
            "{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
            : "=r"(done)
            : "r"(smem_u32(&full[b])), "r"(parity)
            : "memory");
      }
    }
    const int r0 = s * p.rb;
    const int cnt = min(p.rb, n - r0);
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(&full[b])),
                   "r"(cnt * p.seg)
                   : "memory");
    }
    __syncwarp();
    for (int i = lane; i < cnt; i += 32) {
      const uint8_t* src = base + (long long)rows[r0 + i] * p.row_stride;
      uint8_t* dst = smem + ((long long)b * p.rb + i) * p.seg;
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
              smem_u32(dst)),
          "l"(src), "r"(p.seg), "r"(smem_u32(&full[b]))
          : "memory");
    }
  }
  // drain
  for (int s = max(0, nst - p.ns); s < nst; ++s) {
    const int b = s % p.ns;
    const uint32_t parity = (s / p.ns) & 1;
    uint32_t done = 0;
    while (!done) {
      asm volatile(
          "{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
          : "=r"(done)
          : "r"(smem_u32(&full[b])), "r"(parity)
          : "memory");
    }
  }
  if (lane == 0) p.sink[item] = smem[0];
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(smem)), "l"(gmem) : "memory");
}

template <int NS>
__global__ void coop_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int item = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
  const uint8_t* base = p.buf + p.item_base[item];
  const uint16_t* rows = p.rows + p.item_list[item];
  const int n = p.item_n[item];
  const int cps = p.seg / 16;  // chunks per segment
  const int nst = (n + p.rb - 1) / p.rb;
  auto issue = [&](int s) {
    if (s < nst) {
      const int r0 = s * p.rb;
      const int cnt = min(p.rb, n - r0);
      uint8_t* st = smem + (long long)(s % NS) * p.rb * p.seg;
      for (int i = tid; i < cnt * cps; i += nth) {
        const int r = i / cps, c = i - r * cps;
        cp_async16(st + r * p.seg + c * 16, base + (long long)rows[r0 + r] * p.row_stride + c * 16);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  for (int s = 0; s < NS - 1; ++s) issue(s);
  for (int s = 0; s < nst; ++s) {
    issue(s + NS - 1);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 1) : "memory");
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  if (tid == 0) p.sink[item] = smem[0];
}

template <int U>
__global__ void regs_kernel(const Params p) {
  const int item = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
  const uint8_t* base = p.buf + p.item_base[item];
  const uint16_t* rows = p.rows + p.item_list[item];
  const int n = p.item_n[item];
  const int cps = p.seg / 16;
  const int total = n * cps;
  uint32_t x = 0;
  for (int i0 = tid; i0 < total; i0 += nth * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int i = i0 + u * nth;
      if (i < total) {
        const int r = i / cps, c = i - r * cps;
        const uint4* src = reinterpret_cast<const uint4*>(base + (long long)rows[r] * p.row_stride + c * 16);
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                     : "l"(src));
      } else {
        v[u] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (x == 0x12345678u) p.sink[item] = x;
}

}  // namespace gather

using namespace gather;

extern "C" int zskip_gather(int mode, int items, int threads, const void* buf, const long long* item_base,
                            const int* item_list, const int* item_n, const uint16_t* rows, long long row_stride,
                            int seg, int ns, int rb, uint32_t* sink, void* stream) {
  Params p{reinterpret_cast<const uint8_t*>(buf), item_base, item_list, item_n, rows, row_stride, seg, ns, rb, sink};
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const int smem = ns * rb * seg;
  if (mode == 0) {
    if (ns > 32) return -2;
    cudaFuncSetAttribute(tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tma_kernel<<<items, 32, smem, st>>>(p);
  } else if (mode == 1) {
#define C(NS_)                                                                                   \
  if (ns == NS_) {                                                                               \
    cudaFuncSetAttribute(coop_kernel<NS_>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
    coop_kernel<NS_><<<items, threads, smem, st>>>(p);                                           \
    return (int)cudaGetLastError();                                                              \
  }
    C(2) C(4) C(8) C(16)
#undef C
    return -2;
  } else if (mode == 2) {
    if (ns == 4) regs_kernel<4><<<items, threads, 0, st>>>(p);
    else if (ns == 8) regs_kernel<8><<<items, threads, 0, st>>>(p);
    else if (ns == 16) regs_kernel<16><<<items, threads, 0, st>>>(p);
    else return -2;
  } else {
    return -2;
  }
  return (int)cudaGetLastError();
}
