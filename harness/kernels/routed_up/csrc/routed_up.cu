// Routed-up experiment derived from a1c083d7 zskip v4 dense math.
// The complete target up GEMM remains: no K sparsity or reduction split.
// Raw checkpoint NVFP4 bytes/scales are merely permuted at model load.
// Each output owns one ascending-k16 FP32 MMA chain, then original BF16
// GEMM rounding, non-FTZ ReLU/square and RN BF16. Runtime self-check is remote.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
namespace routed_up {
constexpr int K=2688, N=1856, G=K/16, TOP_K=6, TILES=N/64;
struct Params {
  const __nv_bfloat16* A; long long lda; int M;
  const int *ids, *experts, *ntp;
  const uint8_t *W, *S; const float* GS;
  __nv_bfloat16* C; long long ldc;
};
__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// humming fp_to_fp<Float4E2M1, BFloat16>: nibbles at bits 12..15 and 28..31.
__device__ __forceinline__ uint32_t fp4_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x70007000u) >> 6);
}
// humming fp_to_fp<Float8E4M3, BFloat16>: bytes at bits 8..15 and 24..31.
__device__ __forceinline__ uint32_t fp8_to_bf16x2(uint32_t v) {
  return (v & 0x80008000u) | ((v & 0x7F007F00u) >> 4);
}

// prepare_exp_scale_factor<nv_bfloat162, E>: (E * 0x00010001 + 0x007F007F) << 7
constexpr uint32_t kExp124 = (124u * 0x00010001u + 0x007F007Fu) << 7;  // 2^124 in both halves
constexpr uint32_t kExp120 = (120u * 0x00010001u + 0x007F007Fu) << 7;  // 2^120

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t to_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&r);
}


__device__ __forceinline__ uint32_t ldg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldg_u64(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_u32_l1(const void* p) {  // activations: L2 resident, reused
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Match the original Humming fused epilogue: first the GEMM rounds to BF16,
// then max.NaN, non-FTZ FP32 RN square and a second BF16 round.
__device__ __forceinline__ uint16_t relu2_epilogue(float acc, float gs4) {
  const uint32_t b = to_bf16x2(mul_ftz(acc, gs4), 0.0f);
  uint32_t squared;
  asm volatile(
      "{ .reg .b16 lo, hi; .reg .f32 x, y;\n"
      "mov.b32 {lo, hi}, %1; cvt.f32.bf16 x, lo;\n"
      "max.NaN.f32 x, x, 0f00000000;\n"
      "mul.rn.f32 x, x, x; mov.f32 y, 0f00000000;\n"
      "cvt.rn.bf16x2.f32 %0, y, x; }\n"
      : "=r"(squared) : "r"(b));
  return static_cast<uint16_t>(squared);
}

template<int WPC, int R>
__global__ void __launch_bounds__(WPC*32) up_kernel(Params p) {
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  const int g8=lane>>2, t=lane&3;
  const int mb=blockIdx.y, tile=blockIdx.x*WPC+warp;
  // All following collectives are warp-only; the odd final N tile can retire
  // one complete warp without abandoning a CTA barrier or partial collective.
  if (tile>=TILES || mb*16>=*p.ntp) return;
  const int e=p.experts[mb];
  if (e<0 || e>=128) return;
  const int id=lane<16 ? p.ids[mb*16+lane] : p.M*TOP_K;
  const unsigned valid=__ballot_sync(0xffffffffu,
      lane<16 && static_cast<unsigned>(id)<static_cast<unsigned>(p.M*TOP_K));
  const int nrows=__popc(valid);
  if (nrows==0) return;
  // With M<=8 and unique top-six IDs, an expert occurs at most eight times.
  // The Python bridge only admits that original MoE routing contract.
  unsigned left=valid;
  int rows[8];
#pragma unroll
  for (int r=0;r<8;++r) {
    const int src=__ffs(left)-1;
    rows[r]=__shfl_sync(0xffffffffu,id,src<0 ? 0 : src);
    left &= left-1u;
  }
  const __nv_bfloat16* xrow=g8<nrows ? p.A+(long long)(rows[g8]/TOP_K)*p.lda : nullptr;
  const uint8_t* wb=p.W+(long long)e*K*(N/2)+tile*32+g8*4;
  const uint8_t* sb=p.S+(long long)e*G*N+tile*64+g8*8;
  uint32_t wq[R][4], xb[R][2];
  uint2 sq[R];
  auto load=[&](int slot,int grp) {
    if (grp<G) {
      const int kb=grp*16;
      wq[slot][0]=ldg_u32(wb+(long long)(kb+2*t)*(N/2));
      wq[slot][1]=ldg_u32(wb+(long long)(kb+2*t+1)*(N/2));
      wq[slot][2]=ldg_u32(wb+(long long)(kb+2*t+8)*(N/2));
      wq[slot][3]=ldg_u32(wb+(long long)(kb+2*t+9)*(N/2));
      sq[slot]=ldg_u64(sb+(long long)grp*N);
      xb[slot][0]=xrow ? ldg_u32_l1(xrow+kb+2*t) : 0u;
      xb[slot][1]=xrow ? ldg_u32_l1(xrow+kb+2*t+8) : 0u;
    }
  };
  float acc[4][4];
#pragma unroll
  for (int c=0;c<4;++c) {
#pragma unroll
    for (int q=0;q<4;++q) acc[c][q]=0.0f;
  }
#pragma unroll
  for (int j=0;j<R;++j) load(j,j);
  for (int j0=0;j0<G;j0+=R) {
#pragma unroll
    for (int j=0;j<R;++j) {
      if (j0+j<G) {
        const uint32_t w0=wq[j][0],w1=wq[j][1],w2=wq[j][2],w3=wq[j][3];
        const uint2 sc=sq[j];
        const uint32_t b0=xb[j][0],b1=xb[j][1];
        load(j,j0+j+R);
        const uint32_t s01=hmul2_u32(fp8_to_bf16x2(((sc.x<<8)&0x0000ff00u)|((sc.x<<16)&0xff000000u)),kExp120);
        const uint32_t s23=hmul2_u32(fp8_to_bf16x2(((sc.x>>8)&0x0000ff00u)|(sc.x&0xff000000u)),kExp120);
        const uint32_t s45=hmul2_u32(fp8_to_bf16x2(((sc.y<<8)&0x0000ff00u)|((sc.y<<16)&0xff000000u)),kExp120);
        const uint32_t s67=hmul2_u32(fp8_to_bf16x2(((sc.y>>8)&0x0000ff00u)|(sc.y&0xff000000u)),kExp120);
#pragma unroll
        for (int c=0;c<4;++c) {
          const uint32_t sp=c==0?s01:(c==1?s23:(c==2?s45:s67));
          const uint32_t slo=__byte_perm(sp,0,0x1010),shi=__byte_perm(sp,0,0x3232);
          const uint32_t sel=c|(c<<4)|((c+4)<<8)|((c+4)<<12);
          const uint32_t x01=__byte_perm(w0,w1,sel),x23=__byte_perm(w2,w3,sel);
          const uint32_t a0=hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01<<12),kExp124),slo);
          const uint32_t a1=hmul2_u32(hmul2_u32(fp4_to_bf16x2(x01<<8),kExp124),shi);
          const uint32_t a2=hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23<<12),kExp124),slo);
          const uint32_t a3=hmul2_u32(hmul2_u32(fp4_to_bf16x2(x23<<8),kExp124),shi);
          // Every group executes; no zero substitution or identity argument.
          mma_bf16(acc[c],a0,a1,a2,a3,b0,b1);
        }
      }
    }
  }
  const float gs4=mul_ftz(p.GS[e],4.0f);
#pragma unroll
  for (int c=0;c<4;++c) {
    const int col=tile*64+16*c+g8;
#pragma unroll
    for (int q=0;q<4;++q) {
      const int r=2*t+(q&1),cc=col+((q>>1)?8:0);
      if (r<nrows) reinterpret_cast<uint16_t*>(p.C)[(long long)rows[r]*p.ldc+cc]=relu2_epilogue(acc[c][q],gs4);
    }
  }
}
} // namespace routed_up

extern "C" int routed_up_init() {
  cudaFuncAttributes attr;
  return (int)cudaFuncGetAttributes(&attr,routed_up::up_kernel<2,4>);
}
extern "C" int routed_up_run(const void* a,long long lda,int m,
    const int* ids,const int* experts,const int* ntp,int max_blocks,
    const void* w,const void* s,const float* gs,void* c,long long ldc,void* stream) {
  if (m<1 || m>8 || lda<routed_up::K || lda%8 || ldc<routed_up::N || ldc%2 || max_blocks<1 || max_blocks>48) return -2;
  routed_up::Params p{(const __nv_bfloat16*)a,lda,m,ids,experts,ntp,
      (const uint8_t*)w,(const uint8_t*)s,gs,(__nv_bfloat16*)c,ldc};
  routed_up::up_kernel<2,4><<<dim3((routed_up::TILES+1)/2,max_blocks),64,0,(cudaStream_t)stream>>>(p);
  return (int)cudaGetLastError();
}
