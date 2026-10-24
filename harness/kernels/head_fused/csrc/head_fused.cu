// SPDX-License-Identifier: Apache-2.0
// Current hidden vector -> original NVFP4 partial scores -> ranked candidates.
// No target-head change. Same 32/64 group sets and survivor counts as full085.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace head_fused {
constexpr int VOCAB=131072, CHUNK=2048, KEEP=256, FINAL=32;
__device__ __forceinline__ void fp4pair(uint32_t byte,float& lo,float& hi) {
  asm("{ .reg .b8 a,b,c,d; .reg .b32 pair; .reg .b16 l,h; "
      "mov.b32 {a,b,c,d}, %2; cvt.rn.f16x2.e2m1x2 pair, a; "
      "mov.b32 {l,h}, pair; cvt.f32.f16 %0,l; cvt.f32.f16 %1,h; }"
      : "=f"(lo),"=f"(hi) : "r"(byte));
  lo=lo==0.0f?0.0f:lo;hi=hi==0.0f?0.0f:hi;
}
__device__ __forceinline__ float fp8(uint8_t bits) {
  __nv_fp8_e4m3 x; x.__x=bits; return float(x);
}
__device__ __forceinline__ uint64_t key(float score, uint32_t id) {
  const uint32_t b=__float_as_uint(score);
  uint32_t v=(b&0x80000000u)?~b:(b^0x80000000u);
  if (score==0.0f) v=0x80000000u;
  if (score!=score) v=0xffffffffu;
  return (uint64_t(v)<<11)|(2047u^id);
}
__device__ __forceinline__ uint32_t bf16pair(float lo,float hi) {
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo),"f"(hi));
  return out;
}
__device__ __forceinline__ uint32_t dequant_pair(uint8_t byte,float scale) {
  float lo,hi;fp4pair(byte,lo,hi);
  // Finite E2M1 (<=2 significant bits) times E4M3 (<=4 significant
  // bits) fits BF16 exactly. This is a dequantized operand, not a new
  // checkpoint format. Final floating-point association still changes.
  return bf16pair(__fmul_rn(lo,scale),__fmul_rn(hi,scale));
}
__device__ __forceinline__ void mma(float (&c)[4],uint32_t a0,uint32_t a1,
                                   uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
    : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
// Each warp scores16 vocabulary rows. All8 columns of the MMA B tile
// contain the same current hidden vector; only column0 is stored. The
// caller admits this path only for BF16 hidden inputs widened to FP32.
// The ascending K16 F32 chain differs from the original partial screen's
// groupwise FP32 reductions. This is a draft proposal experiment.
template<int GROUPS>
__device__ __forceinline__ float2 dot_mma(const float* h,const uint8_t* w,
                     const uint8_t* scales,int row0,int row1,int t,float gs) {
  float c[4]={0.0f,0.0f,0.0f,0.0f};
#pragma unroll 1
  for(int g=0;g<GROUPS;++g) {
    const float s0=fp8(scales[int64_t(row0)*GROUPS+g]);
    const float s1=fp8(scales[int64_t(row1)*GROUPS+g]);
    const int b=8*g+t;
    const uint32_t a0=dequant_pair(w[int64_t(row0)*(8*GROUPS)+b],s0);
    const uint32_t a1=dequant_pair(w[int64_t(row1)*(8*GROUPS)+b],s1);
    const uint32_t a2=dequant_pair(w[int64_t(row0)*(8*GROUPS)+b+4],s0);
    const uint32_t a3=dequant_pair(w[int64_t(row1)*(8*GROUPS)+b+4],s1);
    const int k=16*g+2*t;
    const uint32_t b0=bf16pair(h[k],h[k+1]);
    const uint32_t b1=bf16pair(h[k+8],h[k+9]);
    mma(c,a0,a1,a2,a3,b0,b1);
  }
  return make_float2(__fmul_rn(c[0],gs),__fmul_rn(c[2],gs));
}

// Four 512-row shards per original chunk. Keeping256 per shard cannot omit
// any top256 member of the complete chunk under this kernel's score/key.
__global__ void coarse(const float* h,const uint8_t* w,const uint8_t* s,
                       const int64_t* cols,float gs,uint64_t* partial,const int* gate) {
  using Sort=cub::BlockRadixSort<uint64_t,256,2>;
  __shared__ uint64_t scores[512];
  __shared__ float hidden[512];
  __shared__ typename Sort::TempStorage temp;
  const int tid=threadIdx.x,t=tid&3,g8=(tid&31)>>2,warp=tid>>5;
  const bool stopped=gate!=nullptr && *gate!=0;
  hidden[tid]=h[cols[tid]];hidden[tid+256]=h[cols[tid+256]];
  __syncthreads();
#pragma unroll 1
  for(int strip=0;strip<4;++strip) {
    const int q=warp*16+128*strip+g8;
    const int row=int(blockIdx.x)*512+q;
    float2 v=make_float2(0.0f,0.0f);
    if(!stopped) v=dot_mma<32>(hidden,w,s,row,row+8,t,gs);
    if(t==0) {
      scores[q]=key(v.x,uint32_t(row%CHUNK));
      scores[q+8]=key(v.y,uint32_t((row+8)%CHUNK));
    }
  }
  __syncthreads();
  uint64_t keys[2]={scores[tid],scores[tid+256]};
  Sort(temp).SortDescendingBlockedToStriped(keys,0,43);
  partial[int64_t(blockIdx.x)*KEEP+tid]=keys[0];
}
__global__ void merge(const uint64_t* partial,int64_t* out) {
  using Sort=cub::BlockRadixSort<uint64_t,256,4>;
  __shared__ typename Sort::TempStorage temp;
  uint64_t keys[4];
#pragma unroll
  for(int j=0;j<4;++j) keys[j]=partial[int64_t(blockIdx.x)*1024+threadIdx.x+j*256];
  Sort(temp).SortDescendingBlockedToStriped(keys,0,43);
  out[int64_t(blockIdx.x)*KEEP+threadIdx.x]=
    int64_t(blockIdx.x)*CHUNK+(2047u^(uint32_t(keys[0])&2047u));
}
__global__ void refine(const float* h,const uint8_t* w,const uint8_t* s,
                       const int64_t* cols,float gs,const int64_t* candidates,
                       int64_t* out,const int* gate) {
  using Sort=cub::BlockRadixSort<uint64_t,256,1>;
  __shared__ uint64_t scores[KEEP];
  __shared__ float hidden[1024];
  __shared__ typename Sort::TempStorage temp;
  const int tid=threadIdx.x,t=tid&3,g8=(tid&31)>>2,warp=tid>>5;
  const bool stopped=gate!=nullptr && *gate!=0;
#pragma unroll
  for(int j=0;j<4;++j) hidden[tid+256*j]=h[cols[tid+256*j]];
  __syncthreads();
#pragma unroll 1
  for(int strip=0;strip<2;++strip) {
    const int q=warp*16+128*strip+g8;
    const int row0=int(candidates[int64_t(blockIdx.x)*KEEP+q]);
    const int row1=int(candidates[int64_t(blockIdx.x)*KEEP+q+8]);
    float2 v=make_float2(0.0f,0.0f);
    if(!stopped) v=dot_mma<64>(hidden,w,s,row0,row1,t,gs);
    if(t==0) {
      scores[q]=key(v.x,uint32_t(row0%CHUNK));
      scores[q+8]=key(v.y,uint32_t(row1%CHUNK));
    }
  }
  __syncthreads();
  uint64_t keys[1]={scores[tid]};
  Sort(temp).SortDescendingBlockedToStriped(keys,0,43);
  if(tid<FINAL) out[int64_t(blockIdx.x)*FINAL+tid]=
    int64_t(blockIdx.x)*CHUNK+(2047u^(uint32_t(keys[0])&2047u));
}
}
extern "C" int head_fused_init() {
  cudaFuncAttributes a;
  cudaError_t e=cudaFuncGetAttributes(&a,head_fused::coarse);
  if(e!=cudaSuccess) return int(e);
  e=cudaFuncGetAttributes(&a,head_fused::merge);
  if(e!=cudaSuccess) return int(e);
  return int(cudaFuncGetAttributes(&a,head_fused::refine));
}
extern "C" int head_fused_coarse(const float* h,const uint8_t* w,const uint8_t* s,
 const int64_t* cols,float gs,uint64_t* partial,int64_t* out,const int* gate,void* stream) {
  cudaStream_t st=static_cast<cudaStream_t>(stream);
  head_fused::coarse<<<256,256,0,st>>>(h,w,s,cols,gs,partial,gate);
  cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
  head_fused::merge<<<64,256,0,st>>>(partial,out);
  return int(cudaGetLastError());
}
extern "C" int head_fused_refine(const float* h,const uint8_t* w,const uint8_t* s,
 const int64_t* cols,float gs,const int64_t* candidates,int64_t* out,const int* gate,void* stream) {
  head_fused::refine<<<64,256,0,static_cast<cudaStream_t>(stream)>>>(h,w,s,cols,gs,candidates,out,gate);
  return int(cudaGetLastError());
}
