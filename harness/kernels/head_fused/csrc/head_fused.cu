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
// Four neighboring lanes cooperatively score one row. Native association can
// differ from Triton's eight-group reduction, so remote token validation is
// authoritative; a boot comparison can decline this optional path.
template<int GROUPS>
__device__ __forceinline__ float dot(const float* h,const uint8_t* w,
                                    const uint8_t* scales,
                                    int row,int l4,float gs) {
  float acc=0.0f;
#pragma unroll 1
  for (int g=0;g<GROUPS;++g) {
    const int b=8*g+2*l4;
    const uint8_t a=w[int64_t(row)*(8*GROUPS)+b];
    const uint8_t c=w[int64_t(row)*(8*GROUPS)+b+1];
    const float h0=h[2*b],h1=h[2*b+1];
    const float h2=h[2*b+2],h3=h[2*b+3];
    float a0,a1,c0,c1;fp4pair(a,a0,a1);fp4pair(c,c0,c1);
    const float p0=__fmaf_rn(a0,h0,__fmul_rn(a1,h1));
    const float p1=__fmaf_rn(c0,h2,__fmul_rn(c1,h3));
    float v=__fadd_rn(p0,p1);
    v=__fadd_rn(v,__shfl_xor_sync(0xffffffffu,v,1,4));
    v=__fadd_rn(v,__shfl_xor_sync(0xffffffffu,v,2,4));
    acc=__fadd_rn(acc,__fmul_rn(v,fp8(scales[int64_t(row)*GROUPS+g])));
  }
  return __fmul_rn(acc,gs);
}

// Four 512-row shards per original chunk. Keeping256 per shard cannot omit
// any top256 member of the complete chunk under this kernel's score/key.
__global__ void coarse(const float* h,const uint8_t* w,const uint8_t* s,
                       const int64_t* cols,float gs,uint64_t* partial,const int* gate) {
  using Sort=cub::BlockRadixSort<uint64_t,256,2>;
  __shared__ uint64_t scores[512];
  __shared__ float hidden[512];
  __shared__ typename Sort::TempStorage temp;
  const int tid=threadIdx.x,l4=tid&3;
  const bool stopped=gate!=nullptr && *gate!=0;
  hidden[tid]=h[cols[tid]];hidden[tid+256]=h[cols[tid+256]];
  __syncthreads();
#pragma unroll 1
  for(int strip=0;strip<8;++strip) {
    const int q=(tid>>2)+64*strip;
    const int row=int(blockIdx.x)*512+q;
    float v=0.0f;
    if(!stopped) v=dot<32>(hidden,w,s,row,l4,gs);
    if(l4==0) scores[q]=key(v,uint32_t(row%CHUNK));
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
  const int tid=threadIdx.x,l4=tid&3;
  const bool stopped=gate!=nullptr && *gate!=0;
#pragma unroll
  for(int j=0;j<4;++j) hidden[tid+256*j]=h[cols[tid+256*j]];
  __syncthreads();
#pragma unroll 1
  for(int strip=0;strip<4;++strip) {
    const int q=(tid>>2)+64*strip;
    const int row=int(candidates[int64_t(blockIdx.x)*KEEP+q]);
    float v=0.0f;
    if(!stopped) v=dot<64>(hidden,w,s,row,l4,gs);
    if(l4==0) scores[q]=key(v,uint32_t(row%CHUNK));
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
