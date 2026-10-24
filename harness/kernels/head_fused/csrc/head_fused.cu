// SPDX-License-Identifier: Apache-2.0
// Current hidden vector -> original NVFP4 partial scores -> ranked candidates.
// Draft fullK only. Original062 partial screens/candidate sets remain unchanged.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace head_fused {
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
template<int GROUPS,int COUNT=GROUPS>
__device__ __forceinline__ float2 dot_mma(const float* h,const uint8_t* w,
                     const uint8_t* scales,int row0,int row1,int t,float gs,
                     int gstart=0,int hbase=0) {
  float c[4]={0.0f,0.0f,0.0f,0.0f};
#pragma unroll 1
  for(int g=gstart;g<gstart+COUNT;++g) {
    const float s0=fp8(scales[int64_t(row0)*GROUPS+g]);
    const float s1=fp8(scales[int64_t(row1)*GROUPS+g]);
    const int b=8*g+t;
    const uint32_t a0=dequant_pair(w[int64_t(row0)*(8*GROUPS)+b],s0);
    const uint32_t a1=dequant_pair(w[int64_t(row1)*(8*GROUPS)+b],s1);
    const uint32_t a2=dequant_pair(w[int64_t(row0)*(8*GROUPS)+b+4],s0);
    const uint32_t a3=dequant_pair(w[int64_t(row1)*(8*GROUPS)+b+4],s1);
    const int k=16*g+2*t-hbase;
    const uint32_t b0=bf16pair(h[k],h[k+1]);
    const uint32_t b1=bf16pair(h[k+8],h[k+9]);
    mma(c,a0,a1,a2,a3,b0,b1);
  }
  return make_float2(__fmul_rn(c[0],gs),__fmul_rn(c[2],gs));
}

// Full-K draft-only rescore: 128 sixteen-candidate tiles x four K slices.
// Current BF16 hidden is widened to F32 at the caller, as for partial MMA.
// Each slice has42 whole NVFP4 groups; no K terms or candidates are removed.
__global__ void full_partial(const float* h,const uint8_t* w,const uint8_t* s,
                            const int64_t* candidates,float* partial,const int* gate) {
  __shared__ float hidden[42*16];
  const int lane=threadIdx.x,t=lane&3,g8=lane>>2;
  const int hbase=int(blockIdx.y)*42*16;
#pragma unroll
  for(int j=lane;j<42*16;j+=32) hidden[j]=h[hbase+j];
  __syncthreads();
  const int offset=int(blockIdx.x)*16;
  const int row0=int(candidates[offset+g8]);
  const int row1=int(candidates[offset+g8+8]);
  float2 v=make_float2(0.0f,0.0f);
  if(gate==nullptr || *gate==0)
    v=dot_mma<168,42>(hidden,w,s,row0,row1,t,1.0f,int(blockIdx.y)*42,hbase);
  if(t==0) {
    partial[int(blockIdx.y)*2048+offset+g8]=v.x;
    partial[int(blockIdx.y)*2048+offset+g8+8]=v.y;
  }
}
__device__ __forceinline__ void choose(float x,int id,float& best,int& best_id) {
  if(x>best || (x==best && id<best_id)) { best=x;best_id=id; }
}
// Merge the four F32 slice sums, apply the original global scale and BF16
// rounding, then original max/min-global-ID plus any-NaN sentinel in one CTA.
// This changes association, so finite admission and remote oracle remain vital.
__global__ void full_select(const float* partial,const int64_t* candidates,
                           float gs,int32_t* out,int* gate,float theta) {
  __shared__ float warp_score[8];
  __shared__ float logits[2048],block_best,warp_sum[8];
  __shared__ int warp_id[8],warp_nan[8];
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  float best=-CUDART_INF_F;int best_id=1<<30,any_nan=0;
#pragma unroll
  for(int i=tid;i<2048;i+=256) {
    float x=__fadd_rn(partial[i],partial[2048+i]);
    x=__fadd_rn(x,partial[4096+i]);
    x=__fadd_rn(x,partial[6144+i]);
    x=__bfloat162float(__float2bfloat16_rn(__fmul_rn(x,gs)));
    if(gate!=nullptr) logits[i]=x;
    any_nan|=(x!=x);
    choose(x,int(candidates[i]),best,best_id);
  }
#pragma unroll
  for(int d=16;d;d>>=1) {
    const float x=__shfl_down_sync(0xffffffffu,best,d);
    const int id=__shfl_down_sync(0xffffffffu,best_id,d);
    const int nan=__shfl_down_sync(0xffffffffu,any_nan,d);
    if(lane+d<32) { choose(x,id,best,best_id);any_nan|=nan; }
  }
  if(lane==0) { warp_score[warp]=best;warp_id[warp]=best_id;warp_nan[warp]=any_nan; }
  __syncthreads();
  if(warp==0) {
    best=lane<8?warp_score[lane]:-CUDART_INF_F;
    best_id=lane<8?warp_id[lane]:(1<<30);
    any_nan=lane<8?warp_nan[lane]:0;
#pragma unroll
    for(int d=16;d;d>>=1) {
      const float x=__shfl_down_sync(0xffffffffu,best,d);
      const int id=__shfl_down_sync(0xffffffffu,best_id,d);
      const int nan=__shfl_down_sync(0xffffffffu,any_nan,d);
      if(lane+d<32) { choose(x,id,best,best_id);any_nan|=nan; }
    }
    if(lane==0) { *out=any_nan?(1<<30):best_id;block_best=best; }
  }
  __syncthreads();
  // Preserve062's active/stop/kept protocol. All threads see one gate state
  // from earlier grids on this stream; this kernel alone updates it below.
  // FP32 exponential/sum association can change the confidence near theta.
  if(gate!=nullptr && gate[2]!=0 && gate[0]==0) {
    float sum=0.0f;
#pragma unroll
    for(int i=tid;i<2048;i+=256) sum=__fadd_rn(sum,__expf(logits[i]-block_best));
#pragma unroll
    for(int d=16;d;d>>=1) {
      const float v=__shfl_down_sync(0xffffffffu,sum,d);
      if(lane+d<32) sum=__fadd_rn(sum,v);
    }
    if(lane==0) warp_sum[warp]=sum;
    __syncthreads();
    if(warp==0) {
      sum=lane<8?warp_sum[lane]:0.0f;
#pragma unroll
      for(int d=16;d;d>>=1) {
        const float v=__shfl_down_sync(0xffffffffu,sum,d);
        if(lane+d<32) sum=__fadd_rn(sum,v);
      }
      if(lane==0) {
        gate[1]=gate[1]+1;
        if(1.0f/sum<theta) gate[0]=1;
      }
    }
  }
}
}
extern "C" int head_fused_init() {
  cudaFuncAttributes a;
  cudaError_t e=cudaFuncGetAttributes(&a,head_fused::full_partial);
  if(e!=cudaSuccess) return int(e);
  return int(cudaFuncGetAttributes(&a,head_fused::full_select));
}
extern "C" int head_fused_finish(const float* h,const uint8_t* w,const uint8_t* s,
 const int64_t* candidates,float gs,float* partial,int32_t* out,int* gate,float theta,void* stream) {
  cudaStream_t st=static_cast<cudaStream_t>(stream);
  head_fused::full_partial<<<dim3(128,4),32,0,st>>>(h,w,s,candidates,partial,gate);
  cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
  head_fused::full_select<<<1,256,0,st>>>(partial,candidates,gs,out,gate,theta);
  return int(cudaGetLastError());
}
