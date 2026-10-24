// SPDX-License-Identifier: Apache-2.0
// Native BF16 MTP hidden-half projection plus the existing FP32 table add.
// W is the original contiguous BF16 [2688,2688], X is already hnorm/BF16.
// Changed draft FP32 reduction tree; target verification is unchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace mtpstart {
constexpr int H=2688,WARPS=4,RING=8;
struct P {const __nv_bfloat16 *x,*w;const float *e;__nv_bfloat16 *y;int m;};
__device__ __forceinline__ void load(const P&p,int warp,int lane,int step,uint32_t(&a)[4],uint32_t(&b)[2]) {
 int n=(blockIdx.x*WARPS+warp)*16+(lane>>2),k=step*16+(lane&3)*2;
 // Ordinary read-only weight loads preserve the highest carrier's keep policy.
 a[0]=__ldg((const uint32_t*)(p.w+n*H+k));
 a[1]=__ldg((const uint32_t*)(p.w+(n+8)*H+k));
 a[2]=__ldg((const uint32_t*)(p.w+n*H+k+8));
 a[3]=__ldg((const uint32_t*)(p.w+(n+8)*H+k+8));
 int token=lane>>2;
 b[0]=token<p.m?__ldg((const uint32_t*)(p.x+token*H+k)):0;
 b[1]=token<p.m?__ldg((const uint32_t*)(p.x+token*H+k+8)):0;
}
__device__ __forceinline__ void mma(const uint32_t(&a)[4],const uint32_t(&b)[2],float(&c)[4]) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
  :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void store(const P&p,int token,int n,float acc) {
 // Keep the original intermediate BF16 boundary BEFORE the FP32 table add.
 float hp=__bfloat162float(__float2bfloat16_rn(acc));
 p.y[token*H+n]=__float2bfloat16_rn(__fadd_rn(hp,__ldg(p.e+token*H+n)));
}
__global__ __launch_bounds__(128,2) void project(P p) {
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 float c[4]={0,0,0,0};uint32_t a[RING][4],b[RING][2];
 #pragma unroll
 for(int j=0;j<RING;++j)load(p,warp,lane,j,a[j],b[j]);
 for(int base=0;base<H/16;base+=RING) {
  #pragma unroll
  for(int j=0;j<RING;++j){
   mma(a[j],b[j],c);
   if(base+j+RING<H/16)load(p,warp,lane,base+j+RING,a[j],b[j]);
  }
 }
 int n=(blockIdx.x*WARPS+warp)*16+(lane>>2),token=(lane&3)*2;
 if(token<p.m){store(p,token,n,c[0]);store(p,token,n+8,c[2]);}
 if(token+1<p.m){store(p,token+1,n,c[1]);store(p,token+1,n+8,c[3]);}
}
}
extern "C" int mtpstart_init() {
 cudaFuncAttributes attr;return (int)cudaFuncGetAttributes(&attr,mtpstart::project);
}
extern "C" int mtpstart_run(const void*x,const void*w,const void*e,void*y,int m,void*stream) {
 if(m<1||m>8)return (int)cudaErrorInvalidValue;
 mtpstart::P p{(const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(const float*)e,(__nv_bfloat16*)y,m};
 mtpstart::project<<<2688/64,128,0,(cudaStream_t)stream>>>(p);
 return (int)cudaGetLastError();
}
