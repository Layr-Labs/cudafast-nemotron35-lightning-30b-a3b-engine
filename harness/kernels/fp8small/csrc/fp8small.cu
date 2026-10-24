// SPDX-License-Identifier: Apache-2.0
// CUTLASS-exact FP8 m16n8k32 chain for 1..8 rows and the two Mamba shapes.
// The original FP8 bytes are reordered at load time for coalesced uint2 reads.
// Each output follows one ascending-K native e4m3 MMA chain, then FP32 scale B,
// FP32 scale A, and RN BF16, matching the pinned SM120 ScaledEpilogue.
// No K split, reduction, activation requantization or weight format change.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8small {
struct P {const uint8_t *a,*b;const float *sa,*sb;__nv_bfloat16 *c;int m;};
template<int N,int K,int W>
__device__ __forceinline__ void load(const P&p,int warp,int lane,int step,uint32_t(&a)[4],uint32_t(&b)[2]) {
 int row=lane>>2, kk=step*32+(lane&3)*4;
 a[0]=row<p.m?__ldg((const uint32_t*)(p.a+row*K+kk)):0;
 a[1]=row+8<p.m?__ldg((const uint32_t*)(p.a+(row+8)*K+kk)):0;
 a[2]=row<p.m?__ldg((const uint32_t*)(p.a+row*K+kk+16)):0;
 a[3]=row+8<p.m?__ldg((const uint32_t*)(p.a+(row+8)*K+kk+16)):0;
 int gw=blockIdx.x*W+warp;
 const uint2* ptr=(const uint2*)p.b+(((gw/4)*(K/32)+step)*128+(gw%4)*32+lane);
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(b[0]),"=r"(b[1]) : "l"(ptr));
}
__device__ __forceinline__ void mma(const uint32_t(&a)[4],const uint32_t(&b)[2],float(&c)[4]) {
 asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
template<int N,int K,int R,int W>
__global__ __launch_bounds__(W*32,2) void gemm(P p) {
 int lane=threadIdx.x%32,warp=threadIdx.x/32;float c[4]={0,0,0,0};
 uint32_t a[R][4],b[R][2];
 #pragma unroll
 for(int j=0;j<R;++j)load<N,K,W>(p,warp,lane,j,a[j],b[j]);
 for(int base=0;base<K/32;base+=R) {
  #pragma unroll
  for(int j=0;j<R;++j){
   if(base+j<K/32)mma(a[j],b[j],c);
   if(base+j+R<K/32)load<N,K,W>(p,warp,lane,base+j+R,a[j],b[j]);
  }
 }
 float sa=__ldg(p.sa),sb=__ldg(p.sb);int row=lane>>2,col=(blockIdx.x*W+warp)*8+(lane&3)*2;
 if(row<p.m){p.c[row*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[0],sb),sa));p.c[row*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[1],sb),sa));}
 if(row+8<p.m){p.c[(row+8)*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[2],sb),sa));p.c[(row+8)*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[3],sb),sa));}
}
}
extern "C" int fp8small_init() {
 cudaFuncAttributes attr;
 cudaError_t rc=cudaFuncGetAttributes(&attr,fp8small::gemm<10304,2688,16,4>);
 if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8small::gemm<2688,4096,16,4>);
}
extern "C" int fp8small_run(const void*a,const void*b,const void*sa,const void*sb,void*c,int m,int n,int k,void*stream) {
 using namespace fp8small;if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)b,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m};
 if(n==10304&&k==2688)gemm<10304,2688,16,4><<<10304/32,128,0,(cudaStream_t)stream>>>(p);
 else if(n==2688&&k==4096)gemm<2688,4096,16,4><<<2688/32,128,0,(cudaStream_t)stream>>>(p);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}

// Native FP8 operand-transpose experiment: W[N,K] * X[K,8] rather than
// X[16,K] * W[K,8]. One warp covers 16 output features and up to 8 tokens.
// The EXISTING packed bytes are read directly, without another weight pack.
// Both paths visit ascending K32 chunks and retain scale-B then scale-A RN.
// Transposing MMA operands is not claimed universally bit-identical; the
// remote loader compares deterministic probes before enabling each pack.
namespace fp8small {
template<int N,int K,int W>
__device__ __forceinline__ void load_weight_a(
 const P&p,int warp,int lane,int step,uint32_t(&a)[4],uint32_t(&b)[2]) {
 int n=(blockIdx.x*W+warp)*16+(lane>>2);
 // Existing pack: groups of32 features, four8-feature warp slices, each
 // lane stores the uint2 for K offsets lane%4*4 and lane%4*4+16.
 int idx=(((n/32)*(K/32)+step)*128)+(n%32/8)*32+(n%8)*4+(lane&3);
 const uint2* ptr=(const uint2*)p.b+idx;
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n"
  :"=r"(a[0]),"=r"(a[2]):"l"(ptr));
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n"
  :"=r"(a[1]),"=r"(a[3]):"l"(ptr+32));
 int token=lane>>2,kk=step*32+(lane&3)*4;
 b[0]=token<p.m?__ldg((const uint32_t*)(p.a+token*K+kk)):0;
 b[1]=token<p.m?__ldg((const uint32_t*)(p.a+token*K+kk+16)):0;
}
template<int N,int K,int R,int W>
__global__ __launch_bounds__(W*32,2) void gemm_weight_a(P p) {
 int lane=threadIdx.x%32,warp=threadIdx.x/32;float c[4]={0,0,0,0};
 uint32_t a[R][4],b[R][2];
 #pragma unroll
 for(int j=0;j<R;++j)load_weight_a<N,K,W>(p,warp,lane,j,a[j],b[j]);
 for(int base=0;base<K/32;base+=R) {
  #pragma unroll
  for(int j=0;j<R;++j){
   if(base+j<K/32)mma(a[j],b[j],c);
   if(base+j+R<K/32)load_weight_a<N,K,W>(p,warp,lane,base+j+R,a[j],b[j]);
  }
 }
 float sa=__ldg(p.sa),sb=__ldg(p.sb);
 int n=(blockIdx.x*W+warp)*16+(lane>>2),token=(lane&3)*2;
 if(token<p.m){
  p.c[token*N+n]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[0],sb),sa));
  p.c[token*N+n+8]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[2],sb),sa));
 }
 if(token+1<p.m){
  p.c[(token+1)*N+n]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[1],sb),sa));
  p.c[(token+1)*N+n+8]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(c[3],sb),sa));
 }
}
}
extern "C" int fp8small_weight_a_init() {
 cudaFuncAttributes attr;
 cudaError_t rc=cudaFuncGetAttributes(&attr,fp8small::gemm_weight_a<10304,2688,16,4>);
 if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8small::gemm_weight_a<2688,4096,16,2>);
}
extern "C" int fp8small_run_weight_a(const void*a,const void*b,const void*sa,const void*sb,void*c,int m,int n,int k,void*stream) {
 using namespace fp8small;if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)b,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m};
 if(n==10304&&k==2688)gemm_weight_a<10304,2688,16,4><<<10304/64,128,0,(cudaStream_t)stream>>>(p);
 else if(n==2688&&k==4096)gemm_weight_a<2688,4096,16,2><<<2688/32,64,0,(cudaStream_t)stream>>>(p);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
