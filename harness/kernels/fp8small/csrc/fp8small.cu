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
// One activation word: the read-only (nc) path, or, for a grid that may start
// before its producer has finished (PDL), an ordinary global load issued after
// griddepcontrol.wait (whose "memory" clobber keeps it there). Same value.
template<bool NC>
__device__ __forceinline__ uint32_t lda(const uint8_t* q) {
 if constexpr (NC) return __ldg((const uint32_t*)q);
 else return *(const uint32_t*)q;
}
// Activation half of one ring step (written by the preceding kernel).
template<int N,int K,int W,bool NC=true>
__device__ __forceinline__ void load_a(const P&p,int lane,int step,uint32_t(&a)[4]) {
 int row=lane>>2, kk=step*32+(lane&3)*4;
 a[0]=row<p.m?lda<NC>(p.a+row*K+kk):0;
 a[1]=row+8<p.m?lda<NC>(p.a+(row+8)*K+kk):0;
 a[2]=row<p.m?lda<NC>(p.a+row*K+kk+16):0;
 a[3]=row+8<p.m?lda<NC>(p.a+(row+8)*K+kk+16):0;
}
// Weight half of one ring step (the load-time pack: never written at run time).
template<int N,int K,int W>
__device__ __forceinline__ void load_b(const P&p,int warp,int lane,int step,uint32_t(&b)[2]) {
 int gw=blockIdx.x*W+warp;
 const uint2* ptr=(const uint2*)p.b+(((gw/4)*(K/32)+step)*128+(gw%4)*32+lane);
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(b[0]),"=r"(b[1]) : "l"(ptr));
}
template<int N,int K,int W,bool NC=true>
__device__ __forceinline__ void load(const P&p,int warp,int lane,int step,uint32_t(&a)[4],uint32_t(&b)[2]) {
 load_a<N,K,W,NC>(p,lane,step,a);
 load_b<N,K,W>(p,warp,lane,step,b);
}
#ifndef FP8SMALL_NO_PDL
// Programmatic dependent launch (sm_90+): wait for the preceding grid's
// completion and memory, and allow the next PDL-launched grid to start.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
 asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#endif
__device__ __forceinline__ void mma(const uint32_t(&a)[4],const uint32_t(&b)[2],float(&c)[4]) {
 asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
// PDL=true (launched with programmatic stream serialization): the weight half
// of the ring prologue is issued before griddepcontrol.wait, every read of the
// activations (the preceding kernel's output) and every store come after it.
// The loaded values, the MMA order and the epilogue are those of PDL=false.
template<int N,int K,int R,int W,bool PDL=false>
__global__ __launch_bounds__(W*32,2) void gemm(P p) {
 int lane=threadIdx.x%32,warp=threadIdx.x/32;float c[4]={0,0,0,0};
 uint32_t a[R][4],b[R][2];
#ifndef FP8SMALL_NO_PDL
 if constexpr (PDL) {
  #pragma unroll
  for(int j=0;j<R;++j)load_b<N,K,W>(p,warp,lane,j,b[j]);
  pdl_wait();
  #pragma unroll
  for(int j=0;j<R;++j)load_a<N,K,W,false>(p,lane,j,a[j]);
 } else
#endif
 {
  #pragma unroll
  for(int j=0;j<R;++j)load<N,K,W>(p,warp,lane,j,a[j],b[j]);
 }
 for(int base=0;base<K/32;base+=R) {
  #pragma unroll
  for(int j=0;j<R;++j){
   if(base+j<K/32)mma(a[j],b[j],c);
   if(base+j+R<K/32)load<N,K,W,!PDL>(p,warp,lane,base+j+R,a[j],b[j]);
  }
 }
#ifndef FP8SMALL_NO_PDL
 if constexpr (PDL) pdl_launch_dependents();
#endif
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
#ifndef FP8SMALL_NO_PDL
namespace fp8small {
template<int N,int K>
cudaError_t launch_pdl(const P&p,cudaStream_t stream) {
 cudaLaunchConfig_t cfg={};
 cfg.gridDim=dim3(N/32);
 cfg.blockDim=dim3(128);
 cfg.dynamicSmemBytes=0;
 cfg.stream=stream;
 cudaLaunchAttribute attr[1];
 attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
 attr[0].val.programmaticStreamSerializationAllowed=1;
 cfg.attrs=attr;
 cfg.numAttrs=1;
 return cudaLaunchKernelEx(&cfg,gemm<N,K,16,4,true>,p);
}
}
extern "C" int fp8small_init_pdl() {
 cudaFuncAttributes attr;
 cudaError_t rc=cudaFuncGetAttributes(&attr,fp8small::gemm<10304,2688,16,4,true>);
 if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8small::gemm<2688,4096,16,4,true>);
}
extern "C" int fp8small_run_pdl(const void*a,const void*b,const void*sa,const void*sb,void*c,int m,int n,int k,void*stream) {
 using namespace fp8small;if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)b,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m};
 cudaError_t rc;
 if(n==10304&&k==2688)rc=launch_pdl<10304,2688>(p,(cudaStream_t)stream);
 else if(n==2688&&k==4096)rc=launch_pdl<2688,4096>(p,(cudaStream_t)stream);
 else return (int)cudaErrorInvalidValue;
 if(rc!=cudaSuccess){cudaGetLastError();return (int)rc;}  // cleared, so the caller can retry without PDL
 return (int)cudaGetLastError();
}
#endif
extern "C" int fp8small_run(const void*a,const void*b,const void*sa,const void*sb,void*c,int m,int n,int k,void*stream) {
 using namespace fp8small;if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)b,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m};
 if(n==10304&&k==2688)gemm<10304,2688,16,4><<<10304/32,128,0,(cudaStream_t)stream>>>(p);
 else if(n==2688&&k==4096)gemm<2688,4096,16,4><<<2688/32,128,0,(cudaStream_t)stream>>>(p);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
