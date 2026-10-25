#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8d6 {
constexpr unsigned FULL = 0xffffffffu;
constexpr int MAIN = 3072;
template<int K> struct Shape {
 static constexpr int C = K / 128;
 static constexpr int HDR = (((4 * C + 1) * 2 + 2048) + 127) / 128 * 128;
 static constexpr int EXC = HDR + C * MAIN;
};
struct P {const uint8_t *a,*w;const float *sa,*sb;__nv_bfloat16 *c;int m;int tile_bytes;int tile0;};
struct Slot {uint32_t c[6];uint32_t x[2];uint32_t q0;};

__device__ __forceinline__ uint4 ld4(const uint8_t* p) {
 uint4 v;
 asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
  : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
 return v;
}
__device__ __forceinline__ uint2 ld2(const uint8_t* p) {
 uint2 v;
 asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x),"=r"(v.y) : "l"(p));
 return v;
}
__device__ __forceinline__ uint32_t ld1(const uint8_t* p) {
 uint32_t v;
 asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
 return v;
}
__device__ __forceinline__ uint32_t bfi8(uint32_t b, uint32_t w, uint32_t pos) {
 uint32_t d;
 asm("bfi.b32 %0, %1, %2, %3, 8;\n" : "=r"(d) : "r"(b), "r"(w), "r"(pos));
 return d;
}
template<int K>
__device__ __forceinline__ void load_main(const uint8_t* tb,int warp,int lane,int c,Slot& s) {
 const uint8_t* mp=tb+Shape<K>::HDR+c*MAIN;
 uint4 A=ld4(mp+warp*512+lane*16);
 uint2 B=ld2(mp+2048+warp*256+lane*8);
 s.c[0]=A.x;s.c[1]=A.y;s.c[2]=A.z;s.c[3]=A.w;s.c[4]=B.x;s.c[5]=B.y;
}
template<int K>
__device__ __forceinline__ void load_x(const uint8_t* tb,int lane,int c,uint32_t ts,uint32_t te,Slot& s) {
 uint32_t off=__shfl_sync(FULL,ts,c),end=__shfl_sync(FULL,te,c);
 uint32_t q0=off&3u;
 s.q0=q0;
 uint32_t length=q0+end-off;
 s.x[0]=(4u*lane<length)?ld1(tb+Shape<K>::EXC+(off-q0)+4*lane):0u;
 s.x[1]=(128u+4u*lane<length)?ld1(tb+Shape<K>::EXC+(off-q0)+128+4*lane):0u;
}
__device__ __forceinline__ uint32_t lanemask_lt() {
 uint32_t m;
 asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
 return m;
}
__device__ __forceinline__ void decode(const Slot& s,const uint8_t* lut,int warp,int lane,uint32_t (&w)[8]) {
 uint32_t cw[8],em[8],pc[8],cnt=0;
 #pragma unroll
 for(int i=0;i<6;++i)cw[i]=s.c[i]&0x3f3f3f3fu;
 cw[6]=((s.c[0]>>6)&0x03030303u)|((s.c[1]>>4)&0x0c0c0c0cu)|((s.c[2]>>2)&0x30303030u);
 cw[7]=((s.c[3]>>6)&0x03030303u)|((s.c[4]>>4)&0x0c0c0c0cu)|((s.c[5]>>2)&0x30303030u);
 const uint8_t* table=lut+(warp*8+(lane>>2))*64;
 #pragma unroll
 for(int i=0;i<8;++i){
  uint32_t x=(cw[i]^0x3f3f3f3fu)&0x3f3f3f3fu;
  em[i]=~(x+0x7f7f7f7fu)&0x80808080u;
  pc[i]=__popc(em[i]);cnt+=pc[i];
  uint32_t word=0;
  #pragma unroll
  for(int j=0;j<4;++j)word|=(uint32_t)table[(cw[i]>>(j*8))&63u]<<(j*8);
  w[i]=word;
 }
 uint32_t lt=lanemask_lt(),pre=0;
 #pragma unroll
 for(int b=0;b<6;++b)pre+=(uint32_t)__popc(__ballot_sync(FULL,(cnt>>b)&1u)&lt)<<b;
 uint32_t q=s.q0+pre,em2[8],three=0;
 #pragma unroll
 for(int i=0;i<8;++i){em2[i]=em[i]&(em[i]-1u);three|=em2[i]&(em2[i]-1u);}
 bool slow=cnt>7u||(q&3u)+cnt>8u||three!=0u;
 if(!__any_sync(FULL,slow)){
  uint32_t xa=__shfl_sync(FULL,s.x[0],(q>>2)&31u),xb=__shfl_sync(FULL,s.x[1],(q>>2)&31u);
  uint32_t qa=((q>>2)+1u)*4u;
  uint32_t ya=__shfl_sync(FULL,s.x[0],(qa>>2)&31u),yb=__shfl_sync(FULL,s.x[1],(qa>>2)&31u);
  uint32_t v0=q<128u?xa:xb,v1=qa<128u?ya:yb,sel=(q&3u)*0x1111u;
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t first=((em[i]^em2[i])>>7)*0xffu,second=(em2[i]>>7)*0xffu;
   w[i]=(w[i]&~first)|(__byte_perm(v0,v1,sel)&first);
   w[i]=(w[i]&~second)|(__byte_perm(v0,v1,sel+0x1111u)&second);
   sel+=pc[i]*0x1111u;
  }
 }else{
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t mk=em[i];
   while(__any_sync(FULL,mk)){
    uint32_t va=__shfl_sync(FULL,s.x[0],(q>>2)&31u);
    uint32_t vb=__shfl_sync(FULL,s.x[1],(q>>2)&31u);
    uint32_t v=q<128u?va:vb;
    if(mk){
     w[i]=bfi8(__byte_perm(v,0u,q&3u),w[i],(uint32_t)__ffs(mk)-8u);
     mk&=mk-1u;++q;
    }
   }
  }
 }
}

template<bool NC>
__device__ __forceinline__ uint32_t lda(const uint8_t* q) {
 if constexpr (NC) return __ldg((const uint32_t*)q);
 else {
  uint32_t value;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(q) : "memory");
  return value;
 }
}
template<int K,bool NC>
__device__ __forceinline__ void load_a(const P&p,int lane,int c,uint32_t (&a)[4][2]) {
 int row=lane>>2;
 #pragma unroll
 for(int s=0;s<4;++s){
  int kk=(c*4+s)*32+(lane&3)*4;
  a[s][0]=row<p.m?lda<NC>(p.a+row*K+kk):0u;
  a[s][1]=row<p.m?lda<NC>(p.a+row*K+kk+16):0u;
 }
}
#ifndef FP8D6_NO_PDL
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
__device__ __forceinline__ void mma(uint32_t a0,uint32_t a2,uint32_t b0,uint32_t b1,float(&c)[4]) {
 asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
 :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
 :"r"(a0),"r"(0u),"r"(a2),"r"(0u),"r"(b0),"r"(b1),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
template<int N,int K,int RC,bool PDL=false>
__global__ __launch_bounds__(128,2) void gemm(P p) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 int tile=blockIdx.x+p.tile0;
 const uint8_t* tb=p.w+(size_t)tile*p.tile_bytes;
 const uint16_t* tab=(const uint16_t*)tb;
 uint32_t ts=0,te=0;
 if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
 __shared__ __align__(16) uint8_t lut[2048];
 uint4 dictionary=ld4(tb+Shape<K>::HDR-2048+threadIdx.x*16);
 ((uint4*)lut)[threadIdx.x]=dictionary;
 __syncthreads();
 Slot s[RC];uint32_t a[RC][4][2];float acc[4]={0.f,0.f,0.f,0.f};
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_main<K>(tb,warp,lane,j,s[j]);
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_x<K>(tb,lane,j,ts,te,s[j]);
#ifndef FP8D6_NO_PDL
 if constexpr (PDL) pdl_wait();
#endif
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_a<K,!PDL>(p,lane,j,a[j]);
 uint32_t wc[8];
 decode(s[0],lut,warp,lane,wc);
 if(RC<C){load_main<K>(tb,warp,lane,RC,s[0]);load_x<K>(tb,lane,RC,ts,te,s[0]);}
 for(int cb=0;cb<C;cb+=RC){
  #pragma unroll
  for(int j=0;j<RC;++j){
   int c=cb+j;
   if(c<C){
    #pragma unroll
    for(int st=0;st<4;++st)mma(a[j][st][0],a[j][st][1],wc[2*st],wc[2*st+1],acc);
    if(c+RC<C)load_a<K,!PDL>(p,lane,c+RC,a[j]);
    if(c+1<C){
     uint32_t wn[8];
     decode(s[(j+1)%RC],lut,warp,lane,wn);
     if(c+1+RC<C){
      load_main<K>(tb,warp,lane,c+1+RC,s[(j+1)%RC]);
      load_x<K>(tb,lane,c+1+RC,ts,te,s[(j+1)%RC]);
     }
     #pragma unroll
     for(int i=0;i<8;++i)wc[i]=wn[i];
    }
   }
  }
 }
#ifndef FP8D6_NO_PDL
 if constexpr (PDL) pdl_launch_dependents();
#endif
 float sa=PDL?__uint_as_float(lda<false>((const uint8_t*)p.sa)):__ldg(p.sa),sb=PDL?__uint_as_float(lda<false>((const uint8_t*)p.sb)):__ldg(p.sb);int row=lane>>2,col=(tile*4+warp)*8+(lane&3)*2;
 if(row<p.m){p.c[row*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[0],sb),sa));p.c[row*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[1],sb),sa));}
}
template<int K>
__global__ __launch_bounds__(128) void decode_kernel(const uint8_t* buf,int tile_bytes,uint32_t* out) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=buf+(size_t)blockIdx.x*tile_bytes;
 const uint16_t* tab=(const uint16_t*)tb;
 uint32_t ts=0,te=0;
 if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
 __shared__ __align__(16) uint8_t lut[2048];
 uint4 dictionary=ld4(tb+Shape<K>::HDR-2048+threadIdx.x*16);
 ((uint4*)lut)[threadIdx.x]=dictionary;
 __syncthreads();
 for(int c=0;c<C;++c){
  Slot s;uint32_t w[8];
  load_main<K>(tb,warp,lane,c,s);
  load_x<K>(tb,lane,c,ts,te,s);
  decode(s,lut,warp,lane,w);
  #pragma unroll
  for(int st=0;st<4;++st){
   size_t i=((((size_t)blockIdx.x*(K/32)+c*4+st)*4+warp)*32+lane)*2;
   out[i]=w[2*st];out[i+1]=w[2*st+1];
  }
 }
}
template<int N,int K,int RC,bool PDL>
cudaError_t launch(const P&p,cudaStream_t stream,int tiles) {
#ifndef FP8D6_NO_PDL
 if constexpr (PDL) {
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(tiles);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=0;cfg.stream=stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attr;cfg.numAttrs=1;
  return cudaLaunchKernelEx(&cfg,gemm<N,K,RC,true>,p);
 }
#endif
 gemm<N,K,RC,false><<<tiles,128,0,stream>>>(p);
 return cudaGetLastError();
}
template<int RC,bool PDL>
cudaError_t dispatch(const P&p,int n,int k,cudaStream_t stream) {
 if(n==10304&&k==2688)return launch<10304,2688,RC,PDL>(p,stream,n/32);
 if(n==2688&&k==4096)return launch<2688,4096,RC,PDL>(p,stream,n/32);
 return cudaErrorInvalidValue;
}
template<int RC,bool PDL>
cudaError_t dispatch_tiles(const P&p,int n,int k,int tiles,cudaStream_t stream) {
 if(n==10304&&k==2688)return launch<10304,2688,RC,PDL>(p,stream,tiles);
 return cudaErrorInvalidValue;
}
}

extern "C" int fp8d6_run(const void*a,const void*w,const void*sa,const void*sb,void*c,int m,int n,int k,
                        int tile_bytes,int pdl,int ring,void*stream) {
 using namespace fp8d6;
 if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)w,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m,tile_bytes,0};
 cudaStream_t st=(cudaStream_t)stream;
 cudaError_t rc;
#ifdef FP8D6_NO_PDL
 if(pdl)return (int)cudaErrorNotSupported;
#endif
 switch(ring){
  case 3: rc=pdl?dispatch<3,true>(p,n,k,st):dispatch<3,false>(p,n,k,st);break;
  case 4: rc=pdl?dispatch<4,true>(p,n,k,st):dispatch<4,false>(p,n,k,st);break;
  case 6: rc=pdl?dispatch<6,true>(p,n,k,st):dispatch<6,false>(p,n,k,st);break;
  default: return (int)cudaErrorInvalidValue;
 }
 if(rc!=cudaSuccess){cudaGetLastError();return (int)rc;}
 return (int)cudaGetLastError();
}
extern "C" int fp8d6_run_tiles(const void*a,const void*w,const void*sa,const void*sb,void*c,int m,int n,int k,
                        int tile_bytes,int pdl,int ring,void*stream,int tile0,int ntiles) {
 using namespace fp8d6;
 if(m<1||m>8||ntiles<1||tile0<0||tile0+ntiles>n/32)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)w,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m,tile_bytes,tile0};
 cudaStream_t st=(cudaStream_t)stream;
 cudaError_t rc;
#ifdef FP8D6_NO_PDL
 if(pdl)return (int)cudaErrorNotSupported;
#endif
 switch(ring){
  case 3: rc=pdl?dispatch_tiles<3,true>(p,n,k,ntiles,st):dispatch_tiles<3,false>(p,n,k,ntiles,st);break;
  case 4: rc=pdl?dispatch_tiles<4,true>(p,n,k,ntiles,st):dispatch_tiles<4,false>(p,n,k,ntiles,st);break;
  case 6: rc=pdl?dispatch_tiles<6,true>(p,n,k,ntiles,st):dispatch_tiles<6,false>(p,n,k,ntiles,st);break;
  default: return (int)cudaErrorInvalidValue;
 }
 if(rc!=cudaSuccess){cudaGetLastError();return (int)rc;}
 return (int)cudaGetLastError();
}
extern "C" int fp8d6_decode(const void*buf,int tile_bytes,void*out,int n,int k,void*stream) {
 using namespace fp8d6;
 cudaStream_t st=(cudaStream_t)stream;
 if(n==10304&&k==2688)decode_kernel<2688><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else if(n==2688&&k==4096)decode_kernel<4096><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
extern "C" int fp8d6_header_bytes(int k) {
 if(k==2688)return fp8d6::Shape<2688>::HDR;
 if(k==4096)return fp8d6::Shape<4096>::HDR;
 return -1;
}
extern "C" int fp8d6_init() {
 cudaFuncAttributes attr;cudaError_t rc;
 rc=cudaFuncGetAttributes(&attr,fp8d6::gemm<10304,2688,3,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8d6::gemm<2688,4096,4,false>);if(rc!=cudaSuccess)return (int)rc;
#ifndef FP8D6_NO_PDL
 rc=cudaFuncGetAttributes(&attr,fp8d6::gemm<10304,2688,3,true>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8d6::gemm<2688,4096,4,true>);if(rc!=cudaSuccess)return (int)rc;
#endif
 rc=cudaFuncGetAttributes(&attr,fp8d6::decode_kernel<2688>);if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8d6::decode_kernel<4096>);
}
extern "C" int fp8d6_has_pdl() {
#ifdef FP8D6_NO_PDL
 return 0;
#else
 return 1;
#endif
}
