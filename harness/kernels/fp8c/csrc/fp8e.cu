// SPDX-License-Identifier: Apache-2.0
// fp8e: fp8c's exact FP8 m16n8k32 chain for 1..8 rows and the two Mamba shapes,
// reading a two-level exponent code instead of fp8c's 7-bit window code. The
// decoder rebuilds in registers the very uint32 words fp8small loads (every
// byte verified at load time by decode_kernel) and feeds the same e4m3 MMA in
// the same ascending-K chain with the same FP32 scale B, scale A, RN BF16
// epilogue. No value is approximated. Built as its own library (libfp8e.so):
// a failed build or a failed load-time proof leaves fp8c untouched.
//
// Code, per output column (one column = the 4 lanes lane>>2 of a warp):
// the column's exponents ranked by count (ties: lower exponent first).
//   P (2 bit, every weight): rank 0..2 -> 0..2, otherwise 3 ("rest").
//   S (2 bit, rest weights only, a per-warp-chunk stream): rank 3..5 -> 0..2,
//     otherwise 3 (exception).
//   X (4 bit, exceptions only, a per-warp-chunk stream): the raw exponent.
//   SM (4 bit, every weight): sign << 3 | mantissa.
// Per column an 8-byte LUT holds (exponent << 3) of ranks 0..5 (bytes 0..5;
// bytes 6..7 are 0, the placeholder an exception is patched over).
// Tile (32 columns = one CTA) layout:
//   [header: u32 xbase | u16 S offsets (chunk,warp)+total, in codes | u16 X
//    offsets (chunk,warp)+total, in nibbles | ... | 32 x 8-byte LUT at HDR-256]
//   [C chunks x 3072 B: SM (4 x u32 per lane, v4), P (2 x u32 per lane, v2)]
//   [S area: 2-bit codes, LSB first, order (chunk, warp, lane, weight)]
//   [X area at xbase: 4-bit exponents, LSB first, same order]
// Weight t (0..31) of a lane's chunk is byte t&3 of fp8small word t>>2; its SM
// nibble is nibble t of the SM plane, its P code bits 2t..2t+1 of the P plane.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8e {
constexpr unsigned FULL = 0xffffffffu;
constexpr int MAIN = 3072;  // bytes per (tile, chunk)
template<int K> struct Shape {
 static constexpr int C = K / 128;
 static constexpr int HDR = ((4 + (4 * C + 1) * 4 + 256) + 127) / 128 * 128;
 static constexpr int SB = HDR + C * MAIN;  // S area
};
struct P {const uint8_t *a,*w;const float *sa,*sb;__nv_bfloat16 *c;int m;int tile_bytes;};
struct Slot {uint32_t sm[4];uint32_t p[2];uint32_t s0,s1,x;uint32_t sq,xq;};
struct Tab {uint32_t so,xo;uint32_t sw,xw,xbase;};

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
__device__ __forceinline__ uint32_t lanemask_lt() {
 uint32_t m;
 asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
 return m;
}
// Exclusive prefix over the warp of a per-lane count < 64 (6 ballot planes).
__device__ __forceinline__ uint32_t warp_prefix(uint32_t cnt,uint32_t lt) {
 uint32_t pre=0;
 #pragma unroll
 for(int b=0;b<6;++b)pre+=(uint32_t)__popc(__ballot_sync(FULL,(cnt>>b)&1u)&lt)<<b;
 return pre;
}
// Per-warp tables: lane c (< C) holds this warp's chunk-c offsets.
template<int K>
__device__ __forceinline__ Tab load_tab(const uint8_t* tb,int warp,int lane) {
 constexpr int C=Shape<K>::C;
 const uint16_t* so=(const uint16_t*)(tb+4);
 const uint16_t* xo=so+(4*C+1);
 Tab t;
 t.so=lane<C?__ldg(so+lane*4+warp):0u;
 t.xo=lane<C?__ldg(xo+lane*4+warp):0u;
 t.sw=(2u*__ldg(so+4*C)+31u)>>5;  // S words in this tile
 t.xw=(__ldg(xo+4*C)+7u)>>3;      // X words in this tile
 t.xbase=__ldg((const uint32_t*)tb);
 return t;
}
template<int K>
__device__ __forceinline__ void load_main(const uint8_t* tb,const Tab& t,int warp,int lane,int c,Slot& s) {
 const uint8_t* mp=tb+Shape<K>::HDR+c*MAIN;
 uint4 A=ld4(mp+warp*512+lane*16);
 uint2 B=ld2(mp+2048+warp*256+lane*8);
 s.sm[0]=A.x;s.sm[1]=A.y;s.sm[2]=A.z;s.sm[3]=A.w;s.p[0]=B.x;s.p[1]=B.y;
 // S window: 64 words from the word holding this warp-chunk's first code.
 uint32_t so=__shfl_sync(FULL,t.so,c),xo=__shfl_sync(FULL,t.xo,c);
 uint32_t sb=(2u*so)>>5,xb=xo>>3;
 s.sq=(2u*so)&31u;s.xq=xo&7u;
 const uint8_t* sp=tb+Shape<K>::SB;
 s.s0=(sb+lane<t.sw)?ld1(sp+4*(sb+lane)):0u;
 s.s1=(sb+32u+lane<t.sw)?ld1(sp+4*(sb+32u+lane)):0u;
 s.x=(xb+lane<t.xw)?ld1(tb+t.xbase+4*(xb+lane)):0u;
}
__device__ __forceinline__ uint32_t win64(uint32_t a0,uint32_t a1,uint32_t w) {
 uint32_t v0=__shfl_sync(FULL,a0,w&31u),v1=__shfl_sync(FULL,a1,w&31u);
 return (w&32u)?v1:v0;
}
// Rebuild the 8 fp8small words of this lane for one chunk.
__device__ __forceinline__ void decode(const Slot& s,uint32_t lutA,uint32_t lutB,int lane,uint32_t (&w)[8]) {
 uint32_t lt=lanemask_lt();
 // Rest weights (P code 3), in weight order; this lane's S codes start after
 // the rest weights of the lanes below it.
 uint32_t r0=s.p[0]&(s.p[0]>>1)&0x55555555u,r1=s.p[1]&(s.p[1]>>1)&0x55555555u;
 uint32_t nrest=(uint32_t)(__popc(r0)+__popc(r1));
 uint32_t bit=s.sq+2u*warp_prefix(nrest,lt);
 uint32_t wi=bit>>5,sh=bit&31u;
 uint32_t sa=win64(s.s0,s.s1,wi),sb=win64(s.s0,s.s1,wi+1u),sc=win64(s.s0,s.s1,wi+2u);
 uint64_t ss=((uint64_t)__funnelshift_r(sb,sc,sh)<<32)|__funnelshift_r(sa,sb,sh);
 uint32_t xm[8],xc=0,kpre=0;
 #pragma unroll
 for(int i=0;i<8;++i){
  uint32_t pb=(s.p[i>>2]>>(8*(i&3)))&0xffu;  // 4 two-bit codes, weights 4i..4i+3
  uint32_t pn=(pb&3u)|((pb&0xcu)<<2)|((pb&0x30u)<<4)|((pb&0xc0u)<<6);  // as 4 nibbles
  uint32_t r4=(pb&(pb>>1)&0x55u);              // bit 2j set: weight 4i+j is rest
  r4=(r4&1u)|((r4>>1)&2u)|((r4>>2)&4u)|((r4>>3)&8u);
  uint32_t sv=(uint32_t)(ss>>(2u*kpre))&0xffu;  // the next 4 S codes
  uint32_t k1=r4&1u,k2=k1+((r4>>1)&1u),k3=k2+((r4>>2)&1u);
  uint32_t sn=((r4&1u)?(sv&3u):0u)
   |((r4&2u)?(((sv>>(2u*k1))&3u)<<4):0u)
   |((r4&4u)?(((sv>>(2u*k2))&3u)<<8):0u)
   |((r4&8u)?(((sv>>(2u*k3))&3u)<<12):0u);
  kpre+=k3+((r4>>3)&1u);
  uint32_t c3=pn+sn;  // 0..2 primary, 3..5 secondary, 6 exception
  uint32_t e=__byte_perm(lutA,lutB,c3);  // exponent << 3 per byte (0 for an exception)
  uint32_t v=c3^0x6666u;                  // nibbles 0..7; zero exactly at exceptions
  xm[i]=~(v+0x7777u)&0x8888u;             // bit 4j+3: weight 4i+j is an exception
  xc+=(uint32_t)__popc(xm[i]);
  uint32_t h=(s.sm[i>>1]>>(16*(i&1)))&0xffffu;
  uint32_t y=__byte_perm(h,0u,0x1100u);
  y=(y&0x000f000fu)|((y>>4)&0x0f000f00u);  // SM nibble j -> byte j
  w[i]=((y&0x08080808u)<<4)|(y&0x07070707u)|e;
 }
 // Exceptions: this lane's exponents follow those of the lanes below it.
 uint32_t q=s.xq+warp_prefix(xc,lt);
 uint32_t v0=__shfl_sync(FULL,s.x,(q>>3)&31u),v1=__shfl_sync(FULL,s.x,((q>>3)+1u)&31u);
 bool slow=(q&7u)+xc>16u;
 if(!__any_sync(FULL,slow)){
  uint64_t xs=((((uint64_t)v1)<<32)|v0)>>(4u*(q&7u));
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t mk=xm[i];
   while(mk){
    uint32_t j=((uint32_t)__ffs(mk)-1u)>>2;
    w[i]|=((uint32_t)xs&15u)<<(8u*j+3u);
    xs>>=4;mk&=mk-1u;
   }
  }
 } else {
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t mk=xm[i];
   while(__any_sync(FULL,mk)){
    uint32_t v=__shfl_sync(FULL,s.x,(q>>3)&31u);
    if(mk){
     uint32_t j=((uint32_t)__ffs(mk)-1u)>>2;
     w[i]|=((v>>(4u*(q&7u)))&15u)<<(8u*j+3u);
     mk&=mk-1u;++q;
    }
   }
  }
 }
}
template<bool NC>
__device__ __forceinline__ uint32_t lda(const uint8_t* q) {
 if constexpr (NC) return __ldg((const uint32_t*)q);
 else return *(const uint32_t*)q;
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
#ifndef FP8E_NO_PDL
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
__device__ __forceinline__ void load_lut(const uint8_t* tb,int hdr,int warp,int lane,uint32_t& lutA,uint32_t& lutB) {
 const uint32_t* lut=(const uint32_t*)(tb+hdr-256)+2*(warp*8+(lane>>2));
 lutA=__ldg(lut);lutB=__ldg(lut+1);
}
// Same schedule as fp8c::gemm: RC chunks in flight per warp, the MMAs of chunk
// c issued before the decode of chunk c+1.
template<int N,int K,int RC,bool PDL=false>
__global__ __launch_bounds__(128,2) void gemm(P p) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=p.w+(size_t)blockIdx.x*p.tile_bytes;
 Tab t=load_tab<K>(tb,warp,lane);
 uint32_t lutA,lutB;load_lut(tb,Shape<K>::HDR,warp,lane,lutA,lutB);
 Slot s[RC];uint32_t a[RC][4][2];float acc[4]={0.f,0.f,0.f,0.f};
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_main<K>(tb,t,warp,lane,j,s[j]);
#ifndef FP8E_NO_PDL
 if constexpr (PDL) pdl_wait();
#endif
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_a<K,!PDL>(p,lane,j,a[j]);
 uint32_t wc[8];
 decode(s[0],lutA,lutB,lane,wc);
 if(RC<C)load_main<K>(tb,t,warp,lane,RC,s[0]);
 for(int cb=0;cb<C;cb+=RC){
  #pragma unroll
  for(int j=0;j<RC;++j){
   int c=cb+j;
   if(c<C){
    #pragma unroll
    for(int st=0;st<4;++st)mma(a[j][st][0],a[j][st][1],wc[2*st],wc[2*st+1],acc);
    if constexpr (K == 2688) {
     if(c+RC<C)load_a<K,!PDL>(p,lane,c+RC,a[j]);
    }
    if(c+1<C){
     uint32_t wn[8];
     decode(s[(j+1)%RC],lutA,lutB,lane,wn);
     if(c+1+RC<C)load_main<K>(tb,t,warp,lane,c+1+RC,s[(j+1)%RC]);
     #pragma unroll
     for(int i=0;i<8;++i)wc[i]=wn[i];
    }
    if constexpr (K == 4096) {
     if(c+RC<C)load_a<K,!PDL>(p,lane,c+RC,a[j]);
    }
   }
  }
 }
#ifndef FP8E_NO_PDL
 if constexpr (PDL) pdl_launch_dependents();
#endif
 float sa=__ldg(p.sa),sb=__ldg(p.sb);int row=lane>>2,col=(blockIdx.x*4+warp)*8+(lane&3)*2;
 if(row<p.m){p.c[row*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[0],sb),sa));p.c[row*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[1],sb),sa));}
}
// Load-time proof: the same loads and decoder, storing the rebuilt words in the
// fp8small pack layout [N/32][K/32][4 warps][32 lanes][2 words].
template<int K>
__global__ __launch_bounds__(128) void decode_kernel(const uint8_t* buf,int tile_bytes,uint32_t* out) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=buf+(size_t)blockIdx.x*tile_bytes;
 Tab t=load_tab<K>(tb,warp,lane);
 uint32_t lutA,lutB;load_lut(tb,Shape<K>::HDR,warp,lane,lutA,lutB);
 for(int c=0;c<C;++c){
  Slot s;uint32_t w[8];
  load_main<K>(tb,t,warp,lane,c,s);
  decode(s,lutA,lutB,lane,w);
  #pragma unroll
  for(int st=0;st<4;++st){
   size_t i=((((size_t)blockIdx.x*(K/32)+c*4+st)*4+warp)*32+lane)*2;
   out[i]=w[2*st];out[i+1]=w[2*st+1];
  }
 }
}
template<int N,int K,int RC,bool PDL>
cudaError_t launch(const P&p,cudaStream_t stream) {
#ifndef FP8E_NO_PDL
 if constexpr (PDL) {
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(N/32);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=0;cfg.stream=stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attr;cfg.numAttrs=1;
  return cudaLaunchKernelEx(&cfg,gemm<N,K,RC,true>,p);
 }
#endif
 gemm<N,K,RC,false><<<N/32,128,0,stream>>>(p);
 return cudaGetLastError();
}
template<int RC,bool PDL>
cudaError_t dispatch(const P&p,int n,int k,cudaStream_t stream) {
 if(n==10304&&k==2688)return launch<10304,2688,RC,PDL>(p,stream);
 if(n==2688&&k==4096)return launch<2688,4096,RC,PDL>(p,stream);
 return cudaErrorInvalidValue;
}
}  // namespace fp8e

extern "C" int fp8e_run(const void*a,const void*w,const void*sa,const void*sb,void*c,int m,int n,int k,
                        int tile_bytes,int pdl,int ring,void*stream) {
 using namespace fp8e;
 if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)w,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m,tile_bytes};
 cudaStream_t st=(cudaStream_t)stream;
 cudaError_t rc;
#ifdef FP8E_NO_PDL
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
extern "C" int fp8e_decode(const void*buf,int tile_bytes,void*out,int n,int k,void*stream) {
 using namespace fp8e;
 cudaStream_t st=(cudaStream_t)stream;
 if(n==10304&&k==2688)decode_kernel<2688><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else if(n==2688&&k==4096)decode_kernel<4096><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
extern "C" int fp8e_header_bytes(int k) {
 if(k==2688)return fp8e::Shape<2688>::HDR;
 if(k==4096)return fp8e::Shape<4096>::HDR;
 return -1;
}
extern "C" int fp8e_init() {
 cudaFuncAttributes attr;cudaError_t rc;
 rc=cudaFuncGetAttributes(&attr,fp8e::gemm<10304,2688,3,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8e::gemm<2688,4096,4,false>);if(rc!=cudaSuccess)return (int)rc;
#ifndef FP8E_NO_PDL
 rc=cudaFuncGetAttributes(&attr,fp8e::gemm<10304,2688,3,true>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8e::gemm<2688,4096,4,true>);if(rc!=cudaSuccess)return (int)rc;
#endif
 rc=cudaFuncGetAttributes(&attr,fp8e::decode_kernel<2688>);if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8e::decode_kernel<4096>);
}
extern "C" int fp8e_has_pdl() {
#ifdef FP8E_NO_PDL
 return 0;
#else
 return 1;
#endif
}
