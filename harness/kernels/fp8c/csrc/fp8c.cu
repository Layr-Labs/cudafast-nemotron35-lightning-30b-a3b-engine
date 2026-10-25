// SPDX-License-Identifier: Apache-2.0
// fp8c: fp8small's exact FP8 m16n8k32 chain for 1..8 rows and the two Mamba
// shapes, reading a lossless 7-bit coding of the weight bytes instead of the
// 8-bit fp8small pack. The decoder rebuilds, in registers, the very uint32
// words fp8small loads (verified for every byte at load time) and hands them
// to the same native e4m3 MMA in the same ascending-K chain, followed by the
// same FP32 scale B, FP32 scale A, RN BF16 epilogue. No value is approximated.
//
// Code (per output column, i.e. per lane): an 8-exponent window [e0, e0+7]. A
// byte s|eeee|mmm with e in the window is the 7-bit code s|(e-e0)|mmm; one code
// (ESC, the least frequent one of the column) marks an escape whose original
// byte is stored raw in the tile's exception area. Layout, one region per
// 32-column tile (= one CTA), in the order the CTA consumes it:
//   [header: u16 offsets (chunk,warp) + final total; u16 e0|esc<<8 per column at HDR-64]
//   [C chunks x 3584 B: planes A (4 x u32/lane), B (2 x u32/lane), D (1 x u32/lane)]
//   [exception bytes, ordered (chunk, warp, lane, word, byte)]
// A chunk is 4 K-steps (128 K). Per lane it holds the 8 fp8small words
// R0..R7 (R[2s+h] = step s, half h) as 7 words: byte j of word i<7 is
// code(R_i byte j) | bit i of code(R_7 byte j) << 7.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace fp8c {
constexpr unsigned FULL = 0xffffffffu;
constexpr int MAIN = 3584;  // bytes per (tile, chunk)
template<int K> struct Shape {
 static constexpr int C = K / 128;
 static constexpr int HDR = (((4 * C + 1) * 2 + 64) + 127) / 128 * 128;
 static constexpr int EXC = HDR + C * MAIN;
};
struct P {const uint8_t *a,*w;const float *sa,*sb;__nv_bfloat16 *c;int m;int tile_bytes;};
struct Slot {uint32_t c[7];uint32_t x;uint32_t q0;};

template<bool KEEP>
__device__ __forceinline__ uint4 ld4(const uint8_t* p) {
 uint4 v;
 if constexpr (KEEP)
  asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v4.u32 {%0,%1,%2,%3}, [%4];\n"
   : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
 else
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
   : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p));
 return v;
}
template<bool KEEP>
__device__ __forceinline__ uint2 ld2(const uint8_t* p) {
 uint2 v;
 if constexpr (KEEP)
  asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v2.u32 {%0,%1}, [%2];\n"
   : "=r"(v.x),"=r"(v.y) : "l"(p));
 else
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x),"=r"(v.y) : "l"(p));
 return v;
}
template<bool KEEP>
__device__ __forceinline__ uint32_t ld1(const uint8_t* p) {
 uint32_t v;
 if constexpr (KEEP)
  asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
 else
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
 return v;
}
__device__ __forceinline__ uint32_t bfi8(uint32_t b, uint32_t w, uint32_t pos) {
 uint32_t d;
 asm("bfi.b32 %0, %1, %2, %3, 8;\n" : "=r"(d) : "r"(b), "r"(w), "r"(pos));
 return d;
}
// Main planes of chunk c (no dependence on the header).
template<int K, bool KEEP>
__device__ __forceinline__ void load_main(const uint8_t* tb,int warp,int lane,int c,Slot& s) {
 const uint8_t* mp=tb+Shape<K>::HDR+c*MAIN;
 uint4 A=ld4<KEEP>(mp+warp*512+lane*16);
 uint2 B=ld2<KEEP>(mp+2048+warp*256+lane*8);
 uint32_t D=ld1<KEEP>(mp+3072+warp*128+lane*4);
 s.c[0]=A.x;s.c[1]=A.y;s.c[2]=A.z;s.c[3]=A.w;s.c[4]=B.x;s.c[5]=B.y;s.c[6]=D;
}
// Exception window of chunk c: lane l holds bytes [wb+4l, wb+4l+4) of the tile's
// exception area, wb = off & ~3, for the words that hold this warp-chunk's bytes.
template<int K, bool KEEP>
__device__ __forceinline__ void load_x(const uint8_t* tb,int lane,int c,uint32_t ts,uint32_t te,Slot& s) {
 uint32_t off=__shfl_sync(FULL,ts,c),end=__shfl_sync(FULL,te,c);
 uint32_t q0=off&3u;
 s.q0=q0;
 s.x=(4u*lane<q0+end-off)?ld1<KEEP>(tb+Shape<K>::EXC+(off-q0)+4*lane):0u;
}
__device__ __forceinline__ uint32_t lanemask_lt() {
 uint32_t m;
 asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
 return m;
}
// Rebuild the 8 fp8small words of this lane for one chunk.
// Escapes of a warp-chunk are stored lane by lane, each lane's in (word, byte)
// order; q = window-relative index of this lane's first one (exclusive prefix
// of the per-lane counts, from 6 bit-plane ballots: a lane has at most 32).
// Fast path: two shuffled window words hold this lane's bytes q..q+7-(q&3),
// enough unless (q&3)+cnt > 8 in some lane (rare: then a shuffle per escape).
__device__ __forceinline__ void decode(const Slot& s,uint32_t e0x,uint32_t esc4,int lane,uint32_t (&w)[8]) {
 uint32_t cw[8],em[8],pc[8],hi=0,cnt=0;
 #pragma unroll
 for(int i=0;i<7;++i){cw[i]=s.c[i];hi|=(s.c[i]>>(7-i))&(0x01010101u<<i);}
 cw[7]=hi;
 #pragma unroll
 for(int i=0;i<8;++i){
  uint32_t x=(cw[i]^esc4)&0x7f7f7f7fu;  // bytes <= 0x7f: x+0x7f carries into bit 7 iff x != 0
  em[i]=~(x+0x7f7f7f7fu)&0x80808080u;
  pc[i]=__popc(em[i]);
  cnt+=pc[i];
  // code s|e'|mmm (bits 6..0) -> s|e'+e0|mmm; e'+e0 <= 15, so no byte carries.
  w[i]=((cw[i]&0x3f3f3f3fu)+e0x)|((cw[i]<<1)&0x80808080u);
 }
 uint32_t lt=lanemask_lt();
 uint32_t pre=(uint32_t)__popc(__ballot_sync(FULL,cnt&1u)&lt)
  +((uint32_t)__popc(__ballot_sync(FULL,cnt&2u)&lt)<<1)
  +((uint32_t)__popc(__ballot_sync(FULL,cnt&4u)&lt)<<2);
 uint32_t q=s.q0+pre;  // exact when no lane has more than 7 escapes
 uint32_t v0=__shfl_sync(FULL,s.x,(q>>2)&31u),v1=__shfl_sync(FULL,s.x,((q>>2)+1u)&31u);
 // Branch-free patch. The k-th escape byte of this lane is byte (q&3)+k of
 // v1:v0; sel replicates it into all four bytes, and a byte mask (0xff per
 // escape byte, from the 0x80 flags) merges it with one LOP3. Pass 1 patches
 // the lowest escape of each word; pass 2 (only if some lane needs it) the
 // second one. Three or more in a word, more than 7 in the chunk or bytes
 // beyond v1:v0 take the per-escape fallback below, which redoes them all.
 uint32_t sel=(q&3u)*0x1111u,em2[8],sl[8],two=0;
 #pragma unroll
 for(int i=0;i<8;++i){
  em2[i]=em[i]&(em[i]-1u);
  two|=em2[i];
  uint32_t fm=((em[i]^em2[i])>>7)*0xffu;
  w[i]=(w[i]&~fm)|(__byte_perm(v0,v1,sel)&fm);
  sl[i]=sel+0x1111u;
  sel+=pc[i]*0x1111u;
 }
 bool slow=cnt>7u||(q&3u)+cnt>8u;
 if(__any_sync(FULL,two)){
  uint32_t three=0;
  #pragma unroll
  for(int i=0;i<8;++i){
   three|=em2[i]&(em2[i]-1u);
   uint32_t fm=(em2[i]>>7)*0xffu;
   w[i]=(w[i]&~fm)|(__byte_perm(v0,v1,sl[i])&fm);
  }
  slow|=three!=0u;
 }
 if(__any_sync(FULL,slow)){
  pre=0;
  #pragma unroll
  for(int b=0;b<6;++b)pre+=(uint32_t)__popc(__ballot_sync(FULL,(cnt>>b)&1u)&lt)<<b;
  q=s.q0+pre;
  #pragma unroll
  for(int i=0;i<8;++i){
   uint32_t mk=em[i];
   while(__any_sync(FULL,mk)){
    uint32_t v=__shfl_sync(FULL,s.x,(q>>2)&31u);
    if(mk){
     w[i]=bfi8(__byte_perm(v,0u,q&3u),w[i],(uint32_t)__ffs(mk)-8u);
     mk&=mk-1u;++q;
    }
   }
  }
 }
}
template<bool KEEP, bool NC>
__device__ __forceinline__ uint32_t lda(const uint8_t* q) {
 uint32_t v;
 if constexpr (KEEP) {
  asm volatile("ld.global.L2::evict_last.u32 %0, [%1];\n" : "=r"(v) : "l"(q));
  return v;
 } else if constexpr (NC) {
  return __ldg((const uint32_t*)q);
 } else {
  return *(const uint32_t*)q;
 }
}
// Activation words (rows 0..7) of the 4 steps of chunk c; rows 8..15 are zero for M<=8.
template<int K, bool KEEP, bool NC>
__device__ __forceinline__ void load_a(const P&p,int lane,int c,uint32_t (&a)[4][2]) {
 int row=lane>>2;
 #pragma unroll
 for(int s=0;s<4;++s){
  int kk=(c*4+s)*32+(lane&3)*4;
  a[s][0]=row<p.m?lda<KEEP,NC>(p.a+row*K+kk):0u;
  a[s][1]=row<p.m?lda<KEEP,NC>(p.a+row*K+kk+16):0u;
 }
}
#ifndef FP8C_NO_PDL
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
// RC chunks (4*RC K-steps) in flight per warp. PDL: header, main planes and
// exception windows (load-time data, never written at run time) are issued
// before griddepcontrol.wait; activations, scales and stores after it.
template<int N,int K,int RC,bool PDL=false,bool KEEP=false>
__global__ __launch_bounds__(128,2) void gemm(P p) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=p.w+(size_t)blockIdx.x*p.tile_bytes;
 const uint16_t* tab=(const uint16_t*)tb;
 uint32_t ts=0,te=0;
 if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
 uint32_t prm=__ldg((const uint16_t*)(tb+Shape<K>::HDR-64)+warp*8+(lane>>2));  // this lane's column
 uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
 Slot s[RC];uint32_t a[RC][4][2];float acc[4]={0.f,0.f,0.f,0.f};
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_main<K,KEEP>(tb,warp,lane,j,s[j]);
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_x<K,KEEP>(tb,lane,j,ts,te,s[j]);
#ifndef FP8C_NO_PDL
 if constexpr (PDL) pdl_wait();
#endif
 #pragma unroll
 for(int j=0;j<RC;++j)if(j<C)load_a<K,KEEP,!PDL>(p,lane,j,a[j]);
 // Software pipeline: wc holds chunk c decoded; the MMAs of chunk c are issued
 // before the decode of chunk c+1, so the two latency chains overlap. Slot
 // (c mod RC) of the ring is refilled with chunk c+RC right after its decode.
 uint32_t wc[8];
 decode(s[0],e0x,esc4,lane,wc);
 if(RC<C){load_main<K,KEEP>(tb,warp,lane,RC,s[0]);load_x<K,KEEP>(tb,lane,RC,ts,te,s[0]);}
 for(int cb=0;cb<C;cb+=RC){
  #pragma unroll
  for(int j=0;j<RC;++j){
   int c=cb+j;
   if(c<C){
    #pragma unroll
    for(int st=0;st<4;++st)mma(a[j][st][0],a[j][st][1],wc[2*st],wc[2*st+1],acc);
    if constexpr (K == 2688) {
     if(c+RC<C)load_a<K,KEEP,!PDL>(p,lane,c+RC,a[j]);
    }
    if(c+1<C){
     uint32_t wn[8];
     decode(s[(j+1)%RC],e0x,esc4,lane,wn);
     if(c+1+RC<C){
      load_main<K,KEEP>(tb,warp,lane,c+1+RC,s[(j+1)%RC]);
      load_x<K,KEEP>(tb,lane,c+1+RC,ts,te,s[(j+1)%RC]);
     }
     #pragma unroll
     for(int i=0;i<8;++i)wc[i]=wn[i];
    }
    // Output projection: decode/refill weights before issuing the next
    // independent activation refill. Keep the original input schedule.
    if constexpr (K == 4096) {
     if(c+RC<C)load_a<K,KEEP,!PDL>(p,lane,c+RC,a[j]);
    }
   }
  }
 }
#ifndef FP8C_NO_PDL
 if constexpr (PDL) pdl_launch_dependents();
#endif
 float sa=__ldg(p.sa),sb=__ldg(p.sb);int row=lane>>2,col=(blockIdx.x*4+warp)*8+(lane&3)*2;
 if(row<p.m){p.c[row*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[0],sb),sa));p.c[row*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[1],sb),sa));}
}
// Load-time proof: the same loads and decoder, storing the rebuilt words in the
// fp8small pack layout [N/32][K/32][4 warps][32 lanes][2 words] instead of an MMA.
template<int K>
__global__ __launch_bounds__(128) void decode_kernel(const uint8_t* buf,int tile_bytes,uint32_t* out) {
 constexpr int C=Shape<K>::C;
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const uint8_t* tb=buf+(size_t)blockIdx.x*tile_bytes;
 const uint16_t* tab=(const uint16_t*)tb;
 uint32_t ts=0,te=0;
 if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
 uint32_t prm=__ldg((const uint16_t*)(tb+Shape<K>::HDR-64)+warp*8+(lane>>2));  // this lane's column
 uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
 for(int c=0;c<C;++c){
  Slot s;uint32_t w[8];
  load_main<K,false>(tb,warp,lane,c,s);
  load_x<K,false>(tb,lane,c,ts,te,s);
  decode(s,e0x,esc4,lane,w);
  #pragma unroll
  for(int st=0;st<4;++st){
   size_t i=((((size_t)blockIdx.x*(K/32)+c*4+st)*4+warp)*32+lane)*2;
   out[i]=w[2*st];out[i+1]=w[2*st+1];
  }
 }
}
template<int N,int K,int RC,bool PDL,bool KEEP>
cudaError_t launch(const P&p,cudaStream_t stream) {
#ifndef FP8C_NO_PDL
 if constexpr (PDL) {
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(N/32);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=0;cfg.stream=stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attr;cfg.numAttrs=1;
  return cudaLaunchKernelEx(&cfg,gemm<N,K,RC,true,KEEP>,p);
 }
#endif
 gemm<N,K,RC,false,KEEP><<<N/32,128,0,stream>>>(p);
 return cudaGetLastError();
}
template<int RC,bool PDL,bool KEEP>
cudaError_t dispatch(const P&p,int n,int k,cudaStream_t stream) {
 if(n==10304&&k==2688)return launch<10304,2688,RC,PDL,KEEP>(p,stream);
 if(n==2688&&k==4096)return launch<2688,4096,RC,PDL,KEEP>(p,stream);
 return cudaErrorInvalidValue;
}
}  // namespace fp8c

// Ring depth in chunks: 3 (in_proj default), 4 (out_proj default) or 6.
extern "C" int fp8c_run(const void*a,const void*w,const void*sa,const void*sb,void*c,int m,int n,int k,
                        int tile_bytes,int pdl,int ring,void*stream,int keep) {
 using namespace fp8c;
 if(m<1||m>8)return (int)cudaErrorInvalidValue;
 P p{(const uint8_t*)a,(const uint8_t*)w,(const float*)sa,(const float*)sb,(__nv_bfloat16*)c,m,tile_bytes};
 cudaStream_t st=(cudaStream_t)stream;
 cudaError_t rc;
#ifdef FP8C_NO_PDL
 if(pdl)return (int)cudaErrorNotSupported;
#endif
 #define FP8C_DISPATCH(RC,PDL_) \
  (keep ? dispatch<RC,PDL_,true>(p,n,k,st) : dispatch<RC,PDL_,false>(p,n,k,st))
 switch(ring){
  case 3: rc=pdl?FP8C_DISPATCH(3,true):FP8C_DISPATCH(3,false);break;
  case 4: rc=pdl?FP8C_DISPATCH(4,true):FP8C_DISPATCH(4,false);break;
  case 6: rc=pdl?FP8C_DISPATCH(6,true):FP8C_DISPATCH(6,false);break;
  default: return (int)cudaErrorInvalidValue;
 }
#undef FP8C_DISPATCH
 if(rc!=cudaSuccess){cudaGetLastError();return (int)rc;}  // cleared: the caller may retry without PDL
 return (int)cudaGetLastError();
}
extern "C" int fp8c_decode(const void*buf,int tile_bytes,void*out,int n,int k,void*stream) {
 using namespace fp8c;
 cudaStream_t st=(cudaStream_t)stream;
 if(n==10304&&k==2688)decode_kernel<2688><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else if(n==2688&&k==4096)decode_kernel<4096><<<n/32,128,0,st>>>((const uint8_t*)buf,tile_bytes,(uint32_t*)out);
 else return (int)cudaErrorInvalidValue;
 return (int)cudaGetLastError();
}
extern "C" int fp8c_header_bytes(int k) {
 if(k==2688)return fp8c::Shape<2688>::HDR;
 if(k==4096)return fp8c::Shape<4096>::HDR;
 return -1;
}
extern "C" int fp8c_init() {
 cudaFuncAttributes attr;cudaError_t rc;
 // the default rings: 3 for in_proj, 4 for out_proj
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<10304,2688,3,false,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<2688,4096,4,false,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<10304,2688,3,false,true>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<2688,4096,4,false,true>);if(rc!=cudaSuccess)return (int)rc;
#ifndef FP8C_NO_PDL
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<10304,2688,3,true,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<2688,4096,4,true,false>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<10304,2688,3,true,true>);if(rc!=cudaSuccess)return (int)rc;
 rc=cudaFuncGetAttributes(&attr,fp8c::gemm<2688,4096,4,true,true>);if(rc!=cudaSuccess)return (int)rc;
#endif
 rc=cudaFuncGetAttributes(&attr,fp8c::decode_kernel<2688>);if(rc!=cudaSuccess)return (int)rc;
 return (int)cudaFuncGetAttributes(&attr,fp8c::decode_kernel<4096>);
}
extern "C" int fp8c_has_pdl() {
#ifdef FP8C_NO_PDL
 return 0;
#else
 return 1;
#endif
}
