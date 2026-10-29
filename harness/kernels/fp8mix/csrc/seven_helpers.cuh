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
// Main planes of chunk c (no dependence on the header).
template<int K>
__device__ __forceinline__ void load_main(const uint8_t* tb,int warp,int lane,int c,Slot& s) {
 const uint8_t* mp=tb+Shape<K>::HDR+c*MAIN;
 uint4 A=ld4(mp+warp*512+lane*16);
 uint2 B=ld2(mp+2048+warp*256+lane*8);
 uint32_t D=ld1(mp+3072+warp*128+lane*4);
 s.c[0]=A.x;s.c[1]=A.y;s.c[2]=A.z;s.c[3]=A.w;s.c[4]=B.x;s.c[5]=B.y;s.c[6]=D;
}
// Exception window of chunk c: lane l holds bytes [wb+4l, wb+4l+4) of the tile's
// exception area, wb = off & ~3, for the words that hold this warp-chunk's bytes.
template<int K>
__device__ __forceinline__ void load_x(const uint8_t* tb,int lane,int c,uint32_t ts,uint32_t te,Slot& s) {
 uint32_t off=__shfl_sync(FULL,ts,c),end=__shfl_sync(FULL,te,c);
 uint32_t q0=off&3u;
 s.q0=q0;
 s.x=(4u*lane<q0+end-off)?ld1(tb+Shape<K>::EXC+(off-q0)+4*lane):0u;
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
template<bool NC>
__device__ __forceinline__ uint32_t lda(const uint8_t* q) {
 if constexpr (NC) return __ldg((const uint32_t*)q);
 else return *(const uint32_t*)q;
}
// Activation words (rows 0..7) of the 4 steps of chunk c; rows 8..15 are zero for M<=8.
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
}  // namespace fp8c (renamed by the including file)
