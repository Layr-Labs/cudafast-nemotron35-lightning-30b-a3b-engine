// Optional per-original-N32-tile choice of the existing six/seven-bit codecs.
// No declared numerical format or original MMA order is changed.
#ifdef FP8MIX_NO_PDL
#define FP8C_NO_PDL
#define FP8SIX_NO_PDL
#endif
#define fp8c fp8mix7
#include "seven_helpers.cuh"
#undef fp8c
#define fp8six fp8mix6
#include "six_helpers.cuh"
#undef fp8six

namespace fp8mix {
using P = fp8mix7::P;
template<int Bits> struct Codec;
template<> struct Codec<7> {
  using Slot = fp8mix7::Slot;
  template<int K> __device__ static __forceinline__ void main(
      const uint8_t* tb,int w,int l,int c,Slot& s) { fp8mix7::load_main<K>(tb,w,l,c,s); }
  template<int K> __device__ static __forceinline__ void exc(
      const uint8_t* tb,int l,int c,uint32_t ts,uint32_t te,Slot& s) {
    fp8mix7::load_x<K>(tb,l,c,ts,te,s);
  }
  __device__ static __forceinline__ void decode(
      const Slot& s,uint32_t e,uint32_t esc,int l,uint32_t (&out)[8]) {
    fp8mix7::decode(s,e,esc,l,out);
  }
};
template<> struct Codec<6> {
  using Slot = fp8mix6::Slot;
  template<int K> __device__ static __forceinline__ void main(
      const uint8_t* tb,int w,int l,int c,Slot& s) { fp8mix6::load_main<K>(tb,w,l,c,s); }
  template<int K> __device__ static __forceinline__ void exc(
      const uint8_t* tb,int l,int c,uint32_t ts,uint32_t te,Slot& s) {
    fp8mix6::load_x<K>(tb,l,c,ts,te,s);
  }
  __device__ static __forceinline__ void decode(
      const Slot& s,uint32_t e,uint32_t esc,int l,uint32_t (&out)[8]) {
    fp8mix6::decode(s,e,esc,l,out);
  }
};

// Four identical immutable descriptor requests, one per warp (16 logical B).
// Payload offsets are multiples of256; bit0 selects six bits, all others zero.
__device__ __forceinline__ uint32_t descriptor(const uint8_t* packet,int lane) {
  uint32_t d = 0;
  if (lane == 0) d = __ldg(reinterpret_cast<const uint32_t*>(packet) + blockIdx.x);
  return __shfl_sync(0xffffffffu,d,0);
}

template<int Bits,int N,int K,int RC,bool PDL>
__device__ __forceinline__ void tile(const P& p,const uint8_t* tb,int lane,int warp) {
  using Coder = Codec<Bits>;
  using Slot = typename Coder::Slot;
  constexpr int C = fp8mix7::Shape<K>::C;
  const uint16_t* tab = reinterpret_cast<const uint16_t*>(tb);
  uint32_t ts=0,te=0;
  if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
  uint32_t prm=__ldg(reinterpret_cast<const uint16_t*>(tb+fp8mix7::Shape<K>::HDR-64)
                    +warp*8+(lane>>2));
  uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
  Slot s[RC]; uint32_t a[RC][4][2]; float acc[4]={0.f,0.f,0.f,0.f};
#pragma unroll
  for(int j=0;j<RC;++j)if(j<C)Coder::template main<K>(tb,warp,lane,j,s[j]);
#pragma unroll
  for(int j=0;j<RC;++j)if(j<C)Coder::template exc<K>(tb,lane,j,ts,te,s[j]);
#ifndef FP8MIX_NO_PDL
  if constexpr(PDL) fp8mix7::pdl_wait();
#endif
#pragma unroll
  for(int j=0;j<RC;++j)if(j<C)fp8mix7::load_a<K,!PDL>(p,lane,j,a[j]);
  uint32_t wc[8];
  Coder::decode(s[0],e0x,esc4,lane,wc);
  if(RC<C){Coder::template main<K>(tb,warp,lane,RC,s[0]);
           Coder::template exc<K>(tb,lane,RC,ts,te,s[0]);}
  for(int cb=0;cb<C;cb+=RC){
#pragma unroll
    for(int j=0;j<RC;++j){
      int c=cb+j;
      if(c<C){
#pragma unroll
        for(int st=0;st<4;++st)
          fp8mix7::mma(a[j][st][0],a[j][st][1],wc[2*st],wc[2*st+1],acc);
        if constexpr(K==2688){if(c+RC<C)fp8mix7::load_a<K,!PDL>(p,lane,c+RC,a[j]);}
        if(c+1<C){
          uint32_t wn[8];
          Coder::decode(s[(j+1)%RC],e0x,esc4,lane,wn);
          if(c+1+RC<C){
            Coder::template main<K>(tb,warp,lane,c+1+RC,s[(j+1)%RC]);
            Coder::template exc<K>(tb,lane,c+1+RC,ts,te,s[(j+1)%RC]);
          }
#pragma unroll
          for(int i=0;i<8;++i)wc[i]=wn[i];
        }
        if constexpr(K==4096){if(c+RC<C)fp8mix7::load_a<K,!PDL>(p,lane,c+RC,a[j]);}
      }
    }
  }
#ifndef FP8MIX_NO_PDL
  if constexpr(PDL)fp8mix7::pdl_launch_dependents();
#endif
  float sa=__ldg(p.sa),sb=__ldg(p.sb);
  int row=lane>>2,col=(blockIdx.x*4+warp)*8+(lane&3)*2;
  if(row<p.m){
    p.c[row*N+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[0],sb),sa));
    p.c[row*N+col+1]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(acc[1],sb),sa));
  }
}

template<int N,int K,int RC,bool PDL>
__global__ __launch_bounds__(128,2) void gemm(P p) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const uint32_t d=descriptor(p.w,lane);
  const uint8_t* tb=p.w+(d&~255u);
  // Mode is uniform for the entire original CTA; every warp follows its own
  // unchanged old decoder, all original K steps and the original PDL points.
  if(d&1u)tile<6,N,K,RC,PDL>(p,tb,lane,warp);
  else tile<7,N,K,RC,PDL>(p,tb,lane,warp);
}

template<int Bits,int K>
__device__ __forceinline__ void decode_tile(const uint8_t* tb,uint32_t* out,int lane,int warp){
  using Coder=Codec<Bits>; using Slot=typename Coder::Slot;
  constexpr int C=fp8mix7::Shape<K>::C;
  const uint16_t* tab=reinterpret_cast<const uint16_t*>(tb);
  uint32_t ts=0,te=0;
  if(lane<C){ts=__ldg(tab+lane*4+warp);te=__ldg(tab+lane*4+warp+1);}
  uint32_t prm=__ldg(reinterpret_cast<const uint16_t*>(tb+fp8mix7::Shape<K>::HDR-64)
                    +warp*8+(lane>>2));
  uint32_t e0x=((prm&0xffu)<<3)*0x01010101u,esc4=((prm>>8)&0xffu)*0x01010101u;
  for(int c=0;c<C;++c){
    Slot s; uint32_t w[8];
    Coder::template main<K>(tb,warp,lane,c,s);
    Coder::template exc<K>(tb,lane,c,ts,te,s);
    Coder::decode(s,e0x,esc4,lane,w);
#pragma unroll
    for(int st=0;st<4;++st){
      size_t i=((((size_t)blockIdx.x*(K/32)+c*4+st)*4+warp)*32+lane)*2;
      out[i]=w[2*st];out[i+1]=w[2*st+1];
    }
  }
}
template<int K>
__global__ __launch_bounds__(128) void decode(const uint8_t* packet,uint32_t* out){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const uint32_t d=descriptor(packet,lane);
  const uint8_t* tb=packet+(d&~255u);
  if(d&1u)decode_tile<6,K>(tb,out,lane,warp);
  else decode_tile<7,K>(tb,out,lane,warp);
}

template<int N,int K,int RC,bool PDL>
int run(const P& p,cudaStream_t st){
#ifndef FP8MIX_NO_PDL
  if constexpr(PDL){
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(N/32);cfg.blockDim=dim3(128);cfg.stream=st;
    cudaLaunchAttribute a[1];a[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    a[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=a;cfg.numAttrs=1;
    auto rc=cudaLaunchKernelEx(&cfg,gemm<N,K,RC,true>,p);
    if(rc!=cudaSuccess)return (int)rc;
    return (int)cudaGetLastError();
  }
#endif
  gemm<N,K,RC,false><<<N/32,128,0,st>>>(p);
  return (int)cudaGetLastError();
}
template<int N,int K,int RC,bool PDL>
int info(int* dst){
  cudaFuncAttributes a;int blocks=0;
  auto rc=cudaFuncGetAttributes(&a,gemm<N,K,RC,PDL>);
  if(rc!=cudaSuccess)return (int)rc;
  rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gemm<N,K,RC,PDL>,128,0);
  if(rc!=cudaSuccess)return (int)rc;
  dst[0]=a.numRegs;dst[1]=(int)a.localSizeBytes;dst[2]=(int)a.sharedSizeBytes;dst[3]=blocks;
  return a.maxThreadsPerBlock>=128?0:-12;
}
template<int N,int K,int RC>
int dispatch(const P& p,cudaStream_t st,int pdl,int* attrs){
#ifndef FP8MIX_NO_PDL
  if(pdl)return attrs?info<N,K,RC,true>(attrs):run<N,K,RC,true>(p,st);
#endif
  return attrs?info<N,K,RC,false>(attrs):run<N,K,RC,false>(p,st);
}
} // namespace fp8mix

extern "C" int fp8mix_run(const void* a,const void* packet,const void* sa,const void* sb,
                         void* c,int m,int n,int k,int ring,int pdl,void* stream,int* attrs){
  if(m<1||m>8||!a||!packet||!sa||!sb||!c)return -1;
  if((reinterpret_cast<uintptr_t>(a)&15)||(reinterpret_cast<uintptr_t>(packet)&255))return -2;
#ifdef FP8MIX_NO_PDL
  if(pdl)return -10;
#endif
  fp8mix::P p{(const uint8_t*)a,(const uint8_t*)packet,(const float*)sa,(const float*)sb,
              (__nv_bfloat16*)c,m,0};
  auto st=(cudaStream_t)stream;
#define MX_DISPATCH(N,K) \
  switch(ring){case 3:return fp8mix::dispatch<N,K,3>(p,st,pdl,attrs); \
               case 4:return fp8mix::dispatch<N,K,4>(p,st,pdl,attrs); \
               case 6:return fp8mix::dispatch<N,K,6>(p,st,pdl,attrs);default:return -3;}
  if(n==10304&&k==2688){MX_DISPATCH(10304,2688)}
  if(n==2688&&k==4096){MX_DISPATCH(2688,4096)}
#undef MX_DISPATCH
  return -4;
}
extern "C" int fp8mix_decode(const void* packet,void* out,int n,int k,void* stream){
  if(!packet||!out||(reinterpret_cast<uintptr_t>(packet)&255))return -1;
  if(n==10304&&k==2688)
    fp8mix::decode<2688><<<n/32,128,0,(cudaStream_t)stream>>>((const uint8_t*)packet,(uint32_t*)out);
  else if(n==2688&&k==4096)
    fp8mix::decode<4096><<<n/32,128,0,(cudaStream_t)stream>>>((const uint8_t*)packet,(uint32_t*)out);
  else return -4;
  return (int)cudaGetLastError();
}
extern "C" int fp8mix_has_pdl(){
#ifdef FP8MIX_NO_PDL
  return 0;
#else
  return 1;
#endif
}

// Private admission only: deliberately publish the dependent before writing
// current A and scales. No prefetch fork/join may hide a missing GDC wait.
template<bool PDL>
__global__ void fp8mix_publish(const uint8_t* src,const float* sa,const float* sb,
                             uint8_t* dst,float* da,float* db,int count){
#ifndef FP8MIX_NO_PDL
  if constexpr(PDL){fp8mix7::pdl_wait();fp8mix7::pdl_launch_dependents();}
#endif
  __nanosleep(10000);
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)dst[i]=src[i];
  if(i==0){*da=*sa;*db=*sb;}
}
extern "C" int fp8mix_publish_run(const void* src,const void* sa,const void* sb,
                                   void* dst,void* da,void* db,int count,int pdl,void* stream){
  if(!src||!sa||!sb||!dst||!da||!db||count<=0)return -1;
#ifndef FP8MIX_NO_PDL
  if(pdl){
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((count+127)/128);
    cfg.blockDim=dim3(128);cfg.stream=(cudaStream_t)stream;
    cudaLaunchAttribute a[1];a[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    a[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=a;cfg.numAttrs=1;
    auto rc=cudaLaunchKernelEx(&cfg,fp8mix_publish<true>,(const uint8_t*)src,
      (const float*)sa,(const float*)sb,(uint8_t*)dst,(float*)da,(float*)db,count);
    if(rc!=cudaSuccess)return (int)rc;
    return (int)cudaGetLastError();
  }
#else
  if(pdl)return -10;
#endif
  fp8mix_publish<false><<<(count+127)/128,128,0,(cudaStream_t)stream>>>(
    (const uint8_t*)src,(const float*)sa,(const float*)sb,(uint8_t*)dst,
    (float*)da,(float*)db,count);
  return (int)cudaGetLastError();
}
