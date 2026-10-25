#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define APK_HD __host__ __device__ __forceinline__
#else
#define APK_HD static inline
#endif

APK_HD uint32_t apk_bits(uint32_t s, uint32_t c, uint32_t b) {
  uint32_t e = c ? b + c - 1 : 0;
  return ((s & 0x80u) << 8) | (e << 7) | (s & 0x7Fu);
}

APK_HD void apk_decode(const uint32_t (&s)[4], const uint32_t (&x)[2], uint32_t b,
                       uint32_t (&w)[8]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int q = 16 * (j & 1) + 8 * h;
      uint32_t lo = apk_bits((s[j] >> (16 * h)) & 0xFFu, (x[j >> 1] >> q) & 15u, b);
      uint32_t hi = apk_bits((s[j] >> (16 * h + 8)) & 0xFFu, (x[j >> 1] >> (q + 4)) & 15u, b);
      w[2 * j + h] = lo | (hi << 16);
    }
  }
}

APK_HD void apk_raw(const uint16_t* blk, int half, int t, uint32_t (&w)[8]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int k = 64 * half + 16 * j + 8 * h + 2 * t;
      w[2 * j + h] = (uint32_t)blk[k] | ((uint32_t)blk[k + 1] << 16);
    }
  }
}
