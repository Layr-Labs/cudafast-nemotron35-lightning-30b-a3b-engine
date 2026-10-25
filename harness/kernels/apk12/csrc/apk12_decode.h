// SPDX-License-Identifier: Apache-2.0
// Decode of the lossless 12-bit BF16 codec (nemotron_h_mtp_packed.load_packed) into
// m16n8k16 B-fragment registers. Shared by the GEMM, the load-time bit check and the
// host-compiled CPU test, so all three run this exact code.
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define APK_HD __host__ __device__ __forceinline__
#else
#define APK_HD static inline
#endif

// One weight from its sign+mantissa byte s, 4-bit exponent code c and block base b:
// code 0 is exponent 0 (zeros, subnormals), code c > 0 is exponent b + c - 1.
APK_HD uint32_t apk_bits(uint32_t s, uint32_t c, uint32_t b) {
  uint32_t e = c ? b + c - 1 : 0;
  return ((s & 0x80u) << 8) | (e << 7) | (s & 0x7Fu);
}

// One lane's eight bf16x2 B registers for one 64-wide K group, lane t = lane % 4.
// Register 2j+h holds K = 16j + 8h + 2t (low half) and that K + 1 (high half), the
// .col B fragment of the group's k16 step j. Its two weights are byte 4j+2h and
// 4j+2h+1 of s and nibble 4j+2h and 4j+2h+1 of x (pack() in py/__init__.py).
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

// An escaped block keeps its 16 original bits per weight in raw[ri * 128 + k % 128];
// blk points at its 128 values and half selects the group's 64 of them.
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
