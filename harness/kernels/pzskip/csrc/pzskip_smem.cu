// pzskip unit: configs 300-307, 320-325, 333, 340-344 (smem4_kernel) and 200-204, 210-214, 220-230
// (smem_kernel). Both kernels use the extern __shared__ smem_raw[], and cicc's code for a kernel that uses
// such an array depends on the array's other users in the unit, so all of them are compiled here.
#include "pzskip.cuh"

namespace pzskip {

int dispatch_smem4(int wpc, const Params& p, cudaStream_t st) {
  switch (wpc) {
    case 300: return pzskip::launch_smem4<1, 4, 2, 4>(p, st);
    case 301: return pzskip::launch_smem4<1, 4, 2, 3>(p, st);
    case 302: return pzskip::launch_smem4<1, 5, 2, 4>(p, st);
    case 303: return pzskip::launch_smem4<1, 4, 3, 2>(p, st);   // 192 columns, 6 warps
    case 304: return pzskip::launch_smem4<2, 3, 2, 4>(p, st);
    case 305: return pzskip::launch_smem4<1, 4, 6, 1>(p, st);   // 384 columns, 12 warps
    case 306: return pzskip::launch_smem4<1, 4, 7, 1>(p, st);   // 448 columns, 14 warps
    case 307: return pzskip::launch_smem4<1, 4, 3, 3>(p, st);
    // 32-row blocks: one warp per 64-column tile, 4 token tiles
    case 320: return pzskip::launch_smem4<1, 4, 2, 8, 32>(p, st);   // 128 columns, 2 warps
    case 321: return pzskip::launch_smem4<1, 4, 3, 6, 32>(p, st);   // 192 columns, 3 warps
    case 322: return pzskip::launch_smem4<1, 4, 6, 3, 32>(p, st);   // 384 columns, 6 warps
    case 323: return pzskip::launch_smem4<1, 4, 7, 2, 32>(p, st);   // 448 columns, 7 warps
    case 324: return pzskip::launch_smem4<1, 6, 3, 4, 32>(p, st);
    case 325: return pzskip::launch_smem4<2, 4, 3, 4, 32>(p, st);
    case 333: {
#ifndef PZSKIP_NO_CACHE_POLICY
      const char* enabled = std::getenv("HUMMING_EVICT_PZ");
      // Fixed target top-6: 496 prompt rows become 2976 routed slots.
      if (p.shape_m == 2976 && (!enabled || enabled[0] != '0')) {
        static int loaded = -1;
        if (loaded < 0) {
          cudaFuncAttributes attr;
          cudaError_t e = cudaFuncGetAttributes(&attr, pzskip::smem4_kernel<2, 3, 6, 3, 32, 1>);
          loaded = e == cudaSuccess;
          if (!loaded) {
            cudaGetLastError();
            std::fprintf(stderr, "pzskip: L2 evict-first load unavailable; stock PZ333 retained\n");
          }
        }
        if (loaded) {
          int rc = pzskip::launch_smem4<2, 3, 6, 3, 32, 1>(p, st);
          if (!rc) return 0;
          loaded = 0;
          cudaGetLastError();
          std::fprintf(stderr, "pzskip: L2 evict-first launch unavailable; stock PZ333 retained\n");
        }
      }
#endif
      return pzskip::launch_smem4<2, 3, 6, 3, 32, 0>(p, st);
    }   // 384 columns, 2 k16 groups per stage, 3 stages
    // 16-row blocks: one warp per 64-column tile, 2 token tiles
    case 340: return pzskip::launch_smem4<1, 4, 2, 8, 16>(p, st);
    case 341: return pzskip::launch_smem4<1, 4, 3, 6, 16>(p, st);
    case 342: return pzskip::launch_smem4<1, 4, 6, 3, 16>(p, st);
    case 343: return pzskip::launch_smem4<1, 4, 7, 2, 16>(p, st);
    case 344: return pzskip::launch_smem4<1, 6, 3, 4, 16>(p, st);
    default: return NOT_HANDLED;
  }
}

int dispatch_smem(int wpc, const Params& p, cudaStream_t st) {
  switch (wpc) {
    case 200: return pzskip::launch_smem<2, 4>(p, st);
    case 201: return pzskip::launch_smem<4, 3>(p, st);
    case 202: return pzskip::launch_smem<2, 3>(p, st);
    case 203: return pzskip::launch_smem<1, 6>(p, st);
    case 204: return pzskip::launch_smem<4, 2>(p, st);
    case 220: return pzskip::launch_smem<2, 3, 2, 4>(p, st);  // 128 cols, <= 128 regs, 4 CTAs/SM
    case 221: return pzskip::launch_smem<2, 3, 1>(p, st);     // 64 cols, 2 warps
    case 222: return pzskip::launch_smem<2, 3, 1, 8>(p, st);  // 64 cols, 2 warps, 8 CTAs/SM
    case 223: return pzskip::launch_smem<2, 2, 2, 4>(p, st);
    case 224: return pzskip::launch_smem<4, 2, 2, 4>(p, st);
    case 225: return pzskip::launch_smem<1, 4, 2, 4>(p, st);
    case 226: return pzskip::launch_smem<1, 4, 2, 5>(p, st);
    case 227: return pzskip::launch_smem<1, 4, 2, 6>(p, st);
    case 228: return pzskip::launch_smem<1, 3, 2, 5>(p, st);
    case 229: return pzskip::launch_smem<1, 5, 2, 4>(p, st);
    case 230: return pzskip::launch_smem<1, 6, 2, 4>(p, st);
    case 210: return pzskip::launch_smem<1, 6, 6>(p, st);   // 384 columns, 12 warps
    case 211: return pzskip::launch_smem<2, 4, 6>(p, st);
    case 212: return pzskip::launch_smem<1, 6, 7>(p, st);   // 448 columns, 14 warps
    case 213: return pzskip::launch_smem<2, 3, 7>(p, st);
    case 214: return pzskip::launch_smem<1, 8, 3>(p, st);   // 192 columns, 6 warps
    default: return NOT_HANDLED;
  }
}

}  // namespace pzskip
