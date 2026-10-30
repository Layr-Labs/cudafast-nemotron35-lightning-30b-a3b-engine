// pzskip unit: configs 2, 3, 6, 7, 13, 16 and 36 (down_kernel, 32 rows per warp)
#include "pzskip.cuh"

namespace pzskip {

int dispatch_down32(int wpc, const Params& p, cudaStream_t st) {
  const int max_blocks = p.max_blocks;
  switch (wpc) {
    case 16: {  // 2 warps, ring 8
      dim3 grid(pzskip::NT64 / 2, max_blocks, 2);
      pzskip::down_kernel<2, 8><<<grid, 64, 0, st>>>(p);
      break;
    }
    case 36: {  // 6 warps, ring 6
      dim3 grid(pzskip::NT64 / 6, max_blocks, 2);
      pzskip::down_kernel<6, 6><<<grid, 192, 0, st>>>(p);
      break;
    }
    case 13: {  // 3 warps, ring 8
      dim3 grid(pzskip::NT64 / 3, max_blocks, 2);
      pzskip::down_kernel<3, 8><<<grid, 96, 0, st>>>(p);
      break;
    }
    case 2: {
      dim3 grid(pzskip::NT64 / 2, max_blocks, 2);
      pzskip::down_kernel<2, 4><<<grid, 64, 0, st>>>(p);
      break;
    }
    case 3: {
      dim3 grid(pzskip::NT64 / 3, max_blocks, 2);
      pzskip::down_kernel<3, 4><<<grid, 96, 0, st>>>(p);
      break;
    }
    case 6: {
      dim3 grid(pzskip::NT64 / 6, max_blocks, 2);
      pzskip::down_kernel<6, 4><<<grid, 192, 0, st>>>(p);
      break;
    }
    case 7: {
      dim3 grid(pzskip::NT64 / 7, max_blocks, 2);
      pzskip::down_kernel<7, 4><<<grid, 224, 0, st>>>(p);
      break;
    }
    default:
      return NOT_HANDLED;
  }
  return (int)cudaGetLastError();
}

}  // namespace pzskip
