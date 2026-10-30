// pzskip unit: configs 102, 103, 106 and 112 (down_kernel, 64 rows per warp)
#include "pzskip.cuh"

namespace pzskip {

int dispatch_down64(int wpc, const Params& p, cudaStream_t st) {
  const int max_blocks = p.max_blocks;
  switch (wpc) {
    case 102: {  // 2 warps, ring 4, 64 rows per warp
      dim3 grid(pzskip::NT64 / 2, max_blocks, 1);
      pzskip::down_kernel<2, 4, 64><<<grid, 64, 0, st>>>(p);
      break;
    }
    case 103: {
      dim3 grid(pzskip::NT64 / 3, max_blocks, 1);
      pzskip::down_kernel<3, 4, 64><<<grid, 96, 0, st>>>(p);
      break;
    }
    case 106: {
      dim3 grid(pzskip::NT64 / 6, max_blocks, 1);
      pzskip::down_kernel<6, 4, 64><<<grid, 192, 0, st>>>(p);
      break;
    }
    case 112: {  // 2 warps, ring 2, 64 rows
      dim3 grid(pzskip::NT64 / 2, max_blocks, 1);
      pzskip::down_kernel<2, 2, 64><<<grid, 64, 0, st>>>(p);
      break;
    }
    default:
      return NOT_HANDLED;
  }
  return (int)cudaGetLastError();
}

}  // namespace pzskip
