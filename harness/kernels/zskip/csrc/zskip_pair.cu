// zskip unit: the paired decode configs 113 / 114 and, with the fused prep, 118 / 119 (zskip_pair.cuh)
#include "zskip_launch.cuh"

#ifndef ZSKIP_NO_PAIR
ZSKIP_INST_PAIR(false, false)
ZSKIP_INST_PAIR(false, true)
#ifndef ZSKIP_NO_FUSED
ZSKIP_INST_PAIR(true, false)
ZSKIP_INST_PAIR(true, true)
#endif
#endif
