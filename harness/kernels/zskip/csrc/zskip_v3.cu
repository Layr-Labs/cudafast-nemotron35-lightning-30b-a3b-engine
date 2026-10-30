// zskip unit: the down3 configs (ZSKIP_CONFIGS) and the fused-prep down4 config (ZSKIP4F_CONFIGS),
// which shares build_lists<64> with the 2-warp down3 configs (see the config lists in zskip.cuh).
#include "zskip_launch.cuh"

ZSKIP_CONFIGS(ZSKIP_INST3)
#ifndef ZSKIP_NO_FUSED
ZSKIP4F_CONFIGS(ZSKIP_INST4F)
#endif
