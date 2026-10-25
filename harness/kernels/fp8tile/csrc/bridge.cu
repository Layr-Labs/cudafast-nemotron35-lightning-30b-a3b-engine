// SPDX-License-Identifier: Apache-2.0
// One host entry, two disjoint output-tile grids. Numerical chains stay local
// to their original 6/7-bit codecs. No arithmetic is split between grids.
#include <cuda_runtime.h>
#define RUN_DECL(name) extern "C" int name(const void*,const void*,const void*,const void*,void*,int,int,int,int,int,int,const void*,int,void*)
RUN_DECL(fp8t6_run);
RUN_DECL(fp8t7_run);
extern "C" int fp8t6_init();
extern "C" int fp8t7_init();
extern "C" int fp8t6_has_pdl();
extern "C" int fp8t7_has_pdl();
extern "C" int fp8tile_init() {
 int rc=fp8t6_init();return rc?rc:fp8t7_init();
}
extern "C" int fp8tile_has_pdl() {
 return fp8t6_has_pdl()&&fp8t7_has_pdl();
}
extern "C" int fp8tile_run(const void*a,const void*w6,const void*w7,
 const void*sa,const void*sb,void*out,const void*ids6,const void*ids7,
 int m,int n,int k,int stride6,int stride7,int count6,int count7,
 int pdl,int ring6,int ring7,void*stream) {
 if(count6<0||count7<0||count6+count7!=n/32)return (int)cudaErrorInvalidValue;
 // Both grids have the same PDL policy. The second waits for the first before
 // mutable input loads; its completion also orders the first's output stores.
 int rc=fp8t6_run(a,w6,sa,sb,out,m,n,k,stride6,pdl,ring6,ids6,count6,stream);
 if(rc)return rc;
 return fp8t7_run(a,w7,sa,sb,out,m,n,k,stride7,pdl,ring7,ids7,count7,stream);
}
