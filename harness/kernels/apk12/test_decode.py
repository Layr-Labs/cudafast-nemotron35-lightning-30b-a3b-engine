# SPDX-License-Identifier: Apache-2.0
"""CPU check of apk12's plane order and decode, no GPU needed.

pack() from py/__init__.py lays out random codec planes (escaped blocks, code-0 weights,
whole code-0 blocks); apk_decode and apk_raw from csrc/apk12_decode.h, compiled for the
host, decode each lane's group at the addresses load_w uses; every register must equal
the bits the record's Triton decode (nemotron_h_attn_packed_kernel._packed_gemm) gives
at the m16n8k16 .col B-fragment positions (K = 2t + (i & 1) + 8 (i >= 2), N = lane / 4).
Run: python3 -I test_decode.py (needs torch and a C++ compiler).
"""
import ctypes
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WRAP = r"""
#include "apk12_decode.h"
extern "C" void decode(const uint32_t* s, const uint32_t* x, uint32_t b, uint32_t* w) {
  apk_decode(*reinterpret_cast<const uint32_t(*)[4]>(s), *reinterpret_cast<const uint32_t(*)[2]>(x),
             b, *reinterpret_cast<uint32_t(*)[8]>(w));
}
extern "C" void raw(const uint16_t* blk, int half, int t, uint32_t* w) {
  apk_raw(blk, half, t, *reinterpret_cast<uint32_t(*)[8]>(w));
}
extern "C" int group_batches(int groups, int width) {
  return apk_group_batches(groups, width);
}
extern "C" int group_index(int batch, int offset, int width) {
  return apk_group_index(batch, offset, width);
}
"""


def host_lib(tmp: Path):
    (tmp / "wrap.cpp").write_text(WRAP)
    subprocess.run(["c++", "-O1", "-std=c++17", "-shared", "-fPIC", "-I", str(HERE / "csrc"),
                    "-o", str(tmp / "dec.so"), str(tmp / "wrap.cpp")], check=True)
    lib = ctypes.CDLL(str(tmp / "dec.so"))
    lib.decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    lib.raw.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.group_batches.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.group_batches.restype = ctypes.c_int
    lib.group_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.group_index.restype = ctypes.c_int
    return lib


def planes(n: int, k: int, gen: torch.Generator):
    nb = n * k // 128
    sm = torch.randint(0, 256, (n * k,), dtype=torch.uint8, generator=gen)
    ex = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, generator=gen)
    zero = torch.rand(n * k // 2, generator=gen) < 0.1
    ex[zero] &= 0xF0
    for b in range(1, nb, 7):
        ex[b * 64:(b + 1) * 64] = 0
    base = torch.randint(1, 241, (nb,), dtype=torch.uint8, generator=gen)
    esc = torch.rand(nb, generator=gen) < 0.05
    esc[0] = esc[-1] = True
    count = int(esc.sum())
    ri = torch.full((nb,), -1, dtype=torch.int32)
    ri[esc] = torch.randperm(count, generator=gen).to(torch.int32)
    raw = torch.randint(-32768, 32768, (count * 128,), dtype=torch.int16, generator=gen)
    return sm, ex, base, ri, raw


def reference(sm, ex, base, ri, raw, flat: torch.Tensor) -> torch.Tensor:
    """_packed_gemm's decode (nemotron_h_attn_packed_kernel.py:28-44) at flat = n * K + k."""
    blk = flat // 128
    rid = ri[blk].long()
    bs = base[blk].long()
    s = sm[flat].long()
    nib = ex[flat >> 1].long()
    code = (nib >> ((flat & 1) * 4)) & 15
    e = torch.where(code == 0, 0, bs + code - 1)
    bits = ((s & 128) << 8) | (e << 7) | (s & 127)
    rw = raw.long()[rid.clamp(min=0) * 128 + flat % 128] & 65535
    return torch.where(rid >= 0, rw, bits)


def run(lib, apk12, n: int, k: int, tiles, seed: int) -> tuple[int, int]:
    gen = torch.Generator().manual_seed(seed)
    sm, ex, base, ri, raw = planes(n, k, gen)
    rows = [t * 32 + c for t in tiles for c in range(32)]
    flat = (torch.tensor(rows)[:, None] * k + torch.arange(k)).view(-1)
    ref = dict(zip(rows, reference(sm, ex, base, ri, raw, flat).view(len(rows), k).tolist()))
    sm4, ex2, meta, ri2, raw2 = apk12.pack((sm, ex, base, ri, raw), n, k)
    assert sm4.numel() == n * k and ex2.numel() == n * k // 2 and meta.numel() == n * k // 128
    sp, xp, rawp = sm4.data_ptr(), ex2.data_ptr(), raw2.data_ptr()
    gn, kb = k // 64, k // 128
    w = (ctypes.c_uint32 * 8)()
    regs = bad = 0
    for tile in tiles:
        for g in range(gn):
            for warp in range(4):
                for lane in range(32):
                    i = ((tile * gn + g) * 4 + warp) * 32 + lane
                    b = int(meta[(tile * kb + g // 2) * 32 + warp * 8 + lane // 4])
                    lib.decode(sp + 16 * i, xp + 8 * i, b, w)
                    col = tile * 32 + warp * 8 + lane // 4
                    if b == 0:
                        rid = int(ri2[col * kb + g // 2])
                        lib.raw(rawp + 2 * 128 * rid, g & 1, lane % 4, w)
                    for j in range(4):
                        for r in range(2):
                            kk = [g * 64 + 16 * j + 2 * (lane % 4) + (e & 1) + 8 * (e >= 2)
                                  for e in (2 * r, 2 * r + 1)]
                            want = ref[col][kk[0]] | (ref[col][kk[1]] << 16)
                            regs += 1
                            bad += w[2 * j + r] != want
    return regs, bad


def main() -> int:
    spec = importlib.util.spec_from_file_location("apk12", HERE / "py" / "__init__.py")
    apk12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apk12)
    with tempfile.TemporaryDirectory() as tmp:
        lib = host_lib(Path(tmp))
        total = 0
        for k in (2688, 4096):
            groups = k // 64
            for width in apk12.PIPE_GROUPS:
                batches = lib.group_batches(groups, width)
                order = [lib.group_index(batch, offset, width)
                         for batch in range(batches) for offset in range(width)]
                valid = [group for group in order if group < groups]
                assert valid == list(range(groups)), (k, width, valid)
                print(f"K={k} pipeline={width}: {len(valid)} groups in ascending order")
        for n, k, tiles, seed in ((96, 384, range(3), 1), (64, 256, range(2), 2),
                                  (4608, 2688, (0, 71, 143), 3), (2688, 4096, (0, 41, 83), 4)):
            regs, bad = run(lib, apk12, n, k, tiles, seed)
            print(f"N={n} K={k} tiles={list(tiles)}: {regs} registers, {bad} differ")
            total += bad
    print("decode test", "PASS" if total == 0 else "FAIL")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
