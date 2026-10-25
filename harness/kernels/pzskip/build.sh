#!/usr/bin/env bash
# pzskip: libpzskip.so (ctypes, static cudart) plus its Python package, into $KERNELS_OUT/pzskip.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/pzskip
mkdir -p "$out"
nvcc=$(command -v nvcc || true)
test -n "$nvcc" || nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
test -x "$nvcc" || { echo "pzskip: nvcc not found" >&2; exit 1; }
# TORCH_CUDA_ARCH_LIST=12.1a -> sm_121a
arch=${TORCH_CUDA_ARCH_LIST:-12.1a}
arch=${arch%%;*}
arch=${arch// /}
arch=${arch%+PTX}
sm=${arch/./}
rm -f "$out/libpzskip.so"  # the fallbacks below test for this build's output, not a stale one
# P2 SASS guard. ptxas (12.8 and 13.0, sm_120a and sm_121a) put the cache-policy descriptor of
# some hinted P2 copies on an odd uniform register (desc[UR1]); a descriptor is 64-bit and needs an
# even UR pair, so that LDGSTS traps (illegal instruction) and the load-time check kills the
# process. A build that contains P2 is kept only if no p2_kernel instruction takes its descriptor
# from an odd UR.
# p2_sass_ok: 0 = clean, 1 = odd descriptor found, 2 = SASS unavailable (no cuobjdump, cuobjdump
# -sass failed: CUDA 13's needs nvdisasm, or no p2_kernel in its output).
cuobjdump=$(command -v cuobjdump || true)
test -n "$cuobjdump" || cuobjdump=$(dirname "$nvcc")/cuobjdump
p2_sass_ok() {
  test -x "$cuobjdump" || { echo "pzskip: cuobjdump not found" >&2; return 2; }
  local sass
  sass=$("$cuobjdump" -sass "$out/libpzskip.so" 2> /dev/null) || { echo "pzskip: cuobjdump -sass failed (nvdisasm missing?)" >&2; return 2; }
  awk '/Function :/ { f = $NF; nf += (f ~ /p2_kernel/); next }
       f ~ /p2_kernel/ && /desc\[UR[0-9]*[13579]\]/ { bad++; print "pzskip: odd descriptor UR: " f ": " $0 > "/dev/stderr" }
       END { exit (nf == 0 ? 2 : (bad > 0 ? 1 : 0)) }' <<< "$sass"
}
# Without SASS the guard cannot check a build. The csrc/pzskip.cu with this sha256 was checked
# clean (no odd descriptor in any p2_kernel, sm_121a and sm_120a, this command line) with nvcc
# V13.0.88, the pinned image's compiler, and ptxas output is deterministic: that exact source with
# that nvcc is kept; anything else (an edited source, another nvcc) gets no P2.
verified_nvcc=V13.0.88
verified_src_sha256=8e1df7b5d444942f432fb4e54de676dfb06a85ce5d63c380e606277e533d75ce
src_sha256=$(sha256sum "$here/csrc/pzskip.cu" 2> /dev/null | cut -d ' ' -f 1 || true)
skip_p2=0
# Full build first. If it fails, or its P2 kernels fail the SASS guard: P2 without the L2 hint
# (-DPZSKIP_P2_NO_HINT), then without the P2 configs 336-338 (-DPZSKIP_NO_P2: the previous source;
# the Python side then keeps config 333), then without the L2 cache policy
# (-DPZSKIP_NO_CACHE_POLICY: stock PZ333), then without both.
for flags in "" "-DPZSKIP_P2_NO_HINT" "-DPZSKIP_NO_P2" "-DPZSKIP_NO_CACHE_POLICY" "-DPZSKIP_NO_P2 -DPZSKIP_NO_CACHE_POLICY"; do
  if (( skip_p2 )); then case "$flags" in *PZSKIP_NO_P2*) ;; *) continue ;; esac; fi
  test -n "$flags" && echo "pzskip: previous build failed or was rejected; rebuilding with $flags" >&2
  # shellcheck disable=SC2086
  "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC $flags \
    -gencode "arch=compute_${sm},code=sm_${sm}" \
    -o "$out/libpzskip.so" "$here/csrc/pzskip.cu" 2>&1 | grep -v '^$' || true
  test -f "$out/libpzskip.so" || continue
  case "$flags" in *PZSKIP_NO_P2*) break ;; esac
  rc=0
  p2_sass_ok || rc=$?
  (( rc == 0 )) && break
  if (( rc == 2 )); then
    if [[ "$src_sha256" == "$verified_src_sha256" ]] && "$nvcc" --version | grep -qF "$verified_nvcc"; then
      echo "pzskip: P2 SASS not checkable here; kept: this source was checked clean with nvcc $verified_nvcc" >&2
      break
    fi
    echo "pzskip: P2 SASS not checkable, and not the source + nvcc $verified_nvcc checked clean; building without P2" >&2
    skip_p2=1
  else
    echo "pzskip: P2 SASS guard rejected the build${flags:+ with $flags}" >&2
  fi
  rm -f "$out/libpzskip.so"
done
test -f "$out/libpzskip.so" || { echo "pzskip: build failed" >&2; exit 1; }
cp "$here/py/__init__.py" "$out/__init__.py"
echo "pzskip: built $out/libpzskip.so for sm_${sm}"
