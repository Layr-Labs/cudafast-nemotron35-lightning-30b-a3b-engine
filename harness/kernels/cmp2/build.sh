#!/usr/bin/env bash
# Driver API allocator only: no arithmetic or model kernels.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$KERNELS_OUT/cmp2
mkdir -p "$out"
cxx=${CXX:-g++}
cuda=${CUDA_HOME:-/usr/local/cuda}
"$cxx" -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC \
 -I"$cuda/include" "$here/csrc/vmm.cpp" -L"$cuda/lib64/stubs" -lcuda \
 -o "$out/libcmp2.so"
cp "$here/py/__init__.py" "$out/__init__.py"
echo "cmp2: built transparent VMM allocator"
