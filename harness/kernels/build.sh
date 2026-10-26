#!/usr/bin/env bash
# Builds every kernel package under harness/kernels/<name>/build.sh into $KERNELS_OUT.
# Runs in the pinned image at setup (no network, checkout read-only); writes only to $KERNELS_OUT.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for sub in "$here"/*/build.sh; do
  test -f "$sub" || continue
  echo "kernels: building $(basename "$(dirname "$sub")")"
  bash "$sub"
done
