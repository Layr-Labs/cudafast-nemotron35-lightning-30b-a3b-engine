#!/usr/bin/env bash
# Builds every kernel package under harness/kernels/<name>/build.sh into $KERNELS_OUT.
# Runs in the pinned image at setup (no network, checkout read-only); writes only to $KERNELS_OUT.
set -euo pipefail
: "${KERNELS_OUT:?KERNELS_OUT must be set}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
max=${KERNELS_JOBS:-$(nproc 2>/dev/null || echo 4)}
logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT
names=()
for sub in "$here"/*/build.sh; do
  test -f "$sub" || continue
  name=$(basename "$(dirname "$sub")")
  names+=("$name")
  while (( $(jobs -rp | wc -l) >= max )); do wait -n || true; done
  ( set +e; bash "$sub" > "$logs/$name.log" 2>&1; echo $? > "$logs/$name.rc" ) &
done
wait
rc=0
for name in "${names[@]}"; do
  echo "kernels: building $name"
  cat "$logs/$name.log"
  code=$(cat "$logs/$name.rc" 2>/dev/null || echo 1)
  if [[ $code != 0 ]]; then
    echo "kernels: $name failed (exit $code)" >&2
    rc=1
  fi
done
exit $rc
