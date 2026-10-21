#!/usr/bin/env bash
# watch.sh -- one fleet snapshot every 30 minutes: runner status, router
# counts, and per-box disk, GPU temperature, containers and serve-log size.
# Under systemd (install-service.sh), output goes to the journal.
#
# Usage: tools/router/watch.sh
#
# Env: ROUTER_REPO  owner/repo (default: `gh repo view` for this checkout)
#      WATCH_HOSTS  required: space-separated ssh hosts to probe
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

# Check the required input before any network call.
read -ra HOSTS <<<"${WATCH_HOSTS:-}"
if [[ ${#HOSTS[@]} -eq 0 ]]; then
  echo "watch.sh: WATCH_HOSTS is required (space-separated ssh hosts to probe)" >&2
  exit 1
fi
ROUTER_REPO="${ROUTER_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
export ROUTER_REPO
R="$ROUTER_REPO"

# Every probe below tolerates its own failure, so one bad snapshot never ends the loop.
while :; do
  echo "=== $(date -u +%FT%TZ)"
  gh api "repos/$R/actions/runners?per_page=100" --jq '.runners[] | "runner \(.name) \(.status) busy=\(.busy)"' \
    || echo "runners: gh api failed"
  python3 tools/router/router.py status || echo "router: status failed"
  for h in "${HOSTS[@]}"; do
    # shellcheck disable=SC2016  # the $(...) below expand on the remote host
    timeout 30 ssh -o ConnectTimeout=10 "$h" 'printf "%s disk=%s temp=%sC containers=%s serve_log=%s\n" "$(hostname -s)" \
      "$(df --output=pcent / | tail -1 | tr -d " ")" \
      "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)" \
      "$(docker ps -q 2>/dev/null | wc -l)" \
      "$(sudo -n du -sh /var/lib/nemotron35/serve-log 2>/dev/null | cut -f1)"' || echo "$h UNREACHABLE"
  done
  sleep 1800
done
