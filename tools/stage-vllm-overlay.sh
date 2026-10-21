#!/usr/bin/env bash
# tools/stage-vllm-overlay.sh -- stage the importable vLLM this tree runs.
#
# The engine is the pinned container's vLLM install with THIS TREE's Python
# source on top. The container supplies what the Python tree cannot: the
# compiled extensions (vllm/_C.abi3.so, _moe_C, the flash-attention build,
# ...) and the files its build generated (_version.py, vllm_flash_attn/).
# The tree supplies every Python and Triton source file, so a participant's
# edit under vllm/vllm/ is what runs.
#
#   .build/vllm-overlay/vllm/  = container site-packages/vllm, then vllm/vllm/
#                                copied over it file by file
#
# Files the tree deleted relative to upstream are NOT deleted from the base, so
# a compiled extension can never be dropped by accident. Rebuilt from scratch
# every run: a stale overlay is the one thing that must never be measured.
#
# STAGE_VLLM_OVERLAY_BASE (TEST SEAM) names a local directory to use as the
# base package instead of the container's install, so the layering is tested
# with no docker. tools/ranked-box-preflight.sh refuses it set.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
die() { printf 'stage-vllm-overlay.sh: %s\n' "$*" >&2; exit 1; }
FIXTURE="${ROOT_DIR}/fixtures/nemotron35_lightning_30b_a3b_track.json"
IMAGE="$(jq -r '.serve_configuration.container_image' "${FIXTURE}")"
DIGEST="$(jq -r '.serve_configuration.container_digest' "${FIXTURE}")"
IMAGE_REF="${IMAGE%%:*}@${DIGEST}"
TAG="$(jq -r '.serve_configuration.engine_tag' "${FIXTURE}")"
OUT="${ROOT_DIR}/.build/vllm-overlay"
[[ -d "${ROOT_DIR}/vllm/vllm" ]] || die "no vendored vLLM source at vllm/vllm"

rm -rf "${OUT}"
mkdir -p "${OUT}"
if [[ -n "${STAGE_VLLM_OVERLAY_BASE:-}" ]]; then
  cp -a "${STAGE_VLLM_OVERLAY_BASE}/vllm" "${OUT}/vllm"
  printf '%s\n' "${TAG#v}" > "${OUT}/.base-version"
else
# Copy the container's installed package out, as the invoking user, with no
# network. `vllm.__file__` locates it without assuming a site-packages path.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${OUT},dst=/out" --env HOME=/tmp \
  --entrypoint bash "${IMAGE_REF}" -c '
    set -euo pipefail
    pkg="$(python3 -c "import importlib.util,os; print(os.path.dirname(importlib.util.find_spec(\"vllm\").origin))")"
    cp -a "${pkg}" /out/vllm
    python3 -c "import vllm; print(vllm.__version__)" > /out/.base-version
  '
fi
base_version="$(cat "${OUT}/.base-version")"
[[ "${base_version}" == "${TAG#v}"* ]] \
  || die "the pinned container carries vLLM ${base_version}, the fixture pins ${TAG}"
# Layer the tree's source over the base.
(cd "${ROOT_DIR}/vllm" && find vllm -type f ! -name '*.pyc' -print0) \
  | (cd "${ROOT_DIR}/vllm" && xargs -0 cp --parents -f -t "${OUT}")
printf '%s\n' "${IMAGE_REF}" > "${OUT}/.base-image"
echo "stage-vllm-overlay.sh: staged ${OUT} (base vLLM ${base_version} from ${IMAGE_REF}, tree source on top)"
