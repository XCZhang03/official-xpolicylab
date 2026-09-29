#!/bin/bash
# Build the auto-research agent image from the official AgentBundle wheelhouse.
# Usage: scripts/build_official_image.sh [tag=robodojo-official:dev]
set -euo pipefail
tag="${1:-robodojo-official:dev}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${root}/official/xpolicylab/common.sh"
runtime="${OFFICIAL_RUNTIME}"
[[ -f "${runtime}/wheels/constraints.txt" ]] || { echo "Build the wheelhouse first (official/xpolicylab)" >&2; exit 1; }
context="$(mktemp -d -p "${runtime}")"; trap 'rm -rf "${context}"' EXIT
cp "${root}/services/controller/agent_runtime.Dockerfile" "${context}/Dockerfile"
# Always ship the toolkit as it is in this checkout (also installed in the policy env).
sync_toolkit
cp -r "${runtime}/wheels" "${context}/wheelhouse"
# Development-only inspection wheels (usd-core for task_source/ assets); never submitted.
inspect="${runtime}/inspect-wheels"
if ! compgen -G "${inspect}/usd_core-*.whl" >/dev/null; then
    mkdir -p "${inspect}"
    "${runtime}/policy-env/bin/pip" download -q --no-deps --only-binary=:all: --python-version 3.11 \
        --platform manylinux_2_28_x86_64 -d "${inspect}" usd-core==26.8
fi
cp -r "${inspect}" "${context}/inspect-wheels"
# Kernel cache: build it with the release wheels if none exists yet.
cache="${runtime}/warp-cache-release"
if [[ ! -d "${cache}" ]]; then
    WARP_CACHE_PATH="${cache}" "${runtime}/policy-env/bin/python" -c "import robodojo_toolkit as t; t.shared_planner()"
fi
cp -r "${cache}" "${context}/warp-cache"
docker build -t "${tag}" "${context}"
docker run --rm --entrypoint python "${tag}" -c "import robodojo_toolkit as tk, curobo, torch, warp, pxr, trimesh; tk.servo; tk.step_eef; import robodojo_toolkit.service; print('image ready', tk.__version__, torch.__version__)"
