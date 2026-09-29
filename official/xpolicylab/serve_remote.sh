#!/bin/bash
# Serve a checkpoint as a remote XPolicyLab policy server for official evaluation.
# The RoboDojo online evaluation client connects to ws://<this host>:<port>.
# Official sweeps use one policy server per task (task_name selects the bundle),
# so start one server, on its own port, for each task you submit.
# Usage: serve_remote.sh <checkpoint> <task> [port=9999] [gpu=0] [bind_host=0.0.0.0]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
checkpoint="$(readlink -f "$1")"; task="$2"; port="${3:-9999}"; gpu="${4:-0}"; bind="${5:-0.0.0.0}"
[[ -f "${checkpoint}/${task}/controller.py" || -f "${checkpoint}/controller.py" ]] || { echo "No bundle for ${task} in ${checkpoint}" >&2; exit 1; }
[[ -x "${POLICY_ENV}/bin/python" ]] || { echo "Run setup_policy_env.sh first (or install.sh from the checkpoint)" >&2; exit 1; }
stage="$(bash "${OFFICIAL_DIR}/stage.sh")"
export PATH="${POLICY_ENV}/bin:${CONDA_ROOT}/bin:${PATH}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
cd "${stage}"
exec bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/AgentBundle --task "${task}" \
    --ckpt "${checkpoint}" --policy-env "${POLICY_ENV}" --policy-port "${port}" --bind-host "${bind}" \
    --action-type joint --policy-gpu "${gpu}"
