#!/bin/bash
# Official XPolicyLab evaluation of a frozen bundle on pristine upstream RoboDojo.
# Usage: run_eval.sh <sim|debug> <task> <checkpoint> [layout_seed=0] [eval_num=1] [gpu=0]
# <checkpoint> holds controller.py, or one <task_name>/controller.py per task.
# The operator has approved Omniverse EULA acceptance for this tooling.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mode="$1"; task="$2"; bundle="$(readlink -f "$3")"; seed="${4:-0}"; eval_num="${5:-1}"; gpu="${6:-0}"
[[ -f "${bundle}/controller.py" || -f "${bundle}/${task}/controller.py" ]] || { echo "No bundle for ${task} in ${bundle}" >&2; exit 1; }
[[ -x "${POLICY_ENV}/bin/python" ]] || { echo "Run setup_policy_env.sh first" >&2; exit 1; }
sync_toolkit  # Evaluate with the toolkit exactly as it is in this checkout.
bash "${OFFICIAL_DIR}/stage.sh" >/dev/null
export PATH="${POLICY_ENV}/bin:${CONDA_ROOT}/bin:${PATH}"
# Our sim env's CUDA activation hook reads these under `set -u`.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y EVAL_NUM="${eval_num}"
if [[ "${mode}" == debug ]]; then export EVAL_ENV_TYPE=debug; else unset EVAL_ENV_TYPE; fi
cd "${STAGE}/XPolicyLab/policy/AgentBundle"
bash eval.sh RoboDojo "${task}" "${bundle}" arx_x5 joint "${seed}" "${gpu}" "${gpu}" "${POLICY_ENV}" "${SIM_ENV}"
echo "Results: ${STAGE}/eval_result/RoboDojo/${task}/AgentBundle"
