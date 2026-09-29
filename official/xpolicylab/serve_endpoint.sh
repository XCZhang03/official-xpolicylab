#!/bin/bash
# Host the agent API that the submitted Mooncake_Agent policy calls.
# Starts one AgentBundle worker per task in the checkpoint (local mode, bound to
# 127.0.0.1, token-protected), then the HTTP endpoint (endpoint/agent_api.py), which
# detects each episode's task and routes it to that task's worker. Put a TLS reverse
# proxy (for example Caddy) in front of the endpoint port for https://.
# Usage: serve_endpoint.sh <checkpoint> [port=8600] [gpus=0] [worker_base_port=9100] [bind=127.0.0.1]
# Environment: MOONCAKE_ENDPOINT_KEYS (client keys, comma-separated, required),
#   OPENROUTER_API_KEY (Gemini for bundles and task detection),
#   ENDPOINT_TASKS=a,b (subset of the checkpoint's tasks), WORKERS_PER_TASK (default 1).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
checkpoint="$(readlink -f "$1")"; port="${2:-8600}"; gpus="${3:-0}"; base="${4:-9100}"; bind="${5:-127.0.0.1}"
[[ -n "${MOONCAKE_ENDPOINT_KEYS:-}" ]] || { echo "Set MOONCAKE_ENDPOINT_KEYS" >&2; exit 1; }
stage="$(bash "${OFFICIAL_DIR}/stage.sh")"
if [[ -n "${ENDPOINT_TASKS:-}" ]]; then IFS=',' read -ra tasks <<< "${ENDPOINT_TASKS}"
else mapfile -t tasks < <(find "${checkpoint}" -mindepth 2 -maxdepth 2 -name controller.py -printf '%h\n' | xargs -n1 basename | sort); fi
[[ ${#tasks[@]} -gt 0 ]] || { echo "No <task>/controller.py bundles in ${checkpoint}" >&2; exit 1; }
IFS=',' read -ra gpu_list <<< "${gpus}"
export AGENTBUNDLE_SERVER_TOKEN="${AGENTBUNDLE_SERVER_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
run="${OFFICIAL_RUNTIME}/endpoint/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "${run}"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT
routes="{}"; index=0
for task in "${tasks[@]}"; do
    for ((copy = 0; copy < ${WORKERS_PER_TASK:-1}; copy++)); do
        worker_port=$((base + index)); gpu="${gpu_list[$((index % ${#gpu_list[@]}))]}"
        bash "${OFFICIAL_DIR}/serve_remote.sh" "${checkpoint}" "${task}" "${worker_port}" "${gpu}" 127.0.0.1 \
            > "${run}/worker-${task}-${copy}.log" 2>&1 &
        pids+=($!)
        routes="$(python3 -c 'import json,sys; r=json.loads(sys.argv[1]); r.setdefault(sys.argv[2], []).append(sys.argv[3]); print(json.dumps(r))' \
            "${routes}" "${task}" "ws://127.0.0.1:${worker_port}")"
        index=$((index + 1))
    done
done
echo "${routes}" > "${run}/workers.json"
echo "Started ${index} workers (logs in ${run}); waiting for them to load"
index=0
for task in "${tasks[@]}"; do
    for ((copy = 0; copy < ${WORKERS_PER_TASK:-1}; copy++)); do
        bash "${stage}/XPolicyLab/utils/wait_for_policy_server.sh" 127.0.0.1 $((base + index)) "${pids[${index}]}" \
            "Worker ${task}" 1200
        index=$((index + 1))
    done
done
cd "${stage}"
PYTHONPATH="${stage}:${stage}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}" "${POLICY_ENV}/bin/python" \
    "${OFFICIAL_DIR}/endpoint/agent_api.py" --workers "${run}/workers.json" --bind "${bind}" --port "${port}" \
    --configs "${stage}/task/RoboDojo/config" --log "${run}/episodes.jsonl"
