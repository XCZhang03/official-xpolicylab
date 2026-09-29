#!/bin/bash
# Assemble a download-and-run AgentBundle checkpoint: nothing is built on the evaluator.
# Usage: build_release.sh <out_dir> <task>[,<task>...]=<bundle_dir> [...]
#   Official sweeps start one policy server per task with its exact task_name, and the
#   adapter loads <checkpoint>/<task_name>/controller.py. Generalization variants
#   (make_toast vs make_toast_random) are separate tasks: give each its own bundle, or
#   list both names for one bundle (make_toast,make_toast_random=<dir>).
#   <out_dir>/<task>/          frozen bundle per task (controller.py + modules)
#   <out_dir>/TASKS.json       task -> bundle source and sha256
#   <out_dir>/wheelhouse/      pinned wheels + cuRobo + toolkit (install.sh installs --no-index)
#   <out_dir>/warp-cache/      prebuilt portable PTX (sm_80+) cuRobo kernels
#   <out_dir>/SHA256SUMS
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
out="$(readlink -f "$1")"; shift
[[ ! -e "${out}" ]] || { echo "Refusing to overwrite ${out}" >&2; exit 1; }
mkdir -p "${out}"
configs="$(bash "${OFFICIAL_DIR}/stage.sh")/task/RoboDojo/config"
declare -A sources=()
for pair in "$@"; do
    [[ "${pair}" == *=* ]] || { echo "Expected <task>[,<task>]=<bundle_dir>, got ${pair}" >&2; exit 1; }
    bundle="$(readlink -f "${pair#*=}")"
    [[ -f "${bundle}/controller.py" ]] || { echo "No controller.py in ${bundle}" >&2; exit 1; }
    IFS=',' read -ra names <<< "${pair%%=*}"
    for task in "${names[@]}"; do
        [[ -f "${configs}/${task}.yml" ]] || { echo "Unknown official task: ${task}" >&2; exit 1; }
        [[ -z "${sources[${task}]:-}" ]] || { echo "Task ${task} given twice" >&2; exit 1; }
        sources[${task}]="${bundle}"
        cp -r "${bundle}" "${out}/${task}"; find "${out}/${task}" -name __pycache__ -prune -exec rm -rf {} +
    done
done
# A task without a bundle fails at policy-server load and scores zero officially.
for task in "${!sources[@]}"; do
    other="${task%_random}"; [[ "${other}" == "${task}" ]] && other="${task}_random"
    if [[ -f "${configs}/${other}.yml" && -z "${sources[${other}]:-}" ]]; then
        echo "WARNING: ${task} has a bundle but its variant ${other} does not; ${other} would score zero." >&2
    fi
done
for task in "${!sources[@]}"; do printf '%s\t%s\n' "${task}" "${sources[${task}]}"; done | sort | python3 -c "
import hashlib, json, pathlib, sys
rows = {}
for line in sys.stdin:
    task, source = line.rstrip('\n').split('\t')
    root = pathlib.Path(sys.argv[1]) / task
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    rows[task] = {'source': source, 'sha256': digest.hexdigest()}
(pathlib.Path(sys.argv[1]) / 'TASKS.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Tasks:', ', '.join(rows))" "${out}"
# RELEASE_BUNDLES_ONLY=1 checks task routing (bundles and TASKS.json) and stops here.
[[ "${RELEASE_BUNDLES_ONLY:-0}" == 1 ]] && exit 0
sync_toolkit
cp -r "${WHEELHOUSE}" "${out}/wheelhouse"
# The agent image must hold exactly these packages (plus dev-only inspection tools), or
# bundles developed there could import something the evaluator does not install.
python3 "${OFFICIAL_DIR}/check_image_sync.py" "${out}/wheelhouse" "${AGENT_IMAGE:-robodojo-official:dev}"
# Prebuild kernels with the release's own wheels (portable PTX, see robodojo_toolkit.planning).
WARP_CACHE_PATH="${out}/warp-cache" "${POLICY_ENV}/bin/python" -c "import robodojo_toolkit as t; t.shared_planner()"
(cd "${out}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
du -sh "${out}"
