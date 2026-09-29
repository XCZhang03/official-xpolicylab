#!/bin/bash
# Stage pristine upstream RoboDojo + XPolicyLab at the pinned commits, plus this adapter.
# Nothing in the shared checkout is modified: sources come from `git archive`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
xpl_commit="$(git -C "${SOURCE_ROBODOJO}" ls-tree "${ROBODOJO_COMMIT}" XPolicyLab | awk '{print $3}')"
[[ -n "${xpl_commit}" ]] || { echo "XPolicyLab submodule pin not found" >&2; exit 1; }
if [[ ! -f "${STAGE}/.staged" ]]; then
    rm -rf "${STAGE}.tmp" && mkdir -p "${STAGE}.tmp/XPolicyLab" "${STAGE}.tmp/third_party"
    git -C "${SOURCE_ROBODOJO}" archive "${ROBODOJO_COMMIT}" | tar -x -C "${STAGE}.tmp"
    git -C "${SOURCE_ROBODOJO}/XPolicyLab" archive "${xpl_commit}" | tar -x -C "${STAGE}.tmp/XPolicyLab"
    # Large read-only trees; both submodules are verified clean at their pins.
    for sub in IsaacLab curobo; do
        pin="$(git -C "${SOURCE_ROBODOJO}" ls-tree "${ROBODOJO_COMMIT}" "third_party/${sub}" | awk '{print $3}')"
        [[ "$(git -C "${SOURCE_ROBODOJO}/third_party/${sub}" rev-parse HEAD)" == "${pin}" ]] || { echo "${sub} not at pin" >&2; exit 1; }
        [[ -z "$(git -C "${SOURCE_ROBODOJO}/third_party/${sub}" status --short --untracked-files=no)" ]] || { echo "${sub} modified" >&2; exit 1; }
        rm -rf "${STAGE}.tmp/third_party/${sub}"
        ln -s "${SOURCE_ROBODOJO}/third_party/${sub}" "${STAGE}.tmp/third_party/${sub}"
    done
    # Assets: symlink overlay of the published Hugging Face bundle. Upstream code reads
    # Robots/x5/curobo.yml, but the public assets ship only its template curobo_tmp.yml
    # (same schema, ${ASSETS_PATH} placeholders meaning the RoboDojo root). Materialize
    # it by substitution alone; RoboDojo code stays exactly upstream.
    assets="$(readlink -f "${SOURCE_ROBODOJO}/Assets")"; overlay="${STAGE}.tmp/Assets"
    rm -rf "${overlay}"; mkdir -p "${overlay}/Robots/x5"
    for entry in "${assets}"/*; do [[ "$(basename "${entry}")" == Robots ]] || ln -s "${entry}" "${overlay}/"; done
    for entry in "${assets}"/Robots/*; do [[ "$(basename "${entry}")" == x5 ]] || ln -s "${entry}" "${overlay}/Robots/"; done
    for entry in "${assets}"/Robots/x5/*; do ln -s "${entry}" "${overlay}/Robots/x5/"; done
    if [[ ! -e "${assets}/Robots/x5/curobo.yml" ]]; then
        sed "s#\${ASSETS_PATH}#${STAGE}#g" "${assets}/Robots/x5/curobo_tmp.yml" > "${overlay}/Robots/x5/curobo.yml"
        echo "asset=Robots/x5/curobo.yml = upstream utils/update_embodiment_config_path.py substitution of curobo_tmp.yml (\${ASSETS_PATH} -> RoboDojo root), x5 only" > "${STAGE}.tmp/.shims"
    fi
    printf 'robodojo=%s\nxpolicylab=%s\n' "${ROBODOJO_COMMIT}" "${xpl_commit}" > "${STAGE}.tmp/.staged"
    mv "${STAGE}.tmp" "${STAGE}"
fi
ln -sfn "${OFFICIAL_DIR}/AgentBundle" "${STAGE}/XPolicyLab/policy/AgentBundle"
ln -sfn "${OFFICIAL_DIR}/Mooncake_Agent" "${STAGE}/XPolicyLab/policy/Mooncake_Agent"
echo "${STAGE}"
