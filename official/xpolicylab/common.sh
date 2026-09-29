# Shared settings for official XPolicyLab tooling. Source; do not execute.
OFFICIAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${OFFICIAL_DIR}/../.." && pwd)"
# Bulky state lives on the SSD runtime, namespaced for this tooling only.
OFFICIAL_RUNTIME="${OFFICIAL_XPL_RUNTIME:-$(readlink -f "${REPO_ROOT}/runtime")/official-xpolicylab}"
POLICY_ENV="${OFFICIAL_POLICY_ENV:-${OFFICIAL_RUNTIME}/policy-env}"
SIM_ENV="${ROBODOJO_SIM_ENV:-$(readlink -f "${REPO_ROOT}/runtime")/envs/robodojo}"
CONDA_ROOT="${CONDA_ROOT:-$(readlink -f "${REPO_ROOT}/runtime")/miniforge3}"
# Read-only source: the shared checkout's git objects (never its working tree).
SOURCE_ROBODOJO="${SOURCE_ROBODOJO:-$(readlink -f "${REPO_ROOT}/RoboDojo")}"
ROBODOJO_COMMIT="$(sed -n 's/^ROBODOJO_COMMIT=//p' "${REPO_ROOT}/dependencies.lock")"
STAGE="${OFFICIAL_RUNTIME}/stage-${ROBODOJO_COMMIT:0:12}"
WHEELHOUSE="${OFFICIAL_RUNTIME}/wheels"

# Rebuild robodojo_toolkit from this checkout into the one wheelhouse shared by the
# agent image and the submission, and install it into the local policy env, so the
# harness, local official evaluation and the downloaded checkpoint run the same code.
sync_toolkit() {
    rm -f "${WHEELHOUSE}"/robodojo_toolkit-*.whl
    "${POLICY_ENV}/bin/pip" wheel -q --no-deps -w "${WHEELHOUSE}" "${REPO_ROOT}/packages/robodojo_toolkit"
    "${POLICY_ENV}/bin/pip" install -q --no-index --no-deps --force-reinstall "${WHEELHOUSE}"/robodojo_toolkit-*.whl
}
