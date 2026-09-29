#!/bin/bash
# Build the offline wheelhouse shared by the agent image and the release checkpoint.
# Usage: build_wheelhouse.sh [--constraints-only] [--force]
#   Run setup_policy_env.sh first. The pinned set is the policy environment itself:
#   constraints.txt = its `pip freeze --all` (without pip, cuRobo and the toolkit) plus
#   packaging, and the wheelhouse holds exactly those wheels, the cuRobo wheel built
#   from RoboDojo's pinned third_party/curobo and the robodojo_toolkit wheel.
#   --constraints-only prints the constraints without downloading (to compare with an
#   existing wheelhouse). --force replaces an existing wheelhouse.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
constraints_only=0; force=0
for arg in "$@"; do
    case "${arg}" in
        --constraints-only) constraints_only=1 ;;
        --force) force=1 ;;
        *) echo "usage: $0 [--constraints-only] [--force]" >&2; exit 2 ;;
    esac
done
PIP="${POLICY_ENV}/bin/pip"
[[ -x "${PIP}" ]] || { echo "Run official/xpolicylab/setup_policy_env.sh first" >&2; exit 1; }
constraints() {
    "${PIP}" freeze --all | grep -viE '^(pip|nvidia[-_]curobo|robodojo[-_]toolkit)==|^-e |@ ' | sort -f
    "${PIP}" freeze --all | grep -qiE '^packaging==' || \
        echo "packaging==$("${POLICY_ENV}/bin/python" -c 'import packaging; print(packaging.__version__)')"
}
if [[ "${constraints_only}" == 1 ]]; then constraints; exit 0; fi
if [[ -e "${WHEELHOUSE}" && "${force}" != 1 ]]; then
    echo "${WHEELHOUSE} exists; pass --force to rebuild it" >&2; exit 1
fi
build="$(mktemp -d "$(dirname "${WHEELHOUSE}")/wheels.build.XXXXXX")"
trap 'rm -rf "${build}"' EXIT
constraints | sort -f > "${build}/constraints.txt"
"${PIP}" download -q --no-deps --only-binary=:all: -d "${build}" -r "${build}/constraints.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128
# cuRobo: a pure-Python wheel from a private copy of the pinned submodule.
src="$(mktemp -d)/curobo"; cp -a "${SOURCE_ROBODOJO}/third_party/curobo" "${src}"
"${PIP}" wheel -q --no-deps --no-build-isolation -w "${build}" "${src}"
rm -rf "$(dirname "${src}")"
"${PIP}" wheel -q --no-deps -w "${build}" "${REPO_ROOT}/packages/robodojo_toolkit"
count="$(ls "${build}"/*.whl | wc -l)"
rm -rf "${WHEELHOUSE}.old"; [[ -e "${WHEELHOUSE}" ]] && mv "${WHEELHOUSE}" "${WHEELHOUSE}.old"
mv "${build}" "${WHEELHOUSE}"; trap - EXIT; rm -rf "${WHEELHOUSE}.old"
echo "Wheelhouse ${WHEELHOUSE}: ${count} wheels, $(du -sh "${WHEELHOUSE}" | cut -f1)"
