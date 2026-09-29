#!/bin/bash
# Create the isolated policy-server environment (XPolicyLab runtime dependencies only).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ ! -x "${POLICY_ENV}/bin/python" ]]; then
    "${CONDA_ROOT}/bin/conda" create -y -q -p "${POLICY_ENV}" python=3.11 pip
fi
# Mirrors XPolicyLab pyproject dependencies; Pillow encodes bridge images.
"${POLICY_ENV}/bin/pip" install -q "numpy>=1.23" "pyyaml>=6" "opencv-python-headless>=4.8" "h5py>=3.8" \
    "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" pillow
"${POLICY_ENV}/bin/python" -c "import websockets.asyncio, msgpack_numpy, yaml, PIL; print('policy env ready')"
# Motion toolkit planning stack: the exact versions RoboDojo's simulator image builds
# (torch 2.7.0+cu128, warp-lang 1.11.0, cuRobo from the pinned third_party submodule).
if [[ "${WITH_PLANNING:-1}" == 1 ]]; then
    "${POLICY_ENV}/bin/pip" install -q "numpy==1.26.0" "scipy==1.15.3" transforms3d "opencv-python-headless>=4.8,<4.11"
    "${POLICY_ENV}/bin/pip" install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    "${POLICY_ENV}/bin/pip" install -q "warp-lang==1.11.0"
    # Non-editable build from a private copy: never writes into the shared submodule.
    src="$(mktemp -d)/curobo"; cp -a "${SOURCE_ROBODOJO}/third_party/curobo" "${src}"
    "${POLICY_ENV}/bin/pip" install -q "${src}[cu12]" --no-build-isolation
    rm -rf "$(dirname "${src}")"
    "${POLICY_ENV}/bin/python" -c "import torch, warp, curobo; print('planning stack ready', torch.__version__, warp.__version__)"
fi
# The motion toolkit itself (repository source package).
"${POLICY_ENV}/bin/pip" install -q "${REPO_ROOT}/packages/robodojo_toolkit"
