#!/bin/bash
# Install the AgentBundle policy environment from the prebuilt wheelhouse only.
# Usage: bash install.sh <policy_env_path> [wheelhouse_dir]
# Nothing is compiled or fetched from a package index; cuRobo ships as a pure-Python
# wheel and compiles its GPU kernels at first use (warmed during policy-server load).
set -euo pipefail
env_path="${1:?policy env path}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheelhouse="${2:-${SCRIPT_DIR}/wheelhouse}"
[[ -f "${wheelhouse}/constraints.txt" ]] || { echo "Missing wheelhouse (run the checkpoint download script first)" >&2; exit 1; }
if [[ ! -x "${env_path}/bin/python" ]]; then
    if command -v conda >/dev/null; then conda create -y -q -p "${env_path}" python=3.11 pip
    else python3.11 -m venv "${env_path}"; fi
fi
# Exact pinned set first (all runtime dependencies), then the two project wheels without
# resolution: cuRobo's metadata lists build-only packages (wheel) not needed at runtime.
"${env_path}/bin/python" -m pip install -q --no-index --find-links "${wheelhouse}" -r "${wheelhouse}/constraints.txt"
"${env_path}/bin/python" -m pip install -q --no-index --no-deps "${wheelhouse}"/nvidia_curobo-*.whl "${wheelhouse}"/robodojo_toolkit-*.whl
"${env_path}/bin/python" -c "import robodojo_toolkit, curobo, torch, warp; assert torch.cuda.is_available(); print('AgentBundle env ready', robodojo_toolkit.__version__, torch.__version__)"
