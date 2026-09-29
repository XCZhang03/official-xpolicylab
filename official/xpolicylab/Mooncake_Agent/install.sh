#!/bin/bash
set -euo pipefail

# No model weights or GPU packages: the policy only needs XPolicyLab itself
# (msgpack, msgpack-numpy, numpy, opencv come with it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}"
python -c "import cv2, msgpack, msgpack_numpy, numpy; print('Mooncake_Agent env ready')"
