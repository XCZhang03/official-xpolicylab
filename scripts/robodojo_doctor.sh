#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robodojo_env.sh"

expected_robodojo="ee67a1468510da7624a089164402359f2afc72c8"
actual_robodojo="$(git -C "$ROBODOJO_SOURCE" rev-parse HEAD)"
[[ "$actual_robodojo" == "$expected_robodojo" ]] || {
    echo "FAIL: RoboDojo commit is $actual_robodojo, expected $expected_robodojo" >&2
    exit 1
}

for path in \
    "$ROBODOJO_SOURCE/Assets/Robots" \
    "$ROBODOJO_SOURCE/Assets/Object" \
    "$ROBODOJO_SOURCE/Assets/Material" \
    "$ROBODOJO_SOURCE/Assets/Eval_Layout" \
    "$ROBODOJO_PYTHON" \
    "$CODEX_BIN"; do
    [[ -e "$path" ]] || { echo "FAIL: missing $path" >&2; exit 1; }
done

echo "RoboDojo commit: $actual_robodojo"
echo "Python: $($ROBODOJO_PYTHON --version 2>&1)"
"$ROBODOJO_PYTHON" -c 'import importlib.metadata as m; import torch, curobo; print("Isaac Sim:", m.version("isaacsim")); print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda); print("CuRobo:", curobo.__file__)'
echo "Codex: $($CODEX_BIN --version 2>&1 | tail -n 1)"
"$CODEX_BIN" login status
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "PASS: source, assets, simulator environment, CuRobo, Codex, and GPUs are present."
