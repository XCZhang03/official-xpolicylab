#!/usr/bin/env bash
# Download RoboDojo's simulator assets into the SSD-backed cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robodojo_env.sh"

cache_root="$ROBODOJO_RUNTIME_ROOT/robodojo-cache"
mkdir -p "$cache_root"

if [[ ! -e "$ROBODOJO_SOURCE/.cache" ]]; then
    ln -s "$cache_root" "$ROBODOJO_SOURCE/.cache"
elif [[ ! -L "$ROBODOJO_SOURCE/.cache" ]]; then
    echo "Refusing to relocate non-symlink $ROBODOJO_SOURCE/.cache" >&2
    exit 2
fi

export HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
export HF_REVISION="${HF_REVISION:-main}"
export PATH="$ROBODOJO_ENV_ROOT/bin:$PATH"
exec bash "$ROBODOJO_SOURCE/scripts/init_assets.sh"
