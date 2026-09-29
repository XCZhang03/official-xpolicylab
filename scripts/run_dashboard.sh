#!/usr/bin/env bash
# Start the trusted, loopback-only auto-research operator dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${ROBODOJO_PYTHON:-$PROJECT_ROOT/runtime/envs/robodojo/bin/python}"

[[ -x "$PYTHON" ]] || { echo "Missing RoboDojo Python: $PYTHON" >&2; exit 2; }
cd "$PROJECT_ROOT"
exec "$PYTHON" -m services.dashboard.server --project-root "$PROJECT_ROOT" "$@"
