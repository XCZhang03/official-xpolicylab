#!/usr/bin/env bash
# Trusted operator launcher; no arbitrary Codex configuration/command passthrough.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
exec "$PROJECT_ROOT/runtime/envs/robodojo/bin/python" -m harness.codex_cli.auto_research "$@"
