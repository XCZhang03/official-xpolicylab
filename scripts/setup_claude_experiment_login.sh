#!/usr/bin/env bash
# Set up (or rotate) the dedicated Claude account used by claude-login experiments.
#
# Signs in to a SEPARATE experiment account in its own CLAUDE_CONFIG_DIR (your
# personal ~/.claude login is never touched), issues a long-lived `claude setup-token`
# token and stores it for the relay. Usage:
#   bash scripts/setup_claude_experiment_login.sh [--email experiment-account@example.com]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${ROBODOJO_PYTHON:-$PROJECT_ROOT/runtime/envs/robodojo/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing RoboDojo Python: $PYTHON" >&2; exit 2; }
command -v claude >/dev/null || { echo "Claude Code (claude) is not installed on the host" >&2; exit 2; }
cd "$PROJECT_ROOT"
helper() { PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -m harness.claude_cli.experiment_login "$@"; }

email_args=()
if [[ "${1:-}" == "--email" && -n "${2:-}" ]]; then
    email_args=(--email "$2")
fi

CONFIG_DIR="$(helper paths | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["config_dir"])')"
mkdir -p "$CONFIG_DIR"
chmod 700 "$(dirname "$CONFIG_DIR")" "$CONFIG_DIR"

echo "== 1/3 Sign in to the EXPERIMENT account (not your personal one)"
echo "   Config dir: $CONFIG_DIR"
CLAUDE_CONFIG_DIR="$CONFIG_DIR" claude auth login --claudeai "${email_args[@]}"
CLAUDE_CONFIG_DIR="$CONFIG_DIR" claude auth status --text

echo
echo "== 2/3 Issue a long-lived token for that account"
echo "   Complete the sign-in in a browser window signed in to the EXPERIMENT account"
echo "   (a private window avoids picking your personal session)."
echo "   Copy the token that setup-token prints; you will paste it in step 3."
CLAUDE_CONFIG_DIR="$CONFIG_DIR" claude setup-token

echo
echo "== 3/3 Store and verify the token"
helper install      # Refuses if the experiment account is also your personal login.
helper check
echo
echo "Done. claude-login experiments now use this account; your ~/.claude login is unchanged."
