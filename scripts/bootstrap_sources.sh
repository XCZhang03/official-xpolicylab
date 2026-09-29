#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/dependencies.lock"

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

if [[ -n "${ROBODOJO_RUNTIME_ROOT:-}" ]]; then
    runtime_target="$(realpath -m "$ROBODOJO_RUNTIME_ROOT")"
elif [[ -d /mnt/ssd8 && -w /mnt/ssd8 ]]; then
    runtime_target="/mnt/ssd8/${USER:-user}/codex-workspaces/$(basename "$PROJECT_ROOT")/runtime"
else
    runtime_target="$PROJECT_ROOT/.local-runtime"
fi

mkdir -p "$runtime_target"
runtime_target="$(readlink -f "$runtime_target")"
if [[ -L "$PROJECT_ROOT/runtime" ]]; then
    [[ "$(readlink -f "$PROJECT_ROOT/runtime")" == "$(readlink -f "$runtime_target")" ]] || {
        echo "runtime points somewhere else: $PROJECT_ROOT/runtime" >&2
        exit 2
    }
elif [[ -e "$PROJECT_ROOT/runtime" ]]; then
    echo "refusing to replace non-symlink runtime path" >&2
    exit 2
else
    ln -s "$runtime_target" "$PROJECT_ROOT/runtime"
fi

mkdir -p "$runtime_target/agent-trials"

apply_patch_once() {
    local checkout="$1"
    local patch="$2"
    local expected_sha="$3"
    local actual_sha
    actual_sha="$(sha256sum "$patch" | awk '{print $1}')"
    [[ "$actual_sha" == "$expected_sha" ]] || {
        echo "patch checksum mismatch: $patch" >&2
        exit 2
    }
    if git -C "$checkout" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "patch already applied: $patch"
    else
        git -C "$checkout" apply --check "$patch"
        git -C "$checkout" apply "$patch"
    fi
}

if [[ ! -d "$PROJECT_ROOT/RoboDojo/.git" ]]; then
    git clone --recursive "$ROBODOJO_URL" "$PROJECT_ROOT/RoboDojo"
fi
git -C "$PROJECT_ROOT/RoboDojo" checkout --detach "$ROBODOJO_COMMIT"
git -C "$PROJECT_ROOT/RoboDojo" submodule sync --recursive
git -C "$PROJECT_ROOT/RoboDojo" submodule update --init --recursive
apply_patch_once \
    "$PROJECT_ROOT/RoboDojo" \
    "$PROJECT_ROOT/patches/robodojo.patch" \
    "$ROBODOJO_PATCH_SHA256"
apply_patch_once \
    "$PROJECT_ROOT/RoboDojo" \
    "$PROJECT_ROOT/patches/robodojo-pr48-drive-reset.patch" \
    "$ROBODOJO_DRIVE_RESET_PATCH_SHA256"

echo "Sources ready. Install the native environment, then run scripts/robodojo_doctor.sh."
