#!/usr/bin/env bash
# Shared native RoboDojo/Isaac Sim environment. Source this file from bash.

ROBODOJO_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_RUNTIME_ROOT="${ROBODOJO_RUNTIME_ROOT:-$ROBODOJO_PROJECT_ROOT/runtime}"
ROBODOJO_ENV_ROOT="${ROBODOJO_ENV_ROOT:-$ROBODOJO_RUNTIME_ROOT/envs/robodojo}"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-$ROBODOJO_PROJECT_ROOT/RoboDojo}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-$ROBODOJO_ENV_ROOT/bin/python}"
if [[ -z "${CODEX_BIN:-}" ]]; then
    if command -v codex >/dev/null 2>&1; then
        CODEX_BIN="$(command -v codex)"
    else
        CODEX_BIN="$ROBODOJO_RUNTIME_ROOT/codex-0.153.4/node_modules/.bin/codex"
    fi
fi

export ROBODOJO_PROJECT_ROOT ROBODOJO_RUNTIME_ROOT ROBODOJO_ENV_ROOT
export ROBODOJO_SOURCE ROBODOJO_PYTHON CODEX_BIN
export CUDA_HOME="${CUDA_HOME:-$ROBODOJO_ENV_ROOT}"
export PATH="$ROBODOJO_ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export PYTHONPATH="$ROBODOJO_SOURCE:$ROBODOJO_SOURCE/XPolicyLab:$ROBODOJO_SOURCE/third_party/curobo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROBODOJO_RUNTIME_ROOT/huggingface-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROBODOJO_RUNTIME_ROOT/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROBODOJO_RUNTIME_ROOT/xdg-cache}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$ROBODOJO_RUNTIME_ROOT/xdg-runtime}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

# Native host graphics configuration. The upstream cluster wrapper instead
# injects a private driver userspace; this workstation already has matching
# 580.173.02 NVIDIA libraries installed system-wide.
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
fi
if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
fi

mkdir -p "$XDG_RUNTIME_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR"
