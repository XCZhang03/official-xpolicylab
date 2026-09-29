"""Codex configuration helpers shared by the auto-research launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tomllib
from services.robodojo.timeouts import task_timeouts



def load_key(path=None):
    """Read the operator's private model-provider key file, or OPENROUTER_API_KEY.

    The key stays in the host model relay; it never enters the agent, MCP
    frontend, bundles or trial state.
    """
    if path is None:
        return os.environ.get("OPENROUTER_API_KEY") or None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "r") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("OpenRouter key file must be a private, owner-readable regular file")
        key = stream.read(16385).strip()
    if not key or len(key) > 16384 or any(c.isspace() for c in key):
        raise ValueError("Invalid OpenRouter key file")
    return key


# Keep costly robot runs legible without inheriting the operator's unrelated TUI
# preferences. The thread ID also gives the operator a stable audit identifier
# when several trials are running concurrently.
ROBOT_STATUS_LINE = [
    "model-with-reasoning",
    "run-state",
    "context-remaining",
    "used-tokens",
    "thread-id",
]


def _toml(value):
    """Encode the JSON-shaped subset used by provider and MCP configuration."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(_toml(str(k)) + " = " + _toml(v) for k, v in value.items()) + "}"
    raise ValueError(f"Unsupported provider configuration type: {type(value).__name__}")


def _write_config(path: Path, value: dict) -> None:
    data = "".join(f"{_toml(key)} = {_toml(item)}\n" for key, item in value.items())
    # Validate our encoding before publishing any configuration.
    tomllib.loads(data)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)
