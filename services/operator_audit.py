"""Private per-trial request accounting shared by trusted services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def empty_request_counts() -> dict[str, int]:
    return {
        "initial_setup_requests": 0,
        "exploration_reset_requests": 0,
        "formal_episode_requests": 0,
        "reset_requests": 0,
        "evaluation_requests": 0,
    }


def request_counts(event_path: Path) -> dict[str, int]:
    """Summarize complete request events; ignore malformed trailing records."""

    counts = empty_request_counts()
    try:
        lines = event_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return counts
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "evaluation_request":
            counts["evaluation_requests"] += 1
            continue
        if event.get("kind") != "setup_request":
            continue
        if event.get("request_kind") == "formal_episode":
            counts["formal_episode_requests"] += 1
            counts["reset_requests"] += 1
        elif event.get("active_episode") is True:
            counts["exploration_reset_requests"] += 1
            counts["reset_requests"] += 1
        else:
            counts["initial_setup_requests"] += 1
    return counts


def append_event(event_path: Path, value: dict[str, Any]) -> None:
    """Durably append one mode-0600 JSONL event."""

    event_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        event_path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    event_path.chmod(0o600)
