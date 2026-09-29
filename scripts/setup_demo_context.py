#!/usr/bin/env python3
"""Cache compact official RoboDojo task videos for visual agent context."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.robodojo.demonstrations import (
    ensure_official_demo,
    ensure_terminal_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="cache one task; repeat for several (default: every task config)",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get("ROBODOJO_RUNTIME_ROOT", PROJECT_ROOT / "runtime")),
    )
    args = parser.parse_args()

    task_root = PROJECT_ROOT / "RoboDojo" / "task" / "RoboDojo" / "config"
    available = {
        path.stem for path in task_root.glob("*.yml") if not path.stem.startswith("_")
    }
    selected = sorted(available if not args.tasks else set(args.tasks))
    unknown = sorted(set(selected) - available)
    if unknown:
        parser.error("unknown RoboDojo task(s): " + ", ".join(unknown))

    cache_root = (
        args.runtime_root.expanduser().resolve() / "reference-demos" / "website"
    )
    total = 0
    terminal_total = 0
    for task in selected:
        result = ensure_official_demo(cache_root, task)
        terminal = ensure_terminal_frame(
            cache_root,
            task,
            video_path=result["path"],
        )
        total += int(result["bytes"])
        terminal_total += int(terminal["bytes"])
        disposition = "cached" if result["cached"] else "downloaded"
        terminal_disposition = "cached" if terminal["cached"] else "extracted"
        print(
            f"{task}: video {disposition} {result['bytes']} bytes; "
            f"terminal {terminal_disposition} {terminal['bytes']} bytes"
        )
    print(
        f"Ready: {len(selected)} exact task clips ({total} bytes) and terminal "
        f"frames ({terminal_total} bytes), {cache_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
