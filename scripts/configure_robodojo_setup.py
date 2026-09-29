#!/usr/bin/env python3
"""Write the next operator-owned RoboDojo agent environment setup."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.robodojo.demonstrations import DEMONSTRATION_CONTEXTS

TRIAL_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def write_setup(
    output: Path,
    *,
    task: str,
    seed: int,
    eval_seed: int = 0,
    sim_gpu: str | int = "0",
    sim_port: int = 0,
    startup_timeout: float = 900,
    include_depth: bool,
    include_camera_parameters: bool,
    enforce_step_limit: bool,
    max_exploration_episodes: int = 3,
    demonstration_context: str = "terminal_state",
    trial_id: str | None = None,
) -> dict[str, object]:
    if trial_id is not None and not TRIAL_ID.fullmatch(trial_id):
        raise ValueError("trial_id contains invalid characters")
    if demonstration_context not in DEMONSTRATION_CONTEXTS:
        raise ValueError(
            "demonstration_context must be one of: " + ", ".join(DEMONSTRATION_CONTEXTS)
        )
    if (isinstance(max_exploration_episodes, bool)
            or not isinstance(max_exploration_episodes, int)
            or not 1 <= max_exploration_episodes <= 100):
        raise ValueError("max_exploration_episodes must be in [1, 100]")
    revision = (
        f"operator-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    setup: dict[str, object] = {
        "revision": revision,
        "task": task,
        "seed": seed,
        "eval_seed": eval_seed,
        "sim_gpu": str(sim_gpu),
        "sim_port": sim_port,
        "startup_timeout": startup_timeout,
        "include_depth": include_depth,
        "include_camera_parameters": include_camera_parameters,
        "enforce_step_limit": enforce_step_limit,
        "max_exploration_episodes": max_exploration_episodes,
        "demonstration_context": demonstration_context,
    }
    if trial_id is not None:
        setup["trial_id"] = trial_id
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(setup, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, output)
        output.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"setup_path": str(output), **setup}


def main() -> int:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--trial-id",
        help="trusted trial identifier when writing a trial-scoped setup",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--sim-gpu", default="0")
    parser.add_argument(
        "--sim-port",
        type=int,
        default=0,
        help="RPC port; 0 asks the simulator to allocate a free loopback port",
    )
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--max-exploration-episodes", type=int, default=3)
    parser.add_argument(
        "--include-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-camera-parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enforce-step-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="legacy mode selector: true starts formal, false exploration; both enforce the native limit",
    )
    parser.add_argument(
        "--demonstration-context",
        choices=DEMONSTRATION_CONTEXTS,
        default="terminal_state",
        help="agent visual context: none, final image, or ordered completion images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runtime" / "operator" / "robodojo_setup.json",
    )
    args = parser.parse_args()
    explicit = {
        "--include-depth": args.include_depth,
        "--include-camera-parameters": args.include_camera_parameters,
        "--enforce-step-limit": args.enforce_step_limit,
    }
    missing = [name for name, value in explicit.items() if value is None]
    if missing:
        parser.error("choose each environment option explicitly: " + ", ".join(missing))
    task_path = root / "RoboDojo" / "task" / "RoboDojo" / "config" / f"{args.task}.yml"
    if not task_path.is_file():
        parser.error(f"unknown RoboDojo task: {args.task}")
    if not 0 <= args.sim_port <= 65535:
        parser.error("--sim-port must be in [0, 65535]")
    if not 30 <= args.startup_timeout <= 1800:
        parser.error("--startup-timeout must be in [30, 1800]")

    setup = write_setup(
        args.output,
        task=args.task,
        seed=args.seed,
        eval_seed=args.eval_seed,
        sim_gpu=args.sim_gpu,
        sim_port=args.sim_port,
        startup_timeout=args.startup_timeout,
        include_depth=args.include_depth,
        include_camera_parameters=args.include_camera_parameters,
        enforce_step_limit=args.enforce_step_limit,
        max_exploration_episodes=args.max_exploration_episodes,
        demonstration_context=args.demonstration_context,
        trial_id=args.trial_id,
    )
    print(json.dumps(setup, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
