"""The fast storage tier that must hold session artifacts, capped workspaces and builds.

`ROBODOJO_ARTIFACT_ROOT` selects it explicitly. Otherwise it is `/mnt/ssd8` when that
exists (the original lab layout), else the target of the repository's `runtime` link,
which `scripts/bootstrap_sources.sh` places on the chosen storage.
"""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "ROBODOJO_ARTIFACT_ROOT"


def artifact_root() -> Path:
    configured = os.environ.get(ENVIRONMENT)
    if configured:
        return Path(configured).resolve()
    if Path("/mnt/ssd8").is_dir():
        return Path("/mnt/ssd8")
    return (PROJECT / "runtime").resolve()


def require_artifact_path(path, what="Controller artifacts") -> Path:
    path = Path(path).resolve()
    root = artifact_root()
    if not path.is_relative_to(root):
        raise ValueError(f"{what} must be under {root} (set {ENVIRONMENT} to change it)")
    return path
