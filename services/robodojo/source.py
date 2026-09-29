"""Which RoboDojo source tree the native simulator imports.

Sessions run the pristine upstream RoboDojo staged by official/xpolicylab/stage.sh
(the exact code, submodules and asset overlay of the official evaluation), selected
with ROBODOJO_SOURCE_ROOT. Without it, tools fall back to the repository's RoboDojo
checkout.
"""
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_ENV = "ROBODOJO_SOURCE_ROOT"


def source_root(project_root=PROJECT):
    value = os.environ.get(SOURCE_ENV)
    return Path(value) if value else Path(project_root) / "RoboDojo"


def official_stage(project_root=PROJECT):
    """Stage (idempotently) and return the pristine official RoboDojo root."""
    result = subprocess.run(["bash", str(Path(project_root) / "official/xpolicylab/stage.sh")],
                            capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError("Staging the official RoboDojo failed: " + result.stderr[-2000:])
    stage = Path(result.stdout.strip().splitlines()[-1])
    if not (stage / ".staged").is_file() or not (stage / "src/eval_client/eval_env.py").is_file():
        raise RuntimeError(f"Official RoboDojo stage is incomplete: {stage}")
    return stage
