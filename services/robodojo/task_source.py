"""Inject the packed official task source (scripts/build_task_sources.py) into a workspace.

The package is reference material for the development agent only: never mounted into
isolated rehearsal or formal runs, and never part of a submitted bundle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[2]
PACKAGES = PROJECT / "runtime/task-sources"
SCHEMA = "robodojo_task_source_v1"
BUILD = "runtime/envs/usd-tools/bin/python scripts/build_task_sources.py --task {task}"


def pinned_commit():
    for line in (PROJECT / "dependencies.lock").read_text().splitlines():
        if line.startswith("ROBODOJO_COMMIT="):
            return line.split("=", 1)[1].strip()
    raise ValueError("dependencies.lock has no ROBODOJO_COMMIT")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_package(task, packages=PACKAGES):
    """The package directory for ``task`` after checking its manifest, commit and hashes."""
    package = Path(packages) / task
    manifest_path = package / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No task source package for {task}; build it with: {BUILD.format(task=task)}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("task") != task:
        raise ValueError(f"Task source package {package} is not a {SCHEMA} package for {task}")
    if manifest.get("robodojo_commit") != pinned_commit():
        raise ValueError(f"Task source package for {task} was built from RoboDojo {manifest.get('robodojo_commit')}, "
                         f"not the pinned {pinned_commit()}; rebuild it with: {BUILD.format(task=task)}")
    present = {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    if present != set(manifest["files"]):
        raise ValueError(f"Task source package for {task} does not match its manifest; rebuild it")
    for relative, digest in manifest["files"].items():
        if _sha(package / relative) != digest:
            raise ValueError(f"Task source file {relative} for {task} changed after packing; rebuild it")
    return package, manifest


def provision_task_source(task, workspace, packages=PACKAGES):
    """Copy the verified package to <workspace>/task_source, read-only; return its manifest."""
    package, manifest = verified_package(task, packages)
    destination = Path(workspace) / "task_source"
    shutil.copytree(package, destination)
    for directory, _, files in os.walk(destination, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        os.chmod(directory, 0o555)
    return manifest
