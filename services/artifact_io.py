"""Publish new artifacts without following agent-controlled filesystem links."""

from __future__ import annotations

import os
from pathlib import Path


def write_artifacts(directory: Path, files: dict[str, bytes]) -> None:
    """Create a fresh directory and files using pinned, no-follow directory FDs.

    All ancestors are walked from / without resolving symlinks. The trusted
    caller must supply a canonical absolute root (not a workspace runtime link).
    An agent can rename an opened directory within its writable area, but cannot
    redirect subsequent opens to a different directory or existing host file.
    """
    directory = Path(directory)
    if not directory.is_absolute() or ".." in directory.parts:
        raise ValueError("Artifact directory must be absolute without '..'")
    for name in files:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Artifact filenames must be single path components")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for index, component in enumerate(directory.parts[1:], start=1):
            try:
                os.mkdir(component, mode=0o700, dir_fd=fd)
            except FileExistsError:
                if index == len(directory.parts) - 1:
                    raise
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        for name, data in files.items():
            output = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=fd,
            )
            with os.fdopen(output, "wb") as stream:
                stream.write(data)
    finally:
        os.close(fd)
