"""Private durable state and race-resistant, content-addressed bundles."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import shutil
import uuid

AGENT_TEMPLATE = Path(__file__).resolve().parents[2] / 'auto_research_agent'
RUNTIME_SOURCE = AGENT_TEMPLATE / 'api/runtime.py'
# The official XPolicyLab adapter's bridge: isolated runs execute bundles through it.
BRIDGE_SOURCE = Path(__file__).resolve().parents[2] / 'official/xpolicylab/AgentBundle/bundle_bridge.py'
DEPLOY_CONFIG = BRIDGE_SOURCE.with_name('deploy.yml')


def official_action_wait_seconds():
    """deploy.yml action_wait_s: how long get_action waits before holding the pose one step."""
    import re
    match = re.search(r'^action_wait_s:\s*([0-9.]+)', DEPLOY_CONFIG.read_text(), re.MULTILINE)
    return float(match.group(1)) if match else 100.0


def project_directory(workspace_path):
    """Allow only a canonical project path; never shadow runtime/API/output mounts."""
    if not isinstance(workspace_path, str) or any(c in workspace_path for c in '\\,\n\r\x00'):
        raise ValueError('Invalid project path')
    relative = PurePosixPath(workspace_path)
    if (relative.is_absolute() or len(relative.parts) < 2 or relative.parts[0] != 'code'
            or '..' in relative.parts or str(relative) != workspace_path):
        raise ValueError('Project must be a directory under /workspace/code/')
    return PurePosixPath('/workspace') / relative


def runtime_digest():
    return hashlib.sha256(RUNTIME_SOURCE.read_bytes() + BRIDGE_SOURCE.read_bytes()).hexdigest()


def persist(path: Path, value):
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def directory_fd(path: Path):
    """Pin every ancestor without following symlinks, including the root."""
    path = Path(os.path.abspath(path))
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def snapshot(source: Path, destination: Path, *, image: str, max_bytes: int, entrypoint=None,
             workspace_path=None):
    """Copy regular files only, then hash copied bytes, not mutable source paths.

    Destination is supervisor-private and never mounted writable. No deserialization
    or execution happens here. A failed partial snapshot is never registered.
    """
    if workspace_path is not None:
        project_directory(workspace_path)
    destination.mkdir(mode=0o755)
    entries = {}
    total = 0

    def visit(fd, output, prefix=""):
        nonlocal total
        for name in sorted(os.listdir(fd)):
            if len(entries) >= 10000:
                raise ValueError("Too many bundle files")
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            relative = f"{prefix}{name}"
            if stat.S_ISDIR(info.st_mode):
                if relative.count("/") > 16:
                    raise ValueError("Bundle nesting too deep")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                (output / name).mkdir(mode=0o755)
                try:
                    visit(child, output / name, relative + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(child, "rb") as src:
                    actual = os.fstat(src.fileno())
                    if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1:
                        raise ValueError("Bundle files must be unlinked regular files")
                    digest = hashlib.sha256()
                    with (output / name).open("xb") as dst:
                        while block := src.read(min(1024 * 1024, max_bytes - total + 1)):
                            total += len(block)
                            if total > max_bytes:
                                raise ValueError("Bundle exceeds artifact budget")
                            dst.write(block)
                            digest.update(block)
                    os.chmod(output / name, 0o444)
                    entries[relative] = digest.hexdigest()
            else:
                raise ValueError("Links and special files are forbidden in bundles")

    try:
        with directory_fd(source) as fd:
            if entrypoint is not None:
                entry = os.stat(entrypoint, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISREG(entry.st_mode):
                    raise ValueError("Entrypoint must be a regular file")
            visit(fd, destination)
    except BaseException:
        # Only this freshly created, supervisor-private directory is removed.
        # It contains our copies, never user files or followed symlinks.
        shutil.rmtree(destination)
        raise
    manifest = {"image": image, "files": entries, "bytes": total, "runtime_sha256": runtime_digest()}
    if workspace_path is not None:
        manifest['workspace_path'] = workspace_path
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def verify(bundle: Path, manifest):
    if manifest["runtime_sha256"] != runtime_digest():
        raise ValueError("Container SDK changed; register and rehearse a new bundle")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("Bundle file set changed")
    for name, expected in manifest["files"].items():
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe bundle")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Bundle changed after submission")
