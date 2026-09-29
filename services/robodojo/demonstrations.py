"""Trusted visual-demonstration download and trial provisioning.

The official task clips contain pixels only.  This module never reads or
copies RoboDojo dataset actions, states, calibration, or evaluator outputs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

DEMONSTRATION_CONTEXTS = (
    "none",
    "terminal_state",
    "completion_sequence",
)
OFFICIAL_DEMO_BASE_URL = "https://robodojo-benchmark.com/doc/videos/tasks"
MAX_DEMO_BYTES = 64 * 1024 * 1024
TASK_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class DemonstrationError(ValueError):
    """A selected visual demonstration cannot be safely supplied."""


def _validate_task(task: str) -> str:
    task = str(task).strip()
    if not TASK_NAME.fullmatch(task):
        raise DemonstrationError("Invalid RoboDojo task name for demonstration")
    return task


def official_demo_url(task: str) -> str:
    """Return the official compact task-video URL for a config name."""

    slug = _validate_task(task).replace("_", "-").lower()
    return f"{OFFICIAL_DEMO_BASE_URL}/{slug}.mp4"


def cached_demo_path(cache_root: Path, task: str) -> Path:
    """Return a traversal-safe cache path for one task clip."""

    cache_root = cache_root.expanduser().resolve()
    path = (cache_root / f"{_validate_task(task)}.mp4").resolve()
    try:
        path.relative_to(cache_root)
    except ValueError as exc:
        raise DemonstrationError("Demonstration cache path escaped its root") from exc
    return path


def cached_terminal_path(cache_root: Path, task: str) -> Path:
    """Return the trusted cached terminal-frame path for one task."""

    cache_root = cache_root.expanduser().resolve()
    path = (cache_root / "terminal" / f"{_validate_task(task)}.jpg").resolve()
    try:
        path.relative_to(cache_root)
    except ValueError as exc:
        raise DemonstrationError("Terminal-frame cache path escaped its root") from exc
    return path


def _validate_mp4(path: Path) -> int:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(12)
    except OSError as exc:
        raise DemonstrationError(f"Cannot read demonstration clip: {path}") from exc
    if not 12 <= size <= MAX_DEMO_BYTES or header[4:8] != b"ftyp":
        raise DemonstrationError("Official demonstration response is not a valid MP4")
    return size


def _validate_jpeg(path: Path) -> int:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(2)
            stream.seek(-2, os.SEEK_END)
            footer = stream.read(2)
    except OSError as exc:
        raise DemonstrationError(f"Cannot read terminal demo image: {path}") from exc
    if (
        not 4 <= size <= 16 * 1024 * 1024
        or header != b"\xff\xd8"
        or footer != b"\xff\xd9"
    ):
        raise DemonstrationError("Cached terminal demonstration is not a valid JPEG")
    return size


def ensure_official_demo(
    cache_root: Path,
    task: str,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Cache exactly one official website clip for ``task`` atomically."""

    path = cached_demo_path(cache_root, task)
    url = official_demo_url(task)
    parsed_url = urllib.parse.urlsplit(url)
    if parsed_url.scheme != "https" or parsed_url.netloc != "robodojo-benchmark.com":
        raise DemonstrationError("Official demonstration URL is not trusted HTTPS")
    if path.is_file():
        return {"path": path, "url": url, "bytes": _validate_mp4(path), "cached": True}

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RoboDojo-Operator/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            content_length = response.headers.get("Content-Length")
            if content_type != "video/mp4":
                raise DemonstrationError(
                    f"Official demonstration returned {content_type}, not video/mp4"
                )
            if content_length is not None and int(content_length) > MAX_DEMO_BYTES:
                raise DemonstrationError(
                    "Official demonstration exceeds the size limit"
                )
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            total = 0
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DEMO_BYTES:
                        raise DemonstrationError(
                            "Official demonstration exceeds the size limit"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        size = _validate_mp4(temporary)
        os.replace(temporary, path)
        path.chmod(0o600)
        return {"path": path, "url": url, "bytes": size, "cached": False}
    except DemonstrationError:
        raise
    except Exception as exc:
        raise DemonstrationError(
            f"Could not download the official demonstration for {task}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_terminal_frame(
    cache_root: Path,
    task: str,
    *,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Extract and cache the final decodable frame of one official task clip."""

    path = cached_terminal_path(cache_root, task)
    if path.is_file():
        return {"path": path, "bytes": _validate_jpeg(path), "cached": True}
    if video_path is None:
        video_path = cached_demo_path(cache_root, task)
    video_path = video_path.expanduser().resolve()
    _validate_mp4(video_path)

    try:
        import cv2
    except ImportError as exc:
        raise DemonstrationError(
            "The RoboDojo environment lacks OpenCV video support"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise DemonstrationError("OpenCV could not decode the demonstration clip")
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
    finally:
        capture.release()
    if last_frame is None:
        raise DemonstrationError("The official demonstration contains no frames")

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.jpg")
    try:
        if not cv2.imwrite(
            str(temporary),
            last_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        ):
            raise DemonstrationError(
                "Could not encode the terminal demonstration image"
            )
        size = _validate_jpeg(temporary)
        os.replace(temporary, path)
        path.chmod(0o600)
        return {"path": path, "bytes": size, "cached": False}
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_directory(staging: Path, target: Path) -> None:
    """Replace the operator-owned trial directory without exposing partial data."""

    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.backup")
    moved_old = False
    try:
        if os.path.lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise DemonstrationError(
                    "Trial demonstration path must be a real directory"
                )
            os.replace(target, backup)
            moved_old = True
        os.replace(staging, target)
    except Exception:
        if moved_old and not os.path.lexists(target):
            os.replace(backup, target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def provision_trial_demonstration(
    *,
    video_path: Path | None,
    terminal_path: Path | None = None,
    target: Path,
    task: str,
    context_kind: str,
) -> dict[str, Any]:
    """Expose only selected demo images inside the current trial runtime."""

    task = _validate_task(task)
    if context_kind not in DEMONSTRATION_CONTEXTS:
        raise DemonstrationError(
            "demonstration_context must be one of: " + ", ".join(DEMONSTRATION_CONTEXTS)
        )
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = target.parent.resolve() / target.name
    # The runtime is agent-writable. Decode in its trusted parent (the trial
    # root), so the agent cannot swap staging files for links while we write.
    staging = target.parent.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    try:
        if context_kind == "none":
            result = {
                "kind": "none",
                "image_count": 0,
                "root": "runtime/demonstrations",
                "manifest_path": None,
            }
            _replace_directory(staging, target)
            return result

        if video_path is None:
            raise DemonstrationError("Selected demonstration clip is not installed")
        video_path = video_path.expanduser().resolve()
        _validate_mp4(video_path)

        if context_kind == "terminal_state" and terminal_path is not None:
            terminal_path = terminal_path.expanduser().resolve()
            _validate_jpeg(terminal_path)
            destination = staging / "terminal.jpg"
            shutil.copy2(terminal_path, destination)
            destination.chmod(0o600)
            ordered_images = ["terminal.jpg"]
            decoded = 1
        else:
            ordered_images = []
            decoded = 0

        capture = None
        if not ordered_images:
            try:
                import cv2
            except ImportError as exc:
                raise DemonstrationError(
                    "The RoboDojo environment lacks OpenCV video support"
                ) from exc
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise DemonstrationError(
                    "OpenCV could not decode the demonstration clip"
                )
        try:
            if context_kind == "completion_sequence" and capture is not None:
                frames = staging / "frames"
                frames.mkdir(mode=0o700)
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    relative = f"frames/frame_{decoded:06d}.jpg"
                    destination = staging / relative
                    if not cv2.imwrite(
                        str(destination),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    ):
                        raise DemonstrationError(
                            "Could not encode a demonstration image"
                        )
                    destination.chmod(0o600)
                    ordered_images.append(relative)
                    decoded += 1
            elif capture is not None:
                last_frame = None
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    last_frame = frame
                    decoded += 1
                if last_frame is not None:
                    destination = staging / "terminal.jpg"
                    if not cv2.imwrite(
                        str(destination),
                        last_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                    ):
                        raise DemonstrationError(
                            "Could not encode the terminal demonstration image"
                        )
                    destination.chmod(0o600)
                    ordered_images.append("terminal.jpg")
        finally:
            if capture is not None:
                capture.release()
        if decoded == 0 or not ordered_images:
            raise DemonstrationError("The official demonstration contains no frames")

        manifest = {
            "schema_version": 1,
            "task": task,
            "context_kind": context_kind,
            "demonstrator": "robot",
            "source": "official_robodojo_task_visualization",
            "image_count": len(ordered_images),
            "ordered_images": ordered_images,
            "contains": ["rgb_images"],
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        manifest_path.chmod(0o600)
        _replace_directory(staging, target)
        return {
            "kind": context_kind,
            "image_count": len(ordered_images),
            "root": "runtime/demonstrations",
            "manifest_path": "runtime/demonstrations/manifest.json",
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
