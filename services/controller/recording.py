"""Publish inspectable run files, not a new artifact API or manifest service.

Only this sanitized subtree is suitable for an agent's read-only file-view grant.
The private supervisor/native directories must never be exposed with it.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

from services.artifact_io import write_artifacts
from services.robodojo.mcp_server import _TASK_EVALUATION_KEYS
from .gateway import PRIVATE
from .storage import directory_fd

IMAGE_NAME = re.compile(r"(?:frame_[0-9]{6}|initial)_cam_(?:high|left_wrist|right_wrist)_(?:rgb|depth)\.png")


def read_regular(path, maximum):
    with directory_fd(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError("Unsafe or oversized recorded file")
            data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise ValueError("Recorded file exceeds quota")
            return data


class RunRecording:
    def __init__(self, root, run_id, maximum, *, logical_prefix="runtime/autonomous_controller", observation_profile="rgbd"):
        if not re.fullmatch(r"[a-z]+-[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run recording ID")
        from services.mcp_contract import ObservationProfile
        self.observation = ObservationProfile(observation_profile)
        self.root = root / "runs" / run_id
        self.logical = PurePosixPath(logical_prefix) / 'runs' / run_id
        self.run_id, self.maximum, self.used = run_id, maximum, 0
        self.calls = 0
        self.errors = []
        self.sequences = {}
        # Published manifest path -> {(kind, camera): logical path} of its last frame.
        self.sequence_final_frames = {}
        self.start_step = self.end_step = self.episode_id = None
        self.paths = {"directory": str(self.logical), "mcp_trace": str(self.logical / "mcp.jsonl"),
                      "frames": str(self.logical / "frames")}
        if not run_id.startswith('interactive-'):
            self.paths.update({"code": str(self.logical / "code"), "exports": str(self.logical / "exports"),
                      "stdout": str(self.logical / "stdout.log"), "stderr": str(self.logical / "stderr.log"),
                      "log_details": str(self.logical / "logs.json")})
        write_artifacts(self.root, {"mcp.jsonl": b""})
        # Only sanitized current-run frames are mounted into robot scripts.
        self.frames = self.root / 'frames'
        self.frames.mkdir(mode=0o755)

    def _charge(self, size):
        if size > self.maximum - self.used:
            raise RuntimeError("Published recording quota exhausted")
        self.used += size

    def _new_files(self, directory, files):
        self._charge(sum(len(data) for data in files.values()))
        write_artifacts(directory, files)

    def _write(self, name, data):
        self._charge(len(data))
        with directory_fd(self.root) as fd:
            out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(out, "wb") as stream:
                stream.write(data)

    def trace(self, event):
        raw = (json.dumps({"time": time.time(), **event}, allow_nan=False) + "\n").encode()
        self._charge(len(raw))
        with directory_fd(self.root) as fd:
            out = os.open("mcp.jsonl", os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(out, "ab") as stream:
                stream.write(raw)
                stream.flush()

    def event(self, event):
        """Trace stays searchable; exact image-bearing responses get sidecars."""
        if event.get("event") != "mcp_result":
            self.trace(event)
            return
        self.calls += 1
        call_dir = PurePosixPath("calls") / f"{self.calls:06d}"
        result = event["result"]
        content = result.get("content", [])
        value = json.loads(content[0]["text"]) if content and content[0].get("type") == "text" else {}
        images = {}
        image_paths = []
        # A step reply's images are its sequence's last frame, already published once.
        final_frames = self.sequence_final_frames.get((value.get("frame_sequence") or {}).get("manifest_path"), {})
        cameras = {a.get("content_index"): (a.get("kind", "rgb"), a.get("camera"))
                   for a in value.get("attachments", []) if isinstance(a, dict)}
        stored = []
        for index, block in enumerate(content):
            if block.get("type") != "image":
                stored.append(block)
                continue
            path = final_frames.get(cameras.get(index))
            if path is None:
                name = f"image_{index:02d}.png"
                images[name] = base64.b64decode(block["data"], validate=True)
                path = str(self.logical / call_dir / name)
            image_paths.append({"content_index": index, "path": path})
            # The response record references image files instead of repeating base64 data.
            stored.append({"type": "image", "mimeType": block.get("mimeType"), "path": path})
        response_name = "response.json"
        images[response_name] = json.dumps({**result, "content": stored}, allow_nan=False).encode()
        self._new_files(self.root / call_dir, images)
        if value.get("step_id") is not None and event.get("tool", "").startswith("robodojo_"):
            if self.start_step is None:
                self.start_step = value["step_id"]
            self.end_step, self.episode_id = value["step_id"], value.get("episode_id")
            self.paths["final_response"] = str(self.logical / call_dir / response_name)
            if image_paths or "final_images" not in self.paths:
                self.paths["final_images"] = image_paths
        self.trace({**{k: v for k, v in event.items() if k != "result"},
                    "response_path": str(self.logical / call_dir / response_name),
                    "returned": value, "images": image_paths})

    def sequence(self, value, workspace):
        """Copy ONLY the existing safe frame sequence, never a simulator run dir."""
        sequence = value.get("frame_sequence")
        if not sequence or not sequence.get("manifest_path"):
            return None
        path = PurePosixPath(sequence["manifest_path"])
        if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("runtime", "frames") or path.name != "manifest.json":
            raise ValueError("Invalid frame sequence location")
        if str(path) in self.sequences:
            return self.sequences[str(path)]
        source = workspace.joinpath(*path.parts)
        original = json.loads(read_regular(source, min(self.maximum - self.used, 32 * 1024 * 1024)))
        original = self.observation.clean(original)
        destination = PurePosixPath("frames") / f"sequence_{len(self.sequences):06d}"
        logical = self.logical / destination
        files = {}
        records = list(original.get("frames", []))
        if original.get("initial_observation"):
            records.append(original["initial_observation"])
        for record in records:
            for item in record.get("files", []):
                filename = PurePosixPath(item["path"])
                if filename.parent != path.parent or not IMAGE_NAME.fullmatch(filename.name):
                    raise ValueError("Frame file is outside the approved sequence")
                if filename.name not in files:
                    remaining = self.maximum - self.used - sum(map(len, files.values()))
                    files[filename.name] = read_regular(source.parent / filename.name, max(0, remaining))

        def clean(item):
            if isinstance(item, dict):
                result = {}
                for key, child in item.items():
                    if key in PRIVATE or key.lower() in _TASK_EVALUATION_KEYS:
                        continue
                    if any(x in key for x in ("path", "directory", "root", "artifact_storage")):
                        # Keep only paths/patterns rebased from this exact sequence.
                        if not isinstance(child, str) or not child.startswith(str(path.parent) + "/"):
                            continue
                        tail = child[len(str(path.parent)) + 1:]
                        if "/" in tail or "\\" in tail or tail in {".", ".."}:
                            raise ValueError("Invalid sequence filename/pattern")
                        result[key] = str(logical / tail)
                    else:
                        result[key] = clean(child)
                return result
            if isinstance(item, list):
                return [clean(x) for x in item]
            return item
        # Existing sequence metadata is retained; there is no new manifest API.
        manifest = clean(original)
        files["manifest.json"] = json.dumps(manifest, allow_nan=False).encode()
        if manifest.get("frames"):
            self.sequence_final_frames[str(logical / "manifest.json")] = {
                (item.get("kind", "rgb"), item.get("camera")): item["path"]
                for item in manifest["frames"][-1].get("files", []) if item.get("path")}
        self._new_files(self.root / destination, files)
        # The container's unprivileged UID can read, but the bind mount is read-only.
        for filename in files:
            (self.root / destination / filename).chmod(0o444)
        (self.root / destination).chmod(0o755)
        published = self.observation.clean(clean(sequence))
        self.sequences[str(path)] = published
        return published

    def finish_interactive(self, summary):
        """Close a shared exploration trace; local source/logs stay in its workspace."""
        info = {'episode_id': self.episode_id, 'start_step_id': self.start_step,
                'end_step_id': self.end_step, 'artifacts': self.paths,
                'recording_errors': list(self.errors)}
        try:
            self._write('result.json', json.dumps({**summary, **info}, allow_nan=False).encode())
        except Exception as exc:
            info['recording_errors'].append(f'result: {type(exc).__name__}')
        return info

    def finish(self, bundle, output, summary):
        errors = list(self.errors)
        # Snapshot and outputs are copied regular files, never private-storage links.
        sources = [("code", bundle), ("exports", output / "files")]
        for label, source in sources:
            try:
                if not source.exists():
                    continue
                for current, directories, names in os.walk(source, followlinks=False):
                    current = Path(current)
                    for directory in directories:
                        if (current / directory).is_symlink():
                            raise ValueError("Linked artifact directory")
                    files = {}
                    for name in names:
                        files[name] = read_regular(current / name, max(0, self.maximum - self.used - sum(map(len, files.values()))))
                    self._new_files(self.root / label / current.relative_to(source), files)
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}")
        for name in ("stdout.log", "stderr.log", "logs.json"):
            try:
                self._write(name, read_regular(output / name, max(0, self.maximum - self.used)))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
        info = {"run_id": self.run_id, "episode_id": self.episode_id,
                "start_step_id": self.start_step, "end_step_id": self.end_step,
                "artifacts": self.paths, "recording_errors": errors}
        try:
            self._write("result.json", json.dumps({**summary, **info}, allow_nan=False).encode())
        except Exception as exc:
            errors.append(f"result: {type(exc).__name__}")
        return info
