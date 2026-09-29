"""Read-only episode review: every simulator episode of a session and its trajectory video.

The native simulator records each episode's cameras as ``sim/sensors.mp4`` (25 fps,
cam_high | left wrist | right wrist side by side), so review streams those files
directly; nothing is re-encoded and nothing in the session is modified.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from services.robodojo.scoring import episode_score

TIMESTAMP = re.compile(r"\d{8}T\d{6}Z")
SIM_PATTERNS = ("native/results/*/sim", "exploration/episode-*/native/results/*/sim",
                "formal-workers/episode-*/native/results/*/sim")


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _results(directory):
    """Controller records by native episode ID, from the session and its formal workers."""
    rows = {}
    for state in (directory / "state.json", *sorted(directory.glob("formal-workers/episode-*/state.json"))):
        for result in _read(state).get("results", []):
            if result.get("episode_id"):
                rows[result["episode_id"]] = result
    return rows


def _started(sim):
    match = TIMESTAMP.search(sim.parent.name)
    if match:
        return datetime.strptime(match[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    return sim.stat().st_mtime


def episodes(directory):
    """Chronological episodes of one session directory (already validated by Sessions)."""
    directory = Path(directory)
    root = directory.resolve()
    results = _results(directory)
    rows = []
    for pattern in SIM_PATTERNS:
        for sim in directory.glob(pattern):
            if sim.is_symlink() or not sim.resolve().is_relative_to(root):
                continue
            summary, state, reset = (_read(sim / name) for name in ("summary.json", "operator_state.json", "reset.json"))
            episode_id = summary.get("episode_id") or state.get("episode_id") or reset.get("episode_id")
            record = results.get(episode_id, {})
            mode = record.get("mode") or state.get("episode_mode") or "unknown"
            video = sim / "sensors.mp4"
            reward = state.get("reward") or {}
            score = episode_score(sim, episode_id) if summary else {"score": None}
            rows.append({
                "episode": str(sim.relative_to(directory)),
                "episode_id": episode_id,
                "mode": "exploration" if mode == "interactive" else mode,
                # A formal worker records itself as episode 1; its directory has the batch index.
                "formal_episode_index": (int(worker[1]) if (worker := re.match(r"formal-workers/episode-(\d+)/",
                                         str(sim.relative_to(directory)))) else record.get("formal_episode_index")),
                "steps": summary.get("step_id", state.get("step_id")),
                "step_limit": state.get("episode_step_limit"),
                "reason": summary.get("reason") or state.get("reason"),
                "native_success": summary.get("success", reward.get("native_success")),
                "native_score": score.get("score"),
                "task_complete": record.get("task_complete"),
                "bundle": record.get("bundle"),
                "live": not summary,
                "video": video.is_file() and not video.is_symlink(),
                "video_bytes": video.stat().st_size if video.is_file() else 0,
                "started_at": _started(sim),
            })
    return sorted(rows, key=lambda row: (row["started_at"], row["episode"]))


def video_path(directory, episode):
    """The sensors.mp4 of a listed episode; anything else is rejected."""
    directory = Path(directory)
    if (not isinstance(episode, str) or Path(episode).is_absolute()
            or not any(Path(episode).match(p) for p in SIM_PATTERNS)):
        raise ValueError("Unknown episode")
    sim = directory / episode
    if (sim.is_symlink() or ".." in Path(episode).parts
            or not sim.resolve().is_relative_to(directory.resolve())):
        raise ValueError("Unknown episode")
    video = sim / "sensors.mp4"
    if video.is_symlink() or not video.is_file():
        raise FileNotFoundError("No video recorded for this episode")
    return video
