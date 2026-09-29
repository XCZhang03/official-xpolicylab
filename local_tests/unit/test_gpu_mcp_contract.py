"""Real private worker -> native MCP -> simulator -> published artifacts."""
import json
from pathlib import Path

import pytest

from gpu_fixtures import gpu_runtime
from services.exploration.pool import Worker


@pytest.mark.gpu
@pytest.mark.parametrize("profile", ["rgbd", "rgb-only"])
def test_real_worker_observation_contract(gpu_runtime, profile):
    _, _, gpu, output = gpu_runtime
    worker = Worker({
        "root": str(output / "worker"), "mode": "auto-research",
        "observation_profile": profile, "task": "make_kong", "sim_gpu": gpu,
        "eval_seed": 0, "seed": 0, "episode": 1, "env_id": 0,
        "episode_seconds": 600, "artifact_bytes": 128 * 1024**2,
        "published_root": str(output / "published"),
    })
    try:
        started = worker.request("start", {})
        reply = started["reply"]
        assert not reply.get("isError"), reply
        initial = json.loads(reply["content"][0]["text"])
        assert len([b for b in reply["content"] if b["type"] == "image"]) == (6 if profile == "rgbd" else 3)
        assert ("camera_parameters" in initial) == (profile == "rgbd")
        assert ("depth" in initial) == (profile == "rgbd")
        action = initial["states"]
        result = worker.request("robodojo_step", {"actions": [action, action]})
        assert not result["reply"].get("isError"), result
        packet = json.loads(result["reply"]["content"][0]["text"])
        assert packet["step_id"] == initial["step_id"] + 2
        assert len(packet["transition"]["steps"]) == 2
        manifest = Path(packet["frame_sequence"]["manifest_path"]).relative_to("runtime/autonomous_controller")
        saved = json.loads((output / "published" / manifest).read_text())
        assert len(saved["frames"]) == 2
        for frame in saved["frames"]:
            assert ("camera_parameters" in frame) == (profile == "rgbd")
            assert {f["kind"] for f in frame["files"]} == ({"rgb", "depth"} if profile == "rgbd" else {"rgb"})
            for file in frame["files"]:
                relative = Path(file["path"]).relative_to("runtime/autonomous_controller")
                assert (output / "published" / relative).is_file()
        rejected = worker.request("robodojo_step", {"env_id": 0, "actions": [action]})
        assert rejected["reply"].get("isError")
        # A routing mistake must not poison trustworthy robot state.
        assert not worker.request("robodojo_observe", {})["reply"].get("isError")
        ended = worker.request("finish", {})
        assert ended["state"]["status"] == "closed"
    finally:
        worker.close()
