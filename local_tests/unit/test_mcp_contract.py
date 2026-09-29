"""Capabilities must agree across discovery, dispatch, files and workspace."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from services.mcp_contract import Contract, MODES, ObservationProfile
from services.mcp_workspace import compose, render
from services.controller.gateway import Gateway
from services.controller.recording import RunRecording
from services.robodojo.mcp_server import RoboDojoMCP

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("profile", ["rgbd", "rgb-only"])
@pytest.mark.parametrize("count", [1, 2])
def test_mode_matrix(mode, profile, count):
    if mode == "teacher-student" and count > 1:
        with pytest.raises(ValueError, match="Multiple environments"):
            Contract(mode, observation_profile=profile, environments=count)
        return
    c = Contract(mode, observation_profile=profile, environments=count)
    definitions = c.robot_definitions()
    assert {t["name"] for t in definitions} == c.robot_names
    for t in definitions:
        expected = count > 1 and t["name"] != "robodojo_pose_math"
        assert ("env_id" in t["inputSchema"].get("properties", {})) == expected
        assert ("env_id" in t["inputSchema"].get("required", [])) == expected
    args = {"actions": [[0.0] * 14]}
    if count > 1:
        with pytest.raises(ValueError, match="env_id"):
            c.require_robot("robodojo_step", args)
        c.require_robot("robodojo_step", {**args, "env_id": count - 1})
        for bad in (True, -1, count, "0", None):
            with pytest.raises(ValueError, match="env_id"):
                c.require_robot("robodojo_step", {**args, "env_id": bad})
    else:
        c.require_robot("robodojo_step", args)
        with pytest.raises(ValueError, match="env_id"):
            c.require_robot("robodojo_step", {**args, "env_id": 0})
    with pytest.raises(ValueError, match="env_id"):
        c.require_robot("robodojo_pose_math", {"env_id": 0})
    if profile == "rgb-only":
        with pytest.raises(PermissionError):
            c.require_robot("robodojo_pixel_to_position", {})


def test_isolated_gateway_discovery_and_forged_calls():
    c = Contract("auto-research", phase="isolated", observation_profile="official")
    calls = []
    backend = SimpleNamespace(
        tools=c.robot_definitions,
        call=lambda name, args: (calls.append((name, args)) or {"step_id": 1}, []))
    gateway = Gateway(backend, contract=c, audit=lambda *_: None)
    names = {t["name"] for t in gateway.definitions()}
    assert names == {"robodojo_observe", "robodojo_status", "robodojo_step", "robodojo_step_ee", "gemini_generate"}
    token = gateway.acquire()
    for name, args in [("robodojo_pixel_to_position", {}), ("robodojo_pose_math", {}),
                       ("gemini_generate", {"messages": []}),  # No provider configured here.
                       ("robodojo_step", {"env_id": 0, "actions": [[0] * 14]}),
                       ("start_episode", {})]:
        response = gateway.handle(token, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args}})
        assert "error" in response or response["result"].get("isError")
    assert calls == [] and not gateway.control_uncertain


def frame():
    return {"cam_high": np.zeros((3, 4, 3), dtype=np.uint8),
            "cam_high_depth_m": np.ones((3, 4), dtype=np.float32),
            "camera_parameters": {"cam_high": {"intrinsic_matrix": [[1]]}},
            "states": np.zeros(14), "eef_positions": np.zeros((2, 3)),
            "eef_quaternions_wxyz": np.zeros((2, 4))}


@pytest.mark.parametrize("profile,images", [("rgbd", 2), ("rgb-only", 1)])
def test_native_images_and_saved_files(tmp_path, profile, images):
    server = RoboDojoMCP.__new__(RoboDojoMCP)
    server.observation_profile = profile
    server.sim = SimpleNamespace(episode_id="episode", step_id=2, task="make_kong",
                                 run_id="run", metadata={"control_frequency_hz": 25})
    server.frame_root = tmp_path / "runtime/frames"
    packet, content = server._observation_sequence_packet([frame(), frame()],
        transition={"steps": [{"step_id": 1}, {"step_id": 2}]})
    assert len(content) == images
    assert [a["content_index"] for a in packet["attachments"]] == list(range(1, images + 1))
    manifest_path = tmp_path / packet["frame_sequence"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    assert ("camera_parameters" in manifest["frames"][0]) == (profile == "rgbd")
    assert ("depth" in packet) == (profile == "rgbd")
    assert len(list(manifest_path.parent.glob("*.png"))) == 2 * images


def test_rgb_recording_filters_before_copying_depth(tmp_path):
    native = tmp_path / "native"
    folder = native / "runtime/frames/seq"
    folder.mkdir(parents=True)
    rgb = "frame_000000_cam_high_rgb.png"
    (folder / rgb).write_bytes(b"rgb")
    # Deliberately absent depth: publication must not even try to read it.
    manifest = {"frames": [{"camera_parameters": {"private": True}, "depth": {"cam_high": {}},
        "files": [{"kind": kind, "path": "runtime/frames/seq/" + name}
                  for kind, name in [("rgb", rgb), ("depth", "frame_000000_cam_high_depth.png")]]}]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    recording = RunRecording(tmp_path / "published", "interactive-" + "a" * 32,
                             100000, observation_profile="rgb-only")
    value = {"frame_sequence": {"manifest_path": "runtime/frames/seq/manifest.json",
             "streams": [{"kind": "rgb"}, {"kind": "depth"}]}}
    published = recording.sequence(value, native)
    assert published["streams"] == [{"kind": "rgb"}]
    exported = next(recording.root.rglob("manifest.json"))
    assert "depth" not in exported.read_text()
    assert "camera_parameters" not in exported.read_text()
    assert len(list(recording.root.rglob("*.png"))) == 1


@pytest.mark.parametrize("count", [1, 2])
def test_workspace_composition_and_references(tmp_path, count):
    c = Contract("auto-research", observation_profile="official", environments=count)
    target = tmp_path / "workspace"
    shutil.copytree(PROJECT / MODES["auto-research"].workspace, target,
                    ignore=shutil.ignore_patterns("runtime", "__pycache__"))
    compose(target, c, task="make_kong")
    assert json.loads((target / "MCP_CONTRACT.json").read_text()) == c.manifest()
    calibration = target / ".agents/skills/rgb-position-calibration"
    assert (calibration / "SKILL.md").is_file() and (calibration / "CALIBRATION.md").is_file()
    assert "(CALIBRATION.md)" in (calibration / "SKILL.md").read_text()
    assert "(SKILL.md)" in (calibration / "CALIBRATION.md").read_text()
    assert not (target / "CALIBRATION.md").exists()
    # robodojo_pose_math may be named as the toolkit function's origin, never as a tool to call.
    retired = ("robodojo_pixel_to_position", "robodojo_step_eef", "robodojo_free_space_move",
               "robot_preview_", 'call("robodojo_pose_math', "call('robodojo_pose_math")
    for path in list(target.glob("*.md")) + list((target / ".agents").rglob("*.md")):
        text = path.read_text()
        assert "<!-- capability:" not in text
        for name in retired:
            assert name not in text, (path, name)
    session = (target / "MCP_SESSION.md").read_text()
    assert "rgb-position-calibration/SKILL.md" in session
    assert ("ctx.env(i)" in session) == (count > 1)
    manual = (target / ".agents/skills/manual-robot-control/SKILL.md").read_text()
    assert "robodojo_step" in manual and "motion-toolkit" in manual


def test_contracts_do_not_mutate_shared_schemas_or_each_other():
    baseline = RoboDojoMCP.tool_definitions()
    original = copy.deepcopy(baseline)
    a = Contract("auto-research", observation_profile="rgb-only", environments=3)
    once = a.select(baseline)
    assert a.select(once) == once
    assert baseline == original
    assert "robodojo_pixel_to_position" not in Contract("auto-research", observation_profile="official").robot_names
    with pytest.raises(ValueError, match="Unknown workspace capability"):
        render("<!-- capability:unknown -->\na\n<!-- otherwise -->\nb\n<!-- end-capability -->", a)


def test_routing_is_pure_and_rejects_global_or_unavailable_tools():
    contract = Contract("auto-research", observation_profile="rgb-only", environments=3)
    args = {"env_id": 2, "actions": [[0.0] * 14]}
    before = copy.deepcopy(args)
    index, payload = contract.route("robodojo_step", args)
    assert index == 2 and payload == {"actions": args["actions"]} and args == before
    for name in ("gemini_generate", "robodojo_pose_math", "robodojo_pixel_to_position", "submit"):
        with pytest.raises(PermissionError):
            contract.route(name, args)
    with pytest.raises(ValueError):
        Contract().route("robodojo_step", args)


def test_lab_entrypoint_uses_one_source_tree(tmp_path):
    from scripts.robot_lab import command
    assert command("dashboard", ["--port", "8000"], tmp_path) == [
        "bash", str(tmp_path / "scripts/run_dashboard.sh"), "--port", "8000"]
    args = command("prepare", ["--task", "make_kong"], tmp_path)
    assert args[1:] == [str(tmp_path / "scripts/configure_auto_research.py"), "--task", "make_kong"]
    assert not any("worktree" in a for a in args)
    for retired in ("control", "review", "direct", "prepare-teacher-student"):
        with pytest.raises(ValueError):
            command(retired, [], tmp_path)
