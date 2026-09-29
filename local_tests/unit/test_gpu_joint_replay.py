"""Real MCP planning, image artifacts, and 25Hz policy replay contract."""
import base64
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from services.robodojo.mcp_server import RoboDojoMCP
from gpu_fixtures import gpu_runtime, _errors


class _MCPHarness:
    """Test-only automatic setup around the original public MCP services."""

    def __init__(self, config):
        if os.environ.get("ROBODOJO_AGENT_TRIAL_ID"):
            raise RuntimeError("Run GPU tests outside an active agent trial")
        native = config.root / "native"
        native.mkdir()
        os.environ.update(
            ROBODOJO_AGENT_WORKSPACE=str(native / "workspace"),
            ROBODOJO_RUNTIME_ROOT=str(native),
            ROBODOJO_MCP_OUTPUT_ROOT=str(native / "results"),
            ROBODOJO_HUMAN_SETUP_PATH=str(native / "operator/setup.json"),
            ROBODOJO_OPERATOR_SETUP_REQUEST_PATH=str(native / "operator/request.json"),
            ROBODOJO_OPERATOR_EVALUATION_REQUEST_PATH=str(native / "operator/evaluation.json"),
            ROBODOJO_SIM_GPU=config.sim_gpu,
            ROBODOJO_TRIAL_GPU=config.sim_gpu,
        )

        class TestMCP(RoboDojoMCP):
            def _validate_trial_task(self, task):
                assert task == config.task

            def _await_human_setup(self, **kwargs):
                return {
                    "revision": "integration-" + config.root.name,
                    "task": config.task, "seed": 0, "eval_seed": 0,
                    "sim_gpu": config.sim_gpu, "sim_port": 0, "startup_timeout": 180,
                    "include_depth": True, "include_camera_parameters": True,
                    "enforce_step_limit": False, "demonstration_context": "none",
                    "max_exploration_episodes": 1,
                }

        self.robot = TestMCP()
        self.frame_workspace = self.robot.agent_workspace

    def start(self):
        return self.robot.call("robodojo_request_human_setup", {})

    def call(self, name, arguments):
        return self.robot.call(name, arguments)

    def close(self):
        self.robot.close()


def _frames(backend, reply):
    path = backend.frame_workspace / reply["frame_sequence"]["manifest_path"]
    return json.loads(path.read_text())["frames"]


def _target_error(frame, target, arm):
    index = 0 if arm == "left" else 1
    q = np.asarray(frame["eef_quaternions_wxyz"][index], dtype=float)
    goal = np.asarray(target["quaternion_wxyz"], dtype=float)
    cosine = abs(np.dot(q, goal) / (np.linalg.norm(q) * np.linalg.norm(goal)))
    return {"position_m": float(np.linalg.norm(np.asarray(frame["eef_positions"][index]) - target["position"])),
            "rotation_rad": float(2 * np.arccos(np.clip(cosine, 0, 1)))}


def _images(content):
    images = [x for x in content if x["type"] == "image"]
    assert len(images) == 6, "Three RGB + three depth final images must be delivered"
    for image in images:
        with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as png:
            assert png.width > 10 and png.height > 10
            assert np.asarray(png).std() > 0


@pytest.mark.gpu
@pytest.mark.skipif(bool(os.environ.get("ROBODOJO_SOURCE_ROOT")), reason=(
    "Legacy native planning tools (robodojo_free_space_move) need the patched RoboDojo planner; "
    "official sessions plan with robodojo_toolkit (official/xpolicylab/examples/planning_smoke_bundle)."))
def test_large_motion_joint_replay(gpu_runtime):
    root, python, gpu, output = gpu_runtime
    previous = dict(os.environ)
    os.environ["ROBODOJO_PYTHON"] = python
    actions, references, segments = [], [], []
    report = {"contract_version": 2, "segments": segments, "artifact_root": str(output)}

    try:
        for mode in ("planned", "joint_replay"):
            config = SimpleNamespace(root=output / mode, task="make_kong", sim_gpu=gpu,
                                     eval_seed=0, exploration_seeds=(0,))
            config.root.mkdir()
            backend = _MCPHarness(config)
            try:
                backend.start()
                initial, _ = backend.call("robodojo_observe", {})
                current, _ = backend.call("robodojo_step", {"actions": [initial["states"]] * 15})
                # Invalid input must not consume a step or break the native RPC.
                with pytest.raises(ValueError):
                    backend.call("robodojo_free_space_move", {
                        "arm": "left", "target": {"position": [0, 0, 1],
                                                  "quaternion_wxyz": [0, 0, 0, 0]}})
                checked, _ = backend.call("robodojo_observe", {})
                assert checked["step_id"] == current["step_id"]
                observed = []
                if mode == "planned":
                    cases = [("left_elevated_combined", "left", [0.5, 0.2, 0.7, 0, 0.5, 0.5]),
                             ("right_wrist_rotation", "right", [0, 0, 0, 0, 0, 1.2])]
                    for label, arm, delta in cases:
                        offset = 0 if arm == "left" else 7
                        proposed = np.asarray(current["states"]).copy()
                        proposed[offset:offset + 6] += delta
                        fk, _ = backend.call("robodojo_fk_preview", {"actions": [proposed.tolist()] * 50})
                        pose = fk["trajectory"][-1][arm]
                        target = {k: pose[k] for k in ("position", "quaternion_wxyz")}
                        target["gripper_opening"] = 0.25 if arm == "left" else 0.8
                        print(f"Planning {label}: {target}", flush=True)
                        cached = arm == "left"
                        plan, images = backend.call("robodojo_free_space_move", {
                            "arm": arm, "target": target, "preview_only": cached, "include_trajectory": True})
                        _images(images)
                        motion = plan["motion_plan"]
                        assert motion["status"] == "Success", motion
                        if cached:
                            assert plan["step_id"] == current["step_id"]
                        planned = motion["trajectory_preview"]["actions"]
                        assert motion["trajectory_preview"]["frequency_hz"] == 25
                        if cached:
                            value, images = backend.call("robodojo_execute_motion_plan", motion["execute_with"])
                            _images(images)
                            execution = value["motion_plan"]
                        else:
                            value = plan
                            execution = motion["execution"]
                        assert execution["status"] == "Success", execution
                        steps = value["transition"]["steps"]
                        assert [x["executed_action"] for x in steps] == planned
                        assert all(x["execution_mode"] == "policy_joint_target" for x in steps)
                        assert all(x["observation_step_id"] + 1 == x["step_id"] for x in steps)
                        assert value["step_id"] - current["step_id"] == len(planned)
                        frames = _frames(backend, value)
                        assert len(frames) == len(planned)
                        begin = len(actions)
                        actions.extend(planned)
                        observed.extend(frames)
                        segment = {"name": label, "arm": arm, "target": target,
                                   "start": begin, "end": len(actions),
                                   "diagnostics": motion["diagnostics"],
                                   "measured_goal_error": execution["measured_goal_error"],
                                   "goal_reached": execution["goal_reached"],
                                   "actual_goal_error": _target_error(value, target, arm)}
                        segments.append(segment)
                        (output / "joint_replay_accuracy.json").write_text(json.dumps(report, indent=2))
                        assert execution["goal_reached"], segment
                        assert segment["actual_goal_error"]["position_m"] <= 0.001
                        assert segment["actual_goal_error"]["rotation_rad"] <= 0.005
                        current = value
                        print(f"{label}: {len(planned)} steps; {segment['actual_goal_error']}", flush=True)
                    # Small closed-loop EEF segment returns exactly the same low-level format.
                    for _ in range(5):
                        targets = {}
                        for i, arm in enumerate(("left", "right")):
                            position = np.asarray(current["eef_positions"][i]).copy()
                            position[2] += 0.005
                            targets[arm] = {"position": position.tolist(),
                                            "quaternion_wxyz": current["eef_quaternions_wxyz"][i],
                                            "gripper_closed": False,
                                            "gripper_opening": current["states"][6 if i == 0 else 13]}
                        current, images = backend.call("robodojo_step_eef", {"targets": [targets]})
                        _images(images)
                        steps = current["transition"]["steps"]
                        assert current["transition"]["joint_target_contract"]["version"] == 2
                        actions.extend(row["executed_action"] for row in steps)
                        observed.extend(_frames(backend, current))
                    references = observed
                    (output / "replay_actions.json").write_text(json.dumps(actions))
                else:
                    for index in range(0, len(actions), 15):
                        value, images = backend.call("robodojo_step", {"actions": actions[index:index + 15]})
                        _images(images)
                        observed.extend(_frames(backend, value))
                    assert len(observed) == len(references)
                    errors = [_errors(a, b) for a, b in zip(references, observed, strict=True)]
                    report["replay_max_error"] = {k: max(e[k] for e in errors) for k in errors[0]}
                    for segment in segments:
                        segment["replay_goal_error"] = _target_error(observed[segment["end"] - 1],
                                                                    segment["target"], segment["arm"])
                    (output / "joint_replay_accuracy.json").write_text(json.dumps(report, indent=2))
                    assert report["replay_max_error"]["joint_rad"] < 1e-4, report
                    assert report["replay_max_error"]["eef_m"] < 1e-4, report
                    assert report["replay_max_error"]["eef_rad"] < 1e-4, report
            finally:
                backend.close()
        print(f"REAL MCP REPORT: {output / 'joint_replay_accuracy.json'}", flush=True)
        print(json.dumps(report), flush=True)
    finally:
        os.environ.clear()
        os.environ.update(previous)
