"""robodojo_step_ee: official native EEF actions through the harness's native take_action."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_fixtures import gpu_runtime
from test_gpu_joint_replay import _MCPHarness

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'packages/robodojo_toolkit/src'))
import robodojo_toolkit as tk  # noqa: E402

pytestmark = pytest.mark.gpu


def test_native_ee_actions_reach_link6_targets(gpu_runtime):
    root, python, gpu, output = gpu_runtime
    previous = dict(os.environ)
    os.environ["ROBODOJO_PYTHON"] = python
    config = SimpleNamespace(root=output / "native-ee", task="make_kong", sim_gpu=gpu,
                             eval_seed=0, exploration_seeds=(0,))
    config.root.mkdir()
    backend = _MCPHarness(config)
    try:
        backend.start()
        meta, _ = backend.call("robodojo_observe", {})
        # Malformed rows are rejected before any motion.
        for bad in ([[0.0] * 14], [[0.0] * 16]):
            with pytest.raises(ValueError):
                backend.call("robodojo_step_ee", {"actions": bad})
        assert backend.call("robodojo_observe", {})[0]["step_id"] == meta["step_id"]
        for index, arm in enumerate(("left", "right")):
            start = {"position": meta["eef_positions"][index], "quaternion_wxyz": meta["eef_quaternions_wxyz"][index]}
            goal = {"position": (np.asarray(start["position"]) + [0.0, 0.12, -0.04]).tolist(),
                    "quaternion_wxyz": start["quaternion_wxyz"], "gripper_opening": 0.3}
            before = meta["step_id"]
            rows = tk.ee_rows(meta, [{arm: goal}] * 15)
            meta, _ = backend.call("robodojo_step_ee", {"actions": rows.tolist()})
            assert meta["step_id"] == before + 15
            position_error, rotation_error = tk.pose_error(meta, arm, goal)
            assert position_error <= 0.001 and rotation_error <= 0.005, (arm, position_error, rotation_error)
            # The commanded gripper follows the EEF row (official state semantics).
            assert meta["states"][6 if arm == "left" else 13] == pytest.approx(0.3, abs=1e-3)
            steps = meta["transition"]["steps"]
            assert steps[-1]["execution_mode"] == "policy_ee_target" and len(steps[-1]["ee_action"]) == 16
    finally:
        backend.close()
        os.environ.clear()
        os.environ.update(previous)
