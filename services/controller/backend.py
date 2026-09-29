"""Trusted adapter to existing MCP implementations, never mounted in containers."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

from .config import layout_of


class NativeBackend:
    def __init__(self, config):
        from services.robodojo.mcp_server import RoboDojoMCP

        root = config.root / "native"
        root.mkdir(mode=0o700, exist_ok=True)
        # This adapter runs in a dedicated supervisor process, not a shared server.
        if os.environ.get("ROBODOJO_AGENT_TRIAL_ID"):
            raise RuntimeError("Controller service must not inherit an interactive trial identity")
        os.environ.setdefault("ROBODOJO_PYTHON", sys.executable)
        # The simulator imports the pristine official RoboDojo, never the patched checkout.
        from services.robodojo.source import SOURCE_ENV, official_stage
        os.environ.setdefault(SOURCE_ENV, str(official_stage()))
        os.environ.update({
            "ROBODOJO_AGENT_WORKSPACE": str(root / "workspace"),
            "ROBODOJO_HUMAN_SETUP_PATH": str(root / "operator" / "setup.json"),
            "ROBODOJO_OPERATOR_SETUP_REQUEST_PATH": str(root / "operator" / "setup-request.json"),
            "ROBODOJO_OPERATOR_EVALUATION_REQUEST_PATH": str(root / "operator" / "evaluation.json"),
            "ROBODOJO_MCP_OUTPUT_ROOT": str(root / "results"),
            "ROBODOJO_RUNTIME_ROOT": str(root),
            "ROBODOJO_SIM_GPU": config.sim_gpu,
            "ROBODOJO_TRIAL_GPU": config.sim_gpu,
        })

        class ManagedBridge(RoboDojoMCP):
            # Includes interactive exploration; isolated runs also have the
            # supervisor's stricter startup-inclusive deadline.
            episode_timeout_seconds = config.formal.wall_seconds

            def _validate_trial_task(self, task):
                # The new supervisor binds identity/configuration durably instead
                # of relying on the legacy Codex workspace's trial.json.
                if task != config.task:
                    raise RuntimeError("Controller task differs from operator configuration")

            def _await_human_setup(self, **kwargs):
                return dict(self.pending_setup)

            def _await_human_evaluation(self, record):
                # Same trusted evidence used by dashboard automatic evaluation.
                # Check identity, task AND step; never accept a stale result.
                state = json.loads((self.sim.output_dir / "sim" / "operator_state.json").read_text())
                for key in ("episode_id", "task", "step_id"):
                    if state.get(key) != record.get(key):
                        raise RuntimeError("Native evaluation identity mismatch")
                success = state.get("reward", {}).get("native_success")
                if type(success) is not bool:
                    raise RuntimeError("Native evaluation unavailable")
                return {**record, "status": "completed", "task_complete": success,
                        "decided_at_unix_s": time.time()}

        from services.mcp_contract import Contract
        self.contract = Contract.from_config(config, isolated=True)
        self.config = config
        self.robot = ManagedBridge()
        self.robot.observation_profile = self.contract.observation_profile
        self.frame_workspace = self.robot.agent_workspace
        self._evaluation = None

    def tools(self):
        return [t for t in self.contract.select(self.robot.tool_definitions())
                if t["name"] in self.contract.robot_names]

    @staticmethod
    def validate(name, arguments):
        from services.robodojo.action_validation import validate_motion_request
        validate_motion_request(name, arguments)

    def call(self, name, arguments):
        self.contract.require_robot(name, arguments)
        from .gateway import MUTATING
        planning_only = name == 'robodojo_free_space_move' and arguments.get('preview_only') is True
        if name in MUTATING and not planning_only:
            # Even a failed motion can change physical state before step_id updates.
            self._evaluation = None
        return self.robot.call(name, arguments)

    def start(self, *, formal, seed):
        self._evaluation = None
        c = self.config
        # Exploration seeds are layout keys that may span every official collection.
        collection, layout = (c.formal_collection, seed) if formal else layout_of(seed)
        self.robot.pending_setup = {
            "revision": uuid.uuid4().hex, "task": c.task, "seed": layout,
            "eval_seed": collection,
            "sim_gpu": c.sim_gpu, "sim_port": 0, "startup_timeout": 180,
            "include_depth": self.contract.observation.geometry,
            "include_camera_parameters": self.contract.observation.geometry,
            # Existing backend uses this legacy field for phase selection, but
            # SimulatorProcess enforces the native horizon in BOTH phases.
            # Visual demos are provisioned once into the development workspace at
            # launch, not recreated in this private native workspace each episode.
            "enforce_step_limit": formal, "demonstration_context": "none",
            "max_exploration_episodes": len(c.exploration_seeds),
        }
        return self.robot.call("robodojo_request_formal_episode" if formal else "robodojo_request_human_setup", {})

    def evaluate(self):
        # Legacy interactive review rejects repeat requests at the same step.
        # Auto-research lifecycle transitions may evaluate after an explicit check;
        # reuse only this process's trusted result for this exact episode/state.
        sim = self.robot.sim
        identity = (sim.episode_id, sim.step_id)
        if self._evaluation is not None and self._evaluation[0] == identity:
            return dict(self._evaluation[1])
        value, _ = self.robot.call("robodojo_request_human_evaluation", {
            "agent_assessment": "Automatic controller supervisor terminal evaluation",
        })
        evaluation = value["human_evaluation_request"]
        self._evaluation = (identity, dict(evaluation))
        return dict(evaluation)

    def close(self):
        self.robot.close()
