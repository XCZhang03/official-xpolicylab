#!/usr/bin/env python3
"""MCP stdio bridge for the native RoboDojo simulator.

The Isaac process remains the owner of simulation state.  This process owns the
MCP connection and talks to Isaac through RoboDojo's existing, lossless local
RPC.  It deliberately starts Isaac lazily, on the first automatic setup
request, so Codex can discover the tools without paying simulator startup cost.

The bridge implements the small MCP surface Codex needs directly with the
newline-delimited JSON-RPC stdio protocol.  That keeps the MCP server usable in
the pinned Isaac Python environment without installing another package into it.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_MODULE_ROOT))

from services.operator_audit import append_event
from services.robodojo.source import source_root
from services.robodojo.trajectory import JOINT_TARGET_CONTRACT, MAX_TRAJECTORY_ACTIONS
from services.robodojo.action_validation import validate_motion_request
from services.artifact_io import write_artifacts
from services.mcp_contract import observation_for
from services.robodojo.demonstrations import (
    DEMONSTRATION_CONTEXTS,
    cached_demo_path,
    cached_terminal_path,
    provision_trial_demonstration,
)

MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "robodojo"
SERVER_VERSION = "0.15.0"
MAX_ACTIONS = MAX_EEF_ACTIONS = MAX_TRAJECTORY_ACTIONS
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TASK_NAME = re.compile(r"^[A-Za-z0-9_]+$")
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SETUP_TOOLS = {
    "robodojo_request_human_setup",
    "robodojo_request_formal_episode",
}
MCP_AGENT_INSTRUCTIONS = (
    "Explore and learn efficiently, then succeed in the single final formal "
    "episode. Target score 100 by satisfying every condition in the supplied "
    "public rubric's 100-point row, not partial progress. Verify each condition "
    "from current observations while steps remain. Always submit formal evaluation "
    "at episode end, even on failure, before calling finish; never "
    "claim a verified native score without evaluator confirmation. "
    "Call robodojo_request_human_setup with no arguments and read "
    "environment_spec. Every episode enforces the original native task step limit. "
    "Exploration has a fixed episode budget, including the initial episode. "
    "Start formal once exploration success is confirmed by "
    "task_complete=true and a repeatable plan is ready, or once the exploration "
    "episodes have been used. Zero remaining episodes means no more resets, "
    "not that the active episode must be abandoned. If success was not confirmed, use the best "
    "tested plan and state the uncertainty honestly. Inspect supplied demo images "
    "before acting. Their layout may differ: learn the requirements, not an exact "
    "action sequence, and use a simpler tested procedure when useful. "
    "If setup starts in formal mode, begin directly without another reset. "
    "Setup and evaluation run automatically through the saved trial profile. "
    "Starting formal consumes the single formal attempt: no reset or retry. Submitting "
    "formal evaluation shuts down the environment before the decision, with no retry. Formal "
    "success is the primary criterion; resets, evaluations, and steps are secondary costs. "
    "Native reward and intermediate success are hidden. When an episode ends, "
    "task_complete reports native success; still submit formal evaluation to record and close it. "
    "EEF link6 poses use environment-frame "
    "metres and unit wxyz quaternions; use robodojo_pose_math. Use cuRobo for a "
    "target pose and bounded EEF chunks for explicit waypoints. Preview only when "
    "useful; the preview server starts with the session but its use is your choice. "
    "Motion calls return the final frame and save the 25 Hz sequence. "
    "Interaction steps count executed 25 Hz blocks, not MCP calls; batching does "
    "not reduce them. Never ask the operator for task or action advice. Report a "
    "mandatory-tool infrastructure blocker instead of bypassing it."
)

# These fields remain available to the simulator-side recorder and evaluator,
# but their raw fields are never part of an agent-visible MCP result. Terminal
# success is disclosed separately as task_complete. Operational status
# strings such as a cuRobo planner's ``status: Success`` are intentionally not
# removed: they describe whether a requested computation ran, not task outcome.
_TASK_EVALUATION_KEYS = {
    "native_end_flag",
    "native_reward",
    "native_score",
    "native_score_percent",
    "native_success",
    "operator_decision_source",
    "reward",
    "rewards",
    "reward_state",
    "score",
    "success",
    "terminated",
    "truncated",
    "valid_for_success_rate",
}


def _write_private_json(path: Path, value: Any) -> None:
    """Atomically write operator-owned state with restrictive permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and paths into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    # Avoid importing numpy during tool discovery.  Isaac's environment has it,
    # but MCP initialization itself only needs the standard library.
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except ValueError:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _agent_safe(value: Any) -> Any:
    """Hide scores and intermediate success; disclose native terminal outcome."""

    if isinstance(value, dict):
        result = {
            str(key): _agent_safe(item)
            for key, item in value.items()
            if str(key).lower() not in _TASK_EVALUATION_KEYS
        }
        if "terminated" in value or "truncated" in value:
            result["episode_ended"] = bool(
                value.get("terminated", False) or value.get("truncated", False)
            )
            if result["episode_ended"] and "success" in value:
                result["task_complete"] = bool(value["success"])
        return result
    if isinstance(value, (list, tuple)):
        return [_agent_safe(item) for item in value]
    return _jsonable(value)


def _png_data(value: Any) -> str:
    """Encode a uint8 RGB array as an MCP image content payload."""

    from PIL import Image

    array = value
    if getattr(array, "ndim", 0) == 2:
        array = array[..., None]
    if array.shape[-1] == 1:
        array = array[..., 0]
    elif array.shape[-1] > 3:
        array = array[..., :3]
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _depth_png_data(value: Any) -> tuple[str, dict[str, Any]]:
    """Encode distance-to-image-plane meters as lossless uint16 millimeters."""

    import numpy as np
    from PIL import Image

    depth = np.asarray(value, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    millimeters = np.zeros(depth.shape, dtype=np.uint16)
    if valid.any():
        scaled = np.rint(np.clip(depth[valid] * 1000.0, 0, 65535)).astype(np.uint32)
        millimeters[valid] = scaled.astype(np.uint16)
    stream = io.BytesIO()
    Image.fromarray(millimeters).save(stream, format="PNG")
    valid_values = depth[valid]
    stats = {
        "shape": [int(depth.shape[1]), int(depth.shape[0])],
        "encoding": "uint16_png",
        "unit": "millimeters",
        "meters_per_unit": 0.001,
        "invalid_value": 0,
        "invalid_pixels": int((~valid).sum()),
        "clipped_pixels": int((valid & (depth * 1000.0 > 65535)).sum()),
        "min_meters": float(valid_values.min()) if valid_values.size else None,
        "max_meters": float(valid_values.max()) if valid_values.size else None,
    }
    return base64.b64encode(stream.getvalue()).decode("ascii"), stats


def _content_text(value: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(_jsonable(value), ensure_ascii=False)}


def _error(message: str, *, code: int = -32000, data: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        result["data"] = _jsonable(data)
    return result


class SimulatorProcess:
    """Own one native RoboDojo server process and its RPC client."""

    def __init__(
        self,
        *,
        project_root: Path,
        python: str,
        output_root: Path,
        enable_depth: bool = True,
        enable_camera_parameters: bool = True,
    ):
        self.project_root = project_root
        self.python = python
        self.output_root = output_root
        self.enable_depth = enable_depth
        self.enable_camera_parameters = enable_camera_parameters
        self.process: subprocess.Popen[str] | None = None
        self.rpc = None
        self.log_thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.ready_metadata: dict[str, Any] | None = None
        self.bound_port: int | None = None
        self.log_path: Path | None = None
        self.output_dir: Path | None = None
        self.run_id: str | None = None
        self.task: str | None = None
        self.episode_id: str | None = None
        self.step_id = 0
        self.metadata: dict[str, Any] | None = None
        self.finished = False
        self.temporary_xdg_runtime: Path | None = None

    @property
    def active(self) -> bool:
        return self.process is not None and self.rpc is not None and not self.finished

    def _read_log(self, stream, log_file) -> None:
        try:
            for line in iter(stream.readline, ""):
                log_file.write(line)
                log_file.flush()
                if '"event": "ready"' not in line:
                    continue
                try:
                    packet = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if packet.get("event") == "ready":
                    self.ready_metadata = packet.get("metadata") or {}
                    self.bound_port = int(packet["port"])
                    self.ready.set()
        finally:
            try:
                stream.close()
            except OSError:
                if os.environ.get("ROBODOJO_MCP_DEBUG"):
                    print(
                        "[robodojo-mcp] failed to close simulator log stream",
                        file=sys.stderr,
                    )
            try:
                log_file.close()
            except OSError:
                if os.environ.get("ROBODOJO_MCP_DEBUG"):
                    print(
                        "[robodojo-mcp] failed to close simulator log file",
                        file=sys.stderr,
                    )

    def start(
        self,
        *,
        task: str,
        seed: int,
        eval_seed: int,
        run_id: str,
        sim_gpu: str | int,
        sim_port: int,
        startup_timeout: float,
        enable_depth: bool | None = None,
        enable_camera_parameters: bool | None = None,
        enforce_step_limit: bool = True,
        episode_timeout_seconds: int = 0,
    ) -> dict[str, Any]:
        if self.active:
            raise RuntimeError(
                "A RoboDojo episode is already active; finish it before starting another"
            )
        if not TASK_NAME.fullmatch(task):
            raise ValueError("task must contain only letters, numbers, and underscores")
        task_config = (
            source_root(self.project_root)
            / "task"
            / "RoboDojo"
            / "config"
            / f"{task}.yml"
        )
        if not task_config.is_file():
            raise ValueError(f"Unknown RoboDojo task: {task}")
        if not RUN_ID.fullmatch(run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, dots, underscores, and hyphens"
            )

        # Artifact storage is created only when an episode is requested; tool
        # discovery must remain usable on a read-only Codex workspace.
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = (self.output_root / run_id).resolve()
        if self.output_dir.exists():
            raise ValueError(f"Artifact identifier already exists: {run_id}")
        self.output_dir.mkdir(parents=True)
        self.run_id = run_id
        native_output = self.output_dir / "sim"
        native_runtime = self.output_dir / "native_runtime"
        native_output.mkdir()
        native_runtime.mkdir()
        self.log_path = self.output_dir / "sim.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(sim_gpu)
        env["PYTHONUNBUFFERED"] = "1"
        # Match scripts/robodojo_env.sh so Isaac never waits for an interactive
        # EULA prompt and uses the native workstation's NVIDIA userspace.
        env["OMNI_KIT_ACCEPT_EULA"] = env.get("OMNI_KIT_ACCEPT_EULA", "Y")
        if Path("/usr/share/vulkan/icd.d/nvidia_icd.json").is_file():
            env.setdefault(
                "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json"
            )
        if Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").is_file():
            env.setdefault(
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
            )
        xdg_runtime = Path(
            env.get(
                "ROBODOJO_XDG_RUNTIME_DIR",
                self.output_root.parent.parent / "xdg-runtime",
            )
        )
        try:
            xdg_runtime.mkdir(parents=True, exist_ok=True)
            xdg_runtime.chmod(0o700)
        except OSError:
            # A host session may export a read-only XDG directory.  This is a
            # small private Kit runtime; rollout evidence remains under
            # output_root and this directory is removed when the process stops.
            xdg_runtime = Path(tempfile.mkdtemp(prefix=f"robodojo-xdg-{os.getuid()}-"))
            self.temporary_xdg_runtime = xdg_runtime
        env["XDG_RUNTIME_DIR"] = str(xdg_runtime)
        source = source_root(self.project_root)
        env["PYTHONPATH"] = os.pathsep.join(
            str(path)
            for path in (
                self.project_root,
                source,
                source / "XPolicyLab",
                source / "third_party" / "curobo",
            )
        ) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [
            self.python,
            "-m",
            "services.robodojo.server",
            "--task",
            task,
            "--output",
            str(native_output),
            "--port",
            str(sim_port),
            "--eval-seed",
            str(eval_seed),
            "--episode-timeout-seconds",
            str(episode_timeout_seconds),
        ]
        if self.enable_depth if enable_depth is None else enable_depth:
            command.append("--camera-depth")
        if (
            self.enable_camera_parameters
            if enable_camera_parameters is None
            else enable_camera_parameters
        ):
            command.append("--camera-calibration")
        command.append(
            "--enforce-step-limit" if enforce_step_limit else "--no-enforce-step-limit"
        )
        log_handle = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=native_runtime,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise
        if self.process.stdout is None:
            log_handle.close()
            self.stop()
            raise RuntimeError("RoboDojo simulator stdout pipe was not created")
        self.ready.clear()
        self.ready_metadata = None
        self.bound_port = None
        self.log_thread = threading.Thread(
            target=self._read_log,
            args=(self.process.stdout, log_handle),
            name="robodojo-sim-log",
            daemon=True,
        )
        self.log_thread.start()
        deadline = time.monotonic() + startup_timeout
        while not self.ready.wait(
            timeout=min(1.0, max(0.0, deadline - time.monotonic()))
        ):
            if self.process.poll() is not None:
                raise RuntimeError(
                    "RoboDojo simulator exited during startup; inspect the service-managed run log"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"RoboDojo simulator did not become ready within {startup_timeout:g}s; "
                    "inspect the service-managed run log"
                )

        try:
            from services.robodojo.protocol import RPCClient

            if self.bound_port is None:
                raise RuntimeError("RoboDojo simulator did not report its RPC port")
            self.rpc = RPCClient(
                "127.0.0.1", self.bound_port, timeout=max(600.0, startup_timeout)
            )
            self.metadata = self.rpc.request("metadata")
            if self.metadata.get("task") != task:
                raise RuntimeError(
                    f"Simulator task mismatch: expected {task!r}, got {self.metadata.get('task')!r}"
                )
            reset = self.rpc.request(
                "reset", seed=int(seed), source="student", policy_version="codex-mcp"
            )
            self.episode_id = reset["episode_id"]
            self.step_id = int(reset["step_id"])
            self.task = task
            self.finished = False
            return reset
        except Exception:
            self.stop()
            raise

    def request(self, op: str, **kwargs: Any) -> Any:
        if self.rpc is None or self.episode_id is None or self.finished:
            raise RuntimeError(
                "No active RoboDojo episode; request human environment setup first"
            )
        return self.rpc.request(
            op, episode_id=self.episode_id, step_id=self.step_id, **kwargs
        )

    def finish(self, reason: str) -> Any:
        if self.rpc is None or self.episode_id is None:
            raise RuntimeError("No active RoboDojo episode")
        if self.finished:
            return {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "reason": reason,
            }
        result = self.rpc.request(
            "finish_pilot",
            episode_id=self.episode_id,
            step_id=self.step_id,
            reason=reason,
        )
        self.finished = True
        return result

    def stop(self) -> None:
        rpc, process = self.rpc, self.process
        self.rpc = None
        self.process = None
        self.finished = True
        if rpc is not None:
            try:
                rpc.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the simulator result
                if os.environ.get("ROBODOJO_MCP_DEBUG"):
                    print(f"[robodojo-mcp] RPC close failed: {exc}", file=sys.stderr)
        if process is not None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    with suppress(OSError):
                        os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        if self.temporary_xdg_runtime is not None:
            shutil.rmtree(self.temporary_xdg_runtime, ignore_errors=True)
            self.temporary_xdg_runtime = None


class RoboDojoMCP:
    # Direct-control episodes; externally supervised modes own their own deadline.
    episode_timeout_seconds = None  # Resolve from the native task at setup.

    def __init__(self) -> None:
        project_root = Path(
            os.environ.get("ROBODOJO_PROJECT_ROOT", Path(__file__).resolve().parents[2])
        ).resolve()
        runtime_root = Path(
            os.environ.get("ROBODOJO_RUNTIME_ROOT", project_root / "runtime")
        ).resolve()
        python = os.environ.get("ROBODOJO_PYTHON", sys.executable)
        self.project_root = project_root
        self._trusted_trial_task = os.environ.get("ROBODOJO_AGENT_TRIAL_TASK")
        self._trusted_trial_id = os.environ.get("ROBODOJO_AGENT_TRIAL_ID")
        self._trusted_sim_gpu = os.environ.get("ROBODOJO_TRIAL_GPU")
        configured_trial_root = os.environ.get("ROBODOJO_AGENT_TRIAL_ROOT")
        self._trusted_trial_root: Path | None = None
        if self._trusted_trial_id:
            if not self._trusted_sim_gpu:
                raise RuntimeError("ROBODOJO_TRIAL_GPU is required for a trusted trial")
            if not configured_trial_root:
                raise RuntimeError(
                    "ROBODOJO_AGENT_TRIAL_ROOT is required for a trusted trial"
                )
            trial_root = Path(configured_trial_root).expanduser().resolve()
            if trial_root.name != self._trusted_trial_id:
                raise RuntimeError("Trusted trial root does not match the trial ID")
            self._trusted_trial_root = trial_root
            expected_workspace = trial_root / "workspace"
            expected_runtime = trial_root / "runtime"
            expected_operator = trial_root / "operator"
            expected_results = trial_root / "results"
            expected_audit = trial_root / "audit"

            def trusted_path(name: str, expected: Path) -> Path:
                configured = Path(os.environ.get(name, expected)).expanduser().resolve()
                if configured != expected.resolve():
                    raise RuntimeError(f"{name} does not match the trusted trial root")
                return configured

            self.agent_workspace = trusted_path(
                "ROBODOJO_AGENT_WORKSPACE", expected_workspace
            )
            self.agent_runtime = trusted_path(
                "ROBODOJO_AGENT_TRIAL_RUNTIME", expected_runtime
            )
            operator_root = trusted_path(
                "ROBODOJO_OPERATOR_TRIAL_ROOT", expected_operator
            )
            results_root = trusted_path("ROBODOJO_RESULT_ROOT", expected_results)
            self.operator_trial_log_path = trusted_path(
                "ROBODOJO_OPERATOR_TRIAL_LOG", expected_audit
            )
            if (self.agent_workspace / "runtime").resolve() != self.agent_runtime:
                raise RuntimeError(
                    "Agent workspace runtime link does not match the trusted trial"
                )
            self.human_setup_path = operator_root / "setup.json"
            self.setup_request_path = operator_root / "setup_request.json"
            self.evaluation_request_path = operator_root / "evaluation_request.json"
            output_root = results_root / "robodojo_mcp"
        else:
            self.agent_workspace = Path(
                os.environ.get(
                    "ROBODOJO_AGENT_WORKSPACE", project_root / "codex_agent"
                )
            ).resolve()
            self.agent_runtime = (self.agent_workspace / "runtime").resolve()
            self.human_setup_path = Path(
                os.environ.get(
                    "ROBODOJO_HUMAN_SETUP_PATH",
                    runtime_root / "operator" / "robodojo_setup.json",
                )
            ).expanduser().resolve()
            self.setup_request_path = Path(
                os.environ.get(
                    "ROBODOJO_OPERATOR_SETUP_REQUEST_PATH",
                    self.human_setup_path.with_name("robodojo_setup_request.json"),
                )
            ).expanduser().resolve()
            self.evaluation_request_path = Path(
                os.environ.get(
                    "ROBODOJO_OPERATOR_EVALUATION_REQUEST_PATH",
                    runtime_root / "operator" / "robodojo_evaluation_request.json",
                )
            ).expanduser().resolve()
            output_root = Path(
                os.environ.get(
                    "ROBODOJO_MCP_OUTPUT_ROOT",
                    runtime_root / "results" / "robodojo_mcp",
                )
            ).expanduser().resolve()
            self.operator_trial_log_path = None
        try:
            self.human_setup_path.relative_to(self.agent_workspace)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "ROBODOJO_HUMAN_SETUP_PATH must be outside the agent workspace"
            )
        self.consumed_setup_path = self.human_setup_path.with_name(
            f"{self.human_setup_path.name}.consumed"
        )
        try:
            self.setup_request_path.relative_to(self.agent_workspace)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "ROBODOJO_OPERATOR_SETUP_REQUEST_PATH must be outside the agent workspace"
            )
        self.human_setup_wait_timeout = float(
            os.environ.get("ROBODOJO_HUMAN_SETUP_APPROVAL_TIMEOUT", "900")
        )
        try:
            self.evaluation_request_path.relative_to(self.agent_workspace)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "ROBODOJO_OPERATOR_EVALUATION_REQUEST_PATH must be outside the "
                "agent workspace"
            )
        self.human_evaluation_wait_timeout = float(
            os.environ.get("ROBODOJO_HUMAN_EVALUATION_TIMEOUT", "900")
        )
        self.frame_root = self.agent_runtime / "frames"
        self.demonstration_root = self.agent_runtime / "demonstrations"
        self.demo_cache_root = runtime_root / "reference-demos" / "website"
        self.trial_metadata_path = self.agent_runtime / "trial.json"
        self._trial_id: str | None = None
        # Codex starts this file by absolute path, so its working directory is
        # not a reliable import root.  Add the checked-out first-party sources
        # before lazily importing the simulator-owned RPC client.
        simulator = source_root(project_root)
        for import_root in (
            project_root,
            simulator,
            simulator / "XPolicyLab",
            simulator / "third_party" / "curobo",
        ):
            if import_root.is_dir() and str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        self._consumed_setup_revisions = self._load_consumed_setup_revisions()
        self._human_evaluation_steps: set[int] = set()
        self._human_evaluation_count = 0
        self._total_human_evaluation_count = 0
        self._environment_setup_count = 0
        self._environment_reset_count = 0
        self._exploration_episodes_started = 0
        self._max_exploration_episodes = 3
        self._formal_episode_started = False
        self._completed_interaction_steps = 0
        self._accounted_episode_ids: set[str] = set()
        self._environment_spec: dict[str, Any] | None = None
        self._demonstration_context: dict[str, Any] = {
            "kind": "none",
            "image_count": 0,
            "root": "runtime/demonstrations",
            "manifest_path": None,
        }
        self.sim = SimulatorProcess(
            project_root=project_root,
            python=python,
            output_root=output_root,
            enable_depth=False,
            enable_camera_parameters=False,
        )

    def _append_operator_trial_event(
        self,
        kind: str,
        **fields: Any,
    ) -> None:
        """Append non-agent-visible request accounting to the trusted trial log."""

        root = getattr(self, "operator_trial_log_path", None)
        if root is None:
            return
        value = {
            "schema": "robodojo_agent_trial_event_v1",
            "kind": kind,
            "trial_id": getattr(self, "_trusted_trial_id", None),
            "task": getattr(self, "_trusted_trial_task", None),
            "at_unix_s": time.time(),
            **fields,
        }
        append_event(root / "events.jsonl", _jsonable(value))

    def _account_current_episode(self) -> None:
        episode_id = self.sim.episode_id
        accounted = getattr(self, "_accounted_episode_ids", set())
        if episode_id is None or episode_id in accounted:
            return
        self._completed_interaction_steps = getattr(
            self, "_completed_interaction_steps", 0
        ) + int(self.sim.step_id)
        accounted.add(episode_id)
        self._accounted_episode_ids = accounted

    def _performance_cost(self) -> dict[str, int]:
        episode_id = self.sim.episode_id
        accounted = getattr(self, "_accounted_episode_ids", set())
        current_steps = (
            int(self.sim.step_id)
            if episode_id is not None and episode_id not in accounted
            else 0
        )
        return {
            "environment_setup_count": getattr(self, "_environment_setup_count", 0),
            "environment_reset_count": getattr(self, "_environment_reset_count", 0),
            "exploration_episodes_started": getattr(self, "_exploration_episodes_started", 0),
            "max_exploration_episodes": getattr(self, "_max_exploration_episodes", 3),
            "current_episode_interaction_steps": current_steps,
            "total_interaction_steps": getattr(self, "_completed_interaction_steps", 0)
            + current_steps,
            "human_evaluation_request_count": getattr(
                self, "_total_human_evaluation_count", 0
            ),
        }

    def _agent_environment_spec(self, setup: dict[str, Any]) -> dict[str, Any]:
        metadata = self.sim.metadata or {}
        formal_episode = bool(setup["enforce_step_limit"])
        return {
            "operator_selected": True,
            "trial_id": getattr(self, "_trial_id", None),
            "revision": setup["revision"],
            "task": setup["task"],
            "seed": setup["seed"],
            "eval_seed": setup["eval_seed"],
            "episode_mode": "formal" if formal_episode else "exploration",
            "episode_timeout_seconds": self.episode_timeout_seconds or None,
            "depth_enabled": setup["include_depth"],
            "camera_parameters_enabled": setup["include_camera_parameters"],
            "demonstration_context": dict(self._demonstration_context),
            "step_limit_enforced": True,
            "native_step_limit_enforced": True,
            "episode_step_limit": metadata.get("episode_step_limit"),
            "native_reference_step_limit": metadata.get("native_reference_step_limit"),
            "exploration_episodes_started": getattr(self, "_exploration_episodes_started", 0),
            "max_exploration_episodes": getattr(self, "_max_exploration_episodes", 3),
            "exploration_episodes_remaining": max(0, getattr(self, "_max_exploration_episodes", 3) - getattr(self, "_exploration_episodes_started", 0)),
            "control_frequency_hz": metadata.get("control_frequency_hz"),
            "physics_frequency_hz": metadata.get("physics_frequency_hz"),
            "cameras": metadata.get("cameras", []),
            "step_accounting": {
                "limit_unit": "executed_25hz_environment_control_block",
                "joint_action_row_steps": 1,
                "eef_waypoint_steps": 1,
                "joint_target_frequency_hz": metadata.get("control_frequency_hz"),
                "curobo_step_formula": "one step per executed 25Hz joint target row",
                "read_only_or_planning_steps": 0,
            },
        }

    def _validate_trial_task(self, task: str) -> None:
        if not self.trial_metadata_path.is_file():
            raise RuntimeError(
                "No isolated trial runtime is active; start the agent with a "
                "trusted --trial-task label"
            )
        metadata = json.loads(self.trial_metadata_path.read_text(encoding="utf-8"))
        trial_id = str(metadata.get("trial_id", ""))
        labelled_task = str(metadata.get("task", ""))
        if (
            getattr(self, "_trusted_trial_root", None) is not None
            and metadata.get("schema_version") != 2
        ):
            raise RuntimeError("Unsupported trusted trial metadata version")
        if (
            getattr(self, "_trusted_trial_root", None) is not None
            and str(metadata.get("sim_gpu")) != self._trusted_sim_gpu
        ):
            raise RuntimeError("Agent trial GPU does not match the trusted launcher")
        if not RUN_ID.fullmatch(trial_id) or not TASK_NAME.fullmatch(labelled_task):
            raise RuntimeError("Invalid agent trial metadata")
        if self._trusted_trial_id and trial_id != self._trusted_trial_id:
            raise RuntimeError("Agent trial ID does not match the trusted launcher")
        if self._trusted_trial_task and labelled_task != self._trusted_trial_task:
            raise RuntimeError("Agent trial task does not match the trusted launcher")
        if labelled_task != task:
            raise RuntimeError(
                f"Operator selected task {task!r}, but this isolated trial is "
                f"labelled {labelled_task!r}; restart the agent with the matching "
                "--trial-task"
            )
        self._trial_id = trial_id

    def _load_consumed_setup_revisions(self) -> set[str]:
        path = self.consumed_setup_path
        if not path.exists():
            return set()
        if not path.is_file() or path.stat().st_mode & 0o022:
            raise PermissionError(
                "Consumed setup ledger must be a private regular file"
            )
        revisions = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if any(not RUN_ID.fullmatch(revision) for revision in revisions):
            raise ValueError("Consumed setup ledger contains an invalid revision")
        return revisions

    def _record_consumed_setup_revision(self, revision: str) -> None:
        if revision in self._consumed_setup_revisions:
            return
        path = self.consumed_setup_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(revision + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o600)
        self._consumed_setup_revisions.add(revision)

    def _reserve_exploration_seed(self, setup: dict[str, Any]) -> dict[str, Any]:
        """Reserve a unique seed before startup, including across bridge restarts."""
        if setup["enforce_step_limit"]:
            return setup
        path = self.human_setup_path.with_name(self.human_setup_path.name + ".exploration-seeds.json")
        used = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(used, list) or any(type(seed) is not int for seed in used):
            raise ValueError("Invalid exploration seed ledger")
        seed = int(setup["seed"])
        used_set = set(used)
        while seed in used_set:
            seed += 1
        _write_private_json(path, sorted(used_set | {seed}))
        selected = {**setup, "seed": seed}
        _write_private_json(self.human_setup_path, selected)
        return selected

    def _load_human_setup(self) -> dict[str, Any]:
        path = self.human_setup_path
        if not path.is_file():
            raise RuntimeError(
                "Automatic setup is not ready; the dashboard must publish "
                "the saved trial profile for this request"
            )
        if path.stat().st_mode & 0o022:
            raise PermissionError(
                "Human setup specification must not be group/world writable"
            )
        try:
            setup = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid human setup specification") from exc
        required = {
            "revision",
            "task",
            "seed",
            "eval_seed",
            "sim_gpu",
            "sim_port",
            "startup_timeout",
            "include_depth",
            "include_camera_parameters",
            "enforce_step_limit",
            "demonstration_context",
        }
        if getattr(self, "_trusted_trial_root", None) is not None:
            required.add("trial_id")
        optional = {"max_exploration_episodes"}
        if not isinstance(setup, dict) or not required <= set(setup) or set(setup) - required - optional:
            raise ValueError(
                "Human setup specification must contain exactly: "
                + ", ".join(sorted(required))
            )
        revision = str(setup["revision"]).strip()
        task = str(setup["task"]).strip()
        if (
            getattr(self, "_trusted_trial_root", None) is not None
            and setup["trial_id"] != self._trusted_trial_id
        ):
            raise RuntimeError("Operator setup does not match this trusted trial")
        if not RUN_ID.fullmatch(revision):
            raise ValueError("Human setup revision has invalid characters")
        if not TASK_NAME.fullmatch(task):
            raise ValueError("Human setup task has invalid characters")
        for field in (
            "include_depth",
            "include_camera_parameters",
            "enforce_step_limit",
        ):
            if not isinstance(setup[field], bool):
                raise TypeError(f"Human setup {field} must be boolean")
        demonstration_context = str(setup["demonstration_context"]).strip()
        if demonstration_context not in DEMONSTRATION_CONTEXTS:
            raise ValueError(
                "Human setup demonstration_context must be one of: "
                + ", ".join(DEMONSTRATION_CONTEXTS)
            )
        sim_port = int(setup["sim_port"])
        sim_gpu = str(setup["sim_gpu"])
        startup_timeout = float(setup["startup_timeout"])
        if not 0 <= sim_port <= 65535:
            raise ValueError("Human setup sim_port must be in [0, 65535]")
        if getattr(self, "_trusted_trial_root", None) is not None and sim_port != 0:
            raise ValueError(
                "Trusted parallel trials require sim_port=0 for automatic allocation"
            )
        if (
            getattr(self, "_trusted_trial_root", None) is not None
            and sim_gpu != self._trusted_sim_gpu
        ):
            raise ValueError("Operator setup GPU does not match the trusted trial")
        if not 30 <= startup_timeout <= 1800:
            raise ValueError("Human setup startup_timeout must be in [30, 1800]")
        max_exploration_episodes = setup.get(
            "max_exploration_episodes", getattr(self, "_max_exploration_episodes", 3)
        )
        if isinstance(max_exploration_episodes, bool) or not isinstance(max_exploration_episodes, int) or not 1 <= max_exploration_episodes <= 100:
            raise ValueError("Human setup max_exploration_episodes must be an integer in [1, 100]")
        return {
            **setup,
            "revision": revision,
            "task": task,
            "seed": int(setup["seed"]),
            "eval_seed": int(setup["eval_seed"]),
            "sim_gpu": sim_gpu,
            "sim_port": sim_port,
            "startup_timeout": startup_timeout,
            "max_exploration_episodes": max_exploration_episodes,
            "demonstration_context": demonstration_context,
        }

    def _validate_episode_start(self, setup: dict[str, Any], *, formal_requested: bool) -> None:
        """Check the immutable trial budget and the requested episode mode."""
        if getattr(self, "_formal_episode_started", False):
            raise RuntimeError("The formal episode is final; no reset or second formal episode is allowed in this run")
        started = getattr(self, "_environment_setup_count", 0)
        if started and setup["max_exploration_episodes"] != self._max_exploration_episodes:
            raise RuntimeError("Exploration episode budget is fixed by the initial setup")
        if formal_requested and not setup["enforce_step_limit"]:
            raise RuntimeError("A formal episode requires the native step limit")
        if not formal_requested and setup["enforce_step_limit"] and started:
            raise RuntimeError("Use robodojo_request_formal_episode for the transition to the final formal episode")
        if not setup["enforce_step_limit"] and getattr(self, "_exploration_episodes_started", 0) >= setup["max_exploration_episodes"]:
            raise RuntimeError("Exploration episode budget exhausted; no further exploration reset is allowed")

    def _await_human_setup(
        self,
        *,
        request_kind: str = "setup_or_reset",
        require_step_limit: bool = False,
    ) -> dict[str, Any]:
        """Wait for the dashboard to apply the saved profile as a fresh revision."""

        if request_kind not in {"setup_or_reset", "formal_episode"}:
            raise ValueError("Unknown human setup request kind")

        active_episode = bool(getattr(self.sim, "episode_id", None))
        current_step_id = int(getattr(self.sim, "step_id", 0))
        self._append_operator_trial_event(
            "setup_request",
            request_kind=request_kind,
            active_episode=active_episode,
            current_step_id=current_step_id,
        )

        def eligible(setup: dict[str, Any] | None) -> bool:
            if setup is None or setup["revision"] in self._consumed_setup_revisions:
                return False
            trusted_task = getattr(self, "_trusted_trial_task", None)
            if trusted_task and setup["task"] != trusted_task:
                return False
            if require_step_limit and not setup["enforce_step_limit"]:
                return False
            return not (
                request_kind == "setup_or_reset"
                and active_episode
                and setup["enforce_step_limit"]
            )

        try:
            setup = self._load_human_setup()
        except RuntimeError:
            setup = None
        if eligible(setup):
            return setup
        wait_timeout = float(getattr(self, "human_setup_wait_timeout", 0.0))
        if wait_timeout <= 0:
            if setup is None:
                setup = self._load_human_setup()
            if setup["revision"] in self._consumed_setup_revisions:
                raise RuntimeError(
                    "This human setup revision was already consumed; the operator "
                    "must provide a new revision for another environment or reset"
                )
            trusted_task = getattr(self, "_trusted_trial_task", None)
            if trusted_task and setup["task"] != trusted_task:
                raise RuntimeError(
                    "The operator setup task does not match this trusted trial"
                )
            if require_step_limit and not setup["enforce_step_limit"]:
                raise RuntimeError(
                    "A formal episode requires a fresh operator setup revision "
                    "with the native step limit enforced"
                )
            if (
                request_kind == "setup_or_reset"
                and active_episode
                and setup["enforce_step_limit"]
            ):
                raise RuntimeError(
                    "An exploration reset cannot consume a formal setup revision; "
                    "use robodojo_request_formal_episode"
                )
            return setup

        request_id = uuid.uuid4().hex
        request = {
            "schema": "robodojo_setup_request_v2",
            "request_id": request_id,
            "request_kind": request_kind,
            "requires_step_limit": require_step_limit,
            "status": "pending",
            "requested_at_unix_s": time.time(),
            "active_episode": active_episode,
            "current_task": getattr(self.sim, "task", None),
            "current_step_id": current_step_id,
            "trial_id": getattr(self, "_trusted_trial_id", None),
            "trial_task": getattr(self, "_trusted_trial_task", None),
        }
        _write_private_json(self.setup_request_path, request)
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            try:
                setup = self._load_human_setup()
            except RuntimeError:
                setup = None
            if eligible(setup):
                _write_private_json(
                    self.setup_request_path,
                    {
                        **request,
                        "status": "approved",
                        "revision": setup["revision"],
                        "decided_at_unix_s": time.time(),
                    },
                )
                return setup
            time.sleep(0.25)
        _write_private_json(
            self.setup_request_path,
            {
                **request,
                "status": "timed_out",
                "decided_at_unix_s": time.time(),
            },
        )
        raise TimeoutError("Automatic environment setup timed out; report an infrastructure failure")

    def _await_human_evaluation(self, record: dict[str, Any]) -> dict[str, Any]:
        """Publish one binary evaluation request and await the trusted dashboard."""

        request = {
            "schema": "robodojo_evaluation_request_v1",
            **record,
            "status": "pending",
        }
        self._append_operator_trial_event(
            "evaluation_request",
            request_id=request.get("request_id"),
            episode_id=request.get("episode_id"),
            episode_mode=request.get("episode_mode"),
            step_id=request.get("step_id"),
        )
        _write_private_json(self.evaluation_request_path, request)
        wait_timeout = float(getattr(self, "human_evaluation_wait_timeout", 0.0))
        if wait_timeout <= 0:
            return request

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            try:
                decision = json.loads(
                    self.evaluation_request_path.read_text(encoding="utf-8")
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                decision = {}
            if decision.get("request_id") != request["request_id"]:
                time.sleep(0.25)
                continue
            if decision.get("trial_id") != request.get("trial_id"):
                raise RuntimeError("Operator evaluation decision trial ID mismatch")
            status = decision.get("status")
            if status == "completed":
                task_complete = decision.get("task_complete")
                if not isinstance(task_complete, bool):
                    raise ValueError(
                        "Operator evaluation decision must contain boolean task_complete"
                    )
                return {
                    **request,
                    "status": "completed",
                    "task_complete": task_complete,
                    "decided_at_unix_s": decision.get("decided_at_unix_s", time.time()),
                }
            if status != "pending":
                raise ValueError(f"Invalid automatic evaluation status: {status!r}")
            time.sleep(0.25)

        result = {
            **request,
            "status": "timed_out",
            "decided_at_unix_s": time.time(),
        }
        _write_private_json(self.evaluation_request_path, result)
        return result

    @staticmethod
    def _append_human_evaluation_log(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(value), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o600)

    @staticmethod
    def tool_definitions() -> list[dict[str, Any]]:
        action = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 14,
            "maxItems": 14,
            "description": "Absolute [left joint1..6 radians, left gripper opening, right joint1..6 radians, right gripper opening]; grippers use 0 closed and 1 open.",
        }
        ee_action = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 16,
            "maxItems": 16,
            "description": "Absolute [left link6 x,y,z (m), qw,qx,qy,qz, left gripper opening, right link6 x,y,z, qw,qx,qy,qz, right gripper opening] in the observation (environment) frame; unit quaternions; grippers 0 closed, 1 open.",
        }
        target = {
            "type": "object",
            "additionalProperties": False,
            "required": ["position", "quaternion_wxyz", "gripper_closed"],
            "properties": {
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "quaternion_wxyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper_closed": {"type": "boolean"},
                "gripper_opening": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
        dual_target = {
            "type": "object",
            "additionalProperties": False,
            "required": ["left", "right"],
            "properties": {"left": target, "right": target},
        }
        inspect_target = {
            "type": "object",
            "additionalProperties": False,
            "required": ["position", "quaternion_wxyz"],
            "properties": target["properties"],
        }
        rotation = {
            "type": "object",
            "additionalProperties": False,
            "required": ["representation", "value"],
            "properties": {
                "representation": {
                    "type": "string",
                    "enum": [
                        "quaternion_wxyz",
                        "quaternion_xyzw",
                        "euler_xyz_extrinsic_rad",
                        "axis_angle_vector_rad",
                        "rotation_matrix_row_major",
                    ],
                },
                "value": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 9,
                    "description": "Length is 4 for quaternion, 3 for Euler/axis-angle vector, and 9 for a row-major 3x3 matrix.",
                },
            },
        }
        pose = {
            "type": "object",
            "additionalProperties": False,
            "required": ["position_m", "rotation"],
            "properties": {
                "position_m": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "rotation": rotation,
            },
        }
        return [
            {
                "name": "robodojo_list_tasks",
                "description": "List task config names for operator reference. This is read-only, does not start Isaac, and does not let the agent select the task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "robodojo_request_human_setup",
                "description": "Request automatic initial setup or an exploration reset from the saved trial profile. No inputs are accepted; read the returned environment_spec. The operator controls task, seed, visual context, sensors, GPU, and episode budgets before launch. This tool is unavailable after formal mode starts or when the exploration budget is spent.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "robodojo_request_formal_episode",
                "description": "Start the single formal episode automatically using the saved trial profile. No inputs are accepted. Request it after successful exploration with a repeatable plan, or after using the last exploration episode. Formal uses the native step limit and permits no reset or retry. Evaluate its terminal state before calling finish.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "robodojo_status",
                "description": "Return episode status, effective environment spec, and cumulative performance cost without advancing simulation or exposing service filesystem paths.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "robodojo_observe",
                "description": "Return the exact cached post-action observation with RGB/depth camera attachments, calibration, and measured dual-arm state; this does not step physics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "robodojo_pixel_to_position",
                "description": "Convert an agent-selected visible image point to a 3-D environment position without advancing physics or consulting task-object state. Auto mode uses observed metric depth when valid; otherwise it triangulates matching selections of the same physical point from at least two calibrated views.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["observation_step_id", "selections"],
                    "properties": {
                        "observation_step_id": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "The step_id of the exact observation on which the pixels were selected; stale selections are rejected.",
                        },
                        "selections": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["camera", "pixel"],
                                "properties": {
                                    "camera": {
                                        "type": "string",
                                        "enum": [
                                            "cam_high",
                                            "cam_left_wrist",
                                            "cam_right_wrist",
                                        ],
                                    },
                                    "pixel": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "description": "Continuous [u,v] pixel-center coordinates in [0,W-1]x[0,H-1], or [x,y] in [0,1] mapped to [x*(W-1),y*(H-1)].",
                                    },
                                },
                            },
                        },
                        "coordinate_space": {
                            "type": "string",
                            "enum": ["image_pixels", "normalized_0_1"],
                            "default": "image_pixels",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["auto", "depth", "triangulation"],
                            "default": "auto",
                        },
                        "depth_window_radius": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                            "default": 1,
                            "description": "Median-filter radius around each selected depth pixel; choose a point well inside the object silhouette.",
                        },
                        "max_position_spread_m": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 1,
                            "default": 0.05,
                        },
                        "max_triangulation_residual_m": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 1,
                            "default": 0.03,
                            "description": "Maximum point-to-ray residual in metres; default 0.03 m.",
                        },
                    },
                },
            },
            {
                "name": "robodojo_pose_math",
                "description": "Convert rotations, add or extract local/environment pose deltas, and format or inspect exact RoboDojo link6 targets using canonical Isaac Lab math. This is read-only robot geometry: it does not require an episode or access scene state.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["operation"],
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "convert_rotation",
                                "compose_pose",
                                "relative_pose",
                                "format_target",
                                "extract_target",
                            ],
                        },
                        "rotation": rotation,
                        "output_representation": {
                            "type": "string",
                            "enum": [
                                "quaternion_wxyz",
                                "quaternion_xyzw",
                                "euler_xyz_extrinsic_rad",
                                "axis_angle_vector_rad",
                                "rotation_matrix_row_major",
                            ],
                            "default": "quaternion_wxyz",
                        },
                        "base_pose": pose,
                        "delta_pose": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "position_m": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "Optional translation delta; omitted means zero.",
                                },
                                "rotation": rotation,
                            },
                            "description": "Pose delta; omitted position or rotation is identity for that component.",
                        },
                        "target_pose": pose,
                        "delta_frame": {
                            "type": "string",
                            "enum": ["local", "environment"],
                            "default": "local",
                            "description": "Local: T_out=T_base*T_delta. Environment: add translation in environment axes and left-multiply rotation q_delta*q_base.",
                        },
                        "pose": pose,
                        "target_kind": {
                            "type": "string",
                            "enum": ["step_eef", "free_space_move"],
                            "default": "step_eef",
                        },
                        "eef_target": inspect_target,
                        "gripper_closed": {"type": "boolean"},
                        "gripper_opening": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
            },
            {
                "name": "robodojo_step",
                "description": "Execute 1..50 absolute 14-D joint targets per call through actions. Split longer sequences into calls; execution stops at native termination or the remaining episode step limit. Joint control does not specify an EEF pose or Cartesian path; prefer pose-level tools and use this only for an established joint-space target or trajectory. Order is [left joints 1..6, left gripper, right joints 1..6, right gripper], with radians and gripper 0 closed/1 open.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["actions"],
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": action,
                            "minItems": 1,
                            "maxItems": MAX_ACTIONS,
                        }
                    },
                },
            },
            {
                "name": "robodojo_step_ee",
                "description": "Execute 1..50 official native EEF actions per call: absolute link6 pose targets for both arms plus gripper openings, one 25 Hz step each. The environment's own cuRobo IK (32 seeds, seeded at the current joints, no step bound or collision check) converts each row to joint targets, exactly as in the official evaluation; if IK fails for an arm, that arm keeps its previous target. Compare the returned measured eef poses with the targets. Split longer sequences into calls; execution stops at native termination or the remaining episode step limit.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["actions"],
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": ee_action,
                            "minItems": 1,
                            "maxItems": MAX_ACTIONS,
                        }
                    },
                },
            },
            {
                "name": "robodojo_step_eef",
                "description": "Execute 1..50 absolute dual-arm EEF waypoints per call through targets. Split longer sequences into calls; the complete batch is validated before motion. Execution stops at native termination or the remaining episode step limit. Each solve starts from the measured state after the preceding waypoint; the service bounds current-to-target translation/rotation norms to 0.02 m/0.1 rad and clamps each DLS joint delta to +/-0.05 rad. Each waypoint counts one 25 Hz interaction step. Use normalized wxyz quaternions.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["targets"],
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": dual_target,
                            "minItems": 1,
                            "maxItems": MAX_EEF_ACTIONS,
                        }
                    },
                },
            },
            {
                "name": "robodojo_free_space_move",
                "description": "Plan and execute a short or long single-arm target EEF pose with cuRobo, including translations, rotations, or combined pose changes. Prefer this for a simple target pose; use explicit EEF action chunks for explicit intermediate waypoints. Executes ordinary 25Hz joint targets, also returned in transition.steps[*].executed_action. Request preview_only=true and include_trajectory=true to obtain the complete trajectory_preview.actions for inspection or robodojo_step replay. Plans are capped at 50 action rows to bound RGB/depth replies; use an intermediate goal for longer moves. Inspect transition.goal_reached separately from command execution status.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["arm", "target"],
                    "properties": {
                        "arm": {"type": "string", "enum": ["left", "right"]},
                        "target": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["position", "quaternion_wxyz"],
                            "properties": {
                                "position": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "description": "Environment-origin link6 position in metres.",
                                },
                                "quaternion_wxyz": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "Environment-origin link6 orientation as unit wxyz.",
                                },
                                "gripper_opening": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Optional target opening; omission holds the measured opening from plan start.",
                                },
                            },
                        },
                        "preview_only": {
                            "type": "boolean",
                            "default": False,
                            "description": "Set true only when uncertainty requires planning/caching without advancing simulation.",
                        },
                        "include_trajectory": {
                            "type": "boolean",
                            "default": False,
                            "description": "Return the complete 25Hz 14D trajectory_preview.actions, suitable for robodojo_step replay. No display downsampling.",
                        },
                    },
                },
            },
            {
                "name": "robodojo_execute_motion_plan",
                "description": "Execute an acceptable previewed plan only if no action has run since planning and every component of the current 14-D measured state remains within 0.002 of its cached start state. Read-only observations do not stale it; Plan_Stale requires replanning. Executes 25Hz 14D joint targets through the same native path as robodojo_step. Executed targets are transition.steps[*].executed_action. Success means commands completed; inspect goal_reached and measured_goal_error for physical accuracy. It attaches the final frame and saves the complete 25 Hz sequence. Scores remain hidden; terminal task_complete reports native success.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["motion_plan_id"],
                    "properties": {"motion_plan_id": {"type": "string"}},
                },
            },
            {
                "name": "robodojo_fk_preview",
                "description": "Preview a 50x14 absolute joint proposal through robot-only FK. This does not advance simulation or plan around objects.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["actions"],
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": action,
                            "minItems": 50,
                            "maxItems": 50,
                        }
                    },
                },
            },
            {
                "name": "robodojo_request_human_evaluation",
                "description": "Request automatic binary completion evaluation. In exploration, request after strong perceptual evidence. In formal, request when complete or at episode end, including failure, before calling finish. Formal submission immediately shuts down the environment; no retry is allowed after any result, timeout, or error. Exploration evaluation keeps the environment open. Native scores remain private.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["agent_assessment"],
                    "properties": {
                        "agent_assessment": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "Concise perceptual assessment of completion, remaining failures, or uncertainty.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Optional self-assessed confidence, not an evaluator score.",
                        },
                    },
                },
            },
            {
                "name": "robodojo_finish",
                "description": "Close an active episode for intentional abandonment or infrastructure cleanup. Submit formal evaluation before finish, including on terminal episode end or failure; evaluation itself closes formal mode. An ended exploration episode can transition/reset directly. Do not call finish when setup failed and no episode exists. On infrastructure failure, use it only if the active core MCP is reachable and trustworthy. It does not evaluate or reset.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reason": {"type": "string", "default": "operator_finish"}
                    },
                },
            },
        ]

    def _require_active(self) -> SimulatorProcess:
        if not self.sim.active:
            raise RuntimeError(
                "No active RoboDojo episode; request human environment setup first"
            )
        return self.sim

    def _observation_packet(
        self, observation: dict[str, Any], *, transition: Any = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        observation = observation_for(self).clean(observation)
        images = []
        depth_maps = {}
        attachments = []
        content: list[dict[str, Any]] = []
        for camera in CAMERAS:
            if camera not in observation:
                continue
            content_index = len(content) + 1  # index 0 is the JSON text block
            content.append(
                {
                    "type": "image",
                    "data": _png_data(observation[camera]),
                    "mimeType": "image/png",
                }
            )
            images.append(camera)
            attachments.append(
                {"kind": "rgb", "camera": camera, "content_index": content_index}
            )
        for camera in CAMERAS:
            depth_key = f"{camera}_depth_m"
            if depth_key not in observation:
                continue
            encoded, stats = _depth_png_data(observation[depth_key])
            content_index = len(content) + 1
            content.append({"type": "image", "data": encoded, "mimeType": "image/png"})
            depth_maps[camera] = dict(
                stats, content_index=content_index, raw_field=depth_key
            )
            attachments.append(
                {"kind": "depth", "camera": camera, "content_index": content_index}
            )
        packet = {
            "episode_id": self.sim.episode_id,
            "step_id": self.sim.step_id,
            "episode_interaction_steps": self.sim.step_id,
            "total_interaction_steps": self._performance_cost()[
                "total_interaction_steps"
            ],
            "performance_cost": self._performance_cost(),
            "task": self.sim.task,
            "instruction": observation.get("instruction"),
            "states": _jsonable(observation.get("states")),
            "eef_positions": _jsonable(observation.get("eef_positions")),
            "eef_quaternions_wxyz": _jsonable(observation.get("eef_quaternions_wxyz")),
            "environment_origin_world_m": _jsonable(
                observation.get("environment_origin_world_m")
            ),
            "images": images,
            "depth": depth_maps,
            "camera_parameters": _jsonable(observation.get("camera_parameters", {})),
            "attachments": attachments,
            "metadata": _jsonable(self.sim.metadata),
            "environment_spec": _jsonable(getattr(self, "_environment_spec", None)),
        }
        if transition is not None:
            packet["transition"] = _jsonable(transition)
        return observation_for(self).clean(packet), content

    def _save_frame_sequence(
        self,
        frame_records: list[dict[str, Any]],
        frame_contents: list[list[dict[str, Any]]],
        *,
        frequency: float,
        transition: dict[str, Any],
    ) -> tuple[str, str]:
        """Persist images and metadata outside model context on SSD-backed storage."""

        run_id = self.sim.run_id or "unscoped"
        sequence_id = f"seq_{self.sim.step_id:06d}_{uuid.uuid4().hex[:8]}"
        logical_dir = Path("runtime") / "frames" / "robodojo" / run_id / sequence_id
        output_dir = self.frame_root / "robodojo" / run_id / sequence_id
        artifacts: dict[str, bytes] = {}

        saved_frames = []
        for record, content in zip(frame_records, frame_contents, strict=True):
            frame_index = int(record["frame_index"])
            files = []
            for attachment, item in zip(record["attachments"], content, strict=True):
                camera = str(attachment["camera"])
                kind = str(attachment["kind"])
                filename = f"frame_{frame_index:06d}_{camera}_{kind}.png"
                artifacts[filename] = base64.b64decode(item["data"])
                files.append(
                    {
                        "kind": kind,
                        "camera": camera,
                        "path": (logical_dir / filename).as_posix(),
                    }
                )
            saved = {
                key: value
                for key, value in record.items()
                if key not in {"attachments", "depth"}
            }
            saved["depth"] = {
                camera: {
                    key: value
                    for key, value in values.items()
                    if key != "content_index"
                }
                for camera, values in record["depth"].items()
            }
            saved["files"] = files
            saved_frames.append(saved)

        manifest = {
            "sequence_id": sequence_id,
            "source": "robodojo",
            "artifact_id": run_id,
            "frequency_hz": frequency,
            "frame_count": len(saved_frames),
            "transition": _agent_safe(transition),
            "frames": saved_frames,
        }
        artifacts["manifest.json"] = (
            json.dumps(observation_for(self).clean(_jsonable(manifest)), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        write_artifacts(output_dir, artifacts)
        return sequence_id, (logical_dir / "manifest.json").as_posix()

    def _observation_sequence_packet(
        self,
        observations: list[dict[str, Any]],
        *,
        transition: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Save every frame, but attach only the final frame to the MCP result."""
        if not observations:
            observation = self.sim.request("teacher_observation")
            return self._observation_packet(observation, transition=transition)
        frame_records = []
        frame_contents: list[list[dict[str, Any]]] = []
        steps = transition.get("steps") if isinstance(transition, dict) else None
        frequency = float((self.sim.metadata or {}).get("control_frequency_hz", 25.0))
        for frame_index, observation in enumerate(observations):
            local, local_content = self._observation_packet(observation)
            step_id = (
                int(steps[frame_index]["step_id"])
                if isinstance(steps, list) and frame_index < len(steps)
                else int(self.sim.step_id - len(observations) + frame_index + 1)
            )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "time_from_execution_start_s": (frame_index + 1) / frequency,
                    "step_id": step_id,
                    "states": local["states"],
                    "eef_positions": local["eef_positions"],
                    "eef_quaternions_wxyz": local["eef_quaternions_wxyz"],
                    "environment_origin_world_m": local["environment_origin_world_m"],
                    "images": local["images"],
                    "depth": local.get("depth", {}),
                    "camera_parameters": local.get("camera_parameters", {}),
                    "attachments": local["attachments"],
                }
            )
            frame_contents.append(local_content)
        final = frame_records[-1]
        final_content = frame_contents[-1]
        sequence_id, manifest_path = self._save_frame_sequence(
            frame_records,
            frame_contents,
            frequency=frequency,
            transition=transition,
        )
        sequence_dir = Path(manifest_path).parent.as_posix()
        packet = {
            "episode_id": self.sim.episode_id,
            "step_id": self.sim.step_id,
            "episode_interaction_steps": self.sim.step_id,
            "total_interaction_steps": self._performance_cost()[
                "total_interaction_steps"
            ],
            "performance_cost": self._performance_cost(),
            "task": self.sim.task,
            "instruction": observations[-1].get("instruction"),
            "states": final["states"],
            "eef_positions": final["eef_positions"],
            "eef_quaternions_wxyz": final["eef_quaternions_wxyz"],
            "environment_origin_world_m": final["environment_origin_world_m"],
            "images": final["images"],
            "depth": final["depth"],
            "camera_parameters": final["camera_parameters"],
            "attachments": final["attachments"],
            "metadata": _jsonable(self.sim.metadata),
            "environment_spec": _jsonable(getattr(self, "_environment_spec", None)),
            "transition": _jsonable(transition),
            "frame_sequence": {
                "sequence_id": sequence_id,
                "frequency_hz": frequency,
                "frame_count": len(frame_records),
                "manifest_path": manifest_path,
                "frame_path_pattern": (
                    f"{sequence_dir}/frame_{{frame_index:06d}}_{{camera}}_{{kind}}.png"
                ),
                "frame_index_range": [0, len(frame_records) - 1],
                "streams": [
                    {
                        "camera": item["camera"],
                        "kind": item["kind"],
                    }
                    for item in final["attachments"]
                ],
                "final_frame": {
                    "frame_index": final["frame_index"],
                    "time_from_execution_start_s": final["time_from_execution_start_s"],
                    "step_id": final["step_id"],
                },
                "returned_images": "final_frame_only",
                "inspection": (
                    "If the final frame is insufficient, choose an index in "
                    "frame_index_range, substitute frame_path_pattern, and use the "
                    "image viewer. No script or interpreter is needed."
                ),
            },
        }
        return observation_for(self).clean(packet), final_content

    def _motion_result_packet(
        self, result: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Separate raw frame arrays from JSON motion metadata before encoding."""
        result_without_frames = dict(result)
        execution = (
            result.get("execution")
            if isinstance(result.get("execution"), dict)
            else result
        )
        frames = execution.get("frames") if isinstance(execution, dict) else None
        execution_without_frames = (
            {key: value for key, value in execution.items() if key != "frames"}
            if isinstance(execution, dict)
            else execution
        )
        if execution is not result:
            result_without_frames["execution"] = execution_without_frames
        else:
            result_without_frames = execution_without_frames
        if isinstance(execution, dict) and execution.get("step_id") is not None:
            self.sim.step_id = int(execution["step_id"])
        if isinstance(frames, list) and frames:
            packet, images = self._observation_sequence_packet(
                frames,
                transition=execution_without_frames,
            )
        else:
            observation = self.sim.request("teacher_observation")
            packet, images = self._observation_packet(observation)
        packet["motion_plan"] = _jsonable(result_without_frames)
        return packet, images

    def _call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")
        if getattr(self, "_formal_evaluation_submitted", False):
            if name == "robodojo_finish":
                return {"status": "already_closed", "formal_attempt_ended": True}, []
            if name not in {"robodojo_status", "robodojo_list_tasks"}:
                raise RuntimeError(
                    "The formal attempt ended when evaluation was submitted; "
                    "the environment is shut down. No actions, evaluation retry, "
                    "reset, or second formal episode are allowed."
                )
        validate_motion_request(name, arguments)
        if name == "robodojo_list_tasks":
            task_dir = source_root(self.project_root) / "task" / "RoboDojo" / "config"
            tasks = sorted(
                path.stem
                for path in task_dir.glob("*.yml")
                if not path.stem.startswith("_")
            )
            return {"tasks": tasks, "count": len(tasks)}, []
        if name in SETUP_TOOLS:
            if arguments:
                raise ValueError("Human setup request does not accept agent parameters")
            formal_requested = name == "robodojo_request_formal_episode"
            if getattr(self, "_formal_episode_started", False):
                raise RuntimeError("The formal episode is final; no reset or second formal episode is allowed in this run")
            if (not formal_requested and getattr(self, "_environment_setup_count", 0)
                    and getattr(self, "_exploration_episodes_started", 0) >= self._max_exploration_episodes):
                raise RuntimeError("Exploration episode budget exhausted; no further exploration reset is allowed")
            setup = self._await_human_setup(
                request_kind=(
                    "formal_episode" if formal_requested else "setup_or_reset"
                ),
                require_step_limit=formal_requested,
            )
            self._validate_trial_task(setup["task"])
            revision = setup["revision"]
            if revision in self._consumed_setup_revisions:
                raise RuntimeError(
                    "This human setup revision was already consumed; the operator "
                    "must provide a new revision for another environment or reset"
                )
            self._validate_episode_start(setup, formal_requested=formal_requested)
            setup = self._reserve_exploration_seed(setup)
            if self.episode_timeout_seconds is None:
                from services.robodojo.timeouts import task_timeouts
                self.episode_timeout_seconds = task_timeouts(setup['task'])['episode_seconds']
            replacing_episode = getattr(self, "_environment_setup_count", 0) > 0
            starting_formal = bool(setup["enforce_step_limit"])
            context_kind = setup["demonstration_context"]
            video_path = (
                None
                if context_kind == "none"
                else cached_demo_path(self.demo_cache_root, setup["task"])
            )
            if video_path is not None and not video_path.is_file():
                raise RuntimeError(
                    "The operator-selected task demonstration is not installed; "
                    "approve setup through the dashboard or run the trusted demo "
                    "setup script"
                )
            terminal_path = (
                cached_terminal_path(self.demo_cache_root, setup["task"])
                if context_kind == "terminal_state"
                else None
            )
            if terminal_path is not None and not terminal_path.is_file():
                raise RuntimeError(
                    "The operator-selected terminal demonstration is not installed; "
                    "approve setup through the dashboard or run the trusted demo "
                    "setup script"
                )
            self._demonstration_context = provision_trial_demonstration(
                video_path=video_path,
                terminal_path=terminal_path,
                target=self.demonstration_root,
                task=setup["task"],
                context_kind=context_kind,
            )
            if getattr(self.sim, "process", None) is not None:
                try:
                    if self.sim.active:
                        self.sim.finish(
                            "human_approved_formal_episode"
                            if starting_formal
                            else "human_approved_reset"
                        )
                finally:
                    self._account_current_episode()
                    self.sim.stop()
            task = setup["task"]
            episode_mode = "formal" if setup["enforce_step_limit"] else "exploration"
            run_id = (
                f"{task}_{episode_mode}_"
                f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_"
                f"{revision}_{uuid.uuid4().hex[:8]}"
            )
            reset = self.sim.start(
                task=task,
                seed=setup["seed"],
                eval_seed=setup["eval_seed"],
                run_id=run_id,
                sim_gpu=setup["sim_gpu"],
                sim_port=setup["sim_port"],
                startup_timeout=setup["startup_timeout"],
                enable_depth=setup["include_depth"],
                enable_camera_parameters=setup["include_camera_parameters"],
                enforce_step_limit=setup["enforce_step_limit"],
                episode_timeout_seconds=self.episode_timeout_seconds,
            )
            try:
                self._record_consumed_setup_revision(revision)
            except Exception:
                try:
                    self.sim.finish("setup_revision_record_failed")
                finally:
                    self.sim.stop()
                raise
            self._environment_setup_count = (
                getattr(self, "_environment_setup_count", 0) + 1
            )
            if not replacing_episode:
                self._max_exploration_episodes = setup["max_exploration_episodes"]
            if not starting_formal:
                self._exploration_episodes_started = getattr(self, "_exploration_episodes_started", 0) + 1
            if replacing_episode:
                self._environment_reset_count = (
                    getattr(self, "_environment_reset_count", 0) + 1
                )
            if starting_formal:
                self._formal_episode_started = True
            self._environment_spec = self._agent_environment_spec(setup)
            observation = self.sim.request("teacher_observation")
            packet, images = self._observation_packet(observation)
            packet.update(
                reset=_jsonable(reset),
                environment_spec=_jsonable(self._environment_spec),
                performance_cost=self._performance_cost(),
                artifact_id=self.sim.run_id,
                artifact_storage="service_managed_not_agent_accessible",
            )
            self._human_evaluation_steps.clear()
            self._human_evaluation_count = 0
            return packet, images
        if name == "robodojo_status":
            if self.sim.process is None:
                return {
                    "active": False,
                    "environment_spec": _jsonable(
                        getattr(self, "_environment_spec", None)
                    ),
                    "performance_cost": self._performance_cost(),
                    "total_interaction_steps": self._performance_cost()[
                        "total_interaction_steps"
                    ],
                }, []
            return {
                "active": self.sim.active,
                "finished": self.sim.finished,
                "task": self.sim.task,
                "episode_id": self.sim.episode_id,
                "step_id": self.sim.step_id,
                "episode_interaction_steps": self.sim.step_id,
                "total_interaction_steps": self._performance_cost()[
                    "total_interaction_steps"
                ],
                "performance_cost": self._performance_cost(),
                "metadata": _jsonable(self.sim.metadata),
                "environment_spec": _jsonable(getattr(self, "_environment_spec", None)),
                "artifact_id": self.sim.run_id,
                "artifact_storage": "service_managed_not_agent_accessible",
            }, []
        if name == "robodojo_pose_math":
            from services.robodojo.pose_math import pose_math

            return _jsonable(pose_math(self.project_root, arguments)), []
        if name == "robodojo_finish":
            sim = self._require_active()
            reason = (
                str(arguments.get("reason", "operator_finish")).strip()
                or "operator_finish"
            )
            result = sim.finish(reason)
            self._account_current_episode()
            sim.stop()
            return {
                "result": _jsonable(result),
                "environment_spec": _jsonable(getattr(self, "_environment_spec", None)),
                "performance_cost": self._performance_cost(),
                "total_interaction_steps": self._performance_cost()[
                    "total_interaction_steps"
                ],
                "artifact_id": sim.run_id,
                "artifact_storage": "service_managed_not_agent_accessible",
            }, []
        sim = self._require_active()
        if name == "robodojo_observe":
            observation = sim.request("teacher_observation")
            return self._observation_packet(observation)
        if name == "robodojo_pixel_to_position":
            requested_step = int(arguments["observation_step_id"])
            if requested_step != sim.step_id:
                raise ValueError(
                    f"Stale pixel selection: observation_step_id={requested_step}, "
                    f"current step_id={sim.step_id}; observe again and reselect"
                )
            observation = sim.request("teacher_observation")
            from services.robodojo.pixel_geometry import (
                locate_pixel_selections,
            )

            result = locate_pixel_selections(
                observation,
                arguments["selections"],
                coordinate_space=str(arguments.get("coordinate_space", "image_pixels")),
                method=str(arguments.get("method", "auto")),
                depth_window_radius=int(arguments.get("depth_window_radius", 1)),
                max_position_spread_m=float(
                    arguments.get("max_position_spread_m", 0.05)
                ),
                max_triangulation_residual_m=float(
                    arguments.get("max_triangulation_residual_m", 0.03)
                ),
            )
            result.update(
                episode_id=sim.episode_id,
                observation_step_id=requested_step,
                observation_is_current=True,
            )
            return _jsonable(result), []
        if name in ("robodojo_step", "robodojo_step_ee"):
            import numpy as np

            actions = arguments["actions"]
            matrix = np.asarray(actions, dtype=np.float32)
            result = sim.request("chunk_step", actions=matrix,
                                 action_type="joint" if name == "robodojo_step" else "ee")
            sim.step_id = int(result["step_id"])
            frames = result.get("frames", [])
            transition = {
                key: value for key, value in result.items() if key != "frames"
            }
            return self._observation_sequence_packet(frames, transition=transition)
        if name == "robodojo_step_eef":
            import numpy as np
            targets = arguments["targets"]  # Entire batch validated before any RPC.
            rows, frames = [], []
            diagnostics = []
            for target in targets:
                proposal = sim.request("eef_joint_target", targets=target)
                diagnostics.append(_jsonable(proposal.get("diagnostics")))
                result = sim.request(
                    "chunk_step",
                    actions=np.asarray([proposal["action"]], dtype=np.float32),
                )
                rows.extend(result["steps"])
                frames.extend(result.get("frames", []))
                sim.step_id = int(result["step_id"])
                if any(
                    item.get("terminated") or item.get("truncated")
                    for item in result["steps"]
                ):
                    break
            return self._observation_sequence_packet(
                frames,
                transition={
                    "episode_id": sim.episode_id,
                    "step_id": sim.step_id,
                    "observation_frequency_hz": float(
                        (sim.metadata or {}).get("control_frequency_hz", 25.0)
                    ),
                    "frame_count": len(frames),
                    "steps": rows,
                    "ik_diagnostics": diagnostics,
                    "joint_target_contract": JOINT_TARGET_CONTRACT,
                },
            )
        if name == "robodojo_free_space_move":
            preview_only = arguments.get("preview_only", False)
            include_trajectory = arguments.get("include_trajectory", False)
            result = sim.request(
                "free_space_move",
                arm=str(arguments["arm"]),
                target=arguments["target"],
                preview_only=preview_only,
                include_trajectory=include_trajectory,
            )
            return self._motion_result_packet(result)
        if name == "robodojo_execute_motion_plan":
            result = sim.request(
                "execute_motion_plan", motion_plan_id=str(arguments["motion_plan_id"])
            )
            return self._motion_result_packet(result)
        if name == "robodojo_fk_preview":
            import numpy as np

            actions = np.asarray(arguments.get("actions"), dtype=np.float32)
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError("actions must be a finite 50x14 matrix")
            return _jsonable(sim.request("fk_preview", actions=actions)), []
        if name == "robodojo_request_human_evaluation":
            assessment = str(arguments.get("agent_assessment", "")).strip()
            if not assessment:
                raise ValueError("agent_assessment must be non-empty")
            if len(assessment) > 2000:
                raise ValueError("agent_assessment must be at most 2000 characters")
            confidence = arguments.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence must be in [0, 1]")
            if sim.step_id in self._human_evaluation_steps:
                raise ValueError(
                    "Human evaluation was already requested for this simulator step; "
                    "gather new perceptual evidence before requesting another costly review"
                )
            formal = (getattr(self, "_environment_spec", {}) or {}).get(
                "episode_mode"
            ) == "formal"
            if formal:
                # Latch before any I/O: even observation/finish/approval errors
                # must not reopen the final attempt.
                self._formal_evaluation_submitted = True
                self._formal_episode_started = True
                try:
                    observation = sim.request("teacher_observation")
                finally:
                    try:
                        sim.finish("formal_evaluation_submitted")
                    finally:
                        self._account_current_episode()
                        sim.stop()
            else:
                observation = sim.request("teacher_observation")
            request_id = uuid.uuid4().hex
            self._human_evaluation_steps.add(sim.step_id)
            self._human_evaluation_count += 1
            self._total_human_evaluation_count = (
                getattr(self, "_total_human_evaluation_count", 0) + 1
            )
            record = {
                "request_id": request_id,
                "trial_id": getattr(self, "_trusted_trial_id", None),
                "episode_id": sim.episode_id,
                "step_id": sim.step_id,
                "task": sim.task,
                "episode_mode": (getattr(self, "_environment_spec", {}) or {}).get(
                    "episode_mode", "unknown"
                ),
                "agent_assessment": assessment,
                "confidence": confidence,
                "requested_at_unix_s": time.time(),
                "request_number": self._human_evaluation_count,
                "evaluation_kind": "binary_task_completion",
                "fixed_evaluation": "Is the instructed task complete in the current perceptual evidence?",
                "allowed_result": {"task_complete": [True, False]},
                "performance_cost": self._performance_cost(),
            }
            if sim.output_dir is None:
                raise RuntimeError("Missing service-managed episode output")
            request_log = sim.output_dir / "human_evaluation_requests.jsonl"
            self._append_human_evaluation_log(
                request_log, {**record, "status": "pending"}
            )
            decision = self._await_human_evaluation(record)
            if decision["status"] != "pending":
                self._append_human_evaluation_log(request_log, decision)
            packet, images = self._observation_packet(observation)
            packet["human_evaluation_request"] = {
                **decision,
                "cost_notice": (
                    "Evaluation counts toward performance cost. Use exploration "
                    "evaluations selectively; always evaluate the final formal attempt."
                ),
                "native_task_evaluation_disclosed": False,
                "result_delivery": "trusted_operator_dashboard",
                "evaluation_mode": "automatic",
            }
            if formal:
                packet["formal_attempt_ended"] = True
                packet["environment_closed"] = True
                packet["next_action"] = (
                    "The formal attempt is over and the environment is shut down. "
                    "Report this evaluation result and stop. No further actions, "
                    "evaluation retries, resets, or formal attempts are allowed."
                )
            elif decision["status"] == "timed_out":
                packet["next_action"] = (
                    "Automatic evaluation timed out. Report an infrastructure "
                    "failure; this is not a task completion result."
                )
            elif (
                decision.get("task_complete") is True
                and record["episode_mode"] == "exploration"
            ):
                packet["next_action"] = (
                    "This exploration result is useful evidence but does not count "
                    "as formal success. Request a fresh formal episode and reproduce "
                    "the successful strategy within its step limit."
                )
            packet["performance_cost"] = self._performance_cost()
            return packet, images
        raise ValueError(f"Unknown MCP tool: {name}")

    def call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        profile = observation_for(self)
        if name == "robodojo_pixel_to_position" and not profile.geometry:
            raise PermissionError("Pixel-to-position is unavailable with RGB-only observations")
        value, images = self._call(name, arguments)
        return profile.clean(_agent_safe(value)), images

    def close(self) -> None:
        try:
            if self.sim.active:
                try:
                    self.sim.finish("mcp_disconnect")
                except Exception as exc:  # noqa: BLE001 - disconnect cleanup is best effort
                    if os.environ.get("ROBODOJO_MCP_DEBUG"):
                        print(
                            f"[robodojo-mcp] finish on disconnect failed: {exc}",
                            file=sys.stderr,
                        )
        finally:
            self.sim.stop()


def _response(
    request_id: Any, result: Any = None, error: dict[str, Any] | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return response


def _serve(server_factory=RoboDojoMCP, instructions=MCP_AGENT_INSTRUCTIONS) -> None:
    server = server_factory()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            request = None
            try:
                request = json.loads(line)
                request_id = request.get("id")
                method = request.get("method")
                if method == "initialize":
                    result = {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                        "instructions": instructions,
                    }
                    response = _response(request_id, result)
                elif method in (
                    "notifications/initialized",
                    "notifications/cancelled",
                    "ping",
                ):
                    if "id" not in request:
                        continue
                    response = _response(request_id, {})
                elif method == "tools/list":
                    definitions = server.tool_definitions()
                    definitions = (server.contract.select(definitions) if hasattr(server, "contract")
                                   else observation_for(server).tools(definitions))
                    response = _response(request_id, {"tools": definitions})
                elif method == "tools/call":
                    params = request.get("params") or {}
                    value, images = server.call(
                        params.get("name", ""), params.get("arguments") or {}
                    )
                    response = _response(
                        request_id,
                        {
                            "content": [_content_text(value), *images],
                            "isError": False,
                            # Codex prefers structuredContent over content; supplying
                            # both can discard image blocks before model ingestion.
                            **({} if images else {"structuredContent": _jsonable(value)}),
                        },
                    )
                else:
                    response = _response(
                        request_id,
                        error=_error(f"Method not found: {method}", code=-32601),
                    )
            except Exception as exc:  # noqa: BLE001 - return JSON-RPC errors to the caller
                if os.environ.get("ROBODOJO_MCP_DEBUG"):
                    traceback.print_exc(file=sys.stderr)
                else:
                    print(
                        f"[robodojo-mcp] {type(exc).__name__}: {exc}", file=sys.stderr
                    )
                if (
                    request_id is None
                    and isinstance(request, dict)
                    and "id" not in request
                ):
                    continue
                response = _response(
                    request_id, error=_error(f"{type(exc).__name__}: {exc}")
                )
            sys.stdout.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            sys.stdout.flush()
    finally:
        server.close()


if __name__ == "__main__":
    _serve()
