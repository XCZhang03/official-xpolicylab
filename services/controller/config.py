"""Operator-owned configuration. No values are accepted from controller code."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

from services.robodojo.demonstrations import DEMONSTRATION_CONTEXTS
from services.robodojo.timeouts import task_timeouts

FORMAL_EPISODES = {'auto-research': 50}


def with_task_timeouts(config):
    seconds = task_timeouts(config.task)['episode_seconds']
    return replace(config, development=replace(config.development, wall_seconds=seconds),
                   formal=replace(config.formal, wall_seconds=seconds))


@dataclass(frozen=True)
class Limits:
    wall_seconds: int
    memory_mb: int
    cpus: float
    pids: int
    scratch_mb: int
    artifact_bytes: int

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == "cpus":
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("cpus must be positive and finite")
            elif type(value) is not int or value < 1:
                raise ValueError(f"Invalid limit: {key}")


# An exploration seed names one saved layout: collection * LAYOUT_STRIDE + layout index.
# Keys below LAYOUT_STRIDE are collection 0, as in earlier configurations.
LAYOUT_STRIDE = 1000
OFFICIAL_COLLECTIONS = (0, 1, 2)
# Default schedule: explore the official collections in order; the formal batch uses
# collection 3, operator-generated novel layouts outside the official set.
EXPLORATION_COLLECTIONS = OFFICIAL_COLLECTIONS
FORMAL_COLLECTION = 3


def layout_key(collection, index):
    if type(collection) is not int or type(index) is not int or collection < 0 or not 0 <= index < LAYOUT_STRIDE:
        raise ValueError('Invalid saved layout')
    return collection * LAYOUT_STRIDE + index


def exploration_layouts(collections, counts, start, episodes):
    """Every layout of the first collection from ``start``, then the next collection."""
    return [layout_key(c, i) for c in collections for i in range(start, counts[c])][:episodes]


def layout_of(key):
    """(Eval_Layout collection, layout index) for an exploration seed."""
    return divmod(key, LAYOUT_STRIDE)


@dataclass(frozen=True)
class Configuration:
    root: Path
    image: str
    task: str
    exploration_seeds: tuple[int, ...]
    formal_seed: int
    eval_seed: int
    sim_gpu: str
    controller_gpu: str | None
    training_gpu: str | None
    development: Limits
    formal: Limits
    training: Limits
    workspace_mb: int = 20480
    demonstration_context: str = 'none'  # Preserve older configurations; operator UI/CLI default to final image.
    formal_episodes: int = 1  # Legacy configurations remain single-episode.
    formal_eval_seed: int | None = None
    formal_workers: int = 1  # Concurrent formal episodes; legacy configurations stay sequential.
    mode: str = 'auto-research'
    observation_profile: str = 'official'
    exploration_envs: int = 1
    agent_cli: str = 'codex'  # 'codex' or 'claude' (harness/claude_cli)
    agent_provider: str = 'openrouter'  # Claude only: openrouter, claude-login or anthropic

    @property
    def formal_collection(self):
        return self.eval_seed if self.formal_eval_seed is None else self.formal_eval_seed

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value.pop("training_seconds", None)  # Retired, unenforced training allowance.
        for key in ('student_model', 'student_reasoning_effort', 'student_max_model_calls'):
            value.pop(key, None)  # Retired Teacher–Student fields.
        if value.pop('teacher_source_session', None) is not None:
            raise ValueError('Teacher–Student sessions are not supported in this release')
        value["root"] = Path(value["root"]).resolve()
        value["exploration_seeds"] = tuple(value["exploration_seeds"])
        for key in ("development", "formal", "training"):
            # Legacy operator files may include superseded per-phase API caps;
            # Gemini admission uses only the shared session dollar ledger.
            limits = dict(value[key])
            limits.pop("gemini_calls", None)
            limits.pop("gemini_tokens", None)
            value[key] = Limits(**limits)
        return cls(**value)

    def __post_init__(self):
        from services.mcp_contract import Contract
        Contract.from_config(self)
        if self.agent_cli not in ('codex', 'claude'):
            raise ValueError('agent_cli must be codex or claude')
        if self.agent_provider not in ('openrouter', 'claude-login', 'anthropic'):
            raise ValueError('agent_provider must be openrouter, claude-login or anthropic')
        if self.mode != 'auto-research':
            raise ValueError('This release supports only auto-research')
        if self.observation_profile != 'official':
            raise ValueError('This release supports only the official observation profile')
        if self.demonstration_context not in DEMONSTRATION_CONTEXTS:
            raise ValueError('Invalid demonstration_context')
        from services.storage_root import artifact_root, ENVIRONMENT
        if not self.root.is_relative_to(artifact_root()):
            raise ValueError(f"Controller artifacts must be under {artifact_root()} (set {ENVIRONMENT} to change it)")
        if type(self.workspace_mb) is not int or not 64 <= self.workspace_mb <= 1048576:
            raise ValueError("workspace_mb must be 64..1048576")
        if not re.fullmatch(r"[A-Za-z0-9_./:-]+@sha256:[a-f0-9]{64}|sha256:[a-f0-9]{64}", self.image):
            raise ValueError("Use a locally provisioned, digest-pinned image")
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.task):
            raise ValueError("Invalid task")
        if not self.exploration_seeds or len(self.exploration_seeds) > 1000:
            raise ValueError("Provide 1..1000 exploration episodes")
        if any(type(s) is not int or s < 0 for s in (*self.exploration_seeds, self.formal_seed, self.eval_seed)):
            raise ValueError("Seeds must be nonnegative integers")
        if type(self.formal_episodes) is not int or not 1 <= self.formal_episodes <= 100:
            raise ValueError('formal_episodes must be 1..100')
        if type(self.formal_workers) is not int or not 1 <= self.formal_workers <= 8:
            raise ValueError('formal_workers must be 1..8')
        if self.formal_eval_seed is not None and (type(self.formal_eval_seed) is not int or self.formal_eval_seed < 0):
            raise ValueError('formal_eval_seed must be a nonnegative integer')
        if len(set(self.exploration_seeds)) != len(self.exploration_seeds):
            raise ValueError("Every exploration episode requires a different seed")
        for gpu in (self.sim_gpu, self.controller_gpu, self.training_gpu):
            if gpu is not None and not re.fullmatch(r"GPU-[a-fA-F0-9-]+", gpu):
                raise ValueError("Use explicit GPU UUIDs, not ordinals or 'all'")
        if not self.sim_gpu:
            raise ValueError("A simulator GPU is required")
        if self.training_gpu == self.sim_gpu or self.controller_gpu == self.sim_gpu:
            raise ValueError("Reserve a separate GPU for untrusted code")
