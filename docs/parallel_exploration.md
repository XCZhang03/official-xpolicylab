# Parallel exploration infrastructure

Opt-in through `exploration_envs` (1–8, default 1). The dashboard's
**Exploration environments** field and `scripts/configure_auto_research.py
--exploration-envs N` set it.

## Execution and budget

Each slot has an independent Isaac process, scene, step counter and recorder on
the operator-selected simulator GPU. This is process-level
parallelism, not Isaac's vectorized scene API. Memory use scales with live slots;
eight is a configuration ceiling, not a promise that eight fit on every GPU.
Startup is lazy: an unused slot consumes no GPU resources or episode.

The host pool is shared by the main agent, all subagents and development Python.
Calls to different slots execute concurrently, including calls pipelined over
one MCP connection. Calls to the same slot serialize.
Agents share a workspace; slot serialization is not exclusive ownership by a
particular subagent. The main agent must assign distinct slots to collaborators.

One durable ledger covers **all initial starts, resets and isolated rehearsals**.
Every reservation consumes a distinct operator-selected seed before simulator
startup. Failed startup is not refunded. Example: four slots and ten episodes
permit ten episode starts total, not ten per slot.
An exhausted budget blocks starts/resets, not actions in existing live episodes.
A rejected reset leaves the current scene intact.

A worker transport failure closes that slot; it cannot silently retry an action.
The host shuts down owned worker/simulator processes, without touching other
sessions. Restarting a pool with interrupted reservations fails closed and
requires operator recovery; it does not grant a fresh budget.

## MCP surface

Single-slot sessions retain their existing schemas. Multi-slot sessions require
`env_id` on robot and exploration lifecycle calls:

```python
ctx.call("start_episode", env_id=0)
ctx.call("robodojo_observe", env_id=0)
ctx.call("robodojo_step", env_id=0, actions=joint_targets)
ctx.call("evaluate", env_id=0)
ctx.call("finish", env_id=0)

# Optional development-Python convenience:
robot = ctx.env(1)
robot.call("start_episode")
robot.call("robodojo_step", actions=joint_targets)
```

`status()` includes the pool summary. Status, registration, rehearsal and
submission are session-wide and take no `env_id`.
Episode-scoped motion arguments and the 25 Hz action contract are unchanged.

Images remain MCP image blocks. Replies identify their slot and global episode
number, shared budget and selected episode's artifact paths. Native recordings
remain private; sanitized traces/images are published under the normal research
artifact tree. The dashboard's live view follows the newest native observation.

## Formal boundaries

Research registration/rehearsal/submission require outstanding exploration calls
to finish. Rehearsal/submission then close all live slots. Rehearsal reserves
from the shared ledger; the exact-successful-rehearsal submission gate is unchanged.
Isolated scripts use the ordinary unbound `ctx`, with no `env_id`.
`ctx.env(i)` is a development-only routing convenience.

Codex subagent support (`multi_agent`) is enabled in every session, with at most
`exploration_envs + 1` concurrent threads: one per exploration slot plus the read-only
layout-overfit auditor (`subagent-audit` skill), which never uses a slot.

## Verification

`local_tests/unit/test_parallel_exploration.py` covers budget races, real socket
concurrency, slot serialization, image-block preservation, shared rehearsal seeds,
and fail-closed recovery.

`scripts/smoke_parallel_exploration.py --gpu GPU-...` starts two real environments,
executes measured-position joint actions concurrently, checks independent step
counters and PNGs, rejects excess resets, verifies published sequences, and closes
both workers. It does not call Codex or evaluate task-solving quality.
