# Agent workspace boundary

`auto_research_agent/` is the source template for everything an agent can see.
`harness/codex_cli/auto_research.py:deploy` copies it into each fresh session, then
adds session-specific files. Edits to the template affect new sessions only.

## Contents of a session workspace (`/workspace` in the container)

- **Instructions:**
  - `AGENTS.md`: workflow, budgets and bundle rules;
  - `START_PROMPT.md`: the injected kickoff;
  - `TASK.md`: the task's public scoring rubric from the reviewed snapshot
    `services/robodojo/task_rubrics.json`, plus the demonstration pointer. A
    missing rubric blocks the launch.
- **Session contract:** `MCP_CONTRACT.json` and `MCP_SESSION.md`, rendered by
  `services/mcp_workspace.py` from the same `Contract` that the MCP endpoint
  enforces (official profile, environment count).
- **Skills in `.agents/skills/`:**
  - `manual-robot-control`, `autonomous-control`, `motion-toolkit`;
  - `rgb-position-calibration` (with a generated task-specific `CALIBRATION.md`);
  - `wrist-camera-inspection`, `in-context-action-learning`, `experience-memory`.
- **`api/runtime.py`:** the development `Context` client for the MCP socket.
- **Writable areas:**
  - `code/<project>/`: the bundle source;
  - `memory/`: notes;
  - `output/`: `ctx.output_dir`.
- **Demonstrations:** `runtime/demonstrations/`. These are RGB images and a
  manifest copied from the cached official task clip; there are no actions or
  states.
- **Task source (`task_source/`, read-only):** the official RoboDojo source for this
  task only. It is packed by `scripts/build_task_sources.py` into
  `runtime/task-sources/<task>/` and injected at launch by
  `services/robodojo/task_source.py`, which verifies the commit and sha256 first; a
  missing package blocks the launch. It holds:
  - the task module whole, plus per-definition excerpts of exactly the RoboDojo
    code it depends on (`scripts/task_source_deps.py`): its reward checks and their
    implementations, the official episode flow, and the framework methods it calls;
  - the task and env configs;
  - the task's own objects only (categories its config declares; no scene fixtures
    or random distractors), each model used by the config or the official layouts
    (collections 0–2): `metadata.json`,
    `description.json`, the original `.usdz`, and an exported `mesh.npz` (triangles per link) and
    `articulation.json`;
  - small scripted trajectories the task code loads.

  Saved layouts are never included. It is for understanding the task and planning
  only: bundles stay closed-loop from RGB and robot state, and isolated runs do not
  receive it.
- **Published run evidence:** `runtime/autonomous_controller/`, a read-only mount
  of the session's sanitized published tree. It holds traces, per-call replies,
  frame sequences and isolated-run exports.

The workspace and Codex state share one fixed-size SSD filesystem
(`workspace_mb`, 20 GiB by default).

## What the agent never receives

- harness, MCP or simulator source;
- private session state (`state.json`, seeds, configuration, scores);
- operator logs, other sessions, host credentials, the Docker socket;
- the network.

Native reward, scores and intermediate success never cross the MCP. Only terminal
`task_complete` and the trusted `evaluate` result do.

## Container

Codex, Bash and Python run in the digest-pinned `robodojo-official:dev` image with:
- `--network=none`, a read-only root, all capabilities dropped,
  `no-new-privileges`, private IPC, a non-root UID and CPU/RAM/PID limits;
- a `/tmp` tmpfs;
- the operator-selected research GPU, used for cuRobo planning.

The image contains the same wheelhouse as the official policy environment
(torch, warp, the cuRobo fork, `robodojo_toolkit`), plus a prebuilt kernel cache.
It also has a development-only `usd-core` layer (`pxr`), which inspects
`task_source/` assets and is not part of submissions. `trimesh` from the wheelhouse
builds meshes from the exported `mesh.npz` arrays.

Bind mounts:
- the workspace and Codex home;
- the published tree (read-only);
- the host model-relay and MCP sockets;
- the container bridge and the Codex binary.

Model credentials stay in the host relay; the robot MCP runs on the host.

Rehearsal and formal runs use a separate fresh container. It holds the frozen
bundle at its registered `/workspace/code/<project>/` path, the runtime client, a
private MCP channel with only the four official robot tools (through the official
AgentBundle bridge), a byte-capped
`/workspace/output`, and read-only frames from the current run. It does not
include the live workspace or memory. See the
[infrastructure contract](controller_backend.md).
