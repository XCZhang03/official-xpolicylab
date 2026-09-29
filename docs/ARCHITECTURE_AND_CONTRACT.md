# Architecture and contract

This is a map of the process and trust boundaries of an auto-research session on
the official interface.

## Process topology

```text
host (trusted)                                         agent container (offline)
─────────────────────────────────────────────────      ──────────────────────────
harness.codex_cli.auto_research  (launcher)
 ├─ ModelRelay ──────── unix socket ─────────────────── Codex (model requests)
 └─ services.controller.frontend  (robot MCP) ───────── Codex tools, api/runtime.py
     └─ Supervisor  (exclusive session lock, state.json)
         ├─ Gateway ─ Contract(official)  filters tools/fields
         ├─ NativeBackend → services.robodojo.mcp_server
         │     └─ RPC → Isaac/RoboDojo episode process (simulator GPU)
         ├─ ExplorationPool  (exploration_envs > 1)
         ├─ DockerSandbox  → isolated rehearsal/formal container (research GPU)
         └─ formal_worker processes  (formal_workers > 1)
```

- **Launcher:** `harness/codex_cli/auto_research.py` deploys a fresh workspace from
  `auto_research_agent/`, writes the private Codex home, starts the model relay and
  the MCP frontend, then runs Codex in Docker.
- **Model credentials:** only the relay holds them. The frontend's environment
  omits `OPENROUTER_API_KEY`.
- **Robot interface:** the frontend serves the robot tools and the lifecycle tools
  (`status`, `start_episode`, `evaluate`, `finish`, `register`, `rehearse`,
  `submit`) over one Unix socket. Up to 16 clients are multiplexed, with calls
  serialized per environment.
- **Contract:** `services/mcp_contract.py` is one immutable object used for tool
  discovery, dispatch and field filtering. The configuration requires
  `mode=auto-research` and `observation_profile=official`.
- **Workspace text:** `services/mcp_workspace.py` renders `MCP_SESSION.md` from the
  same contract.

## Official profile

The profile reproduces what an XPolicyLab policy server receives
(`env_cfg/arx_x5.yml`, `deploy.py`).

- **Robot tools:** `robodojo_observe`, `robodojo_status` and `robodojo_step` only.
- **Observations:** RGB head and wrist cameras. Depth, intrinsics and extrinsics
  are removed.
- **Grippers:** the gripper entries of `states` are replaced by the last commanded
  openings.
- **Actions:** absolute 25 Hz joint rows, counted against the native step limit.

Bundles carry their own motion code through the installed `robodojo_toolkit`
package, so the same code runs under the harness and under
`official/xpolicylab/AgentBundle`. The adapter does four things:
1. It runs the bundle's `main(ctx)` in a thread.
2. It turns each `robodojo_step` chunk into XPolicyLab `get_action` /
   `take_action` calls.
3. It maps `cam_head` to `cam_high`.
4. It warms the planner at load.

See the [official README](../official/xpolicylab/README.md) for the adapter, the
pristine upstream staging and verified accuracy.

## Session storage

The session root lives under `runtime/auto-research/<UTC>_<id>/` on `/mnt/ssd8`.

- **Private files (never mounted):**
  - `operator-input.json`, `configuration.json`: operator configuration;
  - `state.json`: supervisor state, bundles and results;
  - `formal-scores.json`, `mcp.jsonl`: scores and the MCP trace;
  - `native/`, `exploration/`, `formal-workers/`: native simulator results;
  - `bundles/<id>`: frozen bundles;
  - `agent.log`, `frontend.log`.
- **Agent-visible:**
  - `agent-workspace/` (`/workspace`) and `codex-home/`, on the fixed-size SSD
    filesystem;
  - `published/`: sanitized traces, replies, frames, exports and
    `formal_batch.json`, mounted read-only at
    `/workspace/runtime/autonomous_controller`.
- **Shared caches:**
  - `runtime/reference-demos/website/`: official task clips and terminal frames;
  - `runtime/official-xpolicylab/`: the policy environment, wheelhouse, stage
    and releases.

## Evaluation disclosure boundary

The native simulator runs RoboDojo's own evaluator. A recursive filter in the
gateway removes raw reward, score, success and terminated/truncated fields.
It keeps `episode_ended` and maps native success to `task_complete` at episode end.

- **`evaluate`:** returns the trusted binary result for the current exploration
  episode.
- **Formal scoring:** `services/robodojo/scoring.py` follows RoboDojo's `run_eval`
  rule: success counts as 1, otherwise the native process score / 100, with
  missing scores as 0 in a fixed denominator. Scores are written only to the
  private `formal-scores.json` and the operator dashboard. The agent sees success
  counts and rates.

## Why the model does not run the servo loop

The model is an executive with visual feedback, not a 250 Hz controller.
- **Native service:** physics, interpolation and PD effort stay in the simulator.
- **Bundle:** trajectory generation (cuRobo, IK) happens in Python inside the
  bundle.
- **Model:** the Codex agent chooses and debugs motions from 25 Hz evidence and
  writes the controller; it is not in the evaluation loop.
- **Gemini (optional):** a frozen controller may call `gemini_generate` at decision
  points (perception under domain randomization, choosing a step, checking success),
  never per servo step. Officially the bundle runs on our hosted agent API
  (`official/xpolicylab/serve_endpoint.sh`), which holds the key; every Gemini-backed decision keeps a non-Gemini fallback.
