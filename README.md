# RoboDojo auto-research (official interface)

Start with the [overview](OVERVIEW.md): what the system is, how the pieces fit, and what is verified.

A Codex agent develops a frozen controller bundle for one RoboDojo task, and that
bundle is submitted unchanged to the official XPolicyLab evaluation. The agent sees
exactly the official policy interface:

- **Observations:** RGB from `cam_high` and both wrist cameras, joint state with
  last-commanded grippers, `link6` poses and the instruction. There is no depth or
  camera calibration.
- **Actions:** absolute 25 Hz joint rows.
- **Motion:** planning, IK and pose math come from the `robodojo_toolkit` package,
  which is installed identically in the agent image and in the official policy
  environment.

The workflow is:
1. manual task success;
2. autonomous controller development;
3. an isolated rehearsal that succeeds;
4. one submission, scored over a held-out formal batch (50 episodes by default,
   4 running concurrently).

## Layout

- `auto_research_agent/`: the agent workspace template, copied into every session:
  - `AGENTS.md`, `START_PROMPT.md`;
  - the `api/runtime.py` development client;
  - `.agents/skills/`.
- `harness/codex_cli/`: the trusted launcher. It runs Codex in an offline Docker
  container, with a host model relay and robot MCP socket.
- `services/controller/`: supervisor, MCP frontend and gateway, isolated sandbox,
  formal batch workers, recording.
- `services/robodojo/`: the native simulator bridge (Isaac/RoboDojo session, RPC,
  kinematics, scoring).
- `services/exploration/`: parallel exploration slots (`exploration_envs` 1–8).
- `services/mcp_contract.py`, `services/mcp_workspace.py`: the session contract
  (official profile) and workspace rendering.
- `services/dashboard/`: the loopback-only operator dashboard.
- `packages/robodojo_toolkit/`: the motion toolkit. It provides:
  - NumPy/SciPy pose math, X5 FK and bounded IK;
  - 25 Hz row helpers;
  - the vendored RoboDojo cuRobo planner.
- `official/xpolicylab/`: the XPolicyLab `AgentBundle` adapter, the pristine
  upstream staging, `run_eval.sh` and `build_release.sh`. See its
  [README](official/xpolicylab/README.md).
- `RoboDojo/`: the simulator checkout, pinned in `dependencies.lock`.
- `runtime/`: an SSD-backed link to environments, caches, sessions and releases.
  It is never committed.

## Quick start

Set up sources, environments and assets once; see [PORTABLE_SETUP.md](PORTABLE_SETUP.md).
Then:

```bash
# 1. Agent image: the same pinned wheelhouse (torch, warp, cuRobo, toolkit) as the
#    official policy environment. torch is required by cuRobo.
bash scripts/build_official_image.sh            # -> robodojo-official:dev

#    Read-only per-task source packages injected as workspace task_source/
#    (needs runtime/envs/usd-tools: python -m venv + pip install usd-core==26.8 numpy pyyaml).
runtime/envs/usd-tools/bin/python scripts/build_task_sources.py   # -> runtime/task-sources/<task>/

# 2. Dashboard: prepare, launch, monitor and stop sessions.
python3 scripts/robot_lab.py dashboard          # prints http://127.0.0.1:8765/#token=...
```

In the dashboard, **New session** chooses the task, the two GPUs (the simulator's
and a different research GPU), the exploration budget, the formal batch and the
demonstration context. **Prepare** writes a configuration without starting anything.
**Launch** starts the paid agent session (Codex or Claude Code), or you can run the displayed command in a
terminal.

**Trajectory videos** lists every simulator episode of the selected session
(exploration, rehearsals and formal episodes, with native success and score) and plays
each episode's recorded video: 25 fps, cam_high and both wrist cameras side by side,
seekable, with speed control and download. Tick **Earlier sessions** (or open the URL
with `&history=1`) to review sessions that predate the running dashboard; launch and
stop stay limited to current sessions.

Over SSH, forward the port (`ssh -L 8765:127.0.0.1:8765 host`) and open the printed
URL locally. Never give that URL to an agent.

Without the dashboard:

```bash
python3 scripts/robot_lab.py prepare --task make_kong --sim-gpu 0 --research-gpu 1 --episodes 5
bash scripts/start_auto_research_agent.sh --config <printed operator-input.json>
python3 scripts/robot_lab.py contract --exploration-envs 2   # inspect the tool contract
```

### Claude Code instead of Codex

The agent CLI is configuration: `--agent-cli claude` (dashboard: **Agent CLI**) runs
Claude Code with Claude Opus 5.5 in the same offline container, with the same MCP
tools, workspace, skills and budgets (`harness/claude_cli/`). The host relay holds
the model credential; `--agent-provider` picks it:

| Provider | Credential | Model |
| --- | --- | --- |
| `openrouter` (default) | the OpenRouter key (`--key-file`, as for Gemini) | `anthropic/claude-opus-5.5` |
| `claude-login` | a separate Claude account dedicated to experiments: its long-lived `claude setup-token` token in `~/.config/robodojo/claude-experiment.token`, re-read per request. Your personal `~/.claude` login is always refused. | `claude-opus-5-5` |
| `anthropic` | an Anthropic API key file (`--claude-key-file`) | `claude-opus-5-5` |

Set up the experiment account once (and again to rotate its token). The script
signs in to that account in its own config directory, runs `claude setup-token`,
refuses your personal account, and verifies the token:

```bash
bash scripts/setup_claude_experiment_login.sh [--email experiment-account@example.com]
runtime/envs/robodojo/bin/python -m harness.claude_cli.experiment_login check   # account and token age
```

```bash
python3 scripts/robot_lab.py prepare --task make_kong --sim-gpu 0 --research-gpu 1 --episodes 5 \
    --agent-cli claude --agent-provider claude-login
bash scripts/start_auto_research_agent.sh --config <printed operator-input.json>
PYTHONPATH=. runtime/envs/robodojo/bin/python scripts/smoke_auto_research_live_claude.py --provider claude-login
```

- `claude-login` bills only the experiment account. The launcher and the dashboard's
  Prepare check the token first, with a free API call. A missing, revoked or expired
  token stops before any session starts, naming the setup script. Sessions prepared
  earlier with the personal login refuse to launch; prepare a new one. The dashboard
  shows which account each Claude session bills.
- Claude Code runs with `--effort xhigh`, like Codex's `model_reasoning_effort` (Claude Code's
  default is `high`).
- The container gets `CLAUDE.md` (`@AGENTS.md`), `.claude/skills` (a link to
  `.agents/skills`), a read-only `settings.json` (bypass permissions, web tools
  denied; subagents inherit Opus 5.5 unless spawned with `sonnet`, which is Sonnet 5.5 on
  OpenRouter and Haiku 4.5 on first-party providers, for cheap auditors) and the MCP
  config. `HOME=/claude-home`: Claude Code skips project skills when the working
  directory is the home directory.
- The relay refuses server tools (web search/fetch, code execution), URL or file
  media, `mcp_servers`, other models and calls beyond `--max-model-calls`. It writes
  `<session>/model-usage.jsonl`, which the dashboard shows as token usage.
- Verified 2026-09-29 with `smoke_auto_research_live_claude.py` on `claude-login`
  and `openrouter`: MCP tools, Python `api.runtime`, sandbox isolation, skill loading,
  a subagent and no web tools. A full research session has not been run yet.

## Official submission

1. **In the session**, the agent calls `register` → `rehearse` → `submit`. The
   harness formal batch uses fresh containers and simulators, with no intervention
   and no retries.
2. **Check on the official stack.** Run the frozen bundle through XPolicyLab on
   pristine upstream RoboDojo:

   ```bash
   bash official/xpolicylab/run_eval.sh debug <task> <session>/bundles/<bundle-id>
   ```

3. **Build the checkpoint.** It holds a per-task bundle, the offline wheelhouse,
   prebuilt portable cuRobo kernels and `SHA256SUMS`:

   ```bash
   bash official/xpolicylab/build_release.sh <out_dir> <task>=<session>/bundles/<bundle-id>
   ```

The dashboard's **Official release** panel shows both commands, filled in for the
selected session.

## Verify

```bash
runtime/envs/robodojo/bin/python -m pytest -m "not upstream and not gpu" -q local_tests
```

See [test setup](local_tests/README.md) for GPU and upstream tests.

## Documentation

- [Overview](OVERVIEW.md)
- [Agent workspace boundary](docs/AGENT_WORKSPACE.md)
- [Architecture and contract](docs/ARCHITECTURE_AND_CONTRACT.md)
- [Auto-research infrastructure contract](docs/controller_backend.md)
- [Parallel exploration](docs/parallel_exploration.md)
- [Robot MCP guide](docs/ROBODOJO_MCP.md)
- [Official XPolicyLab mode](official/xpolicylab/README.md)
- Reference material:
  - [control and cuRobo audit](docs/ROBODOJO_CONTROL_AUDIT.md);
  - [joint replay benchmark](docs/JOINT_REPLAY_BENCHMARK.md);
  - [RGB table calibration](docs/RGB_TABLE_CALIBRATION.md).
