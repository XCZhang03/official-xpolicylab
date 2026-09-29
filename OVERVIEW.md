# Overview: agent-developed controllers for the official RoboDojo evaluation

This repository lets an AI coding agent (Codex or Claude Code) develop a **frozen
controller bundle** for one RoboDojo task, under exactly the interface the official
evaluation gives a policy, and ships that bundle to the official XPolicyLab
evaluation unchanged. The goal is the highest official test score of the frozen
bundle; nothing agent-driven runs during evaluation.

Status as of 2026-09-29. Operating details live in [README.md](README.md); this
document explains what the system is, how the pieces fit, and what is verified.

## Status at a glance

| Area | State |
| --- | --- |
| Official interface in the harness | RGB only (`cam_high`, two wrist cameras), joint state, link6 poses, instruction; absolute 25 Hz joint or native end-effector actions. No depth, no calibration. |
| Simulator | Pristine upstream RoboDojo (`ee67a14`) and XPolicyLab (`432f82b1`), staged byte-identical from the pinned commits. |
| Agent CLIs | Codex (GPT-6 Astra, effort xhigh) and Claude Code (Opus 5.5, effort xhigh), selected per session. |
| Harness results | Claude Code on `make_kong`: formal batch **32/32** on official layout collection 2 (16 exploration layouts). Harness formal batch, not yet an official XPolicyLab run. |
| Official path | Bundles run through the real XPolicyLab adapter (`AgentBundle`) end to end on the official stack; the submitted client (`Mooncake_Agent`) runs through the official simulator against our hosted endpoint on localhost. |
| Not yet done | An official leaderboard submission; the hosted endpoint on a public server with TLS; accuracy measurement of the endpoint's Gemini task detector. |

## End-to-end flow

```text
 research session (one task)                          official evaluation
 ─────────────────────────────                        ─────────────────────────────────
 1 manual success   agent drives the robot            XPolicyLab env client (organisers)
 2 controller       agent writes controller.py              │ ws, per action chunk
 3 register         frozen bundle (hashed)                  ▼
 4 rehearse         isolated run, fresh episode        Mooncake_Agent (submitted, no weights)
 5 submit           formal batch: N held-out layouts        │ HTTPS POST /v1/act (observation)
                    fresh containers, no retries            ▼
        │                                              our agent API (endpoint/agent_api.py)
        ▼                                                ├ task_router: which task is this?
 run_eval.sh      bundle through the real adapter        └ AgentBundle worker for that task
 build_release.sh checkpoint (bundles, wheels, kernels)     (frozen bundle + cuRobo + Gemini)
```

## 1. What the agent works with

**Interface.** The agent sees only what an official XPolicyLab policy receives and
may emit only what the official environment accepts:
- Observations: RGB `cam_high`, `cam_left_wrist`, `cam_right_wrist`; joint state with
  last-commanded grippers; `link6` poses; the task instruction.
- Actions: `robodojo_step` (absolute 25 Hz joint rows) and `robodojo_step_ee`
  (absolute link6 pose rows, solved by the environment's own IK, as officially).
- Lifecycle tools: `exploration_status`, `start_episode`, `evaluate`, `finish`,
  `register`, `rehearse`, `submit`.
- `gemini_generate`: Gemini 3.8 Flash (vision, function calling), $10 per session,
  brokered by the host; the agent never holds a key.

**Toolkit.** `packages/robodojo_toolkit` (0.2.0) is installed identically in the agent
image and in the official policy environment: pose math, X5 FK and bounded IK, 25 Hz
row helpers, and the vendored RoboDojo cuRobo planner. Bundles may use only the
official tools, this toolkit, and their own code and data.

**Workspace** (`auto_research_agent/`, copied into every session):
- `AGENTS.md` (workflow and rules), `START_PROMPT.md`, `TASK.md` (public rubric,
  demonstration, and for `*_random` tasks the variant's randomization), `MCP_SESSION.md`.
- `task_source/`: the read-only official source for this task only (task module,
  the reward predicates it uses, configs, object geometry), no evaluation layouts.
- `api/runtime.py`: Python access to the same MCP endpoint, and `run_official`, which
  runs a bundle through the exact official adapter bridge during development.
- Skills (`.agents/skills/`): manual-robot-control, autonomous-control, motion-toolkit,
  rgb-position-calibration (camera models, table-plane points, multi-view
  triangulation script), wrist-camera-inspection, gemini, in-context-action-learning,
  experience-memory, subagent-audit (a fresh reviewer flags layout overfitting before
  registration, on a cheaper model).

**Generalization.** Official evaluation uses held-out layouts, so the instructions
require observation-driven controllers and an overfitting audit. `*_random`
variants are separate tasks with their own bundles; their `TASK.md` states what the
RoboDojo source randomizes (instances, clutter, placement, table, floor, lighting,
background).

## 2. Agent CLIs

| | Codex | Claude Code |
| --- | --- | --- |
| Launcher | `harness/codex_cli/` | `harness/claude_cli/` (`agent_cli: claude`) |
| Model | GPT-6 Astra via OpenRouter, effort xhigh | Opus 5.5, effort xhigh |
| Provider | host Responses relay | host Messages relay: OpenRouter, a dedicated Claude experiment account (`claude-login`, never the personal login), or an Anthropic key |
| Subagents | inherit Astra; auditors may use GPT-6 Sol (relay allows it; untested in a session) | inherit Opus; `sonnet` alias for auditors (Sonnet 5.5 on OpenRouter, Haiku 4.5 first-party) |
| Usage shown | Codex rollout logs | relay's `model-usage.jsonl` |

Both run in the same offline container with the same workspace, MCP tools and
budgets. Relays hold every credential; they allowlist models, refuse server tools
(web search/fetch, code execution) and URL media, and cap model calls (4000 default).

## 3. Host architecture and trust boundaries

```text
host (trusted)                                   agent container (offline, read-only root)
launcher ─ model relay ─── unix socket ───────── agent CLI (Codex / Claude Code)
        └─ MCP frontend ── unix socket ───────── agent tools, api/runtime.py
            └─ Supervisor (session lock, state.json)
                ├─ Gateway: official contract filters tools and fields
                ├─ NativeBackend → RoboDojo episode process (simulator GPU)
                ├─ ExplorationPool (1–8 concurrent exploration environments)
                ├─ DockerSandbox → isolated rehearsal/formal containers (research GPU)
                └─ formal workers (concurrent formal episodes)
```

- The container has no network, host filesystem, Docker socket or credentials; its
  writable storage is a capped filesystem. Details: [architecture and
  contract](docs/ARCHITECTURE_AND_CONTRACT.md), [workspace
  boundary](docs/AGENT_WORKSPACE.md), [infrastructure contract](docs/controller_backend.md).
- Episodes run on the pristine official stage, so harness physics matches official
  evaluation (including the unfixed upstream `swap_blocks` button reset, RoboDojo #48).
- The dashboard (`python3 scripts/robot_lab.py dashboard`, loopback only) prepares,
  launches, monitors and stops sessions, plays every episode's video, shows model and
  Gemini spend, and prints the official check and release commands per session.

## 4. Evaluation inside the harness

- **Exploration** uses official layout collections 0, 1, 2 in order (any budget up to
  their total). **Formal** runs on a held-out collection: by default collection 3,
  novel layouts generated by `scripts/generate_layouts.py` from each task's
  randomization; a collection that is also explored is rejected.
- `register` freezes a directory; `rehearse` runs it alone in a fresh container and
  episode through the official adapter; `submit` requires a successful rehearsal of
  the exact bundle and runs the formal batch (fresh containers and simulators,
  concurrent workers, no intervention, no retries). The success rate is successes over
  the fixed episode count.

## 5. Official evaluation path

**Adapter** (`official/xpolicylab/AgentBundle/`): an XPolicyLab policy that runs a
frozen bundle. The bundle's `main(ctx)` runs in a thread; each `robodojo_step` or
`robodojo_step_ee` call becomes the next `get_action` chunk and returns once the
environment has executed it. The official client waits 120 s per request and treats a
timeout as fatal, so the bridge answers with a hold step after 90 s.

**Local official check and packaging:**
```bash
bash official/xpolicylab/run_eval.sh sim <task> <session>/bundles/<id>     # real XPolicyLab + RoboDojo
bash official/xpolicylab/build_release.sh <out> make_toast=<bundle> make_toast_random=<bundle>
```
The checkpoint holds one bundle per task, `TASKS.json`, the offline wheelhouse (same
pins as the agent image, checked), prebuilt portable cuRobo kernels and `SHA256SUMS`.

**Hosted agent API** (the planned submission, modelled on RoboProbe's GPT-6 Astra
harness, which calls a model API from inside an ordinary XPolicyLab policy):
- Submitted: `official/xpolicylab/Mooncake_Agent/`, upstream's `demo_policy`
  `deploy.py` byte for byte and an HTTP client that posts the latest observation per
  action chunk (PNG, lossless) with a bearer key to `MOONCAKE_BASE_URL`. No weights,
  no GPU, no task names, no routing.
- Hosted by us: `official/xpolicylab/serve_endpoint.sh` starts one AgentBundle worker
  per task (localhost, token-protected) and `endpoint/agent_api.py`. On an episode's
  first request, `endpoint/task_router.py` picks the task from the instruction and head
  camera with Gemini (fallback: instruction word match, preferring the `_random`
  bundle); every decision is logged. Worker failures answer with hold steps; retries are
  answered from cache; a client's lease is freed when its connection closes.
- Verified locally: official simulator → `Mooncake_Agent` → endpoint → worker, a full
  600-step `make_kong` episode and both XPolicyLab debug loops.

Details: [official mode README](official/xpolicylab/README.md).

## 6. Leaderboard rules we follow

From the RoboDojo leaderboard page (checked 2026-09-29):
- Evaluation runs on the RoboDojo system, from a deployable package or a remote
  policy server; scores are computed by the organisers.
- Three evaluation seeds; hidden verification layouts, and a large gap from public
  layouts invalidates a submission.
- Generalization tasks: base and `_random` run 25 trials per seed each and are
  merged into one score.
- Verified-board entries need released code, checkpoint and configuration, plus
  methodological novelty and a paper or report.
- XPolicyLab PRs follow its CONTRIBUTING standard: adapter files, policy README
  template, debug closed loop, and an eval-only declaration when there is no training
  code.

Unconfirmed from public sources: whether a hosted package may call external APIs
during evaluation (the organisers' own GPT-6 Astra evaluation did), and whether a
non-learned agent-written controller qualifies for the verified board.

## 7. Repository map

| Path | Contents |
| --- | --- |
| `auto_research_agent/` | agent workspace template: instructions, skills, `api/runtime.py` |
| `harness/codex_cli/`, `harness/claude_cli/` | launchers and host model relays |
| `services/controller/` | supervisor, MCP frontend, gateway, sandbox, formal batch, Gemini broker |
| `services/robodojo/` | simulator session, MCP server, kinematics, task variants, task source packing |
| `services/exploration/` | parallel exploration environments |
| `services/dashboard/` | operator dashboard |
| `packages/robodojo_toolkit/` | motion toolkit (bundles and development) |
| `official/xpolicylab/` | AgentBundle adapter, Mooncake_Agent client, hosted endpoint, staging, eval, release |
| `scripts/` | configure, launch, image build, task sources, layout generation, smokes |
| `local_tests/` | unit and GPU tests |
| `RoboDojo/` (pinned), `runtime/` | simulator checkout and SSD-backed runtime; neither committed |

## 8. Setup, running and tests

- One-time setup: [PORTABLE_SETUP.md](PORTABLE_SETUP.md), including the generated inputs that are not
  committed: the per-task source packages (`scripts/build_task_sources.py`, about 11 GB) and
  the held-out formal layouts (`scripts/generate_layouts.py`). Then build the agent image
  (`scripts/build_official_image.sh`).
- Run sessions from the dashboard, or `robot_lab.py prepare` and
  `start_auto_research_agent.sh` ([README](README.md)).
- Tests: `runtime/envs/robodojo/bin/python -m pytest -m "not upstream and not gpu" -q local_tests`;
  GPU and upstream tests in [local_tests/README.md](local_tests/README.md).

## 9. Known limitations

- The Gemini task detector's std vs `_random` accuracy is unmeasured, notably for
  `pack_objects_into_box_random` and `sweep_blocks_random`, which add no clutter.
- The hosted endpoint has run only on localhost; direct TLS is implemented but untested.
- Held-out generated layouts exist only for tasks without clutter (the generator refuses
  clutter). A native layout-stability checker was written but is untested and not on this branch.
- Bundles from the older RGBD harness (for example the 56% `make_toast` bundle) used
  depth and simulator calibration and cannot run on the official interface.
