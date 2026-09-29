# Auto-research infrastructure contract

`scripts/start_auto_research_agent.sh` runs Codex and ordinary development tools
inside offline Docker. Every session uses the official observation profile
(see the [MCP guide](ROBODOJO_MCP.md)).

## Agent image

The agent image `robodojo-official:dev` (`scripts/build_official_image.sh`, from
`services/controller/agent_runtime.Dockerfile`) installs the same pinned wheelhouse
as the official policy environment. The wheelhouse contains:
- NumPy 1.26, SciPy, Pillow and headless OpenCV 4.10;
- torch 2.7.0+cu128 and warp-lang 1.11.0;
- the pinned cuRobo fork and `robodojo_toolkit`.

torch is there because cuRobo's solvers are torch-based. The pose math, FK, IK and
row helpers need only NumPy/SciPy. OpenCV decodes with `cv2.imdecode` to BGR,
unlike the RGB observation and Pillow arrays.

Dependencies are installed by the trusted operator, never fetched by the agent.
New sessions pin the image digest when they are prepared. Existing configurations
and frozen bundles keep theirs and are never silently upgraded.

## Two execution settings

| Setting | Process / files | Episode and control |
| --- | --- | --- |
| Exploration | Codex, Bash and Python share the development container and persistent workspace | Direct calls and scripts share one scene; one MCP request executes at a time |
| Rehearsal / formal | Frozen bundle in a fresh isolated container; safe outputs copied back | Fresh episode, script alone, no agent intervention |

In exploration, `from api.runtime import Context` and `with Context() as ctx:`
connect ordinary Python to the same external MCP endpoint used by Codex.
`ctx.call(name, **arguments)` forwards native tool arguments and returns MCP
content blocks. There is no special probe/operate execution service, shell RPC,
server recorder, artifact-read RPC or explicit file-export RPC.

A function may send literal action numbers, plan with `robodojo_toolkit`, or run
an observation/action feedback loop with model-free perception.
It can also run the full `controller.main(ctx)` in the current exploration episode.
Returning hands control back without resetting or undoing actions. Such development
tests may involve agent intervention and never qualify for formal submission.

The Unix-socket server multiplexes up to 16 clients with one main-thread request
dispatcher. An idle Codex connection does not block another Python connection.
Calls, not entire scripts, are serialized: authors must coordinate background
controllers before intervening. Disconnecting a client does not cancel a dispatched
robot action. Observe its outcome before continuing.

`robodojo_step` accepts 1..50 rows per call; the native episode horizon still
applies. Split longer sequences across
calls. Complete motion arguments are validated before dispatch. A rejected
request reports its specific reason and `no_action_executed: true` without
locking the episode. Failures after motion dispatch (including lost responses
or failed artifact publication) set `control_uncertain: true` and block further
motion. Inspect status/observations instead of blindly retrying. Error replies
include operation, stage, reason, recovery guidance and, for gateway calls,
call ID. Credentials and private host paths are not included. Lifecycle/evaluation failures also retain sanitized diagnostics.

## Public tools and lifecycle

The MCP exposes the four official robot tools (`robodojo_observe`, `robodojo_status`,
`robodojo_step`, `robodojo_step_ee`) plus:

| Tool | Contract |
| --- | --- |
| `status` | Episode state, manual-success evidence, episode timeout, exploration episodes left, registered bundles with `rehearsal_qualified`, storage, current artifact paths |
| `start_episode` | Closes previous exploration and starts next distinct-seed episode |
| `evaluate` | Trusted current exploration evaluation |
| `finish` | Closes exploration and returns its artifact pointers |
| `register(source)` | Freeze a workspace-relative directory containing `controller.py` |
| `rehearse(bundle)` | Run frozen full script in a fresh exploration episode under the formal per-episode wall limit |
| `submit(bundle)` | One reserved formal batch of exactly the successfully rehearsed bundle; defaults to 50 episodes, 4 concurrent, in new configurations |

Instructions ask the agent to achieve manual success before building
automation. This is soft workflow guidance, not a programming access gate.
Every exploration episode, including rehearsal, uses a distinct seed and the
original native step limit. Formal's seed is held out. Starting an episode or
attempting motion invalidates cached evaluation, including ambiguous motion.
Repeated evaluation of an unchanged episode/step reuses trusted evidence.

`Supervisor` exclusively locks the trial directory and durably reserves an episode
or formal attempt before starting it. Formal reservation survives
restart. Successful rehearsal requires native success of the exact frozen bundle
and runtime fingerprint, a clean completion, and no uncertain control, evaluation,
output or recording error. Registration does not require manual success.
Changing code requires registering and successfully rehearsing a new bundle.

New configurations use formal collection (`formal_eval_seed`) 3, novel layouts
generated by `scripts/generate_layouts.py`, and 50 consecutive layout IDs starting
at `formal_seed=0`. Exploration exhausts official collection 0, then 1, then 2; the
formal collection must not be explored. The identity of a layout is the pair (collection, layout ID);
formal/exploration pairs cannot overlap. Existing configurations without batch fields retain their original one
episode. Each formal episode has a fresh isolated container and simulator, the
native step limit, and its own configured formal wall timeout. Agent development
is paused for the entire submission. No per-episode failure is retried or excluded from the denominator.

`formal_workers` sets how many formal episodes run at once (1..8). New
configurations default to 4; configurations without the field stay sequential.
Each concurrent episode is a separate `services.controller.formal_worker` process
under `formal-workers/episode-NNN/`, with its own supervisor lock, simulator,
controller container and private MCP trace. Its published run still appears under
the session's `runtime/autonomous_controller/runs/`. Workers need no channel to the
parent: each reads the parent's saved configuration and imports its rehearsed bundle. Every layout is launched at most once. The worker's own episode record is authoritative. A worker that started its
episode but died without a result
counts as an error, like a sequential timeout. A worker that fails before starting
any episode (spawn, preflight or bundle import) interrupts the batch, as that
failure would in the sequential loop. A worker still running ten minutes past the
formal wall timeout gets one SIGTERM, then SIGKILL three minutes later.
`formal_batch.active_episodes` lists running 1-based episode numbers; `state.active`
is unused during a concurrent batch. On stop, workers get about 20 seconds before
the supervisor exits; any still cleaning up stay in `formal_worker_pids`. If the
supervisor restarts mid-batch, the batch is marked interrupted, this session's
surviving workers are terminated and started episodes are recorded as interrupted.
Each worker adds a simulator on the simulator GPU and a controller container, with
the formal memory/CPU limits, on the research GPU; controller errors increased at
8 workers. Wall timeouts are unchanged under contention. The task-scaled session
launcher deadline also bounds the batch: ceil(episodes / workers) × formal
seconds, plus startup, must fit in the time remaining after development.
The dashboards read concurrent episodes from `formal-workers/*/native/results`.

`formal_batch` in trusted state tracks completed episodes, successes, ordinary
failures and execution/evaluation/recording errors. The final `success_rate` is
successes / scheduled episodes; success requires native success and clean execution.
An abnormal worker exit after a finalized episode does not invalidate its task
result. The supervisor records `worker_returncode` and `infrastructure_errors`
separately in state and the published batch report. `infrastructure_error_count`
counts completed episodes with these diagnostics and overlaps the task-outcome
counts; it is not an additional failure category in the success-rate denominator.
Environment-shutdown failures are also recorded without erasing an already obtained
evaluation, and still propagate to stop the affected supervisor. A worker lost
before a finalized result has no confirmed success and remains an error.
An interrupted batch has no final rate and cannot be resubmitted. Full per-episode
records, logs and frames remain available in the operator dashboard.

An operator may explicitly evaluate a historical frozen bundle in a **new archive**:
`python -m services.controller.batch --source-session SESSION --bundle ID --output NEW_SSD_SESSION`.
This validates and imports its successful rehearsal evidence and exact bundle hash,
then runs 50 episodes in collection 3 by default. The original result is untouched.
This post-hoc command starts no Codex.

For concurrent operator evaluation, use
`python -m services.controller.parallel_batch --source-session SESSION --bundle ID --output NEW_SSD_SESSION --workers 4`.
It launches up to eight independent single-episode supervisors, each with a fresh
simulator, controller container, ports and artifact directory. All use the saved
simulator/controller GPU assignments. `formal-report.json` contains progress,
per-layout outcomes and source artifact directories. The 50-layout schedule is
reserved before work starts; failed or interrupted layouts are never retried.
Agent `submit` uses `formal_workers` instead of this operator-only route.
Concurrency can increase resource contention; episode timeouts remain unchanged.

With explicit operator approval, `--source-batch STOPPED_BATCH` (instead of
`--source-session`/`--bundle`) migrates only layouts that never started. A durable
claim prevents another migration. Completed results are preserved; a started,
interrupted layout is separately disclosed and counts as zero in the fixed
denominator, not as a measured controller failure. The original archive is never
rewritten. Restarting an interrupted parallel coordinator is not supported.

## Rehearsal and formal isolation

Only these full runs create a fresh execution container. Registration copies and
hashes regular bundle files with no-follow path handling and size/count limits.
The development container, including background writers, is paused during
registration and isolated evaluation. The evaluation container receives:

- Frozen source and data files at their registered `/workspace/code/<project>/` path,
  immutable runtime at `/workspace/api/runtime.py`.
- A private stdio MCP channel limited to the four robot tools; no episode lifecycle. The
  bundle runs through the official AgentBundle bridge (`api/runtime.py:run_official`).
- Read-only sanitized current-run frames under `/workspace/runtime/autonomous_controller/...`.
- A byte-capped writable `/workspace/output` and capped temporary storage.
  Both exploration and isolated runs start in `/workspace`.

There is no editable workspace, development socket, prior-run artifact mount,
host credential, general network or simulator access. `main(ctx)` receives its
context; it must not open a development `Context()` itself. Bundle-relative paths
based on `__file__` allow the same controller to run in both settings.
Only canonical `code/<project>` registration paths are accepted, and the path is
included in the bundle hash. Generated `/workspace/output` files return to
`/workspace/runtime/autonomous_controller/runs/<run-id>/exports/` in the development
workspace. Isolation uses an empty read-only workspace scaffold, not a live workspace mount.

Rehearsal uses the same **6 seconds per native step** wall limit as formal,
with no cumulative rehearsal-time cap. Each rehearsal consumes one
exploration episode. The deadline includes environment startup, controller
execution and trusted final evaluation; nested operation deadlines cannot extend it.
Timeout returns `isError=true`, `reason="timeout"`, `error_type="TimeoutError"`,
the configured timeout and artifact pointers. It never qualifies for formal.
Cleanup may take additional time after execution is stopped.

Every rehearsal/formal episode result records `timing` (`episode_wall_timing_v1`),
including interrupted operations. All durations are monotonic wall seconds, not CPU
time. `environment_setup_seconds` includes connection, simulator startup/reset,
and the initial observation. `environment_step_seconds` measures end-to-end motion
MCP calls, including simulation, rendering and frame publication; it is
not physics-only time. Read-only robot calls are separate in
`environment_other_mcp_seconds`.
`controller_processing_seconds` is container execution wall time outside MCP
handling, including process startup, local computation, sleeps, IPC and logging.
Container preparation/export/cleanup, evaluation, environment shutdown and other
supervisor overhead are recorded separately. These categories sum to
`total_seconds`, which ends before final result publication and can exceed the
execution deadline because cleanup is included. `mcp_tools` contains per-tool
call counts and elapsed seconds; it is a breakdown, not an additional duration.
Timeout results also identify `timeout_stage` (setup, execution, or evaluation).

On native episode end, further motion calls are rejected. The script has up
to ten seconds within the original deadline to finish local file writes and logs.
Then the whole container is stopped. Crashes, deadlines and formal completion all
close the environment; there is no post-failure retry.

## Ordinary files and logs

Development scripts use normal workspace files. Redirect stdout/stderr yourself
when retained logs are needed. Robot MCP requests/results are recorded automatically;
`status` returns the active trace path. Motion replies retain inline final images
and pointers to their complete 25 Hz sequences.

Full evaluation containers write ordinary files under `/workspace/output`. The trusted runner
kills the writer before copying safe regular files to the published exports directory.
Links, special files and oversized outputs are rejected; an output error blocks
qualification. The output filesystem itself enforces its byte cap. A short-lived
trusted mount helper provisions/unmounts it; generated code has no mount privilege.
No Python object/checkpoint from outputs is executed or deserialized on the host.

Sanitized runs appear under `/workspace/runtime/autonomous_controller/runs/<run-id>/`:

- `mcp.jsonl`, per-call `response.json` and images, complete frame sequences.
- `result.json` with outcome and pointers.
- For isolated runs: frozen code, safe `exports/`, `stdout.log`, `stderr.log`
  and log truncation metadata.

Exploration does not claim to snapshot mutable local source or capture shell logs.
Its source and user-saved logs remain ordinary workspace files. Published files are
read-only to the agent. Private session paths, seed/evaluation internals and
simulator files are never mounted or published.

## Credentials and containment

Both containers use locally provisioned digest-pinned images, non-root UIDs,
no capabilities, no-new-privileges, default Docker seccomp, private IPC/network,
read-only root, CPU/RAM/PID limits and operator-selected GPU UUIDs. The research
GPU runs cuRobo planning in both the development and the isolated containers. The research
GPU must differ from the simulator GPU. Shared kernel/GPU-driver risks remain;
containers are not a hardware trust boundary.

Development mounts writable `/workspace` and `/codex-home` on one fixed-size SSD
filesystem (default 20 GiB), read-only published artifacts,
and two host-owned Unix sockets: robot MCP and Codex's fixed inference relay.
The relay only accepts bounded Responses requests for the configured model; it is
not an HTTP proxy. All credentials are added outside the container. Codex's native
shell/file/image tools run with full access **inside** this outer container.
No host home, Docker socket, simulator package or Internet route is exposed.
Images and dependencies are provisioned by the operator, never downloaded by
the agent.

The model relay's body limit is 256 MiB (including decompressed requests and the
response stream). New auto-research sessions configure automatic compaction at
256,000 total active-context tokens, subject to Codex's model-context bound.
These are separate byte and token limits; they do not change image delivery,
or robot MCP limits. Existing relay processes retain their loaded
limits until restarted; changing the source does not hot-reconfigure a session.

Rehearsal and submission freeze the whole development container, including Codex
and its model-stream reader. New sessions set the provider's
`stream_idle_timeout_ms=(session_seconds+300)*1000`, matching the
MCP tool allowance. The host relay allows the same interval for response writes
to a paused client; incoming request reads still time out after 120 seconds and
upstream provider I/O after 600 seconds. This prevents intentional isolation
pauses from being mistaken for model-stream failures, including long formal
batches. It does not extend the task-scaled launcher deadline, the per-episode
rehearsal limit, or formal episode limits. No heartbeat or action replay is used.

New auto-research sessions set Codex's provider `request_max_retries=2` and
`stream_max_retries=2`. Codex owns model-request/stream recovery, but its
backoff lasts seconds while provider rate-limit windows last minutes. The relay
therefore retries upstream HTTP 429/502/503/504 itself, before any response byte
reaches Codex (so no partial stream is replayed), honouring numeric
`Retry-After` or jittered exponential backoff capped at 60 s, for up to 600 s per
incoming request. Those retries count as one model call. Other statuses, and a
retryable status that outlasts the window, are passed through without provider
error bodies or headers. This keeps authentication/validation errors distinct
from retryable provider failures. HTTP and stream budgets are separate and can
nest: persistent HTTP 503 errors allow at most nine relay requests for one model
continuation; persistent stream failures allow three. The installed Codex also
retries HTTP 401 through its outer stream budget (three attempts), despite not
retrying it in the HTTP layer. Every request Codex sends consumes the model-call
cap (`--max-model-calls`, default 4000). An exhausted cap returns HTTP 402 with
a plain-text reason: Codex does not retry it, and it is not reported as provider
rate limiting. Retries do not extend the task-scaled launcher deadline.
Completed tool results remain in Codex's history during recovery. There is no
launcher restart loop or automatic replay of MCP motion, rehearsal,
submission or failed episodes. Exhausted retry budgets still surface an error.
Existing sessions retain their generated configuration; no live session is patched
or resumed automatically.

## Operator configuration and testing

```bash
runtime/envs/robodojo/bin/python scripts/configure_auto_research.py --task make_kong --sim-gpu 0 --research-gpu 1 --episodes 5
bash scripts/start_auto_research_agent.sh --config /path/printed/above/operator-input.json
```

Choose available GPUs; these ordinals are examples. The generator resolves immutable
image/GPU identities and distinct seeds, then saves private configuration on SSD.
The launcher copies the task rubric, instructions, skills and API client
into a fresh workspace. Existing workspaces are not silently rewritten.
`--prepare-only` consumes that fresh deployment root without starting a model or
episode; generate another configuration to launch.

Backend diagnostics:
`python -m services.controller --config FILE check|status`;
`register --source DIR`, `rehearse --bundle ID`, `formal --bundle ID` are operator-only.
The dashboard (`python3 scripts/robot_lab.py dashboard`) wraps preparation,
launch, stop and monitoring.

Configuration retains `development`, `formal` and `training` resource objects.
`training` CPU/RAM/PID/tmpfs and `training_gpu` size the development container;
`controller_gpu` selects isolated inference GPU. The legacy `development.wall_seconds`
and saved `development_reserved_seconds` do not limit rehearsals; the formal
per-episode wall limit applies instead. New launches derive all episode deadlines
as **6 seconds × native task steps**, including manual exploration, rehearsal and
formal. The overall session deadline is **24 hours × native task steps / 400**.
For Make Toast (1400 steps), these are 140 minutes and 84 hours respectively.
Saved legacy absolute timeout fields cannot override these new-launch budgets.
Running sessions retain their loaded deadlines. Native step limits are unchanged.
The retired `training_seconds` allowance
is ignored when loading old configurations. Native-shell training has no separate
time allowance; `training.wall_seconds` is unused by the development container.
`workspace_mb` caps workspace/Codex state; `--workspace-gib` sets it.
Published/audit/bundle/output allowances are separate; native simulator artifact
retention still needs operator management. Bulky files belong on SSD, not in Git.

Normal cleanup removes owned containers and closes simulator processes.
After supervisor SIGKILL, immediate cleanup requires external service supervision.
On restart, persisted owned execution containers are removed and interrupted attempts
remain consumed. No finally block can guarantee cleanup after host/power failure.

See [test instructions](../local_tests/README.md). Real Docker/fake-task checks
verify transport and containment, not learned task success. For the official
XPolicyLab check and the checkpoint build, see the
[official README](../official/xpolicylab/README.md).
