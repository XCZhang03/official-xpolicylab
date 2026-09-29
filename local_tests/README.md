# Regression tests

Run from the repository root with Python 3.11 or newer:

```bash
python -m pip install pytest numpy scipy pillow msgpack
python -m pytest -m "not upstream and not gpu" -q
```

Most portable tests use temporary directories and fake simulators. The optional
Docker tests also execute real containers when the daemon and a local
`python:3.12-slim` image are available; the development-container and Codex
checks additionally use the locally built `robodojo-official:dev` image
(`scripts/build_official_image.sh`) when it is installed. The non-GPU suite does
not start Isaac, use a GPU, call a model, download data, or require credentials.
Tests never pull images.

After bootstrapping the pinned RoboDojo checkout and its submodules, run
`python -m pytest -q` to include the `upstream` tests of native source contracts
and Isaac Lab's CPU pose math. Those tests skip if the source files are absent.
The existing `runtime/envs/robodojo/bin/python` environment can run the suite.

## GPU integration

`gpu` tests skip unless native dependencies, an idle visible NVIDIA GPU with at
least 24 GiB free, and writable SSD-backed `runtime/` storage are available. Once
preflight passes, simulator startup or accuracy failures fail the test; they are
not silently skipped.

```bash
runtime/envs/robodojo/bin/python -m pytest -m gpu -vs
```

Use `CUDA_VISIBLE_DEVICES` to restrict eligible GPUs. Shared fixtures live in
`unit/gpu_fixtures.py`. `unit/test_gpu_mcp_contract.py` checks real worker
observations per profile, including the official profile's commanded-gripper
states. `unit/test_gpu_joint_replay.py` plans large combined and wrist motions
through real MCP handlers, executes them, checks final goal error
(1 mm / 0.005 rad), and replays the executed 25 Hz actions through
`robodojo_step` in a fresh episode, with inline images and full-frame manifests.
All owned services close on failure; small reports stay under
`runtime/integration-tests/`. Historical executor comparisons are documented in
[benchmark methods](../docs/JOINT_REPLAY_BENCHMARK.md).

Only `unit/` and this guide are committed. Local experiments, recordings,
datasets, credentials, and simulator outputs remain ignored.

## Official interface and toolkit

`test_official_profile.py` checks that only auto-research exists, that sessions
accept only the official observation profile, that the contract offers exactly
`robodojo_observe`/`robodojo_status`/`robodojo_step`, commanded-gripper
substitution and field stripping, workspace composition, and the XPolicyLab
`bundle_bridge.py` under the official `update_obs`/`get_action` loop, including
cancellation when the episode ends mid-bundle. `test_robodojo_toolkit.py` checks
toolkit pose math against the harness, FK/IK and 25 Hz row helpers.
`test_mcp_contract.py` checks discovery, dispatch, routing, frame publication and
workspace rendering agree for one and several exploration environments, and that
retired tools never appear in the rendered workspace.

## Auto-research backend

`test_controller_backend.py` checks ownership handoff, frozen submissions,
rejection of non-official tools, private provider-key loading, timing and
inspectable artifacts without Docker. `test_controller_docker.py` checks actual
isolation, offline Python, logs, exports, timeouts and full-script
rehearsal/formal lifecycles against a deterministic fake task; its GPU cases
additionally run a real MakeKong episode through the official joint-step loop.
`test_motion_gateway_errors.py` checks the 50-row cap, complete-batch preflight,
actionable errors without path leakage, and uncertainty only after motion dispatch.

`test_formal_batch.py` runs a fake 50-episode schedule through the production
supervisor: distinct layout IDs, clean-success criteria, a fixed denominator
including errors, interruption/no-retry rules, and verified import of historical
rehearsal evidence. `test_parallel_submit.py` runs the concurrent `submit` path
with inline workers: each layout launched once, systemic worker failures
interrupting the batch like the sequential loop, lost episodes counted as errors,
watchdog SIGTERM-then-SIGKILL, and restart recovery. A four-process test uses real
isolated Docker controllers and the production worker entrypoint with a fake
robot backend. `test_parallel_exploration.py` covers slot routing, the shared
episode budget and the episode worker.

`test_auto_research_frontend.py` checks bundle paths, the offered tool surface,
native image delivery, the stdio wire limit, protected configuration, the deployed
skill set and isolated launcher settings, for all three demonstration modes.
`test_development_python.py` runs ordinary Python repeatedly inside one
development container against the shared MCP socket, including `robodojo_toolkit`
imports, direct-agent handoff and recovery after a script exception, then runs the
same nested project as a rehearsal and formal submission. Robot feedback is
synthetic (`unit/native_fakes.py`); these tests do not claim task success.

`test_auto_research_codex.py` starts the installed Codex CLI with the real stdio
MCP frontend and a localhost fake model endpoint. It inspects the actual deferred
tool registry (official robot tools only, no web or model API), network isolation,
host-only credentials and imports of the deployed API and toolkit. Retry fixtures
inject transient and persistent HTTP/stream failures and check bounded recovery
and exactly-once tool execution; pause fixtures freeze the Codex container while a
model stream is open. When the official MakeKong clip is already cached, it also
deploys final-state and full-sequence demonstrations; it never downloads media.
`test_auto_research_relay.py` checks fixed endpoints, authentication, streaming,
call limits and rejection of provider-side network tools.

`test_research_dashboard.py` checks setup validation against the configuration
CLI's options, session stages, release commands for the submitted bundle, session
scoping and token-authenticated endpoints of the dedicated dashboard.

`test_workspace_storage.py` provisions a disposable 64 MiB SSD filesystem using
the trusted storage helper. A real unprivileged Docker writer verifies ENOSPC,
space recovery, unmount/remount persistence, and rejection of uncapped paths.

```bash
runtime/envs/robodojo/bin/python -m pytest local_tests/unit/test_controller_backend.py local_tests/unit/test_controller_docker.py -q -rs
```
