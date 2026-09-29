# Official RoboDojo mode (XPolicyLab)

Develop agent bundles against the same interface as the official RoboDojo
leaderboard, then submit them unchanged through an XPolicyLab adapter.

## How official evaluation drives a policy

- One policy server per task. Sweeps call `run_policy_eval.sh` per task, and
  the server receives `task_name` in its config.
- The environment loop is `deploy.py`:
  `reset()` → `get_obs()` → `update_obs(obs)` → `get_action()` returns a chunk →
  `take_action(a)` for each action → `update_obs` after each action.
  The episode ends inside the environment; the policy sees no success signal.
- Observation (`env_cfg/arx_x5.yml`): RGB `cam_head` and both wrist cameras;
  depth, intrinsics and extrinsics are disabled.
  State: `*_arm_joint_state` (6), `*_ee_joint_state` (last commanded gripper,
  in [0, 1]), `*_ee_pose` (`[x, y, z, qw, qx, qy, qz]`, environment frame),
  and `instruction`.
- Actions, one environment step each at 25 Hz, native per-task `step_lim`.
  `EvalEnv.take_action` infers the type of every action from its keys, so both
  kinds work in any episode:
  - joint: `*_arm_joint_state` + `*_ee_joint_state`;
  - native EEF: `*_ee_pose` (`[x,y,z,qw,qx,qy,qz]`, link6, observation frame) +
    `*_ee_joint_state`, solved by the env's own cuRobo IK (`robot_manager.solve_ik`);
    if IK fails, that arm keeps its previous target.
- Scoring: 3 eval seeds (layout collections 0/1/2) × native episodes per task;
  success rate plus partial average score; hidden-layout verification.

## Harness profile: `observation_profile: official`

Set in `services/mcp_contract.py`; auto-research only.

- Robot tools are exactly `robodojo_observe`, `robodojo_status`, `robodojo_step`
  (joint rows) and `robodojo_step_ee` (native EEF rows). Everything else, including
  pose math, is not listed and is rejected before dispatch
  (`no_action_executed: true`). There is no model API.
- Observations are RGB-only. `states[6]` and `states[13]` report the
  **commanded** gripper, as official observations do. The session supplies
  `commanded_gripper_openings`; the contract swaps it in for this profile and
  strips it for every other profile.
- Our `eef_positions` / `eef_quaternions_wxyz` already come from the official
  `*_ee_pose`, so no frame conversion is needed. The camera is renamed
  `cam_head` → `cam_high` in both directions.
- The workspace template contains only official-interface skills, and
  `MCP_SESSION.md` states the rules. Planning, bounded IK and pose math come from
  `robodojo_toolkit`.
- The native simulator imports the pristine staged upstream RoboDojo
  (`stage.sh`, selected via `ROBODOJO_SOURCE_ROOT` by `NativeBackend`), not the
  repository's patched checkout.
- Isolated rehearsal and formal episodes run the bundle through this adapter's
  `bundle_bridge.py` (`api/runtime.py:run_official`).

## Adapter: `AgentBundle/`

A standard XPolicyLab adapter. `deploy.py` and the scripts are copied unchanged
from `demo_policy`.

- `bundle_bridge.py` runs the bundle's `main(ctx)` in a thread.
  - `robodojo_step` (joint) and `robodojo_step_ee` (native EEF) rows become the
    next `get_action()` chunk. The call returns once the environment has executed
    the chunk and sent a fresh observation. Replies carry `transition: {"steps": []}`
    for compatibility; there is never an episode-end signal.
  - `robodojo_observe` answers from the latest observation.
  - When the bundle finishes, the bridge holds the pose until the episode ends.
  - A reset cancels the bundle (`EpisodeCancelled`).
  - `ctx.output_dir` is a writable scratch directory. Bundles must not assume
    `/workspace/output` exists.
- Bundle selection (`model.py:resolve_bundle`):
  `bundle_path`, or `<ckpt>/<task_name>/controller.py`, or `<ckpt>/controller.py`.
  One checkpoint can therefore carry one frozen bundle per task.

## Motion toolkit (`packages/robodojo_toolkit`)

A source package that bundles depend on. It isn't copied into skills; skills only
show its entry points. It runs on the policy side with observations as its only input.

- `pose_math(arguments)`: the same dictionary API and results as
  `robodojo_pose_math`, in NumPy/SciPy. It matches the harness's Isaac-based
  implementation on 37,500 random cases (maximum difference 1.6e-14).
- `DualArm()`: X5 forward kinematics, `check()` against an observation, and
  `step_toward()`, the bounded DLS update behind `step_eef`. The code is copied
  from `services/robodojo/kinematics.py` and parity-tested against it. It uses the packaged URDF (Apache-2.0
  assets), the arm roots from `env_cfg/robot/dual_x5.yml`, and the joint limits
  read from `ARX.usd`, which equal the URDF's (±10 rad on joints 1–5, ±3.14 on joint 6).
- `policy_actions(...)`: planner samples → 25 Hz 14-D rows. Identical to
  `services/robodojo/trajectory.py`.
- `Planner(config="official")` / `shared_planner()`: cuRobo free-space planning
  with the vendored RoboDojo planner (MIT; hash in its header), built exactly as
  RoboDojo's robot manager builds it:
  - identity origin;
  - `dt = 0.004` (so every 10th planner sample becomes a 25 Hz row);
  - table height `0.74 − root z`;
  - the same 1 mm / 0.005 rad endpoint acceptance as `free_space_move`.

  `official` uses the published `curobo_tmp.yml`, with paths filled in the way
  upstream `utils/update_embodiment_config_path.py` fills them. The adapter
  builds and warms it while the policy server loads, before the port opens, so
  warp's first-use kernel compilation stays out of the 120 s call limit.

**Accuracy on the official path (2026-09-28).** The `examples/planning_smoke_bundle`
run went through `run_eval.sh sim make_kong` on upstream RoboDojo:
- 8/8 planned moves reached their goals;
- worst errors were 0.13 mm and 0.00033 rad (bar: 1 mm / 0.005 rad, as in
  `test_gpu_joint_replay.py`);
- each row advanced the environment by exactly one step;
- packaged FK matched the observed pose within 0.03 mm.

**Harness vs official collision model.** The harness's `free_space_move` uses the
vendored `x5_v2` config, which differs from the official template in its collision
spheres and self-collision settings. On six test moves both configs accepted and
rejected the same ones.

## Hosted agent API: what we submit and what we host

The submission is modelled on RoboProbe's GPT-6 Astra harness: an ordinary XPolicyLab
adapter that loads no checkpoint and calls a hosted model API from inside the policy
server for every action chunk, with the endpoint and key given at runtime.

```text
evaluator machine                              our machine (e.g. near HK)
official env client --ws--> Mooncake_Agent  --HTTPS /v1/act-->  endpoint/agent_api.py
                            (no checkpoint,                     |  task_router.py: which task
                             no GPU, no task                    v  (Gemini on instruction + head camera)
                             logic)                             AgentBundle worker per task, 127.0.0.1
                                                                (frozen bundle, cuRobo, Gemini key)
```

**Submitted (`Mooncake_Agent/`):** the upstream `demo_policy` `deploy.py` byte for byte,
and a `model.py` that stores the latest observation and, on `get_action`, POSTs it
(PNG-encoded cameras, lossless) with a bearer key, returning the action chunk. It has
no task names, bundles, routing or Gemini key. Configuration is only
`MOONCAKE_BASE_URL` and `MOONCAKE_API_KEY`, like `L3_INSPECT_BASE_URL` for RoboProbe.
One request per chunk: the intermediate `update_obs` calls stay on the evaluator's
machine, since our bridge only uses the observation before each `get_action`.

**Hosted (`serve_endpoint.sh`, never submitted):**

```bash
MOONCAKE_ENDPOINT_KEYS=<client key> OPENROUTER_API_KEY=<key> \
  bash official/xpolicylab/serve_endpoint.sh <checkpoint> 8600 0,1 9100 127.0.0.1
# then a TLS reverse proxy: https://<host>/v1/act -> 127.0.0.1:8600
```

- It starts one AgentBundle worker per `<task>/controller.py` in the checkpoint
  (`ENDPOINT_TASKS` selects a subset, `WORKERS_PER_TASK` adds parallel copies). Workers
  are bound to 127.0.0.1 and accept only the endpoint's `agentbundle_hello` token.
- **Routing, on our side only:** the client never sends a task name. On an episode's
  first request, `task_router.py` asks Gemini to choose among the served tasks from the
  instruction and the head camera: the instruction names the base task, and clutter or
  a randomized scene marks `*_random`. Invalid answers or an unavailable Gemini fall
  back to a word match on the instruction, preferring the `_random` bundle. Every
  decision is appended to `<run>/episodes.jsonl` for audit.
- **Failure handling:** a worker error answers with a hold step, never an HTTP error,
  because an error reaching the evaluator's client is fatal for the whole trial. A
  repeated `request_id` (client retry after a dropped reply) is answered from cache.
- **Timing:** the client waits 110 s per request, below the environment client's 120 s.
  The worker answers within `action_wait_s` (90 s) by holding. The first request of
  an episode adds one Gemini call for task detection.
- **Gemini:** bundles call `gemini_generate` on the workers, each with its own
  `gemini_budget_usd` cap; task detection has its own cap (`--gemini-budget-usd`, $2).
  Keys stay on our machine.
- **Security:** client keys (`MOONCAKE_ENDPOINT_KEYS`) are checked on every request;
  serve only through TLS.

## Generalization variants (`*_random`)

RoboDojo ships 54 task configs: 42 base tasks and 12 `_random` generalization
variants (for example `make_toast` and `make_toast_random`). Checked in upstream
`ee67a146`:

- **Separate tasks.** Each variant has its own config, task class, saved layouts
  (`Eval_Layout/.../<collection>/make_toast_random_<n>.json`, 45 per collection) and
  `eval_nums`. Sweeps start one policy server per task with the exact `task_name`
  (`setup_eval_policy_server.sh` passes `task_name=...`), and results are recorded
  per task.
- **Nothing in the observation says which variant runs.** The instruction is
  identical in 11 of 12 pairs. The exception is `sweep_blocks`: "sweep the blocks"
  versus "sweep the objects".
- **What changes** (make_toast): unseen object models (toaster `index: [2, 4]`
  instead of `[0, 1]`), 15 random `Clutter` distractors, and slightly different
  success thresholds in the task class (bread z bounds).

The task name is the only reliable selector, so route by it. `resolve_bundle` loads
`<checkpoint>/<task_name>/controller.py`, so give each variant its own bundle,
developed in its own session (`--task make_toast_random`):

```bash
bash official/xpolicylab/build_release.sh <out> make_toast=<std_bundle> make_toast_random=<random_bundle>
bash official/xpolicylab/build_release.sh <out> make_toast,make_toast_random=<bundle>   # one bundle for both
```

`build_release.sh` rejects unknown task names. It warns when one variant of a pair
has no bundle: that task would fail to load and score zero. It records each task's
source and sha256 in `TASKS.json`. The dashboard's release panel offers the combined
command, using the newest counterpart session's bundle.

## Download-and-run packaging

Everything is built beforehand; the evaluator only downloads and installs.

```text
<checkpoint>/                       # Hugging Face repo, fetched by the PR's download script
  <task_name>/controller.py (+ modules)   # one frozen bundle per task
  wheelhouse/                       # 93 pinned wheels + nvidia_curobo + robodojo_toolkit
    constraints.txt                 #   (torch 2.7.0+cu128, warp-lang 1.11.0: RoboDojo's own pins)
```

`build_release.sh <out> <task>[,<task>]=<bundle> ...` assembles the checkpoint:
- the per-task bundles;
- the wheelhouse, with the toolkit wheel rebuilt;
- `warp-cache/`: portable sm_80 PTX kernels, which Ampere-and-newer drivers load
  without compiling;
- `SHA256SUMS`.

`AgentBundle/install.sh <env> <wheelhouse>` installs from `--no-index` only:
first the pinned set, then the two project wheels with `--no-deps`. The cuRobo
fork (`d17b54c`) is pure Python and needs no CUDA toolchain. Build the
wheelhouse with `setup_policy_env.sh` plus
`pip wheel`/`pip download -r constraints.txt` (run 2026-09-28; 4.1 GB).

**Download-and-run drill (2026-09-28).** Evaluator-side sequence:
1. `sha256sum -c` passes.
2. `conda create --offline` plus `install.sh` build the env from the checkpoint
   wheelhouse only.
3. The adapter copies the shipped kernel cache and warms the planner at load.
4. The official `run_eval.sh sim make_kong` smoke passes 8/8 (0.13 mm /
   0.00033 rad), 136 s wall including Isaac startup.

Planner load cost: 29 s cold (about 11 s of it is warp compilation) versus 18 s
with the shipped cache. The remainder is cuRobo's per-process GPU
initialization, which falls inside the client's 900 s load window. A single
plan takes about 0.03 s.

## Local tooling (self-contained)

State lives under `runtime/official-xpolicylab/`. Shared checkouts are never
modified.

```bash
bash official/xpolicylab/setup_policy_env.sh   # isolated policy env (XPolicyLab deps only)
bash official/xpolicylab/stage.sh              # pristine sources at the pinned commits
bash official/xpolicylab/run_eval.sh debug stack_bowls official/xpolicylab/examples/probe_bundle
bash official/xpolicylab/run_eval.sh sim make_kong <checkpoint> 0 1 1   # seed, eval_num, GPU
```

`stage.sh` exports RoboDojo `ee67a146` and its XPolicyLab submodule
`432f82b1` with `git archive`. It symlinks the clean `third_party` submodules
and the assets, and links `AgentBundle` from this worktree.

**RoboDojo code is exactly upstream.** Upstream reads `Assets/Robots/x5/curobo.yml`,
but the public Hugging Face assets ship only its template `curobo_tmp.yml`: the same
schema, with `${ASSETS_PATH}` placeholders that mean the RoboDojo root. The stage
materializes `curobo.yml` from that template by substitution alone, into a symlinked
`Assets` overlay. No patch is applied. `run_eval.sh` accepts the Omniverse EULA
(operator-approved).

Verified 2026-09-28:
- **Debug loop:** 10 episodes, `[MAIN] eval finished`.
- **Simulator, `make_kong`:** one episode on the staged sources.
  - The bundle came from a two-task checkpoint.
  - It received 3 × 640×480 RGB images and a 14-D state.
  - Its steps counted against the official 600-step limit.
  - The episode ended with `eval finished`.
- **Unit tests:** `local_tests/unit/test_official_profile.py`.

## Official leaderboard protocol (robodojo-benchmark.com/leaderboard, checked 2026-09-29)

Quoted from the leaderboard page's publication rules:

- **Remote evaluation:** "Policies must be evaluated through the RoboDojo online
  evaluation system, either by submitting a deployable policy package or by
  connecting a remote policy server." "Participants can run a remote policy server
  and communicate with the official RoboDojo evaluation client through the
  standardized deployment protocol." "Reported scores are computed by the official
  evaluation system rather than self-reported by participants."
  - We submit a deployable package (`Mooncake_Agent/`) that calls our hosted agent
    API, the pattern of RoboProbe's GPT-6 Astra harness (see "Hosted agent API").
- **Seeds:** "each submitted policy is evaluated under three random seeds" (one
  checkpoint under three evaluation seeds is allowed); mean and standard deviation
  are reported.
- **Hidden verification:** "each submitted model is also evaluated on hidden
  verification layouts … If the hidden-layout performance differs significantly
  from the public-layout performance, the submission is considered invalid."
- **What to submit:** inference code and weights following XPolicyLab standards
  (pull request, private repository or directly); complete local results from
  `bash robodojo.sh summerize`; a description with model type(s) "Agent, VLA, WAM,
  LLM, VLM, or IL. Multiple types may be combined."; a commitment to release the
  artifact within one week of listing.
- **Verified-board eligibility:** at least one of "architectural or methodological
  innovation" or "distinct data or pre-training methodology", plus a released or
  planned paper or technical report. Results without released code, checkpoint and
  configuration "are reported separately and are not considered verified".
- **Scoring:** Generalization base and `_random` tasks run 25 trials per seed each;
  the summary "merges each Generalization base task with its `_random` sibling"
  (Quick Evaluation, Benchmark rules).

## Official rules found (public sources, 2026-09-28)

- **Submission:** an XPolicyLab PR (`policy/<POLICY>/`) plus an application at
  robodojo-benchmark.com/eval. A remote policy server is also accepted.
- **Verified entries** must release the checkpoint, the training and
  deployment code, and instructions at publication. Unreleased results are
  "reported separately". An eval-only first PR is allowed with a training
  release timeline.
- **Not permitted:** "agents or other high-level orchestrators that coordinate
  multiple explicitly distinct policies or models". A single frozen controller
  is one policy.
- **Network/API:** not documented for hosted packages. The RoboDojo team
  itself listed API-driven LLM entries (GPT-6-Astra and others) as reported
  results.
- **Hardware and time:** hardware is not documented. Every websocket request
  (`update_obs`, `get_action`, …) has a 120 s timeout: `PolicyEvalClientConfig`'s
  default, which `eval_env.py` does not override, and which a policy's `deploy.yml`
  does not reach. A timeout is fatal for the trial. The bridge therefore answers
  `get_action` with one hold step after `action_wait_s` = 90 s.

## Community practice (public sources, 2026-09-28)

- **Network:** merged adapters download checkpoints from the Hugging Face Hub
  at model load (InternVLA_A1 `snapshot_download`, MolmoAct2 #110), so outbound
  HTTP at load time appears to work. API calls in hosted packages remain
  unconfirmed. This bundle needs no network at load or run time.
- **GPU:** the simulator needs at least 16 GB VRAM and RT cores. The hosted
  policy GPU is unpublished; submissions report that 24 GB cards may OOM
  (OpenWAM #113) and validate on an L40 (48 GB) (#110). Isaac Sim 5.1 needs a
  580-series driver on Blackwell (RoboDojo #23).
- **RoboDojo PR #48 (button drive-target reset):** still open and unmerged,
  and independently reproduced on the unmodified image. Official evaluation
  therefore undercounts `swap_blocks`.
  - Historical: harness sessions before 2026-09-28 used a checkout with the fix
    backported, so their `swap_blocks` results were optimistic relative to official.
  - Current: `NativeBackend` runs the pristine official stage without it, so
    harness physics matches official evaluation.
  - Open PRs #58/#60/#61/#62 also change reset behaviour.

## Open issues

1. **Ask RoboDojoCommittee@gmail.com:**
   - Is an agent-generated, non-learned controller bundle (no checkpoint)
     eligible for the verified board?
   - Does the hosted policy server have outbound network access?
   - Is the official `curobo.yml` generated from `curobo_tmp.yml` by the same substitution?
2. **Layouts:** exploration uses every saved layout of official collection 0, then
   collection 1, then 2, so any budget up to their total is valid (135 for `*_random`
   tasks, which have 45 per collection). The harness formal batch runs on
   collection 3, novel layouts from `scripts/generate_layouts.py` under
   `Eval_Layout/RoboDojo/arx_x5/3/`; configuration fails until it holds enough
   layouts for the task.

Resolved on this branch: the instructions teach only official tools plus
`robodojo_toolkit`; the generated `CALIBRATION.md` link test passes.
