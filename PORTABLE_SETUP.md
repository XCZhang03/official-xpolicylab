# Portable setup

The repository holds only harness source. Simulator checkouts, assets, Python
environments, wheelhouses, images, secrets and experiment outputs are machine-local
under the ignored `runtime/` link. Never commit them.

## 1. Sources

```bash
git clone <this-repository> robot-agent
cd robot-agent
bash scripts/bootstrap_sources.sh
```

The bootstrap script:
- creates the `runtime` link, on `/mnt/ssd8` when that is writable, or at
  `$ROBODOJO_RUNTIME_ROOT`;
- clones RoboDojo at the revision in `dependencies.lock`, with its pinned submodules;
- verifies and applies the versioned simulator patches.

## 2. Simulator environment and assets

Install RoboDojo and Isaac Sim following the upstream instructions, so that their
Python environment is available at `runtime/envs/robodojo`. `dependencies.lock`
records the known-good versions. Accept NVIDIA's license as the installing user.
Then:

```bash
bash scripts/setup_assets.sh                                   # RoboDojo assets
runtime/envs/robodojo/bin/python scripts/setup_demo_context.py # optional: pre-cache task demos
bash scripts/robodojo_doctor.sh                                # check pins and assets
```

When a session is prepared, the official task-page demonstration clip for its task
is cached if it is missing.

### Task source packages

Every session workspace gets a read-only `task_source/` for its task: the task code,
the reward checks it uses, its configs and its objects' geometry. These packages are
generated from the pinned RoboDojo checkout and its assets, not committed (about
11 GB for all 54 tasks, mostly `object.usdz` originals). Build them once:

```bash
python3 -m venv runtime/envs/usd-tools       # verified with Python 3.12
runtime/envs/usd-tools/bin/pip install usd-core==26.8 numpy pyyaml
runtime/envs/usd-tools/bin/python scripts/build_task_sources.py              # every task
runtime/envs/usd-tools/bin/python scripts/build_task_sources.py --task make_toast   # or one task
```

Each package in `runtime/task-sources/<task>/` records the RoboDojo commit and every
file's sha256 in `MANIFEST.json`. Deployment refuses a missing, modified or
out-of-date package and prints this build command. Rebuild after changing the
RoboDojo pin.

### Held-out formal layouts

The harness formal batch runs on collection 3 by default: novel layouts generated
from each task's randomization, so they differ from the official collections 0–2
used for exploration. Generate them per task before configuring a session; the
generator refuses tasks with clutter. For example, 50 layouts for `make_kong`:

```bash
runtime/envs/robodojo/bin/python scripts/generate_layouts.py --task make_kong --count 50 \
    --output RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/3
```

Configuration fails until the chosen formal collection holds enough layouts for the
task. A formal collection that is also explored is rejected.

## 3. Official policy environment and wheelhouse

The official submission and the agent image share one pinned package set: torch
2.7.0+cu128, warp-lang 1.11.0, the pinned cuRobo fork and `robodojo_toolkit`.

```bash
bash official/xpolicylab/setup_policy_env.sh   # runtime/official-xpolicylab/policy-env
bash official/xpolicylab/stage.sh              # pristine upstream RoboDojo + XPolicyLab
```

Build the offline wheelhouse in `runtime/official-xpolicylab/wheels/`, with
`constraints.txt`, as described in
[official/xpolicylab/README.md](official/xpolicylab/README.md#download-and-run-packaging).
Internet access is needed only for this local build. The evaluator and the agent
containers install from the wheelhouse with `--no-index`.

## 4. Agent image

```bash
bash scripts/build_official_image.sh   # robodojo-official:dev
```

This installs the wheelhouse and the prebuilt portable (sm_80+ PTX) cuRobo kernel
cache into a `python:3.11-slim` image, then checks that the toolkit, cuRobo, torch
and warp import. Session configurations pin the image digest; launchers never pull.

## 5. Credentials and run

Codex reaches its model through a host-owned relay. You need:
- a `codex` binary on `PATH`;
- a Responses-API provider in `$CODEX_HOME/config.toml` or
  `$CODEX_HOME/openrouter.config.toml` (`--profile` selects the file);
- the private key file `~/.codex/secrets/openrouter_api_key`: mode 0600, owned by
  you. `--key-file` overrides the path. The key never enters the agent
container or the robot MCP. Then start the dashboard (see [README.md](README.md)):

```bash
python3 scripts/robot_lab.py dashboard
```
