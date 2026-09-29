# Mooncake_Agent

**Contributor:** Mooncake | **Paper:** Not yet available | **arXiv:** Not yet available | **Original code:** Not yet available

`Mooncake_Agent` is a hosted agent policy. The policy server loads no checkpoint and uses no GPU: for every action chunk it sends the latest observation (head and wrist RGB cameras, proprioception and the official instruction) to a remote agent API over HTTPS and executes the returned chunk of absolute joint or end-effector targets. Serving uses XPolicyLab's websocket contract and the `demo_policy` evaluation loop unchanged.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

No model weights or GPU packages; installs XPolicyLab in editable mode (`msgpack`, `msgpack-numpy`, `numpy` and `opencv` come with it):

```bash
cd XPolicyLab/policy/Mooncake_Agent
bash install.sh
conda activate <policy_env>
```

## Data Processing

Not supported: this is an eval-only submission. The policy is a hosted agent and uses no XPolicyLab training data.

## Training

Not supported: eval-only submission (no `process_data.sh` / `train.sh`).

## Configuration

Required environment variables on the policy-server machine:

```bash
export MOONCAKE_BASE_URL=https://<agent-api-host>   # required, no default host
export MOONCAKE_API_KEY=<key>                       # variable name set by api_key_env
```

`deploy.yml` keys: `base_url` (used when `MOONCAKE_BASE_URL` is unset), `api_key_env`, `api_timeout_s` (110 s per request, below the environment client's 120 s), `api_retries` (1, the same request id is resent after a dropped connection), `image_format` (`png` lossless by default; `jpg` or `raw`).

## Evaluation

`ckpt_name` is ignored (pass any placeholder such as `none`):

```bash
cd XPolicyLab/policy/Mooncake_Agent
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo make_kong none arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Notes

- Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint` (chunks may also carry `*_ee_pose` targets, which `EvalEnv` solves with its own IK).
- The policy server needs outbound HTTPS to the agent API. Each `get_action` is one request, answered within the client timeout; while the agent is still deciding it answers with a hold step.
- API: `POST {MOONCAKE_BASE_URL}/v1/act`, msgpack body `{session_id, episode_id, request_id, observation}` with PNG-encoded camera frames and `Authorization: Bearer <key>`; the reply is `{actions: [<action dict>, ...]}`.
