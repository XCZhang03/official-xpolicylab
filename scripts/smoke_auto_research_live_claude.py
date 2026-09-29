"""Operator-only, bounded paid smoke of the Claude Code launcher; artifacts stay on SSD.

No simulator episode is started: this checks the Claude-specific plumbing (host
Messages relay, MCP over the container bridge, Python api.runtime, skills, subagents,
offline sandbox) that the Codex smokes already cover for Codex.
Usage: smoke_auto_research_live_claude.py [--provider claude-login|openrouter|anthropic] [--max-model-calls N]
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import signal
import subprocess

from harness.claude_cli.auto_research import deploy, launch
from services.controller.config import Configuration, Limits

PROMPT = """Run a bounded infrastructure smoke of this agent environment, NOT a task attempt.
Do not start an episode, register, rehearse, submit or call gemini_generate. Keep every
output short and batch independent checks. Write /workspace/memory/claude-smoke.json with
one entry per check below: {"check", "passed", "evidence"}; then give a one-paragraph report.

1. mcp: list the auto_research MCP tools you have and call exploration_status once.
2. python: with Bash, run python3 -c "from api.runtime import Context\\nwith Context() as c: print(c.call('exploration_status')['content'][0]['text'][:200])"
   (as a small script file under /workspace/code/smoke/) and record its output.
3. sandbox: with Bash, check /.dockerenv exists, no /var/run/docker.sock, no /home/xiangcheng,
   no OPENROUTER_API_KEY / ANTHROPIC_API_KEY env var holding a real key, no file under
   /claude-home containing a credential, and that a TCP connection to 1.1.1.1:443 fails.
4. skills: invoke the motion-toolkit skill with the Skill tool (not by reading files) and quote
   its first heading; list which of the workspace skills the Skill tool offers you.
5. subagent: spawn one subagent with model "sonnet" (as the subagent-audit skill does) to read
   AGENTS.md and return its first heading only.
6. web: state whether WebFetch/WebSearch tools are available to you (they should not be).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', default='claude-login', choices=('claude-login', 'openrouter', 'anthropic'))
    parser.add_argument('--max-model-calls', type=int, default=25)
    parser.add_argument('--key-file', type=Path, help='openrouter/anthropic key, or a setup-token file')
    args = parser.parse_args()
    root = Path('/mnt/ssd8/xiangcheng/codex-workspaces/robot_agent/runtime/integration-tests') / (
        f'live-claude-{args.provider}-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    image = subprocess.check_output(['docker', 'image', 'inspect', 'robodojo-official:dev',
                                     '--format', '{{.Id}}'], text=True).strip()
    limits = Limits(wall_seconds=300, memory_mb=8192, cpus=4, pids=512, scratch_mb=512, artifact_bytes=128*1024**2)
    config = Configuration(root=root, image=image, task='make_kong', exploration_seeds=(0,), formal_seed=1,
        eval_seed=0, sim_gpu='GPU-c4909b1f-facf-4d9f-049d-3d58cb2ca630',
        controller_gpu='GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
        training_gpu='GPU-17fcc761-79d6-cba8-b2db-b8609a55c0c6',
        development=limits, formal=limits, training=limits, workspace_mb=512,
        agent_cli='claude', agent_provider=args.provider)
    key = args.key_file or (Path.home()/'.codex/secrets/openrouter_api_key' if args.provider == 'openrouter' else None)
    # claude-login without --key-file: the dedicated experiment token (never ~/.claude).
    deployed = deploy(config, provider=args.provider, key_file=key)
    (root/'smoke-prompt.txt').write_text(PROMPT)
    print(json.dumps({'root': str(root), 'model': deployed['model']}), flush=True)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    result = launch(config, deployed, gemini_key_file=None, prompt=PROMPT, capture=True,
                    max_model_calls=args.max_model_calls)
    (root/'agent-stdout.jsonl').write_text(result.stdout or '')
    (root/'agent-stderr.txt').write_text(result.stderr or '')
    print(json.dumps({'returncode': result.returncode, 'root': str(root)}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
