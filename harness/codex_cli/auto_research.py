"""Run the complete agent (Codex, or Claude Code via harness/claude_cli) in Docker; keep robot
services and the model relay on the host."""
from dataclasses import asdict
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid

from harness.codex_cli.configuration import _write_config, load_key, ROBOT_STATUS_LINE
from harness.codex_cli.model_relay import ModelRelay, MODEL

AUDITOR_MODEL = 'openai/gpt-6-sol'  # A fifth of MODEL's price per token (OpenRouter, 2026-09-29).
from harness.codex_cli import workspace_storage
from services.controller.config import Configuration
from services.controller.demonstrations import provision_demonstration, task_demonstration_context
from services.controller.sandbox import DockerSandbox
from services.controller.storage import persist
from services.robodojo.task_rubrics import task_rubric_context
from services.robodojo.task_variants import task_variant_context

PROJECT = Path(__file__).resolve().parents[2]
from services.robodojo.timeouts import task_timeouts


def session_timeout(config):
    return task_timeouts(config.task)['session_seconds']
DISABLED_FEATURES = ('apps', 'plugins', 'browser_use',
    'browser_use_external', 'computer_use', 'image_generation', 'artifact',
    'skill_mcp_dependency_install', 'hooks', 'enable_request_compression')


def prepare_workspace(config, home_name):
    """Capped storage, the agent workspace (template, task source, TASK.md), the private
    configuration and the container bridge; shared by the Codex and Claude launchers."""
    root = config.root
    storage = workspace_storage.provision(root, config.image, config.workspace_mb)
    workspace, home = root/'agent-workspace', root/home_name
    for link, name in ((workspace, 'workspace'), (home, home_name)):
        (storage/name).mkdir(mode=0o700)
        link.symlink_to(storage/name, target_is_directory=True)
    for name in ('code', 'memory', 'output', 'runtime/autonomous_controller'):
        (workspace/name).mkdir(parents=True)
    (root/'published').mkdir(mode=0o700)
    from services.mcp_contract import Contract, MODES
    template = PROJECT / MODES[config.mode].workspace
    shutil.copytree(template, workspace, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # api.runtime.run_official uses the exact official adapter bridge in development too.
    from services.controller.storage import BRIDGE_SOURCE
    shutil.copyfile(BRIDGE_SOURCE, workspace/'api/bundle_bridge.py')
    from services.mcp_workspace import compose
    compose(workspace, Contract.from_config(config), task=config.task)
    demonstration = provision_demonstration(config, workspace, PROJECT/'runtime/reference-demos/website')
    # Read-only official task source and object assets for understanding the task only.
    from services.robodojo.task_source import provision_task_source
    provision_task_source(config.task, workspace)
    (workspace/'TASK.md').write_text(task_rubric_context(config.task, include_source_url=False)
                                   + task_variant_context(config.task)
                                   + task_demonstration_context(demonstration))
    private_config = root/'configuration.json'
    with private_config.open('x') as stream:
        json.dump(asdict(config), stream, default=str)
    private_config.chmod(0o600)
    bridge = root/'container_bridge.py'
    shutil.copyfile(PROJECT/'harness/codex_cli/container_bridge.py', bridge)
    return {'workspace': workspace, 'home': home, 'configuration': private_config, 'bridge': bridge}


def deploy(config, *, provider_home, profile='openrouter', key_file=None, codex=None):
    """Prepare a new SSD workspace without starting a model or simulator."""
    if not profile or Path(profile).name != profile:
        raise ValueError('Invalid provider profile')
    codex = Path(codex or shutil.which('codex') or '').resolve()
    if not codex.is_file():
        raise RuntimeError('Codex binary is required')
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    provider = {}
    for path in (provider_home/'config.toml', provider_home/f'{profile}.config.toml'):
        if path.exists():
            data = tomllib.loads(path.read_text())
            for key in ('model_provider', 'model_providers'):
                if key in data:
                    if key == 'model_providers':
                        provider.setdefault(key, {}).update(data[key])
                    else:
                        provider[key] = data[key]
    selected = provider['model_providers'][provider['model_provider']]
    if selected.get('wire_api', 'responses') != 'responses':
        raise ValueError('Responses provider required')
    headers = dict(selected.get('http_headers', {}))
    key = load_key(key_file) if key_file else None
    if key:
        headers['Authorization'] = 'Bearer '+key
    elif selected.get('env_key'):
        headers['Authorization'] = 'Bearer '+os.environ[selected['env_key']]
    elif selected.get('auth'):
        raise ValueError('Auth commands are not copied or executed; supply --key-file for the host relay')
    # Provider/auth material stays in a private host file, never in a mounted config.
    relay_file = root/'model-relay.json'
    # Spawned subagents may use the cheaper AUDITOR_MODEL (subagent-audit skill); Codex's model
    # catalog names it without the provider prefix. Main-agent calls keep MODEL.
    relay_file.write_text(json.dumps({'base_url': selected['base_url'], 'headers': headers,
                                      'additional_models': {AUDITOR_MODEL: None},
                                      'model_aliases': {AUDITOR_MODEL.split('/', 1)[1]: AUDITOR_MODEL}}))
    relay_file.chmod(0o600)
    prepared = prepare_workspace(config, 'codex-home')
    workspace, home = prepared['workspace'], prepared['home']
    private_config, bridge = prepared['configuration'], prepared['bridge']
    settings = {'model_provider': 'host_relay', 'model': MODEL, 'model_reasoning_effort': 'xhigh',
        'model_providers': {'host_relay': {'name': 'Host model relay',
            'base_url': 'http://127.0.0.1:17777', 'wire_api': 'responses', 'requires_openai_auth': False,
            # Codex owns model recovery and retains completed tool results. Do
            # not wrap launch(), MCP calls, or robot episodes in retry loops.
            'request_max_retries': 2, 'stream_max_retries': 2,
            # Docker pause freezes Codex's stream reader, not its wall clock.
            # A formal batch can pause it for most of the session. The host
            # still enforces provider I/O and the absolute launcher deadline.
            'stream_idle_timeout_ms': (session_timeout(config)+300)*1000}},
        'model_auto_compact_token_limit': 256000, 'model_auto_compact_token_limit_scope': 'total',
        'approval_policy': 'never', 'sandbox_mode': 'danger-full-access', 'web_search': 'disabled',
        'features': {**{name: False for name in DISABLED_FEATURES}, 'shell_tool': True,
                     'unified_exec': True, 'code_mode_host': True, 'skip_host_skill_discovery': True},
        'projects': {'/workspace': {'trust_level': 'trusted'}}, 'tui': {'status_line': ROBOT_STATUS_LINE},
        'mcp_servers': {'auto_research': {'command': '/usr/local/bin/python',
            'args': ['/opt/agent/container_bridge.py', 'mcp'], 'required': True,
            'tool_timeout_sec': session_timeout(config)+300,
            'default_tools_approval_mode': 'approve'}}}
    # Subagents: one per exploration environment, plus the read-only overfit auditor
    # (subagent-audit skill), which every session needs.
    settings['features'].update(multi_agent=True, multi_agent_v2=True)
    settings['agents'] = {'enabled': True, 'max_concurrent_threads_per_session': config.exploration_envs + 1}
    _write_config(home/'config.toml', settings)
    resume_session = None
    return {'workspace': workspace, 'codex_home': home, 'codex': codex, 'resume_session': resume_session,
            'configuration': private_config, 'relay_configuration': relay_file, 'bridge': bridge}


def container_command(config, deployed, relay_directory, name, *, prompt, interactive=False, json_output=False):
    if os.getuid() == 0:
        raise RuntimeError('Run the launcher as a non-root operator')
    lim = config.training
    # Docker --env replaces the image's PYTHONPATH; robodojo_toolkit is installed in the image.
    python_path = '/workspace'
    command = ['docker', 'run', '--pull=never', '--name', name, '--interactive',
        '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--ipc=private', '--init',
        '--pids-limit', str(lim.pids), '--memory', f'{lim.memory_mb}m', '--memory-swap', f'{lim.memory_mb}m',
        '--cpus', str(lim.cpus), '--ulimit', 'core=0', '--log-driver=none',
        '--tmpfs', f'/tmp:rw,nosuid,nodev,size={lim.scratch_mb}m,mode=1777',
        '--workdir', '/workspace', '--env', 'HOME=/workspace', '--env', 'CODEX_HOME=/codex-home',
        '--env', 'PIP_NO_INDEX=1', '--env', 'HF_HUB_OFFLINE=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH='+python_path,
        '--env', 'PATH=/opt/codex:/usr/local/bin:/usr/bin:/bin', '--env', 'TERM=xterm-256color',
        # Development scripts share one warmed cuRobo planner process (robodojo_toolkit.service);
        # isolated runs and the official server build theirs in-process, unaffected.
        '--env', 'ROBODOJO_TOOLKIT_PLANNER_SOCKET=/tmp/robodojo-toolkit/planner.sock']
    if interactive:
        command.append('--tty')
    mounts = [(deployed['workspace'], '/workspace', False), (deployed['codex_home'], '/codex-home', False),
        (deployed['codex_home']/'config.toml', '/codex-home/config.toml', True),
        (config.root/'published', '/workspace/runtime/autonomous_controller', True),
        (relay_directory, '/run/agent-relay', True), (deployed['bridge'], '/opt/agent/container_bridge.py', True),
        (deployed['codex'], '/opt/codex/codex', True)]
    helper = deployed['codex'].with_name('codex-code-mode-host')
    if helper.is_file():
        mounts.append((helper, '/opt/codex/codex-code-mode-host', True))
    for source, destination, readonly in mounts:
        if any(c in str(source) for c in (',', '\n')):
            raise ValueError('Invalid mount path')
        command += ['--mount', f'type=bind,src={source},dst={destination}'+(',readonly' if readonly else '')]
    if config.training_gpu:  # The development container plans with cuRobo on this GPU.
        command += ['--gpus', 'device='+config.training_gpu, '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility']
    command += ['--entrypoint', '/usr/local/bin/python', config.image,
                '/opt/agent/container_bridge.py', '--no-daemon', '-C', '/workspace']
    if not interactive:
        command += ['exec']
        if deployed.get('resume_session'):
            command += ['resume']
        command += ['--skip-git-repo-check']
        if json_output:
            command.append('--json')
    elif deployed.get('resume_session'):
        command.append('resume')
    if deployed.get('resume_session'):
        command.append(deployed['resume_session'])
    command.append(prompt)
    return command


def launch(config, deployed, *, key_file=None, prompt, interactive=False, capture=False, max_model_calls=4000):
    """Host owns both relays and cleanup; Docker owns every agent-executed command."""
    DockerSandbox(config.image, config.training).preflight()
    workspace_storage.validate(config.root, config.workspace_mb,
                               deployed['workspace'], deployed['codex_home'])
    provider = json.loads(deployed['relay_configuration'].read_text())
    name = 'robodojo-agent-'+uuid.uuid4().hex
    server = model = agent = None
    # Only tiny socket files use /tmp, to stay below Linux's Unix-socket path limit.
    with tempfile.TemporaryDirectory(prefix='robodojo-relay-') as temporary:
        relay_dir = Path(temporary)
        with (config.root/'frontend.log').open('wb') as log:
            try:
                model = ModelRelay(relay_dir/'model.sock', **provider, max_calls=max_model_calls,
                                   client_write_timeout=session_timeout(config)+300)
                model.start()
                args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
                        '--workspace', str(deployed['workspace']), '--socket', str(relay_dir/'mcp.sock'),
                        '--agent-container', name]
                # The host MCP frontend brokers gemini_generate: it reads the provider key
                # from the private key file (or the host environment). The agent container
                # never receives it.
                if key_file:
                    args += ['--key-file', str(key_file)]
                environment = {k: v for k, v in os.environ.items() if not k.startswith('ROBODOJO_')}
                environment['PYTHONPATH'] = str(PROJECT)
                server = subprocess.Popen(args, cwd=PROJECT, env=environment, stdout=log, stderr=log)
                deadline = time.monotonic()+30
                while not (relay_dir/'mcp.sock').exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Host MCP startup failed; inspect frontend.log')
                    time.sleep(.05)
                command = container_command(config, deployed, relay_dir, name, prompt=prompt,
                                            interactive=interactive, json_output=capture)
                agent = subprocess.Popen(command, stdout=subprocess.PIPE if capture else None,
                                         stderr=subprocess.PIPE if capture else None, text=capture)
                stdout, stderr = agent.communicate(timeout=session_timeout(config))
                return subprocess.CompletedProcess(command, agent.returncode, stdout, stderr)
            finally:
                # Exact owned identity; never remove containers from other sessions.
                subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
                if agent and agent.poll() is None:
                    agent.terminate()
                    agent.wait(timeout=10)
                if server:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
                if model:
                    model.close()


def startup_prompt(workspace, override=None):
    """Keep the default kickoff editable alongside the standing instructions."""
    return override or (workspace/'START_PROMPT.md').read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--provider-home', type=Path, default=Path(os.environ.get('CODEX_HOME', Path.home()/'.codex')))
    parser.add_argument('--profile', default='openrouter')
    parser.add_argument('--key-file', type=Path, default=Path.home()/'.codex/secrets/openrouter_api_key')
    parser.add_argument('--prepare-only', action='store_true', help='Disposable deployment check; starts no model/episode')
    parser.add_argument('--exec-prompt', help='Run noninteractively instead of opening the terminal UI')
    parser.add_argument('--non-interactive', action='store_true', help='Run the default research prompt without a terminal UI')
    parser.add_argument('--max-model-calls', type=int, default=4000)
    # agent_cli: claude (operator configuration) runs Claude Code instead; see harness/claude_cli.
    parser.add_argument('--claude-key-file', type=Path,
                        help='anthropic: an API key file; claude-login: override the experiment setup-token '
                             'file (default: harness/claude_cli/experiment_login.py). The personal '
                             '~/.claude login is always refused.')
    parser.add_argument('--claude-model', help='Override the provider default (Opus 5.5)')
    args = parser.parse_args()
    if args.max_model_calls < 1:
        parser.error('max-model-calls must be positive')
    config = Configuration.from_dict(json.loads(args.config.read_text()))
    from services.controller.config import with_task_timeouts
    config = with_task_timeouts(config)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    config.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.root/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lifecycle = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing'}
        record = config.root/'agent-lifecycle.json'
        # A repeated launch must not overwrite the original run's lifecycle.
        if record.exists():
            raise FileExistsError('Use a fresh auto-research configuration')
        persist(record, lifecycle)
        try:
            if config.agent_cli == 'claude':
                from harness.claude_cli import auto_research as claude
                provider = config.agent_provider
                key_file = {'openrouter': args.key_file, 'anthropic': args.claude_key_file,
                            'claude-login': args.claude_key_file}[provider]
                if provider == 'claude-login' and not args.claude_key_file:
                    # Fail before provisioning if the experiment token is missing or rejected.
                    from harness.claude_cli import experiment_login
                    experiment_login.check()
                deployed = claude.deploy(config, provider=provider, model=args.claude_model, key_file=key_file)
            else:
                deployed = deploy(config, provider_home=args.provider_home, profile=args.profile, key_file=args.key_file)
            if args.prepare_only:
                print(json.dumps({k: str(v) for k, v in deployed.items()}, indent=2))
                lifecycle['status'] = 'prepared-only'
                return
            prompt = startup_prompt(deployed['workspace'], args.exec_prompt)
            lifecycle['status'] = 'running'
            persist(record, lifecycle)
            interactive = not (args.non_interactive or args.exec_prompt)
            if config.agent_cli == 'claude':
                result = claude.launch(config, deployed, gemini_key_file=args.key_file, prompt=prompt,
                                       interactive=interactive, max_model_calls=args.max_model_calls)
            else:
                result = launch(config, deployed, key_file=args.key_file, prompt=prompt,
                                interactive=interactive, max_model_calls=args.max_model_calls)
            lifecycle.update(status='exited', returncode=result.returncode)
            raise SystemExit(result.returncode)
        except KeyboardInterrupt:
            lifecycle['status'] = 'stopped'
            raise SystemExit(130)
        except Exception as exc:
            lifecycle.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            lifecycle['finished_at'] = time.time()
            persist(record, lifecycle)


if __name__ == '__main__':
    main()
