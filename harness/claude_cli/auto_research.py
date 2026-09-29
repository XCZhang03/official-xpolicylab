"""Run Claude Code as the auto-research agent in Docker, like the Codex launcher.

The container is the same (offline, read-only root, capped workspace, one GPU); the
host keeps the robot MCP frontend and a Messages relay that holds the provider
credential (messages_relay.py). Selected by ``agent_cli: claude`` in the operator
configuration; everything else about a session is shared with the Codex launcher.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from harness.claude_cli.messages_relay import Credential, MessagesRelay, PROVIDERS
from harness.codex_cli import workspace_storage
from harness.codex_cli.auto_research import PROJECT, prepare_workspace, session_timeout
from services.controller.sandbox import DockerSandbox

HOME_NAME = 'claude-home'
DENIED_TOOLS = ('WebFetch', 'WebSearch')
# Same reasoning effort as Codex sessions (model_reasoning_effort: xhigh); Claude Code's
# default is high.
EFFORT = 'xhigh'
RELAY_TOKEN = 'host-relay'  # Placeholder the relay replaces; not a credential.


def claude_binary(path=None):
    binary = Path(path or shutil.which('claude') or '').resolve()
    if not binary.is_file():
        raise RuntimeError('Claude Code binary is required (install claude on the host)')
    return binary


def deploy(config, *, provider='openrouter', key_file=None, model=None, claude=None):
    """Prepare a new SSD workspace for Claude Code without starting a model or simulator."""
    binary = claude_binary(claude)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    model = model or PROVIDERS[provider]['model']
    auditor, fast = PROVIDERS[provider]['auditor'], PROVIDERS[provider]['fast']
    account = None
    if provider == 'claude-login':
        from harness.claude_cli import experiment_login
        key_file = key_file or experiment_login.token_file()  # Never the personal ~/.claude login.
        account = experiment_login.status()
    Credential(provider, key_file=key_file)  # Validate before provisioning.
    # Only paths and names: the relay reads the credential itself at each request.
    relay_file = root/'model-relay.json'
    relay_file.write_text(json.dumps({
        'agent_cli': 'claude', 'provider': provider, 'base_url': PROVIDERS[provider]['base_url'], 'model': model,
        'models': sorted({model, auditor, fast}),
        'key_file': str(key_file) if key_file else None,
        # Identity only (no token): which dedicated account this session bills.
        'account': {k: account.get(k) for k in ('email', 'organization')} if account else None}))
    relay_file.chmod(0o600)
    prepared = prepare_workspace(config, HOME_NAME)
    workspace, home = prepared['workspace'], prepared['home']
    # Claude Code reads CLAUDE.md and .claude/skills; the template keeps AGENTS.md and
    # .agents/skills, shared with Codex.
    (workspace/'CLAUDE.md').write_text('@AGENTS.md\n')
    (workspace/'.claude').mkdir(exist_ok=True)
    (workspace/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)
    timeout_ms = (session_timeout(config)+300)*1000
    settings = {
        'permissions': {'defaultMode': 'bypassPermissions', 'deny': list(DENIED_TOOLS)},
        'skipDangerousModePermissionPrompt': True,
        'autoUpdates': False, 'includeCoAuthoredBy': False, 'cleanupPeriodDays': 3650,
        'env': {
            'ANTHROPIC_BASE_URL': 'http://127.0.0.1:17777', 'ANTHROPIC_AUTH_TOKEN': RELAY_TOKEN,
            # Subagents inherit the agent's model unless spawned with model "sonnet"
            # (cheaper auditors, subagent-audit skill) or "haiku".
            'ANTHROPIC_MODEL': model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': auditor, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': fast,
            # A formal batch pauses the container; the host relay keeps provider I/O alive.
            'API_TIMEOUT_MS': str(timeout_ms), 'MCP_TOOL_TIMEOUT': str(timeout_ms), 'MCP_TIMEOUT': '60000',
            'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': str(session_timeout(config)*1000),
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY': '1',
            # Every MCP tool stays a local tool definition (the relay refuses server tools).
            'ENABLE_TOOL_SEARCH': 'false'}}
    (home/'settings.json').write_text(json.dumps(settings, indent=2))
    (home/'.claude.json').write_text(json.dumps({
        'hasCompletedOnboarding': True, 'bypassPermissionsModeAccepted': True,
        'projects': {'/workspace': {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {'auto_research': {
        'type': 'stdio', 'command': '/usr/local/bin/python', 'args': ['/opt/agent/container_bridge.py', 'mcp']}}}))
    return {'workspace': workspace, 'claude_home': home, 'claude': binary, 'model': model,
            'configuration': prepared['configuration'], 'relay_configuration': relay_file,
            'bridge': prepared['bridge']}


def claude_arguments(deployed, *, prompt, interactive=False, json_output=False):
    args = ['--model', deployed['model'], '--effort', EFFORT, '--mcp-config', '/claude-home/mcp.json', '--strict-mcp-config',
            '--permission-mode', 'bypassPermissions', '--disallowedTools', *DENIED_TOOLS]
    if not interactive:
        args = ['-p', *args, '--output-format', 'stream-json' if json_output else 'text', '--verbose']
    return [*args, '--', prompt]


def container_command(config, deployed, relay_directory, name, *, prompt, interactive=False, json_output=False):
    if os.getuid() == 0:
        raise RuntimeError('Run the launcher as a non-root operator')
    lim = config.training
    command = ['docker', 'run', '--pull=never', '--name', name, '--interactive',
        '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--ipc=private', '--init',
        '--pids-limit', str(lim.pids), '--memory', f'{lim.memory_mb}m', '--memory-swap', f'{lim.memory_mb}m',
        '--cpus', str(lim.cpus), '--ulimit', 'core=0', '--log-driver=none',
        '--tmpfs', f'/tmp:rw,nosuid,nodev,size={lim.scratch_mb}m,mode=1777',
        # HOME must differ from the project directory: Claude Code skips project skills
        # (.claude/skills) when the working directory is the home directory.
        '--workdir', '/workspace', '--env', 'HOME=/claude-home', '--env', 'CLAUDE_CONFIG_DIR=/claude-home',
        '--env', 'AGENT_BINARY=/opt/claude/claude',
        '--env', 'PIP_NO_INDEX=1', '--env', 'HF_HUB_OFFLINE=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH=/workspace', '--env', 'PATH=/opt/claude:/usr/local/bin:/usr/bin:/bin',
        '--env', 'TERM=xterm-256color',
        '--env', 'ROBODOJO_TOOLKIT_PLANNER_SOCKET=/tmp/robodojo-toolkit/planner.sock']
    if interactive:
        command.append('--tty')
    mounts = [(deployed['workspace'], '/workspace', False), (deployed['claude_home'], '/claude-home', False),
        (deployed['claude_home']/'settings.json', '/claude-home/settings.json', True),
        (deployed['claude_home']/'mcp.json', '/claude-home/mcp.json', True),
        (config.root/'published', '/workspace/runtime/autonomous_controller', True),
        (relay_directory, '/run/agent-relay', True), (deployed['bridge'], '/opt/agent/container_bridge.py', True),
        (deployed['claude'], '/opt/claude/claude', True)]
    for source, destination, readonly in mounts:
        if any(c in str(source) for c in (',', '\n')):
            raise ValueError('Invalid mount path')
        command += ['--mount', f'type=bind,src={source},dst={destination}'+(',readonly' if readonly else '')]
    if config.training_gpu:
        command += ['--gpus', 'device='+config.training_gpu, '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility']
    command += ['--entrypoint', '/usr/local/bin/python', config.image, '/opt/agent/container_bridge.py',
                *claude_arguments(deployed, prompt=prompt, interactive=interactive, json_output=json_output)]
    return command


def launch(config, deployed, *, gemini_key_file=None, prompt, interactive=False, capture=False, max_model_calls=4000):
    """Host owns both relays and cleanup; Docker owns every agent-executed command."""
    DockerSandbox(config.image, config.training).preflight()
    workspace_storage.validate(config.root, config.workspace_mb, deployed['workspace'], deployed['claude_home'])
    relay = json.loads(deployed['relay_configuration'].read_text())
    if relay.get('credentials_file'):
        raise RuntimeError('This session was prepared with the personal Claude login, which experiments no '
                           'longer use; prepare a new session (claude-login uses the experiment token)')
    credential = Credential(relay['provider'], key_file=relay['key_file'])
    name = 'robodojo-agent-'+uuid.uuid4().hex
    server = model = agent = None
    with tempfile.TemporaryDirectory(prefix='robodojo-relay-') as temporary:
        relay_dir = Path(temporary)
        with (config.root/'frontend.log').open('wb') as log:
            try:
                model = MessagesRelay(relay_dir/'model.sock', credential, base_url=relay['base_url'],
                                      models=relay.get('models') or [relay['model']], max_calls=max_model_calls,
                                      client_write_timeout=session_timeout(config)+300,
                                      usage_log=config.root/'model-usage.jsonl')
                model.start()
                args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
                        '--workspace', str(deployed['workspace']), '--socket', str(relay_dir/'mcp.sock'),
                        '--agent-container', name]
                if gemini_key_file:  # gemini_generate stays brokered by the host frontend.
                    args += ['--key-file', str(gemini_key_file)]
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
