"""Agent frontend permissions, tool contracts, and fixed Codex launcher config."""
import json
from pathlib import Path
import tomllib
from dataclasses import replace

import pytest

from test_controller_backend import configuration, supervisor
from services.controller.frontend import ResearchFrontend
from harness.codex_cli.auto_research import deploy, startup_prompt, DISABLED_FEATURES


def data(reply):
    return json.loads(reply['content'][0]['text'])


@pytest.fixture
def frontend(supervisor, tmp_path):
    f = ResearchFrontend(supervisor[0], tmp_path/'agent')
    yield f
    f.close()


def test_frontend_soft_guidance_and_native_images(supervisor, frontend):
    s, backend = supervisor
    f = frontend
    initial = {t['name'] for t in f.tools()}
    assert {'robodojo_observe', 'robodojo_status', 'robodojo_step', 'register'} <= initial
    assert not {'robodojo_free_space_move', 'robodojo_step_eef', 'robodojo_pose_math'} & initial
    assert 'gemini_generate' in initial
    assert not {'operate', 'enter_recorder', 'exit_recorder', 'discard_recorder'} & initial
    assert not {'shell_exec', 'shell_poll', 'shell_stop', 'read_file', 'view_image'} & initial
    assert not any(name.startswith('research_') for name in initial)
    assert 'research_write_file' not in initial and 'train' not in initial
    assert 'robodojo_request_formal_episode' not in initial
    assert not f.status()['manual_success_confirmed']
    bundle = f.workspace/'code/test'
    bundle.mkdir(parents=True)
    (bundle/'controller.py').write_text('def main(ctx): pass')
    first = f.call('start_episode', {})
    assert first['content'][1] == backend.image and 'structuredContent' not in first
    registered = data(f.call('register', {'source': 'code/test'}))
    assert registered['bundle'] in s.state['bundles'] and s.state['active']['mode'] == 'interactive'
    with pytest.raises(RuntimeError, match='rehearsal'):
        f.call('submit', {'bundle': registered['bundle']})
    assert s.state['active']['mode'] == 'interactive'
    backend.success = True
    assert data(f.call('evaluate', {}))['manual_success_confirmed']
    assert initial == {t['name'] for t in f.tools()}


def test_frontend_paths_and_native_recording_bypass_are_rejected(supervisor, tmp_path, frontend):
    s, backend = supervisor
    f = frontend
    for name in ('/etc/passwd', '../private/state.json', 'code/../../private/configuration.json'):
        with pytest.raises((ValueError, PermissionError)):
            f.source(name)
    (f.workspace/'link').symlink_to(tmp_path/'private')
    with pytest.raises(ValueError):
        f.call('register', {'source': 'link'})
    f.call('start_episode', {})
    backend.success = True
    f.call('evaluate', {})
    f.call('robodojo_step', {'actions': [[0]*14]})
    assert backend.step == 1


def test_protocol_errors_do_not_disclose_private_paths(supervisor, tmp_path, frontend):
    f = frontend
    reply = f.handle({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
                     'params': {'name': 'read_file', 'arguments': {'path': 'missing'}}})
    assert reply['id'] == 7 and reply['result']['isError']
    assert str(tmp_path) not in json.dumps(reply)


def test_public_result_preserves_execution_and_evaluation_diagnostics():
    source = {'error_type': 'RuntimeError', 'error': {'reason': 'Container exited early'},
              'evaluation_error': 'TimeoutError', 'evaluation_failure': {'reason': 'Evaluation timed out'},
              'container': 'private-container', 'workspace_path': '/private'}
    result = ResearchFrontend.public_result(source)
    assert result['error'] == source['error'] and result['evaluation_failure'] == source['evaluation_failure']
    assert 'container' not in result and 'workspace_path' not in result


def test_large_inline_image_through_frontend_stdio(frontend):
    import io
    import os
    from PIL import Image
    import base64
    raw = io.BytesIO()
    Image.frombytes('RGB', (768, 768), os.urandom(768*768*3)).save(raw, format='PNG')
    block = {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(raw.getvalue()).decode()}
    # A multi-megabyte line below the wire limit is parsed and answered, not truncated.
    request = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'register', 'arguments': {'source': 'code/missing', 'image': block}}}) + '\n'
    assert len(request) > 2*1024*1024
    output = io.StringIO()
    frontend.serve(io.StringIO(request + json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}) + '\n'), output)
    first, second = map(json.loads, output.getvalue().splitlines())
    assert first['id'] == 1 and first['result']['isError']
    assert 'wire limit' not in json.dumps(first)
    assert second['id'] == 2 and second['result']['tools']


def test_large_inline_gemini_image_through_frontend_stdio(frontend):
    import io
    import os
    from PIL import Image
    import base64
    from services.controller.gemini import GeminiRouter, MODEL
    raw = io.BytesIO()
    Image.frombytes('RGB', (768, 768), os.urandom(768*768*3)).save(raw, format='PNG')
    block = {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(raw.getvalue()).decode()}
    captured = []
    def send(payload):
        captured.append(payload)
        return {'model': MODEL, 'choices': [{'message': {'content': 'seen'}}],
                'usage': {'total_tokens': 10}}
    frontend.api.gemini = GeminiRouter('host-only-test-key', send=send)
    request = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'gemini_generate', 'arguments': {
            'messages': [{'role': 'user', 'content': [block]}]}}}) + '\n'
    assert len(request) > 2*1024*1024
    output = io.StringIO()
    frontend.serve(io.StringIO(request), output)
    result = json.loads(output.getvalue())['result']
    assert not result.get('isError'), result
    assert data(result)['usage']['total_tokens'] == 10
    assert captured[0]['messages'][0]['content'][0]['image_url']['url'].endswith(block['data'])


def test_frontend_oversized_wire_request_closes_stream(frontend, monkeypatch):
    import io
    monkeypatch.setattr('services.controller.frontend.MAX_REQUEST_BYTES', 16)
    output = io.StringIO()
    frontend.serve(io.StringIO('x'*17 + '\n' + '{}\n'), output)
    assert len(output.getvalue().splitlines()) == 1
    assert json.loads(output.getvalue())['error']['message'] == 'Request exceeds wire limit'


@pytest.fixture
def demo_cache(tmp_path, monkeypatch):
    """Deterministic video decoder; real JPEG export, no OpenCV/network dependency."""
    import sys
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    cache = tmp_path/'runtime/reference-demos/website'
    cache.mkdir(parents=True)
    for task in ('make_kong', 'pour_by_language', 'swap_blocks', 'make_toast'):
        (cache/f'{task}.mp4').write_bytes(b'\x00\x00\x00\x18ftypisom')
    class Capture:
        def __init__(self, _):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            self.index += 1
            return (True, np.full((8, 12, 3), self.index*50, dtype=np.uint8)) if self.index <= 3 else (False, None)
        def release(self):
            pass
    def write(path, frame, _):
        Image.fromarray(frame).save(path, format='JPEG')
        return True
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=Capture, imwrite=write, IMWRITE_JPEG_QUALITY=1))
    import urllib.request
    def no_network(*a, **k):
        pytest.fail('Cached demonstration must not require network')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    return cache


@pytest.mark.parametrize('kind', ['none', 'terminal_state', 'completion_sequence'])
@pytest.mark.parametrize('task', ['make_kong', 'pour_by_language', 'swap_blocks', 'make_toast'])
def test_deploy_copies_skills_rubric_and_disables_other_execution(tmp_path, monkeypatch, demo_cache, kind, task):
    def fake_storage(root, *_):
        path = root/'agent-storage'
        path.mkdir()
        return path
    monkeypatch.setattr('harness.codex_cli.workspace_storage.provision', fake_storage)
    provider = tmp_path/'provider'
    provider.mkdir()
    (provider/'openrouter.config.toml').write_text('model_provider="fixture"\n[model_providers.fixture]\nname="Fixture"\nbase_url="http://127.0.0.1:1"\nwire_api="responses"\n')
    config = configuration(tmp_path)
    object.__setattr__(config, 'task', task)
    object.__setattr__(config, 'demonstration_context', kind)  # Fixture root intentionally bypasses SSD validation.
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _: provision_demonstration(config, workspace, demo_cache))
    deployed = deploy(config, provider_home=provider, codex='/usr/bin/true')
    settings = tomllib.loads((deployed['codex_home']/'config.toml').read_text())
    assert set(settings['mcp_servers']) == {'auto_research'}
    assert all(settings['features'][key] is False for key in DISABLED_FEATURES)
    assert settings['web_search'] == 'disabled' and settings['approval_policy'] == 'never'
    assert settings['sandbox_mode'] == 'danger-full-access'  # Docker is the outer boundary.
    assert settings['features']['shell_tool'] and settings['features']['unified_exec']
    # Subagents are always available: exploration workers plus the overfit auditor.
    assert settings['features']['multi_agent'] and settings['agents']['enabled']
    assert settings['agents']['max_concurrent_threads_per_session'] == config.exploration_envs + 1
    assert settings['model_auto_compact_token_limit'] == 256000
    assert settings['model_auto_compact_token_limit_scope'] == 'total'
    assert settings['model_providers']['host_relay']['base_url'] == 'http://127.0.0.1:17777'
    from services.robodojo.timeouts import task_timeouts
    allowance = task_timeouts(task)['session_seconds'] + 300
    assert settings['model_providers']['host_relay']['stream_idle_timeout_ms'] == allowance*1000
    assert settings['mcp_servers']['auto_research']['tool_timeout_sec'] == allowance
    assert not (deployed['codex_home']/'auth.json').exists()
    from services.controller.storage import AGENT_TEMPLATE
    skills = {p.name for p in (deployed['workspace']/'.agents/skills').iterdir()}
    assert skills == {'autonomous-control', 'experience-memory', 'gemini', 'in-context-action-learning',
                      'manual-robot-control', 'motion-toolkit', 'rgb-position-calibration',
                      'subagent-audit', 'wrist-camera-inspection'}
    assert (deployed['workspace']/'.agents/skills/rgb-position-calibration/CALIBRATION.md').is_file()
    for name in sorted(skills):
        relative = Path('.agents/skills')/name
        assert (deployed['workspace']/relative/'SKILL.md').is_file()
        for source in (AGENT_TEMPLATE/relative).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                copied = deployed['workspace']/source.relative_to(AGENT_TEMPLATE)
                if source.suffix == '.md':
                    from services.mcp_contract import Contract
                    from services.mcp_workspace import render
                    assert copied.read_text() == render(source.read_text(), Contract.from_config(config))
                else:
                    assert copied.read_bytes() == source.read_bytes()
    from services.controller.storage import RUNTIME_SOURCE, runtime_digest
    copied_api = deployed['workspace']/'api/runtime.py'
    assert copied_api.read_bytes() == RUNTIME_SOURCE.read_bytes()
    original_digest = runtime_digest()
    copied_api.write_text('# Agent can edit its own copy, not the trusted runner SDK.\n')
    assert runtime_digest() == original_digest
    assert not (deployed['workspace']/'services').exists()
    assert '100' in (deployed['workspace']/'TASK.md').read_text()
    assert 'https://' not in (deployed['workspace']/'TASK.md').read_text()
    task_text = (deployed['workspace']/'TASK.md').read_text()
    if task == 'pour_by_language':
        assert 'return that bottle upright' in task_text
        assert 'return both robot arms to their initial poses' in task_text
        assert 'each pour, including the final one' in task_text
        assert 'Both arms must leave their home regions' in task_text
        assert 'Tilt each bottle far enough and hold it tilted long enough' in task_text
        assert 'Aim to nearly fill the bowl without spilling' in task_text
        assert 'outside those regions at the same time' in task_text
        assert 'raise the idle arm more than 15 cm above its initial end-effector height' in task_text
    elif task == 'swap_blocks':
        assert 'more than 3 cm' in task_text
        assert 'less than 3 cm apart in 3D' in task_text
        assert 'above 95% to below 50%' in task_text
        assert 'fourth counted press fails' in task_text
        assert 'both arms to their initial end-effector poses' in task_text
        assert 'not a return-home transition' in task_text
    elif task == 'make_toast':
        assert 'one bread slice in each toaster slot' in task_text
        assert 'all four slices upright' in task_text
        assert 'lever fully down' in task_text
        assert 'ensure it stays down' in task_text
    else:
        assert '"target tile" in the rubric, which is moved from the left pile' in task_text
    assert 'Python' in (deployed['workspace']/'AGENTS.md').read_text()
    context = json.loads((config.root/'demonstration-context.json').read_text())
    assert context['kind'] == kind
    target = deployed['workspace']/'runtime/demonstrations'
    assert not list(target.rglob('*.mp4'))  # Only pixels, never video/data/actions.
    if kind == 'none':
        assert context['manifest_path'] is None and not list(target.iterdir())
    else:
        from PIL import Image
        manifest = json.loads((target/'manifest.json').read_text())
        assert manifest['contains'] == ['rgb_images']
        assert manifest['image_count'] == (1 if kind == 'terminal_state' else 3)
        assert manifest['ordered_images'] == (['terminal.jpg'] if kind == 'terminal_state' else
            [f'frames/frame_{i:06d}.jpg' for i in range(3)])
        assert 'runtime/demonstrations/manifest.json' in (deployed['workspace']/'TASK.md').read_text()
        for index, name in enumerate(manifest['ordered_images']):
            with Image.open(target/name) as image:
                assert image.size == (12, 8)
                assert abs(image.getpixel((0, 0))[0]-(150 if kind == 'terminal_state' else (index+1)*50)) <= 2
    for name in ('AGENTS.md', 'START_PROMPT.md'):
        assert (deployed['workspace']/name).read_bytes() == (AGENT_TEMPLATE/name).read_bytes()
    assert startup_prompt(deployed['workspace']) == (AGENT_TEMPLATE/'START_PROMPT.md').read_text().strip()
    assert (deployed['configuration'].stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        deploy(config, provider_home=provider, codex='/usr/bin/true')


def test_demo_context_validation_and_legacy_default(tmp_path):
    from dataclasses import asdict
    from services.controller.config import Configuration
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match='demonstration_context'):
        replace(config, demonstration_context='video_url')
    legacy = asdict(config)
    legacy['root'] = '/mnt/ssd8/demo-configuration-test'
    legacy.pop('demonstration_context')
    assert Configuration.from_dict(legacy).demonstration_context == 'none'


def test_demo_failure_is_not_silently_downgraded(tmp_path, monkeypatch):
    from services.controller import demonstrations
    def unavailable(*a, **k):
        raise ValueError('Selected demo missing')
    monkeypatch.setattr(demonstrations, 'ensure_official_demo', unavailable)
    with pytest.raises(ValueError, match='Selected demo missing'):
        demonstrations.cache_demonstration(tmp_path, 'make_kong', 'completion_sequence')


def test_status_returns_only_trusted_demo_descriptor(supervisor, frontend):
    context = {'kind':'completion_sequence', 'image_count':3, 'root':'runtime/demonstrations',
        'manifest_path':'runtime/demonstrations/manifest.json'}
    (supervisor[0].config.root/'demonstration-context.json').write_text(json.dumps(context))
    assert data(frontend.call('exploration_status', {}))['demonstration_context'] == context


def test_startup_prompt_uses_deployed_draft_and_preserves_explicit_override(tmp_path):
    prompt = tmp_path/'START_PROMPT.md'
    prompt.write_text('Custom draft for this deployment.\n')
    assert startup_prompt(tmp_path) == 'Custom draft for this deployment.'
    prompt.unlink()
    assert startup_prompt(tmp_path, 'Diagnostic only; do not start an episode.') == 'Diagnostic only; do not start an episode.'


@pytest.mark.parametrize('times_out', [False, True])
def test_launcher_uses_task_scaled_hours_and_preserves_cleanup(tmp_path, monkeypatch, times_out):
    from types import SimpleNamespace
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    config.root.mkdir(parents=True)
    provider = tmp_path/'relay.json'
    provider.write_text('{}')
    deployed = {'relay_configuration': provider, 'workspace': tmp_path,
                'codex_home': tmp_path, 'configuration': tmp_path/'config.json'}
    events = []

    class Relay:
        def __init__(self, socket, **kwargs):
            assert kwargs['client_write_timeout'] == 129_900
            self.socket = socket

        def start(self):
            (self.socket.parent/'mcp.sock').touch()

        def close(self):
            events.append('relay closed')

    class Process:
        returncode = None

        def __init__(self, command, **kwargs):
            self.name = 'agent' if command == ['fake-agent'] else 'server'

        def communicate(self, timeout):
            assert timeout == 129_600  # make_kong: 600 native steps, 36 hours.
            events.append('task-scaled wait')
            if times_out:
                raise auto_research.subprocess.TimeoutExpired('fake-agent', timeout)
            self.returncode = 0
            return '', ''

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append(self.name+' terminated')

        def wait(self, timeout):
            self.returncode = 0

    monkeypatch.setattr(auto_research, 'DockerSandbox', lambda *a: SimpleNamespace(preflight=lambda: None))
    monkeypatch.setattr(auto_research.workspace_storage, 'validate', lambda *a: None)
    monkeypatch.setattr(auto_research, 'ModelRelay', Relay)
    monkeypatch.setattr(auto_research, 'container_command', lambda *a, **k: ['fake-agent'])
    monkeypatch.setattr(auto_research.subprocess, 'Popen', Process)
    monkeypatch.setattr(auto_research.subprocess, 'run', lambda command, **k: events.append(command[:3]))
    if times_out:
        with pytest.raises(auto_research.subprocess.TimeoutExpired):
            auto_research.launch(config, deployed, prompt='test')
        assert 'agent terminated' in events
    else:
        assert auto_research.launch(config, deployed, prompt='test').returncode == 0
    assert 'task-scaled wait' in events
    assert ['docker', 'rm', '--force'] in events
    assert 'server terminated' in events and 'relay closed' in events


@pytest.mark.parametrize('options,interactive,prompt', [([], True, 'Draft startup.'),
    (['--non-interactive'], False, 'Draft startup.'), (['--exec-prompt', 'Diagnostic.'], False, 'Diagnostic.')])
def test_launcher_mode_and_durable_lifecycle(tmp_path, monkeypatch, options, interactive, prompt):
    from types import SimpleNamespace
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'START_PROMPT.md').write_text('Draft startup.\n')
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)  # Local fixture storage.
    monkeypatch.setattr(auto_research, 'deploy', lambda *a, **k: {'workspace':workspace})
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    calls = []
    def launch(*args, **kwargs):
        calls.append(kwargs)
        record = json.loads((config.root/'agent-lifecycle.json').read_text())
        assert record['status'] == 'running' and not record.get('finished_at')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(auto_research, 'launch', launch)
    monkeypatch.setattr(auto_research.sys, 'argv', ['auto-research', '--config', str(source), *options])
    with pytest.raises(SystemExit) as exit:
        auto_research.main()
    assert exit.value.code == 0
    assert calls[0]['interactive'] is interactive and calls[0]['prompt'] == prompt
    path = config.root/'agent-lifecycle.json'
    record = json.loads(path.read_text())
    assert record['status'] == 'exited' and record['finished_at'] >= record['started_at']
    with pytest.raises(FileExistsError):
        auto_research.main()
    assert json.loads(path.read_text()) == record
