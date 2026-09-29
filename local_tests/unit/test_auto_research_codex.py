"""Actual Codex + stdio MCP startup against a local fake Responses endpoint."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import replace
import gzip
import json
import shutil
import shlex
import subprocess
import threading
import time
import tomllib

import pytest

from test_controller_docker import docker_runtime
from test_controller_backend import configuration
from gpu_fixtures import agent_gpu
from harness.codex_cli.auto_research import deploy, launch
from harness.codex_cli import workspace_storage


@pytest.mark.parametrize('use_gpu,demo_kind,retry_case', [
    (False, 'none', 'none'), pytest.param(True, 'none', 'none', marks=pytest.mark.gpu),
    (False, 'terminal_state', 'none'), (False, 'completion_sequence', 'none'),
    *[(False, 'none', case) for case in (
        'http_once', 'stream_once', 'stream_truncated_once', 'http_exhausted', 'stream_exhausted',
        'auth_error', 'stream_after_tool', 'pause', 'pause_short_timeout')]])
def test_real_codex_native_shell_inside_container(docker_runtime, request, monkeypatch, use_gpu, demo_kind, retry_case):
    # A regression must fail this fixture, not inherit the 24-hour run budget.
    monkeypatch.setattr('harness.codex_cli.auto_research.session_timeout', lambda config: 120)
    if retry_case == 'http_exhausted':
        # Measure Codex's own nested budget; the relay's retry window is unit-tested.
        monkeypatch.setattr('harness.codex_cli.model_relay.RETRY_WINDOW_SECONDS', 0)
    paused_case = retry_case in ('pause', 'pause_short_timeout')
    container_names, pauses = [], []
    if paused_case:
        from harness.codex_cli import auto_research
        original_command = auto_research.container_command
        def record_container(config, deployed, relay_directory, name, **kwargs):
            container_names.append(name)
            return original_command(config, deployed, relay_directory, name, **kwargs)
        monkeypatch.setattr(auto_research, 'container_command', record_container)
    image, root = docker_runtime
    installed = subprocess.run(['docker', 'image', 'inspect', 'robodojo-official:dev'],
                               capture_output=True, text=True, timeout=15)
    if installed.returncode == 0:
        image = json.loads(installed.stdout)[0]['Id']
    codex = shutil.which('codex')
    if not codex:
        pytest.skip('Codex CLI is not installed')
    if demo_kind != 'none':
        from harness.codex_cli.auto_research import PROJECT
        if not (PROJECT/'runtime/reference-demos/website/make_kong.mp4').is_file():
            pytest.skip('Cached official demo required; tests never download')
        pytest.importorskip('cv2')
    requests = []
    gpu = request.getfixturevalue('agent_gpu') if use_gpu else None
    if use_gpu and installed.returncode:
        pytest.skip('Development image required for native Codex CUDA smoke')
    smoke = '''from pathlib import Path
import os, socket, struct, zlib, importlib.util
assert Path('/.dockerenv').exists()
assert not Path('/var/run/docker.sock').exists()
assert not Path('/home/xiangcheng').exists()
assert not Path('/codex-home/auth.json').exists()
assert not os.environ.get('OPENROUTER_API_KEY')
assert importlib.util.find_spec('isaacsim') is None
from api.runtime import Context
assert callable(Context.call) and callable(Context.close)
with Context() as ctx:
    assert 'manual_success_confirmed' in ctx.call('exploration_status')['content'][0]['text']
Path('python-mcp-ready.txt').write_text('shared-endpoint-ready')
assert importlib.util.find_spec('api.runtime').origin == '/workspace/api/runtime.py'
assert 'fixture-private-secret' not in Path('/codex-home/config.toml').read_text()
try:
    socket.create_connection(('1.1.1.1', 443), timeout=.2)
except OSError:
    pass
else:
    raise AssertionError('Internet accessible')
Path('startup.txt').write_text('container-python-ready')
with Path('execution-count.txt').open('a') as stream:
    stream.write('executed\\n')
def chunk(kind, data):
    return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
png = bytes([137,80,78,71,13,10,26,10])
png += chunk(b'IHDR', struct.pack('!2I5B', 32, 32, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress((bytes([0])+bytes([255,0,0])*32)*32))
png += chunk(b'IEND', b'')
Path('smoke.png').write_bytes(png)
print('container-python-ready')
'''
    if use_gpu:
        smoke += '\nimport torch\nassert torch.cuda.device_count() == 1\nx = torch.ones(16, device="cuda")\nassert x.sum().item() == 16\nPath("gpu-ready.txt").write_text("ready")\n'
    if installed.returncode == 0:
        smoke += '\nimport subprocess\nsubprocess.run(["git", "--version"], check=True)\nsubprocess.run(["rg", "--version"], check=True)\n'
        smoke += '\nimport robodojo_toolkit as tk\nassert not importlib.util.find_spec("robodojo_toolkit").origin.startswith("/workspace/")\nassert len(tk.hold([0.0]*14, 2)) == 2\nPath("package-ready.txt").write_text("toolkit-ready")\n'
    if demo_kind != 'none':
        smoke += '''
import json
manifest_path=Path('/workspace/runtime/demonstrations/manifest.json')
manifest=json.loads(manifest_path.read_text())
assert manifest['context_kind']==DEMO_KIND
assert manifest['image_count']==len(manifest['ordered_images'])
assert manifest['image_count'] == 1 if DEMO_KIND=='terminal_state' else manifest['image_count'] > 1
for name in manifest['ordered_images']:
    assert (manifest_path.parent/name).is_file()
assert 'runtime/demonstrations/manifest.json' in Path('/workspace/TASK.md').read_text()
with Context() as ctx:
    context=json.loads(ctx.call('exploration_status')['content'][0]['text'])['demonstration_context']
    assert context['kind']==DEMO_KIND and context['image_count']==manifest['image_count']
Path('demo-ready.json').write_text(json.dumps(manifest))
print('DEMO_IMAGE_READY')
'''.replace('DEMO_KIND', repr(demo_kind))
    action = '''text({inventory: ALL_TOOLS.map(t => t.name),
process: typeof process, require: typeof require, fetch: typeof fetch});
const status = ALL_TOOLS.find(t => t.name.endsWith('__exploration_status'));
text(await tools[status.name]({}));
text(await tools.exec_command({cmd: COMMAND, yield_time_ms: 10000}));
const result = await tools.view_image({path: '/workspace/smoke.png'});
image(result.image_url);
'''.replace('COMMAND', json.dumps('python -c '+shlex.quote(smoke)))
    if demo_kind != 'none':
        demo_image = '/workspace/runtime/demonstrations/'+('terminal.jpg' if demo_kind == 'terminal_state' else 'frames/frame_000000.jpg')
        action += f"const demo = await tools.view_image({{path: {json.dumps(demo_image)}}}); image(demo.image_url);\n"
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            assert self.headers.get('Authorization') == 'Bearer fixture-private-secret'
            raw = self.rfile.read(int(self.headers['Content-Length']))
            if self.headers.get('Content-Encoding') == 'gzip':
                raw = gzip.decompress(raw)
            request = json.loads(raw)
            requests.append(request)
            message = {'id': 'msg_local', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'output_text', 'text': 'Local inventory check complete.', 'annotations': []}]}
            if len(requests) == 1:
                message = {'id': 'tool_local', 'type': 'custom_tool_call', 'call_id': 'call_local',
                           'name': 'exec', 'namespace': 'functions',
                           'input': action}
            response = {'id': 'resp_local', 'object': 'response', 'created_at': 1, 'status': 'completed',
                        'model': request['model'], 'output': [message],
                        'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}
            events = [
                {'type': 'response.created', 'response': {**response, 'status': 'in_progress', 'output': []}},
                {'type': 'response.output_item.added', 'output_index': 0, 'item': {**message, 'status': 'in_progress', 'content': []}},
                {'type': 'response.output_item.done', 'output_index': 0, 'item': message},
                {'type': 'response.completed', 'response': response}]
            fail = ((retry_case.endswith('_once') and len(requests) == 2) or
                    (retry_case.endswith('_exhausted') and len(requests) >= 2) or
                    (retry_case == 'auth_error' and len(requests) >= 2))
            if fail and (retry_case.startswith('http_') or retry_case == 'auth_error'):
                self.send_response(401 if retry_case == 'auth_error' else 503)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            failure = {'type': 'response.failed', 'response': {
                **response, 'status': 'failed', 'output': [],
                'error': {'code': 'server_error', 'message': 'Injected transient model failure'}}}
            if fail:
                events = [events[0]] if retry_case == 'stream_truncated_once' else [events[0], failure]
            elif retry_case == 'stream_after_tool' and len(requests) == 1:
                # The tool has been emitted, then the same model stream fails.
                # Recovery must keep its result, not execute the tool again.
                events[-1] = failure
            data = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if paused_case and len(requests) == 1:
                # Emit the tool while keeping its model stream open, as in the
                # failed real rehearsal. Freeze via the actual MCP isolation
                # helper, not SIGSTOP or a simulated tool sleep.
                from services.controller.frontend import ResearchFrontend
                prefix = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events[:-1]).encode()
                self.wfile.write(prefix)
                self.wfile.flush()
                deadline = time.monotonic()+20
                while not (deployed['workspace']/'startup.txt').exists():
                    if time.monotonic() >= deadline:
                        pauses.append('tool did not start')
                        return
                    time.sleep(.02)
                frontend = object.__new__(ResearchFrontend)
                frontend.agent_container = container_names[0]
                with frontend.frozen():
                    pauses.append('paused')
                    time.sleep(7)  # Exceeds the scaled legacy five-second idle timer.
                pauses.append('unpaused')
                time.sleep(.2)  # Allow expired timers to fire before the final event.
                self.wfile.write(data[len(prefix):])
                return
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = root/'provider'
        provider.mkdir()
        (provider/'openrouter.config.toml').write_text(f'''model_provider="local_fixture"
[model_providers.local_fixture]
name="Local fixture"
base_url="http://127.0.0.1:{server.server_port}"
wire_api="responses"
request_max_retries=0
stream_max_retries=0
''')
        config = configuration(root)
        config = replace(config, image=image, training_gpu=gpu, workspace_mb=256, demonstration_context=demo_kind,
                         training=replace(config.training, memory_mb=2048, pids=256, scratch_mb=128))
        secret = root/'test-key'
        secret.write_text('fixture-private-secret')
        secret.chmod(0o600)
        deployed = deploy(config, provider_home=provider, codex=codex, key_file=secret)
        settings = tomllib.loads((deployed['codex_home']/'config.toml').read_text())
        assert settings['model_auto_compact_token_limit'] == 256000
        assert settings['model_auto_compact_token_limit_scope'] == 'total'
        assert settings['model_providers']['host_relay']['request_max_retries'] == 2
        assert settings['model_providers']['host_relay']['stream_max_retries'] == 2
        assert settings['model_providers']['host_relay']['stream_idle_timeout_ms'] == 420000
        if paused_case:
            # No retry can hide a timeout in the fixed case. The control case
            # reproduces the previous idle failure at a shorter time scale.
            from harness.codex_cli.configuration import _write_config
            provider_settings = settings['model_providers']['host_relay']
            provider_settings['request_max_retries'] = provider_settings['stream_max_retries'] = 0
            if retry_case == 'pause_short_timeout':
                provider_settings['stream_idle_timeout_ms'] = 5000
            (deployed['codex_home']/'config.toml').rename(deployed['codex_home']/'default-config.toml')
            _write_config(deployed['codex_home']/'config.toml', settings)
        result = launch(config, deployed, prompt='Local infrastructure smoke. Do not start a robot episode.', capture=True)
        stdout, stderr = result.stdout, result.stderr
        (root/'codex-stdout.jsonl').write_text(stdout)
        (root/'codex-stderr.log').write_text(stderr)
        assert requests, stderr
        (root/'model-requests.json').write_text(json.dumps(requests))
        if paused_case:
            assert pauses == ['paused', 'unpaused'], (pauses, stdout, stderr)
            assert (deployed['workspace']/'execution-count.txt').read_text() == 'executed\n'
            if retry_case == 'pause_short_timeout':
                assert result.returncode != 0 and len(requests) == 1, (stdout, stderr)
                assert 'idle timeout waiting for SSE' in stdout+stderr
                return
        exhausted = retry_case.endswith('_exhausted') or retry_case == 'auth_error'
        assert (result.returncode != 0) == exhausted, (stdout, stderr)
        # Every model retry includes the completed call result exactly once.
        for retry_request in requests[1:]:
            outputs = [item for item in retry_request.get('input', [])
                       if item.get('type') == 'custom_tool_call_output' and item.get('call_id') == 'call_local']
            assert len(outputs) == 1
        assert (deployed['workspace']/'execution-count.txt').read_text() == 'executed\n'
        if retry_case in ('http_once', 'stream_once', 'stream_truncated_once'):
            assert len(requests) == 3
        elif retry_case in ('none', 'stream_after_tool', 'pause'):
            assert len(requests) == 2
        elif retry_case in ('stream_exhausted', 'auth_error'):
            # This CLI's outer stream budget also retries HTTP 401, but its
            # HTTP retry layer does not multiply those attempts.
            assert len(requests) == 4  # Tool response + initial failure + two retries.
        else:
            # HTTP and stream retry budgets can nest; the total remains bounded.
            assert len(requests) == 10  # Tool response + 3 * 3 HTTP attempts.
        events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
        status_calls = [event for event in events if event.get('type') == 'item.completed'
                        and event.get('item', {}).get('type') == 'mcp_tool_call'
                        and event['item'].get('tool') == 'exploration_status']
        assert len(status_calls) == 1  # The direct MCP call wasn't replayed either.
        results = [item for request in requests for item in request.get('input', [])
                   if item.get('type') == 'custom_tool_call_output']
        assert results, (stdout, stderr)
        serialized = json.dumps(results)
        assert 'exploration_status' in serialized and 'manual_success_confirmed' in serialized
        unescaped = serialized.replace('\\"', '"')
        for global_name in ('process', 'require', 'fetch'):
            assert f'"{global_name}":"undefined"' in unescaped.replace(' ', '')
        # Inspect the actual deferred registry returned by the hermetic executor,
        # not just the outer tools (which recent Codex sends as additional_tools).
        for banned in ('shell_exec', 'shell_poll', '__research_', 'web__run'):
            assert banned not in serialized
        assert 'spawn_agent' not in serialized
        assert 'start_episode' in serialized and 'exec_command' in serialized and 'write_stdin' in serialized
        assert 'apply_patch' in serialized and 'view_image' in serialized
        # The official robot tools plus the host-brokered model API; no planner or preview over MCP.
        assert 'robodojo_step' in serialized and 'robodojo_observe' in serialized
        assert 'gemini_generate' in serialized
        for retired in ('robodojo_free_space_move', 'robodojo_step_eef', 'robodojo_pose_math'):
            assert retired not in serialized
        assert '__operate' not in serialized
        assert (deployed['workspace']/'startup.txt').read_text() == 'container-python-ready'
        assert (deployed['workspace']/'python-mcp-ready.txt').read_text() == 'shared-endpoint-ready'
        if use_gpu:
            assert (deployed['workspace']/'gpu-ready.txt').read_text() == 'ready'
        if installed.returncode == 0:
            assert (deployed['workspace']/'package-ready.txt').read_text() == 'toolkit-ready'
        assert 'data:image/' in serialized  # Native file image reached the next model request.
        if demo_kind != 'none':
            assert 'DEMO_IMAGE_READY' in serialized
            assert serialized.count('data:image/') >= 2  # Synthetic smoke plus the real demo JPEG.
            assert (deployed['workspace']/'demo-ready.json').is_file()
        state = json.loads((config.root/'state.json').read_text())
        assert state['exploration_started'] == 0 and not state['formal_reserved']
        assert not state['interactive_success']
        (root/'codex-tool-inventory.json').write_text(serialized)
    finally:
        if (root/'private/agent-storage.json').exists():
            workspace_storage.unmount(root/'private')
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
