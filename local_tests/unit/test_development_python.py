"""Real development-container Python and direct MCP share one exploration scene.

Robot physics are deterministic test doubles here; no task-success claim.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import tempfile
import time
import uuid

import pytest

from auto_research_agent.api.runtime import Context
from services.controller.frontend import ResearchFrontend
from services.controller.supervisor import Supervisor
from test_controller_backend import configuration
from test_controller_docker import docker_runtime
from native_fakes import NativeFrames


@contextmanager
def client(path):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(15)
        connection.connect(str(path))
        with connection.makefile('r') as reader, connection.makefile('w') as writer:
            yield Context(reader, writer)


def data(result):
    return json.loads(result['content'][0]['text'])


def server(listener, config, workspace, ready):
    def stop(*_):
        raise SystemExit()
    signal.signal(signal.SIGTERM, stop)
    class Backend(NativeFrames):
        def evaluate(self):
            return {'task_complete': self.step >= 4}  # Fake task for lifecycle/path checks only.

        def call(self, name, arguments):
            if name == 'robodojo_step':
                time.sleep(.02)  # Give concurrent callers an opportunity to overlap.
            return super().call(name, arguments)
    supervisor = Supervisor(config, lambda: Backend(config.root/'native'/uuid.uuid4().hex, terminal_at=100))
    frontend = ResearchFrontend(supervisor, workspace)
    ready.send(True)
    ready.close()
    try:
        frontend.serve_socket(listener)
    finally:
        frontend.close()
        listener.close()


@pytest.fixture
def live_endpoint(docker_runtime):
    image, root = docker_runtime
    policy_image = subprocess.run(['docker', 'image', 'inspect', 'robodojo-official:dev'],
                                  capture_output=True, text=True, timeout=15)
    if policy_image.returncode == 0:
        image = json.loads(policy_image.stdout)[0]['Id']
    config = replace(configuration(root), image=image, controller_gpu=None)
    limits = replace(config.development, memory_mb=2048, pids=128)
    config = replace(config, development=limits, formal=limits)
    workspace = root/'workspace'
    workspace.mkdir()
    published = config.root/'published'
    published.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='robot-mcp-') as relay:
        path = Path(relay)/'mcp.sock'
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            listener.listen(16)
            context = multiprocessing.get_context('fork')
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=server, args=(listener, config, workspace, child))
            process.start()
            child.close()
            try:
                assert parent.poll(15) and parent.recv(), 'MCP server did not start'
                yield image, root, workspace, published, path
            finally:
                process.terminate()
                process.join(15)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                parent.close()


def test_socket_clients_are_serialized_and_bad_json_does_not_stop_server(live_endpoint):
    _, _, _, _, path = live_endpoint
    with client(path) as agent:
        assert data(agent.call('start_episode'))['step_id'] == 0
        def move(_):
            with client(path) as ctx:
                return data(ctx.call('robodojo_step', actions=[[0]*14]))['step_id']
        with ThreadPoolExecutor(2) as pool:
            assert sorted(pool.map(move, range(2))) == [1, 2]
        assert data(agent.call('robodojo_observe'))['step_id'] == 2
        with socket.socket(socket.AF_UNIX) as invalid:
            invalid.connect(str(path))
            invalid.sendall(b'not-json\n')
            assert json.loads(invalid.recv(4096))['error']['code'] == -32700
        assert data(agent.call('exploration_status'))['exploration_remaining'] == 2


def test_ordinary_python_full_script_recording_and_agent_handoff(live_endpoint):
    _, root, workspace, published, path = live_endpoint
    image = subprocess.run(['docker', 'image', 'inspect', 'robodojo-official:dev'],
                           capture_output=True, text=True, timeout=15)
    if image.returncode:
        pytest.skip('Locally provisioned development image required')
    image = json.loads(image.stdout)[0]['Id']
    repo = Path(__file__).resolve().parents[2]
    name = 'robodojo-test-development-'+uuid.uuid4().hex
    command = ['docker', 'run', '--detach', '--pull=never', '--name', name,
        '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit', '128',
        '--memory', '2g', '--tmpfs', '/tmp:size=64m', '--workdir', '/workspace',
        '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONPATH=/workspace',
        '--mount', f'type=bind,src={workspace},dst=/workspace',
        '--mount', f'type=bind,src={repo}/auto_research_agent/api,dst=/workspace/api,readonly',
        '--mount', f'type=bind,src={published},dst=/workspace/runtime/autonomous_controller,readonly',
        '--mount', f'type=bind,src={path.parent},dst=/run/agent-relay,readonly',
        '--entrypoint', '/bin/sleep', image, '300']
    # Same function is usable as controller.py:main(ctx), with no special operate RPC.
    (workspace/'controller.py').write_text('''
import json
def main(ctx):
    for _ in range(2):
        observation = json.loads(ctx.call('robodojo_observe')['content'][0]['text'])
        ctx.call('robodojo_step', actions=[observation['states']]*2)
    print('full-script-finished', flush=True)
''')
    def python(code, *, success=True):
        result = subprocess.run(['docker', 'exec', name, 'python', '-c', code],
                                capture_output=True, text=True, timeout=60)
        if success:
            assert result.returncode == 0, result.stderr
        return result
    try:
        subprocess.run(command, capture_output=True, check=True, timeout=30)
        with client(path) as agent:  # Remains open/idle while scripts connect.
            agent.call('start_episode')
            result = python('''
import robodojo_toolkit as tk
from api.runtime import Context
from controller import main
with Context() as ctx:
    main(ctx)
    meta, images = tk.parse_reply(ctx.call('robodojo_observe'))
    assert meta['step_id'] == 4 and len(images) == 3
''')
            assert 'full-script-finished' in result.stdout
            assert data(agent.call('robodojo_observe'))['step_id'] == 4
            python('''
import robodojo_toolkit as tk
from api.runtime import Context
with Context() as ctx:
    meta, _ = tk.parse_reply(ctx.call('robodojo_observe'), images=False)
    rows = tk.hold(meta['states'], 2)
    reply = ctx.call('robodojo_step', actions=[list(map(float, row)) for row in rows])
    assert tk.parse_reply(reply, images=False)[0]['step_id'] == 6
''')
            assert data(agent.call('robodojo_step', actions=[[0]*14]))['step_id'] == 7
            failed = python('''
from api.runtime import Context
with Context() as ctx:
    ctx.call('robodojo_step', actions=[[0]*14])
    raise RuntimeError('intentional-development-error')
''', success=False)
            assert failed.returncode != 0 and 'intentional-development-error' in failed.stderr
            assert data(agent.call('robodojo_observe'))['step_id'] == 8
            status = data(agent.call('exploration_status'))
            assert status['exploration_remaining'] == 2
            assert not status['formal_reserved']
            # Full development scripts never produce a qualifying rehearsal.
            state = json.loads((root/'private/state.json').read_text())
            assert not any(r['mode'] == 'rehearsal' for r in state['results'])
            trace = published/Path(status['artifacts']['mcp_trace']).relative_to('runtime/autonomous_controller')
            assert 'robodojo_step' in trace.read_text()

            # Run identical nested-project code in development, rehearsal and formal.
            # Published frame manifests must resolve inside each isolated run.
            project = workspace/'code/nested/reach'
            project.mkdir(parents=True)
            (workspace/'output').mkdir()
            (workspace/'private-development-marker').write_text('must not enter isolated runs')
            (project/'asset.txt').write_text('frozen-input')
            controller = '''
import json
from pathlib import Path

def main(ctx):
    assert Path.cwd() == Path('/workspace')
    assert Path(__file__).parent == Path('/workspace/code/nested/reach')
    assert Path(__file__).with_name('asset.txt').read_text() == 'frozen-input'
    assert Path('/workspace/api/runtime.py').is_file()
    frames = None
    for _ in range(2):
        observation = json.loads(ctx.call('robodojo_observe')['content'][0]['text'])
        feedback = json.loads(ctx.call('robodojo_step', actions=[observation['states']]*2)['content'][0]['text'])
        # Development replies carry the harness frame sequence; isolated runs go through
        # the official bridge, whose replies are exactly the official observation.
        if 'frame_sequence' in feedback:
            frames = Path('/workspace') / feedback['frame_sequence']['manifest_path']
            assert frames.is_file()
        else:
            assert feedback['transition'] == {'steps': []} and feedback['step_id'] == 2 * (_ + 1)
    Path('/workspace/output/paths.json').write_text(json.dumps({
        'project': str(Path(__file__).parent), 'frames': frames and str(frames),
        'development_visible': Path('/workspace/private-development-marker').exists(),
        'socket_visible': Path('/run/agent-relay/mcp.sock').exists(),
        'visible_runs': sorted(p.name for p in Path('/workspace/runtime/autonomous_controller/runs').iterdir()),
    }))
'''
            (project/'controller.py').write_text(controller)
            python("import sys; sys.path.insert(0, '/workspace/code/nested/reach'); "
                   "from api.runtime import Context; from controller import main; "
                   "ctx = Context(); main(ctx); ctx.close()")
            development = json.loads((workspace/'output/paths.json').read_text())
            assert development['development_visible'] and development['socket_visible']
            assert development['frames']  # The container checked the manifest exists.
            bundle = data(agent.call('register', source='code/nested/reach'))
            assert bundle['project_directory'] == '/workspace/code/nested/reach'
            for tool in ('rehearse', 'submit'):
                result = data(agent.call(tool, bundle=bundle['bundle']))
                assert result['returncode'] == 0 and result['task_complete'], result
                run = published/Path(result['artifacts']['directory']).relative_to('runtime/autonomous_controller')
                exported = json.loads((run/'exports/paths.json').read_text())
                assert exported['project'] == development['project']
                assert not exported['development_visible'] and not exported['socket_visible']
                assert exported['visible_runs'] == [run.name]
                assert exported['frames'] is None  # No harness-only fields under the official bridge.
                # The harness still publishes every frame of the isolated run.
                assert any(p.suffix == '.png' for p in (run/'frames').rglob('*'))
                assert (run/'code/controller.py').read_text() == controller
    finally:
        subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
