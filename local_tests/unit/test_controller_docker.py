"""Real quarantine checks. Reuse a locally installed image; never pull one."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from services.controller.config import Limits
from services.controller.gateway import Gateway
from services.controller.sandbox import DockerSandbox as _DockerSandbox
from services.controller.storage import snapshot


def _png():
    import base64, io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (4, 3), (42, 16, 8)).save(buffer, format='PNG')
    return {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(buffer.getvalue()).decode()}


IMAGE = _png()


def official_value(step, ended=False):
    """An official-profile observation; ``transition`` carries the native episode end."""
    return {'step_id': step, 'episode_id': 'fake-episode', 'states': [0.0] * 14,
            'eef_positions': [[0.0, 0.0, 1.0]] * 2, 'eef_quaternions_wxyz': [[1.0, 0.0, 0.0, 0.0]] * 2,
            'attachments': [{'kind': 'rgb', 'camera': 'cam_high', 'content_index': 1}],
            'transition': {'steps': [{'episode_ended': ended}]}}


class OfficialBackend:
    """Fake robot whose episode ends after ``limit`` executed rows.

    Isolated runs execute bundles through the official bridge, which observes first
    and keeps stepping (holding the pose) until the episode ends.
    """
    def __init__(self, limit=1):
        self.limit, self.step, self.calls, self.rows = limit, 0, [], []

    def tools(self):
        return []

    def start(self, **kwargs):
        self.step = 0
        return {}, []

    def call(self, name, arguments):
        self.calls.append(name)
        if name == 'robodojo_step':
            rows = arguments.get('actions') or [[0.0] * 14]
            self.rows.extend(rows)
            self.step += len(rows)
        return official_value(self.step, self.step >= self.limit), [IMAGE]

    def evaluate(self):
        return {'task_complete': False}

    def close(self):
        pass


class DockerSandbox(_DockerSandbox):
    def run(self, *args, **kwargs):
        if kwargs.get('gateway') is None:
            gateway = Gateway(OfficialBackend(), audit=lambda _: None)
            kwargs.update(gateway=gateway, token=gateway.acquire())
        return super().run(*args, **kwargs)


@pytest.fixture
def docker_runtime():
    try:
        # Isolated runs execute through the official bridge, which needs numpy and Pillow:
        # use the agent image (override with ROBODOJO_TEST_IMAGE).
        name = os.environ.get("ROBODOJO_TEST_IMAGE", "robodojo-official:dev")
        query = subprocess.run(["docker", "image", "inspect", name], capture_output=True, timeout=15)
        if query.returncode:
            pytest.skip(f"Docker access and local {name} image required")
        image = json.loads(query.stdout)[0]["Id"]
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker unavailable")
    root = Path(__file__).resolve().parents[2] / "runtime"
    # Repository may be reached through a symlink; use the actual project root.
    root = Path(__file__).resolve().parents[3] / "runtime" if not root.exists() else root
    root = root.resolve()
    if not root.is_relative_to("/mnt/ssd8") or not os.access(root, os.W_OK):
        pytest.skip("Writable SSD runtime required")
    parent = root / "integration-tests"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="controller-docker-", dir=parent))
    return image, output


def test_actual_container_quarantine_and_artifact_export(docker_runtime):
    image, output = docker_runtime
    src = output / "source"
    src.mkdir()
    (src / "controller.py").write_text('''
import json, os, socket, importlib.util, subprocess, sys
from pathlib import Path
def main(ctx):
    assert os.getuid() == 65534
    assert not Path("/var/run/docker.sock").exists()
    assert not Path("/home/xiangcheng").exists()
    assert not os.environ.get("OPENROUTER_API_KEY")
    assert importlib.util.find_spec("isaacsim") is None
    assert importlib.util.find_spec("isaaclab") is None
    print("python-print-checkpoint")
    print("python-stderr-checkpoint", file=sys.stderr)
    subprocess.run([sys.executable, "-c", "print('child-print-checkpoint')"], check=True)
    try:
        Path("/workspace/api/runtime.py").write_text("modified")
        raise AssertionError("SDK was writable")
    except OSError:
        pass
    for address in [("1.1.1.1", 443), ("127.0.0.1", 80)]:
        try:
            connection = socket.create_connection(address, timeout=0.2)
        except OSError:
            continue
        connection.close()
        raise AssertionError("Network was reachable")
    Path("/workspace/output/report.json").write_text(json.dumps({"isolated": True}))
''')
    snapshot(src, output / "bundle", image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 100000))
    runner.preflight()
    result = runner.run(output / "bundle", mode="rehearsal", entrypoint="controller.py", output=output / "run")
    assert result["returncode"] == 0, (output / "run/stderr.log").read_text()
    assert json.loads((output / "run/files/report.json").read_text()) == {"isolated": True}
    assert "python-print-checkpoint" in (output / "run/stdout.log").read_text()
    assert "python-stderr-checkpoint" in (output / "run/stderr.log").read_text()
    assert "child-print-checkpoint" in (output / "run/stderr.log").read_text()
    assert not result["logs"]["stdout"]["truncated"]


def test_actual_container_mcp_closed_loop(docker_runtime):
    image, output = docker_runtime
    src = output / "source"
    src.mkdir()
    (src / "controller.py").write_text('''
import json
def main(ctx):
    def data(reply):
        return json.loads(reply["content"][0]["text"])
    for i in range(3):
        state = data(ctx.call("robodojo_observe"))
        assert state["step_id"] == i  # The bridge's own step count, as officially.
        row = list(state["states"]); row[0] = 0.1 * (i + 1)
        next_state = data(ctx.call("robodojo_step", actions=[row]))
        assert next_state["step_id"] == i+1 and next_state["transition"] == {"steps": []}
''')
    backend = OfficialBackend(limit=5)
    gateway = Gateway(backend, audit=lambda e: None)
    token = gateway.acquire()
    snapshot(src, output / "bundle", image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 100000))
    result = runner.run(output / "bundle", mode="rehearsal", entrypoint="controller.py",
                        gateway=gateway, token=token, output=output / "run")
    assert result["returncode"] == 0, (output / "run/stderr.log").read_text()
    # Three bundle rows, then the bridge holds the measured pose until the episode ends.
    assert backend.calls[0] == "robodojo_observe" and backend.step == 5
    assert [row[0] for row in backend.rows] == pytest.approx([0.1, 0.2, 0.3, 0.0, 0.0])


def test_terminal_rehearsal_cancels_bundle_and_copies_files(docker_runtime):
    image, output = docker_runtime
    src = output/'source'
    src.mkdir()
    (src/'controller.py').write_text('''
from pathlib import Path
def main(ctx):
    Path('/workspace/output/final.txt').write_text('saved before the episode ended')
    ctx.call('robodojo_step', actions=[[0.0]*14])
    Path('/workspace/output/after.txt').write_text('the official bridge cancels the bundle first')
''')
    backend = OfficialBackend(limit=1)
    gateway = Gateway(backend, audit=lambda e: None)
    token = gateway.acquire()
    snapshot(src, output/'bundle', image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 100000))
    result = runner.run(output/'bundle', mode='rehearsal', entrypoint='controller.py',
                        gateway=gateway, token=token, output=output/'run')
    assert result['returncode'] == 0, (output/'run/stderr.log').read_text()
    assert backend.calls == ['robodojo_observe', 'robodojo_step']
    assert (output/'run/files/final.txt').read_text() == 'saved before the episode ended'
    assert not (output/'run/files/after.txt').exists()


def test_actual_container_timeout_kills_descendants(docker_runtime):
    image, output = docker_runtime
    src = output / "source"
    src.mkdir()
    (src / "controller.py").write_text("import subprocess, time\ndef main(ctx):\n    subprocess.Popen(['sleep','300'])\n    time.sleep(300)\n")
    snapshot(src, output / "bundle", image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(3, 128, 1, 32, 16, 100000))
    result = runner.run(output / "bundle", mode="rehearsal", entrypoint="controller.py", output=output / "run")
    assert result["reason"] == "timeout"
    assert not os.path.ismount(output/'run/writable')


@pytest.mark.parametrize('unsafe', ['symlink', 'hardlink'])
def test_evaluation_outputs_reject_links_and_release_mount(docker_runtime, unsafe):
    image, output = docker_runtime
    src = output/'source'
    src.mkdir()
    link = ('os.symlink("/workspace/code/controller/controller.py", "/workspace/output/link.py")' if unsafe == 'symlink'
            else 'Path("/workspace/output/normal.txt").write_text("data"); os.link("/workspace/output/normal.txt", "/workspace/output/alias.txt")')
    (src/'controller.py').write_text('import os\nfrom pathlib import Path\ndef main(ctx):\n    '+link+'\n')
    snapshot(src, output/'bundle', image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 100000))
    result = runner.run(output/'bundle', mode='rehearsal', entrypoint='controller.py', output=output/'run')
    assert result['returncode'] == 0 and result['output_error']
    assert not (output/'run/files').exists()
    assert not os.path.ismount(output/'run/writable')


def test_evaluation_output_filesystem_enforces_byte_cap(docker_runtime):
    image, output = docker_runtime
    src = output/'source'
    src.mkdir()
    (src/'controller.py').write_text('''
import errno
from pathlib import Path
def main(ctx):
    try:
        Path('/workspace/output/oversized.bin').write_bytes(b'x'*(1024*1024))
    except OSError as error:
        assert error.errno == errno.ENOSPC
        Path('/workspace/output/oversized.bin').unlink()
        Path('/workspace/output/recovered.txt').write_text('cap-enforced')
    else:
        raise AssertionError('Output cap not enforced')
''')
    snapshot(src, output/'bundle', image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 65536))
    result = runner.run(output/'bundle', mode='rehearsal', entrypoint='controller.py', output=output/'run')
    assert result['returncode'] == 0 and not result['output_error']
    assert (output/'run/files/recovered.txt').read_text() == 'cap-enforced'


def test_actual_container_crash_and_truncation_logs(docker_runtime):
    image, output = docker_runtime
    src = output / "source"
    src.mkdir()
    (src / "controller.py").write_text("def main(ctx):\n    print('checkpoint-before-crash')\n    print('x' * (1024 * 1024 + 10))\n    raise RuntimeError('intentional-test-error')\n")
    snapshot(src, output / "bundle", image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 128, 1, 32, 16, 100000))
    result = runner.run(output / "bundle", mode="rehearsal", entrypoint="controller.py", output=output / "run")
    assert result["returncode"] == 1
    assert (output / "run/stdout.log").read_text().startswith("checkpoint-before-crash")
    assert result["logs"]["stdout"]["truncated"] is True
    assert result["logs"]["stdout"]["bytes_saved"] == 1024 * 1024
    assert "Traceback" in (output / "run/stderr.log").read_text()
    assert "intentional-test-error" in (output / "run/stderr.log").read_text()
    assert "checkpoint-before-crash" not in json.dumps(result)


@pytest.mark.gpu
def test_actual_container_gpu_driver_access(docker_runtime):
    """Check GPU isolation/compute without installing the future policy stack."""
    image, output = docker_runtime
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        pytest.skip('NVIDIA GPU unavailable')
    candidates = [row.split(',')[0].strip() for row in query.stdout.splitlines()
                  if int(row.split(',')[1]) > 1024 and int(row.split(',')[2]) < 20]
    if not candidates:
        pytest.skip('No idle GPU')
    src = output / 'source'
    src.mkdir()
    (src / 'controller.py').write_text('''
import ctypes, os
from pathlib import Path
def main(ctx):
    assert not os.environ.get('OPENROUTER_API_KEY')
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    count = ctypes.c_int()
    assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1
    device = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
    context = ctypes.c_void_p()
    assert cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0
    try:
        pointer = ctypes.c_uint64()
        assert cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(1024 * 1024)) == 0
        assert cuda.cuMemFree_v2(pointer) == 0
    finally:
        assert cuda.cuCtxDestroy_v2(context) == 0
    Path('/workspace/output/gpu.txt').write_text('one isolated CUDA device; allocation successful')
''')
    snapshot(src, output / 'bundle', image=image, max_bytes=100000)
    runner = DockerSandbox(image, Limits(30, 512, 1, 32, 16, 100000), gpu=candidates[0])
    runner.preflight()
    result = runner.run(output / 'bundle', mode='rehearsal', entrypoint='controller.py', output=output / 'run')
    assert result['returncode'] == 0, (output / 'run/stderr.log').read_text()
    assert 'allocation successful' in (output / 'run/files/gpu.txt').read_text()


def test_actual_container_rehearsal_and_formal(docker_runtime):
    """Real script lifecycle, deterministic fake task; no claimed robot success."""
    from services.controller.config import Configuration
    from services.controller.supervisor import Supervisor
    image, output = docker_runtime
    limits = Limits(30, 128, 1, 32, 16, 1000000)
    config = Configuration(output / 'trial', image, 'make_kong', (0, 1), 2, 0,
                           'GPU-aaaa', None, None, limits, limits, limits)
    class Backend(OfficialBackend):
        def __init__(self):
            super().__init__(limit=4)
        def evaluate(self):
            return {'task_complete': self.step >= 3}
    src = output / 'source'
    src.mkdir()
    (src / 'controller.py').write_text('''
import json
def main(ctx):
    for i in range(3):
        value = json.loads(ctx.call('robodojo_observe')['content'][0]['text'])
        assert value['step_id'] == i
        ctx.call('robodojo_step', actions=[[0.0]*14])
    print('full-script-completed')
''')
    supervisor = Supervisor(config, Backend)
    try:
        gateway, token = supervisor.start_interactive()
        for i in range(3):
            reply = gateway.handle(token, {'jsonrpc': '2.0', 'id': i, 'method': 'tools/call',
                                          'params': {'name': 'robodojo_step', 'arguments': {}}})
            assert 'error' not in reply
        assert supervisor.finish_interactive()['evaluation']['task_complete']
        bundle, _ = supervisor.register(src)
        for formal in (False, True):
            result = supervisor.run(bundle, formal=formal)
            assert result['returncode'] == 0 and result['task_complete']
            # Three bundle rows, then one held step ends the four-step fake episode.
            assert result['start_step_id'] == 0 and result['end_step_id'] == 4
            assert not result['recording_errors']
            timing = result['timing']
            assert timing['controller_processing_seconds'] > 0
            assert timing['environment_setup_seconds'] > 0
            assert timing['environment_step_seconds'] > 0
            assert timing['mcp_tools']['robodojo_step']['calls'] == 4
            assert timing['mcp_tools']['robodojo_observe']['calls'] == 1  # The bridge answers the bundle's.
            assert timing['container_overhead_seconds'] > 0
            components = [v for k, v in timing.items() if k.endswith('_seconds')
                          and k not in {'total_seconds', 'wall_limit_seconds'}]
            assert sum(components) == pytest.approx(timing['total_seconds'])
            assert supervisor.backend is None
        assert supervisor.state['exploration_started'] == 2
        with pytest.raises(RuntimeError, match='no retry'):
            supervisor.run(bundle, formal=True)
    finally:
        supervisor.close()


@pytest.mark.gpu
def test_actual_container_native_closed_loop_rehearsal(docker_runtime, monkeypatch):
    """Docker -> MCP -> Isaac official observation/joint-step loop and artifacts."""
    import sys
    from services.controller.backend import NativeBackend
    from services.controller.config import Configuration
    from services.controller.supervisor import Supervisor
    image, output = docker_runtime
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10, check=True)
        candidates = []
        for row in query.stdout.splitlines():
            gpu, free, utilization = map(str.strip, row.split(","))
            if int(free) >= 24 * 1024 and int(utilization) < 20:
                candidates.append((int(free), gpu))
        if not candidates:
            pytest.skip("No idle GPU with 24 GiB free")
        gpu = max(candidates)[1]
        ready = subprocess.run([sys.executable, "-c", "import importlib.util; assert importlib.util.find_spec('isaaclab'); assert importlib.util.find_spec('isaacsim')"], capture_output=True)
        if ready.returncode:
            pytest.skip("Native Isaac dependencies unavailable")
    except (OSError, subprocess.SubprocessError):
        pytest.skip("GPU unavailable")
    # NativeBackend configures a dedicated-process environment. Restore the test
    # process environment afterward, as pytest hosts multiple independent tests.
    previous = dict(os.environ)
    # The official bridge holds the pose until the native step limit after main returns,
    # so the run records a full episode of frames (about 0.6 MB per step).
    limits = Limits(1800, 512, 1, 32, 16, 2 * 1024**3)
    config = Configuration(output / "trial", image, "make_kong", (0, 1, 2), 3, 0,
                           gpu, None, None, limits, limits, limits)
    supervisor = Supervisor(config, lambda: NativeBackend(config))
    try:
        monkeypatch.setenv("ROBODOJO_PYTHON", sys.executable)
        gateway, original_token = supervisor.start_interactive()
        # This test verifies native transport, not task-solving ability.
        assert not supervisor.state['interactive_success']  # Soft workflow guidance only.
        original_episode = supervisor.backend.robot.sim.episode_id
        src = output / "source"
        src.mkdir()
        (src / "controller.py").write_text('''
import base64, json
from pathlib import Path
def main(ctx):
    for _ in range(2):
        response = ctx.call("robodojo_observe")
        state = json.loads(response["content"][0]["text"])
        images = [b for b in response["content"] if b["type"] == "image"]
        assert len(images) >= 3
        assert all(base64.b64decode(b["data"]).startswith(b"\\x89PNG") for b in images)
        response = ctx.call("robodojo_step", actions=[state["states"]] * 3)
        feedback = json.loads(response["content"][0]["text"])
        assert feedback["step_id"] == state["step_id"] + 3
    print('official-joint-steps-verified', feedback['step_id'])
    Path("/workspace/output/feedback.json").write_text(json.dumps(feedback))
''')
        supervisor.finish_interactive()
        ident, _ = supervisor.register(src)
        result = supervisor.run(ident, formal=False)
        assert result["returncode"] == 0, list(output.rglob("stderr.log"))[0].read_text()
        assert supervisor.backend is None
        assert supervisor.state["exploration_started"] == 2
        assert result["start_step_id"] == 0 and result["end_step_id"] >= 6
        assert not result["recording_errors"]
        view = config.root / "agent-view"
        trace = [json.loads(line) for line in (view / result["artifacts"]["mcp_trace"]).read_text().splitlines()]
        motions = [e for e in trace if e.get("event") == "mcp_result" and e.get("tool") == "robodojo_step"]
        # The bundle's two 3-row chunks, then the official adapter holds one step per
        # chunk until the native step limit ends the episode.
        counts = [json.loads((view / m["returned"]["frame_sequence"]["manifest_path"]).read_text())["frame_count"]
                  for m in motions]
        assert counts[:2] == [3, 3] and set(counts[2:]) == {1}
        assert result["end_step_id"] == 6 + len(counts) - 2
        # Step replies reference their sequence's frames instead of storing copies.
        assert all(image["path"].split("/")[-3] == "frames" for m in motions for image in m["images"])
        for motion in motions[:2]:
            sequence = motion["returned"]["frame_sequence"]
            manifest = json.loads((view / sequence["manifest_path"]).read_text())
            assert manifest["frequency_hz"] == 25 and manifest["frame_count"] == 3
            for frame in manifest["frames"]:
                assert len(frame["states"]) == 14
                for file in frame["files"]:
                    assert (view / file["path"]).read_bytes().startswith(b"\x89PNG")
        request = {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "robodojo_observe", "arguments": {}}}
        assert "error" in gateway.handle(original_token, request)
        assert type(result["evaluation"]["task_complete"]) is bool
        # A real unsuccessful rehearsal must not authorize formal submission.
        full = output / "formal-source"
        full.mkdir()
        (full / "controller.py").write_text("def main(ctx):\n    ctx.call('robodojo_observe')\n")
        bundle_id, _ = supervisor.register(full)
        rehearsal = supervisor.run(bundle_id, formal=False)
        assert rehearsal["returncode"] == 0
        assert rehearsal["task_complete"] is False
        assert not supervisor.state["formal_reserved"] and supervisor.backend is None
        with pytest.raises(RuntimeError, match="successful rehearsal"):
            supervisor.run(bundle_id, formal=True)
    finally:
        supervisor.close()
        os.environ.clear()
        os.environ.update(previous)
