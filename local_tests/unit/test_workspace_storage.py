"""Real kernel ENOSPC enforcement, persistence and recovery; no simulator/model."""
import json
import subprocess
import pytest

from test_controller_docker import docker_runtime
from harness.codex_cli import workspace_storage


def test_real_storage_cap_and_recovery(docker_runtime):
    image, root = docker_runtime
    target = workspace_storage.provision(root, image, 64)
    try:
        workspace_storage.validate(root, 64, target, target)
        with pytest.raises(RuntimeError, match='outside'):
            workspace_storage.validate(root, 64, root, target)
        with pytest.raises(RuntimeError, match='not mounted'):
            workspace_storage.validate(root, 128, target, target)
        script = '''import errno, os
from pathlib import Path
block = b'x'*(1024*1024)
try:
    with open('/workspace/full', 'wb', buffering=0) as stream:
        for _ in range(100):
            stream.write(block)
except OSError as exc:
    assert exc.errno == errno.ENOSPC, exc
else:
    raise AssertionError('Storage cap did not enforce ENOSPC')
Path('/workspace/full').unlink()
Path('/workspace/retained.txt').write_text('retained after unmount')
print('ENOSPC and recovery verified')
'''
        import os
        result = subprocess.run(['docker', 'run', '--rm', '--pull=never', '--network=none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user', f'{os.getuid()}:{os.getgid()}', '--mount',
            f'type=bind,src={target},dst=/workspace', image, 'python', '-c', script],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'ENOSPC' in result.stdout
        assert workspace_storage.inspect(root)['capacity_bytes'] <= 64*1024**2
        workspace_storage.unmount(root)
        assert not workspace_storage.inspect(root)['mounted']
        with pytest.raises(RuntimeError, match='not mounted'):
            workspace_storage.validate(root, 64, target, target)
        workspace_storage.remount(root)
        assert (target/'retained.txt').read_text() == 'retained after unmount'
        (root/'storage-test.json').write_text(json.dumps({'enospc': True, 'recovered': True}))
    finally:
        workspace_storage.unmount(root)


def test_concurrent_session_mounts_are_independent(docker_runtime):
    from concurrent.futures import ThreadPoolExecutor
    image, parent = docker_runtime
    roots = [parent/f'session-{i}' for i in range(3)]
    for root in roots:
        root.mkdir(mode=0o700)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(lambda root: workspace_storage.provision(root, image, 64), roots))
        assert len({path.stat().st_dev for path in targets}) == 3
        for root, target in zip(roots, targets):
            workspace_storage.validate(root, 64, target, target)
    finally:
        for root in roots:
            if (root/'agent-storage.json').exists():
                workspace_storage.unmount(root)
