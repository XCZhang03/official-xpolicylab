"""Trusted fixed-size session filesystem; never expose backing images to agents."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import uuid

from services.storage_root import artifact_root


MOUNT_SCRIPT = '''import fcntl, os, stat, subprocess
# Docker snapshots /dev at container creation. LOOP_CTL_GET_FREE may allocate a
# new kernel loop device afterward, so its node must be created in this helper.
fd = os.open('/dev/loop-control', os.O_RDWR)
try:
    number = fcntl.ioctl(fd, 0x4C82, 0)
finally:
    os.close(fd)
device = '/dev/loop'+str(number)
try:
    os.mknod(device, stat.S_IFBLK | 0o600, os.makedev(7, number))
except FileExistsError:
    pass
subprocess.run(['mount', '-t', 'ext4', '-o', 'loop='+device+',nosuid,nodev',
    '/storage/agent-storage.ext4', '/storage/agent-storage'], check=True)
'''


def mounted(path):
    return os.path.ismount(path)


def helper(root, image, action, *, output_bytes=None):
    # Loop-device selection is not atomic across mount processes. Serialize this
    # project's allocation/release even when different trial roots launch together.
    lock_path = Path(__file__).resolve().parents[2]/'runtime/workspace-storage.lock'
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _helper(root, image, action, output_bytes=output_bytes)


def _helper(root, image, action, *, output_bytes=None):
    root = Path(root).resolve()
    if not root.is_relative_to(artifact_root()) or any(c in str(root) for c in ',\n'):
        raise ValueError(f'Storage helper requires a private trial directory under {artifact_root()}')
    # This short-lived operator helper runs only mount/umount, never generated code.
    # Shared propagation publishes this one submount back to the SSD parent.
    name = 'robodojo-storage-'+uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--name', name, '--pull=never', '--network=none',
        '--privileged', '--read-only', '--user=0:0', '--log-driver=none',
        '--mount', f'type=bind,src={root},dst=/storage,bind-propagation=rshared']
    if action == 'mount':
        command += ['--entrypoint', '/usr/local/bin/python', image, '-c', MOUNT_SCRIPT]
    elif action == 'unmount':
        command += ['--entrypoint', '/usr/bin/umount', image, '/storage/agent-storage']
    elif action in {'output_mount', 'output_mount_exec'}:
        if type(output_bytes) is not int or output_bytes < 1:
            raise ValueError('Invalid output cap')
        command += ['--entrypoint', '/usr/bin/mount', image, '-t', 'tmpfs',
                    '-o', f'size={output_bytes},mode=1777,nosuid,nodev'+(',noexec' if action == 'output_mount' else ''),
                    'tmpfs', '/storage/writable']
    elif action == 'output_unmount':
        command += ['--entrypoint', '/usr/bin/umount', image, '/storage/writable']
    else:
        raise ValueError('Unsupported storage operation')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError('Workspace storage helper failed: '+result.stderr.strip())
    finally:
        subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=15)


def provision(root, image, megabytes):
    root = Path(root).resolve()
    if not root.is_relative_to(artifact_root()) or any(c in str(root) for c in ',\n'):
        raise ValueError(f'Storage must be a private trial directory under {artifact_root()}')
    if type(megabytes) is not int or not 64 <= megabytes <= 1024*1024:
        raise ValueError('workspace_mb must be 64..1048576')
    backing, target = root/'agent-storage.ext4', root/'agent-storage'
    target.mkdir(mode=0o700)  # Refuse reuse/overwriting of any existing directory.
    with backing.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.truncate(megabytes*1024**2)
    subprocess.run(['mkfs.ext4', '-q', '-m', '0', '-E',
        f'root_owner={os.getuid()}:{os.getgid()}', str(backing)],
        check=True, capture_output=True, timeout=60)
    helper(root, image, 'mount')
    if not mounted(target):
        raise RuntimeError('Workspace mount did not propagate; refusing uncapped launch')
    metadata = {'workspace_mb': megabytes, 'image': image, 'backing': backing.name}
    (root/'agent-storage.json').write_text(json.dumps(metadata))
    os.chmod(target, 0o700)
    return target


def inspect(root):
    root = Path(root).resolve()
    metadata = json.loads((root/'agent-storage.json').read_text())
    target = root/'agent-storage'
    result = {**metadata, 'mounted': mounted(target)}
    if result['mounted']:
        stat = os.statvfs(target)
        result.update(capacity_bytes=stat.f_blocks*stat.f_frsize,
                      available_bytes=stat.f_bavail*stat.f_frsize)
    return result


def validate(root, megabytes, workspace, home):
    """Fail closed if either writable mount escaped the capped filesystem."""
    root = Path(root).resolve()
    status = inspect(root)
    target = root/'agent-storage'
    backing = root/'agent-storage.ext4'
    if (not status['mounted'] or status['workspace_mb'] != megabytes or
        status['capacity_bytes'] > megabytes*1024**2 or backing.is_symlink() or
        backing.stat().st_size != megabytes*1024**2):
        raise RuntimeError('Expected capped workspace is not mounted')
    for path in (workspace, home):
        path = Path(path).resolve()
        if not path.is_relative_to(target) or path.stat().st_dev != target.stat().st_dev:
            raise RuntimeError('Agent writable path is outside the capped filesystem')
    return status


def unmount(root):
    root = Path(root).resolve()
    metadata = inspect(root)
    if metadata['mounted']:
        # No lazy/forced unmount: a live agent must not lose its storage mount.
        helper(root, metadata['image'], 'unmount')


def remount(root):
    root = Path(root).resolve()
    metadata = inspect(root)
    if not metadata['mounted']:
        helper(root, metadata['image'], 'mount')
        if not mounted(root/'agent-storage'):
            raise RuntimeError('Workspace mount did not propagate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'mount', 'unmount'])
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.action == 'mount':
        remount(args.root)
    elif args.action == 'unmount':
        unmount(args.root)
    print(json.dumps(inspect(args.root), indent=2))


if __name__ == '__main__':
    main()
