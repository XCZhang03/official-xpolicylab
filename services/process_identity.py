"""PID-reuse-safe lifecycle identities for trusted harness processes."""
import ctypes
import os
from pathlib import Path
import signal


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19]}


def alive(record):
    if not record or record.get('finished_at') or type(record.get('pid')) is not int or record['pid'] <= 1:
        return False
    try:
        return identity(record['pid'])['start_ticks'] == record.get('start_ticks')
    except (OSError, IndexError):
        return False


def terminate(record):
    if not alive(record):
        raise ValueError('No matching live direct-control launcher; nothing was signaled')
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        pin, send = libc.pidfd_open, libc.pidfd_send_signal
    except AttributeError as exc:
        raise ValueError('Host lacks PID-safe stop support; use the launch terminal') from exc
    pin.argtypes, pin.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    send.argtypes, send.restype = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int
    fd = pin(record['pid'], 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Cannot pin launcher identity')
    try:
        if not alive(record):
            raise ValueError('Launcher exited; nothing was signaled')
        if send(fd, signal.SIGTERM, None, 0) < 0:
            raise OSError(ctypes.get_errno(), 'Cannot signal pinned launcher')
    finally:
        os.close(fd)
