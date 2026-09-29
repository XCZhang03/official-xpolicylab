"""Read-only native camera snapshots shared by operator views; never robot RPC."""
from pathlib import Path


def process_matches(pid, run_path):
    try:
        pid = int(pid)
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (TypeError, ValueError, OSError):
        return False
    return b'services.robodojo.server' in command and str(run_path.resolve()).encode() in command
