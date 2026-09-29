"""Machine-independent settings shared by the live smoke scripts."""
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def integration_root():
    """runtime/integration-tests on the repository's runtime link (bootstrap_sources.sh)."""
    return (PROJECT / 'runtime').resolve() / 'integration-tests'


def gpu_uuid(selector):
    """GPU ordinal or UUID -> UUID, from nvidia-smi."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {}
    for line in rows.splitlines():
        index, uuid = (part.strip() for part in line.split(','))
        mapping[index] = mapping[uuid] = uuid
    if str(selector) not in mapping:
        raise SystemExit(f'Unknown GPU {selector!r}; available: {sorted(k for k in mapping if k.isdigit())}')
    return mapping[str(selector)]


def add_gpu_arguments(parser):
    parser.add_argument('--sim-gpu', default='0', help='Simulator GPU ordinal or UUID')
    parser.add_argument('--research-gpu', default='1', help='Separate GPU for the agent container')
