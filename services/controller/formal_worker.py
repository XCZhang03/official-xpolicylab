"""One formal episode of a parallel agent submission, in its own process.

The parent session supervisor keeps the Gemini key and the single session ledger;
this worker reaches both only through its inherited socket channel. It evaluates
exactly one held-out layout and is never restarted or retried by the parent.
"""
import argparse
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import urllib.error

from .backend import NativeBackend
from .batch import import_rehearsed_bundle
from .config import Configuration
from .gemini import MAX_REQUEST_BYTES
from .sandbox import DockerSandbox
from .supervisor import Supervisor

PROJECT = Path(__file__).resolve().parents[2]
MAX_MESSAGE_BYTES = 2 * MAX_REQUEST_BYTES
ERRORS = {e.__name__: e for e in (ValueError, RuntimeError, PermissionError, TimeoutError)}


def worker_root(parent_root, index):
    return Path(parent_root)/'formal-workers'/f'episode-{index+1:03d}'


def worker_configuration(parent, index):
    config = replace(parent, formal_episodes=1, formal_seed=parent.formal_seed+index,
                     formal_eval_seed=parent.formal_collection, formal_workers=1)
    # A child of the already validated session root, so the storage rule still holds.
    object.__setattr__(config, 'root', worker_root(parent.root, index))
    return config


class Channel:
    """Worker side of the broker: one request/response at a time, never reused after failure."""
    def __init__(self, connection):
        self.connection = connection
        self.reader = connection.makefile('rb')
        self.broken = False

    def request(self, op, **arguments):
        if self.broken:
            raise RuntimeError('Session Gemini broker is unavailable')
        try:
            self.connection.sendall(json.dumps({'op': op, 'args': arguments}, allow_nan=False).encode()+b'\n')
            line = self.reader.readline(MAX_MESSAGE_BYTES+1)
            if not line or len(line) > MAX_MESSAGE_BYTES:
                raise RuntimeError('Session Gemini broker closed the channel')
            reply = json.loads(line)
        except BaseException:
            # A deadline can interrupt mid-response; later replies would be misaligned.
            self.broken = True
            raise
        if 'error' in reply:
            error = reply['error']
            if isinstance(error.get('code'), int):
                exc = urllib.error.HTTPError('', error['code'], error['message'], None, None)
            else:
                exc = ERRORS.get(error['type'], RuntimeError)(error['message'])
            raise exc
        return reply['value']


class RemoteGemini:
    def __init__(self, channel):
        self.channel = channel

    def prepare(self, arguments):
        return tuple(self.channel.request('prepare', arguments=arguments))

    def execute(self, payload, reservation):
        return tuple(self.channel.request('execute', payload=payload, reservation=reservation))


class WorkerSupervisor(Supervisor):
    """Charges the parent's session ledger instead of a private one."""
    def __init__(self, config, backend_factory, channel, **kwargs):
        self.channel = channel
        super().__init__(config, backend_factory, RemoteGemini(channel), **kwargs)

    def budget(self):
        return self.channel.request('budget')

    def charge(self, tokens, cost_reservation):
        self.channel.request('charge', tokens=tokens, cost_reservation=cost_reservation)

    def refund(self, tokens, reported, cost_reservation):
        self.channel.request('refund', tokens=tokens, reported=reported, cost_reservation=cost_reservation)


def serve_broker(supervisor, connection):
    """Parent side: expose only Gemini validation/execution and the shared ledger."""
    with connection, connection.makefile('rb') as reader:
        while line := reader.readline(MAX_MESSAGE_BYTES+1):
            try:
                if len(line) > MAX_MESSAGE_BYTES:
                    raise ValueError('Broker message exceeds limit')
                request = json.loads(line)
                op, arguments = request['op'], request['args']
                if op == 'budget':
                    value = supervisor.budget()
                elif op == 'charge':
                    value = supervisor.charge(arguments['tokens'], arguments['cost_reservation'])
                elif op == 'refund':
                    value = supervisor.refund(arguments['tokens'], arguments['reported'], arguments['cost_reservation'])
                elif op not in {'prepare', 'execute'}:
                    raise PermissionError('Unsupported broker operation')
                elif supervisor.gemini is None:
                    raise RuntimeError('Gemini is not configured')
                elif op == 'prepare':
                    value = supervisor.gemini.prepare(arguments['arguments'])
                else:
                    value = supervisor.gemini.execute(arguments['payload'], arguments['reservation'])
                reply = {'value': value}
            except Exception as exc:
                code = getattr(exc, 'code', None)
                reply = {'error': {'type': type(exc).__name__, 'message': str(exc),
                                   'code': code if type(code) is int else None}}
            try:
                connection.sendall(json.dumps(reply, allow_nan=False).encode()+b'\n')
            except OSError:
                return


def spawn(parent_root, bundle_id, index, connection):
    """Default worker factory: a detached process whose finally blocks own cleanup."""
    root = worker_root(parent_root, index)
    root.parent.mkdir(mode=0o700, exist_ok=True)
    # The worker never needs the key; the parent brokers every provider request.
    environment = {k: v for k, v in os.environ.items() if k != 'OPENROUTER_API_KEY'}
    with (root.parent/f'{root.name}.log').open('xb') as log:
        return subprocess.Popen([sys.executable, '-m', 'services.controller.formal_worker',
            '--parent-root', str(parent_root), '--bundle', bundle_id, '--index', str(index),
            '--channel-fd', str(connection.fileno()), '--parent-pid', str(os.getpid())],
            pass_fds=(connection.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, cwd=PROJECT, env=environment, start_new_session=True)


def run_worker(parent_config, bundle_id, index, channel, *, backend_factory=None,
               runner_factory=DockerSandbox):
    config = worker_configuration(parent_config, index)
    supervisor = WorkerSupervisor(config, backend_factory or (lambda: NativeBackend(config)), channel,
                                  runner_factory=runner_factory,
                                  published_root=parent_config.root/'published')
    try:
        bundle = import_rehearsed_bundle(supervisor, parent_config.root, bundle_id)
        return supervisor.run(bundle, formal=True)
    finally:
        try:
            supervisor.close()
        finally:
            # The parent's frozen bundle and the published run copy remain the evidence.
            shutil.rmtree(config.root/'bundles', ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--channel-fd', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    # Stop, with normal cleanup, if the parent supervisor dies without terminating us.
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    if os.getppid() != args.parent_pid:
        raise SystemExit('Parent supervisor exited before the worker started')
    def stop(*_):
        # Interrupt the episode once; a repeated signal must not break its cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    state = json.loads((args.parent_root/'state.json').read_text())
    parent = Configuration.from_dict(state['configuration'])
    if parent.root != args.parent_root.resolve():
        raise SystemExit('Parent configuration root mismatch')
    channel = Channel(socket.socket(fileno=args.channel_fd))
    result = run_worker(parent, args.bundle, args.index, channel)
    print(json.dumps({k: result.get(k) for k in ('reason', 'returncode', 'task_complete')}), flush=True)


if __name__ == '__main__':
    main()
