"""One durable episode ledger, independent workers, serialized control per slot.

The ledger counts starts (including initial starts and failed startups), not tool
calls. A reset reserves a new seed before replacing the old episode. Exhaustion
never closes another slot's live episode. No agent can select seeds or create slots.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from services.controller.storage import persist as _write_private_json


def slot_count(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError('exploration_envs must be an integer in [1, 8]')
    return value




class Worker:
    """Private socket transport. Worker executes native code on its main thread."""
    def __init__(self, config):
        root = Path(config['root'])
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'worker.json'
        _write_private_json(path, config)
        parent, child = socket.socketpair()
        parent.settimeout(config['episode_seconds'] + 300)
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('ROBODOJO_', 'OPENROUTER_', 'OPENAI_'))
                       and not k.endswith(('API_KEY', 'ACCESS_TOKEN'))}
        project = Path(__file__).resolve().parents[2]
        self.log = (root / 'worker.log').open('ab')
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'services.exploration.worker', '--config', str(path), '--fd', str(child.fileno())],
                cwd=project, env=environment, pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
        except BaseException:
            parent.close()
            self.log.close()
            raise
        finally:
            child.close()
        self.socket = parent
        self.stream = parent.makefile('rwb')
        self.sequence = 0
        self.root = root

    def interrupt(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self.process.poll() is None:
            self.process.terminate()

    def request(self, name, arguments):
        self.sequence += 1
        self.stream.write(json.dumps({'id': self.sequence, 'name': name, 'arguments': arguments}).encode() + b'\n')
        self.stream.flush()
        line = self.stream.readline(128 * 1024 * 1024 + 1)
        if not line or len(line) > 128 * 1024 * 1024:
            raise RuntimeError('Exploration worker exited or exceeded its reply limit')
        response = json.loads(line)
        if response.get('id') != self.sequence:
            raise RuntimeError('Exploration worker response identity mismatch')
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    def close(self):
        try:
            self.stream.close()
        except OSError:
            pass
        finally:
            self.socket.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        finally:
            self.log.close()
            # A SIGKILLed worker cannot run its finally block. Clean up only its
            # exact native output tree, using pinned process identities.
            from services.process_identity import identity, terminate
            prefix = str(self.root.resolve()).encode() + b'/'
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    record = identity(int(entry.name))
                    args = (entry / 'cmdline').read_bytes().split(b'\0')
                    if (any(module in args for module in
                            (b'services.robodojo.server',))
                            and any(arg.startswith(prefix) for arg in args)):
                        terminate(record)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass


class ExplorationPool:
    def __init__(self, root, *, count, seeds, worker_config, worker_factory=Worker):
        self.root = Path(root)
        self.count = slot_count(count)
        self.seeds = tuple(seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('Exploration seeds must be distinct nonnegative integers')
        self.root.mkdir(parents=True, exist_ok=True)
        self._owner = (self.root / 'owner.lock').open('a')
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner.close()
            raise RuntimeError('Exploration pool already has an owner')
        self._lock = threading.RLock()
        self._slots = [threading.RLock() for _ in range(count)]
        self.workers = {}
        self.worker_config, self.worker_factory = worker_config, worker_factory
        self.path = self.root / 'state.json'
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state['seeds'] != list(seeds) or self.state['env_count'] != count:
                    raise ValueError('Cannot change an existing exploration pool')
                # Never silently recreate a live scene or refund its seed.
                if any(row['status'] in {'starting', 'active', 'closing'} for row in self.state['episodes']):
                    raise RuntimeError('Interrupted exploration pool requires operator recovery; no automatic retry')
            else:
                self.state = {'schema': 1, 'env_count': count, 'seeds': list(seeds), 'episodes': [], 'sealed': False}
                self._save()
        except BaseException:
            self._owner.close()
            raise

    def _save(self):
        _write_private_json(self.path, self.state)

    def validate_slot(self, env_id):
        if type(env_id) is not int or not 0 <= env_id < self.count:
            raise ValueError(f'env_id must be an integer in [0, {self.count - 1}]')
        return env_id

    def _reserve(self, env_id, kind='interactive'):
        with self._lock:
            if self.state['sealed']:
                raise RuntimeError('Exploration is closed')
            index = len(self.state['episodes'])
            if index >= len(self.seeds):
                raise RuntimeError('Shared exploration episode budget exhausted; existing episodes may continue')
            row = {'episode': index + 1, 'env_id': env_id, 'seed': self.seeds[index],
                   'kind': kind, 'status': 'starting', 'started_at': time.time()}
            self.state['episodes'].append(row)
            self._save()
            return row

    def reserve_rehearsal(self):
        return self._reserve(None, 'rehearsal')

    def finish_reservation(self, row, *, status='closed', **details):
        with self._lock:
            row.update(status=status, finished_at=time.time(), **details)
            self._save()

    def status(self):
        with self._lock:
            rows = self.state['episodes']
            slots = []
            for env_id in range(self.count):
                row = next((r for r in reversed(rows) if r['env_id'] == env_id), {})
                slots.append({'env_id': env_id, **{k: row[k] for k in
                    ('episode', 'episode_id', 'status', 'task_complete', 'artifacts', 'error') if k in row}})
                if 'artifacts' in slots[-1]:
                    slots[-1]['artifacts'] = {k: v for k, v in row['artifacts'].items()
                                              if k in {'directory', 'mcp_trace'}}
                if 'error' in slots[-1]:
                    from services.controller.errors import public_failure
                    slots[-1]['error'] = public_failure(RuntimeError(slots[-1]['error']))['reason']
            remaining = len(self.seeds) - len(rows)
            active = sum(r['status'] in {'starting', 'active', 'closing'} for r in rows)
            return {'env_count': self.count, 'exploration_episodes_started': len(rows),
                    'exploration_episodes_remaining': remaining, 'exploration_episode_budget': len(self.seeds),
                    'active_episode_count': active, 'exploration_complete': remaining == 0 and active == 0,
                    'sealed': self.state['sealed'], 'environments': slots}

    def start(self, env_id):
        self.validate_slot(env_id)
        with self._slots[env_id]:
            row = self._reserve(env_id)  # Reserve BEFORE closing an existing episode.
            try:
                self._finish(env_id)
                config = {**self.worker_config, 'seed': row['seed'], 'env_id': env_id,
                          'episode': row['episode'], 'root': str(self.root / f'episode-{row["episode"]:03d}')}
                with self._lock:
                    if self.state['sealed']:
                        raise RuntimeError('Exploration shut down during startup')
                    worker = self.worker_factory(config)
                    self.workers[env_id] = (worker, row)
                packet = worker.request('start', {})
                self._update(row, packet)
                return self._reply(env_id, packet)
            except BaseException as exc:
                pair = self.workers.pop(env_id, None)
                if pair:
                    pair[0].close()
                self.finish_reservation(row, status='error', error=f'{type(exc).__name__}: {exc}')
                raise

    def _update(self, row, packet):
        with self._lock:
            row.update(packet['state'])
            self._save()

    def _reply(self, env_id, packet):
        result = deepcopy(packet['reply'])
        content = result.setdefault('content', [])
        # Do not stringify image blocks or turn them into structuredContent.
        content.append({'type': 'text', 'text': json.dumps({'env_id': env_id,
            'episode': packet['state'].get('episode'), 'artifacts': packet['state'].get('artifacts'),
            'exploration': {k: v for k, v in self.status().items() if k != 'environments'}})})
        return result

    def call(self, env_id, name, arguments):
        self.validate_slot(env_id)
        with self._slots[env_id]:
            if env_id not in self.workers:
                raise RuntimeError('No active episode in this env_id; start an episode first')
            worker, row = self.workers[env_id]
            try:
                packet = worker.request(name, arguments)
                self._update(row, packet)
                if packet['state']['status'] == 'closed':
                    self.workers.pop(env_id)
                    worker.close()
                return self._reply(env_id, packet)
            except BaseException as exc:
                self.workers.pop(env_id, None)
                worker.close()
                self.finish_reservation(row, status='error', error=f'{type(exc).__name__}: {exc}')
                raise

    def _finish(self, env_id, arguments=None):
        pair = self.workers.pop(env_id, None)
        if pair is None:
            return {'content': [{'type': 'text', 'text': json.dumps({'env_id': env_id, 'active': False})}]}
        worker, row = pair
        try:
            packet = worker.request('finish', arguments or {})
            self._update(row, packet)
            return self._reply(env_id, packet)
        except BaseException as exc:
            self.finish_reservation(row, status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            worker.close()

    def finish(self, env_id, arguments=None):
        self.validate_slot(env_id)
        with self._slots[env_id]:
            return self._finish(env_id, arguments)

    def finish_all(self):
        errors = []
        for env_id in range(self.count):
            try:
                self.finish(env_id)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))

    def seal(self):
        with self._lock:
            self.state['sealed'] = True
            self._save()

    def close(self):
        try:
            self.finish_all()
        finally:
            self._owner.close()

    def interrupt(self):
        with self._lock:
            self.seal()
            for worker, _ in list(self.workers.values()):
                worker.interrupt()


class LifecycleGate:
    """Reject overlapping global lifecycle changes instead of racing slot calls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.exclusive = False

    @contextmanager
    def enter(self, *, exclusive=False):
        with self.lock:
            if self.exclusive or (exclusive and self.active):
                raise RuntimeError('Another operation is in progress; wait before changing session lifecycle')
            self.active += 1
            self.exclusive = exclusive
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
                self.exclusive = False
