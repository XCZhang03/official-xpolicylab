"""Budget races, slot isolation and native-worker contracts without GPU/API calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time

import pytest

from services.exploration.pool import ExplorationPool, LifecycleGate
from services.exploration.worker import EpisodeWorker
from test_controller_backend import Backend, configuration, supervisor


class FakeWorker:
    barrier = None

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.step = 0
        self.inside = False

    def request(self, name, arguments):
        assert not self.inside, 'same slot raced'
        self.inside = True
        try:
            if name == 'fail':
                raise RuntimeError('native crash')
            if name == 'step':
                if self.barrier is not None:
                    self.barrier.wait(timeout=2)
                time.sleep(.01)
                self.step += 1
            state = {'episode': self.config['episode'], 'episode_id': f"episode-{self.config['episode']}",
                     'status': 'closed' if name == 'finish' else 'active'}
            return {'state': state, 'reply': {'content': [
                {'type': 'text', 'text': json.dumps({'step_id': self.step})},
                {'type': 'image', 'mimeType': 'image/png', 'data': 'image-bytes'}]}}
        finally:
            self.inside = False

    def close(self):
        self.closed = True


@pytest.fixture
def pool(tmp_path):
    p = ExplorationPool(tmp_path/'pool', count=2, seeds=[10, 11, 12], worker_config={}, worker_factory=FakeWorker)
    yield p
    p.close()


def test_global_budget_reset_and_images(pool):
    first = pool.start(0)
    assert first['content'][1]['type'] == 'image'
    assert 'structuredContent' not in first
    assert 'seed' not in json.dumps(first)
    pool.start(1)
    other = pool.workers[1][0]
    pool.start(0)
    assert not other.closed
    assert [r['seed'] for r in pool.state['episodes']] == [10, 11, 12]
    with pytest.raises(RuntimeError, match='budget exhausted'):
        pool.start(1)
    assert not other.closed
    pool.call(1, 'step', {})
    assert not pool.status()['exploration_complete']
    pool.finish(0)
    assert not pool.status()['exploration_complete']
    pool.finish(1)
    assert pool.status()['exploration_complete']


def test_parallel_slots_really_overlap_and_same_slot_serializes(pool, monkeypatch):
    pool.start(0)
    pool.start(1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        monkeypatch.setattr(FakeWorker, 'barrier', threading.Barrier(2))
        list(executor.map(lambda i: pool.call(i, 'step', {}), [0, 1]))
        monkeypatch.setattr(FakeWorker, 'barrier', None)
        list(executor.map(lambda _: pool.call(0, 'step', {}), range(4)))
    assert pool.workers[0][0].step == 5
    assert pool.workers[1][0].step == 1


def test_budget_race_never_oversubscribes(pool):
    def start(i):
        try:
            pool.start(i % 2)
            return True
        except RuntimeError as exc:
            assert 'budget exhausted' in str(exc)
            return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(start, range(12))) == 3
    assert pool.status()['exploration_episodes_remaining'] == 0


def test_rehearsal_consumes_the_same_budget(pool):
    pool.start(1)
    row = pool.reserve_rehearsal()
    assert row['seed'] == 11
    assert row['kind'] == 'rehearsal'
    pool.finish_reservation(row)
    pool.start(0)
    assert pool.status()['exploration_episodes_started'] == 3


def test_failed_worker_is_not_refunded_or_retried(pool):
    pool.start(0)
    pool.start(1)
    with pytest.raises(RuntimeError, match='native crash'):
        pool.call(0, 'fail', {})
    assert pool.state['episodes'][0]['status'] == 'error'
    assert pool.status()['exploration_episodes_remaining'] == 1
    pool.call(1, 'step', {})


@pytest.mark.parametrize('env_id', [None, True, -1, 2, '0', 0.0])
def test_invalid_slot_has_no_effect(pool, env_id):
    with pytest.raises(ValueError):
        pool.start(env_id)
    assert not pool.state['episodes']


def test_crash_recovery_fails_closed_and_configuration_is_bound(tmp_path):
    root = tmp_path/'pool'
    p = ExplorationPool(root, count=2, seeds=[10, 11], worker_config={}, worker_factory=FakeWorker)
    p._reserve(0)
    p._owner.close()  # Simulate a lost supervisor after a durable reservation.
    with pytest.raises(RuntimeError, match='recovery'):
        ExplorationPool(root, count=2, seeds=[10, 11], worker_config={})
    with pytest.raises(ValueError, match='Cannot change'):
        ExplorationPool(root, count=2, seeds=[10, 12], worker_config={})


def test_duplicate_owner_and_sealed_pool(pool):
    with pytest.raises(RuntimeError, match='owner'):
        ExplorationPool(pool.root, count=2, seeds=pool.seeds, worker_config={})
    pool.seal()
    with pytest.raises(RuntimeError, match='closed'):
        pool.start(0)


def test_global_lifecycle_excludes_inflight_operations():
    gate = LifecycleGate()
    with gate.enter():
        with gate.enter():
            with pytest.raises(RuntimeError, match='in progress'):
                with gate.enter(exclusive=True):
                    pytest.fail('entered lifecycle during a motion')
    with gate.enter(exclusive=True):
        with pytest.raises(RuntimeError):
            with gate.enter():
                pytest.fail('entered motion during lifecycle')


def test_schemas_require_env_id_without_mutating_single_env_contract():
    from services.robodojo.mcp_server import RoboDojoMCP
    original = RoboDojoMCP.tool_definitions()
    snapshot = deepcopy(original)
    from services.mcp_contract import Contract
    tools = Contract('auto-research', observation_profile='official', environments=2).select(original)
    step = next(t for t in tools if t['name'] == 'robodojo_step')
    assert 'env_id' in step['inputSchema']['required']
    assert step['inputSchema']['properties']['env_id']['maximum'] == 1
    assert original == snapshot


def test_episode_worker_preserves_images_and_publishes_trace(tmp_path):
    backend = Backend()
    backend.frame_workspace = tmp_path
    config = {'root': str(tmp_path/'private'), 'task': 'make_kong', 'sim_gpu': 'GPU-aaaa',
              'eval_seed': 0, 'seed': 7, 'episode': 1, 'env_id': 0, 'episode_seconds': 60,
              'published_root': str(tmp_path/'published'), 'artifact_bytes': 1024**2}
    worker = EpisodeWorker(config, backend_factory=lambda _: backend)
    try:
        reply = worker.call('start', {})['reply']
        assert reply['content'][1] == backend.image
        assert backend.starts == [(False, 7)]
        backend.success = True
        result = worker.call('evaluate', {})
        assert result['state']['task_complete']
        assert 'native_score' not in json.dumps(result)
        with pytest.raises(PermissionError):
            worker.call('start_episode', {})
        with pytest.raises(RuntimeError, match='cannot reset'):
            worker.call('start', {})
    finally:
        worker.finish()
    assert backend.closed
    assert list((tmp_path/'published').glob('runs/*/mcp.jsonl'))


def test_research_frontend_routes_slots_and_shares_supervisor_budget(supervisor, tmp_path):
    from services.exploration.research import ParallelResearchFrontend
    s, _ = supervisor
    object.__setattr__(s.config, 'exploration_envs', 2)
    frontend = ParallelResearchFrontend(s, tmp_path/'workspace',
        pool_factory=lambda *a, **kw: ExplorationPool(*a, **kw, worker_factory=FakeWorker))
    try:
        with pytest.raises(ValueError, match='explicit env_id'):
            frontend.call('start_episode', {})
        frontend.call('start_episode', {'env_id': 0})
        frontend.call('start_episode', {'env_id': 1})
        assert s.state['exploration_started'] == 2
        assert frontend.status()['exploration_remaining'] == 1
        assert frontend.status()['exploration']['active_episode_count'] == 2
        with pytest.raises(ValueError):
            frontend.call('start_episode', {'env_id': 0, 'seed': 7})
        assert s.state['exploration_started'] == 2
    finally:
        frontend.pool.close()
        frontend.api.revoke()


def test_rehearsal_uses_next_global_seed_after_parallel_exploration(supervisor, tmp_path):
    from services.exploration.research import ParallelResearchFrontend
    s, backend = supervisor
    object.__setattr__(s.config, 'exploration_envs', 2)
    frontend = ParallelResearchFrontend(s, tmp_path/'workspace',
        pool_factory=lambda *a, **kw: ExplorationPool(*a, **kw, worker_factory=FakeWorker))
    try:
        frontend.call('start_episode', {'env_id': 0})
        frontend.call('start_episode', {'env_id': 1})
        project = frontend.workspace/'code/test'
        project.mkdir(parents=True)
        (project/'controller.py').write_text('def main(ctx): pass')
        registered = json.loads(frontend.call('register', {'source': 'code/test'})['content'][0]['text'])
        with pytest.raises(RuntimeError, match='successful rehearsal'):
            frontend.call('submit', {'bundle': registered['bundle']})
        assert frontend.pool.status()['active_episode_count'] == 2
        frontend.call('rehearse', {'bundle': registered['bundle']})
        assert backend.starts[-1] == (False, 3)
        assert frontend.pool.status()['exploration_episodes_started'] == 3
        assert frontend.pool.status()['exploration_complete']
    finally:
        frontend.pool.close()
        frontend.api.revoke()


def test_pipelined_socket_calls_run_in_parallel_and_ids_images_survive(tmp_path):
    import socket
    from services.exploration.transport import serve_socket
    barrier = threading.Barrier(2)

    class Service:
        def parallel_request(self, request):
            return request['method'] == 'step'

        def handle(self, request):
            if request['method'] == 'stop':
                raise SystemExit()
            barrier.wait(timeout=3)
            return {'jsonrpc': '2.0', 'id': request['id'],
                    'result': {'content': [{'type': 'image', 'data': 'bytes', 'mimeType': 'image/png'}]}}

        def interrupt(self):
            pass

    errors = []
    listener = socket.socket(socket.AF_UNIX)
    path = str(tmp_path/'mcp.sock')
    listener.bind(path)
    listener.listen()
    def run():
        try:
            serve_socket(Service(), listener, slots=2)
        except SystemExit:
            pass
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(5)
            client.connect(path)
            client.sendall(b'{"jsonrpc":"2.0","id":1,"method":"step"}\n{"jsonrpc":"2.0","id":2,"method":"step"}\n')
            with client.makefile('rb') as stream:
                replies = [json.loads(stream.readline()), json.loads(stream.readline())]
            assert {r['id'] for r in replies} == {1, 2}
            assert all(r['result']['content'][0]['type'] == 'image' for r in replies)
            client.sendall(b'{"jsonrpc":"2.0","id":3,"method":"stop"}\n')
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
    finally:
        listener.close()


def test_startup_failure_reserves_exactly_once(tmp_path):
    def fail(_):
        raise RuntimeError('startup failed')
    pool = ExplorationPool(tmp_path/'pool', count=2, seeds=[0, 1], worker_config={}, worker_factory=fail)
    try:
        with pytest.raises(RuntimeError, match='startup failed'):
            pool.start(0)
        assert pool.status()['exploration_episodes_started'] == 1
        assert pool.status()['active_episode_count'] == 0
        assert pool.state['episodes'][0]['status'] == 'error'
    finally:
        pool.close()


