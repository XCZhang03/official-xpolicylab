"""Hosted agent API: the submitted Mooncake_Agent client, our endpoint and its workers."""
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import numpy as np
import pytest

OFFICIAL = Path(__file__).resolve().parents[2] / 'official/xpolicylab'


def load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def xpl(monkeypatch):
    template = types.ModuleType('XPolicyLab.model_template')
    template.ModelTemplate = object
    monkeypatch.setitem(sys.modules, 'XPolicyLab', types.ModuleType('XPolicyLab'))
    monkeypatch.setitem(sys.modules, 'XPolicyLab.model_template', template)
    for key in ('MOONCAKE_BASE_URL', 'MOONCAKE_API_KEY', 'AGENTBUNDLE_SERVER_TOKEN', 'OPENROUTER_API_KEY'):
        monkeypatch.delenv(key, raising=False)


def observation(instruction='Put the bread into the toaster', value=0.1):
    rng = np.random.default_rng(0)
    state = {f'{arm}_{kind}': np.full(n, value, dtype=np.float32) for arm in ('left', 'right')
             for kind, n in (('arm_joint_state', 6), ('ee_joint_state', 1))}
    vision = {cam: {'color': rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)}
              for cam in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}
    return {'state': state, 'vision': vision, 'instruction': instruction}


class FakeClient:
    """WsModelClient stand-in for an AgentBundle worker."""
    log = []

    def __init__(self, **kwargs):
        self.url = kwargs['url']
        self.fail = False

    def call(self, name, payload=None):
        FakeClient.log.append((self.url, name, payload))
        if name == 'agentbundle_hello':
            return {'ok': payload['token'] == 'worker-secret'}
        if name == 'get_action':
            if 'broken' in self.url:
                raise ConnectionError('worker died')
            return [{'left_arm_joint_state': np.arange(6, dtype=np.float32)}]
        return None

    def close(self):
        pass


@pytest.fixture
def endpoint(xpl, monkeypatch):
    api_module = load('agent_api', OFFICIAL / 'endpoint/agent_api.py', monkeypatch)
    router_module = load('task_router', OFFICIAL / 'endpoint/task_router.py', monkeypatch)
    FakeClient.log = []
    urls = {'make_toast': 'ws://w/toast', 'make_toast_random': 'ws://w/toast-random', 'hang_mugs': 'ws://w/broken'}
    workers = {task: [api_module.Worker(task, url, token='worker-secret', client_factory=FakeClient)]
               for task, url in urls.items()}
    answers = []

    def gemini(request):
        if not answers:
            raise RuntimeError('provider down')
        text = json.dumps({'task': answers.pop(0), 'reason': 'clutter'})
        return {'content': [{'type': 'text', 'text': json.dumps(
            {'finish_reason': 'stop', 'message': {'content': text}})}]}
    router = router_module.TaskRouter(list(workers), gemini=gemini)
    api = api_module.AgentAPI(api_module.WorkerPool(workers, lease_wait_s=0.1), router, log=lambda *_, **__: None)
    server = ThreadingHTTPServer(('127.0.0.1', 0), api_module.make_handler(api, ['client-key']))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('MOONCAKE_BASE_URL', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setenv('MOONCAKE_API_KEY', 'client-key')
    yield api, answers, load('mooncake_model', OFFICIAL / 'Mooncake_Agent/model.py', monkeypatch)
    server.shutdown()


def test_client_needs_an_endpoint_and_holds_no_task_logic(xpl, monkeypatch):
    model = load('mooncake_model', OFFICIAL / 'Mooncake_Agent/model.py', monkeypatch)
    with pytest.raises(ValueError, match='MOONCAKE_BASE_URL'):
        model.Model({})
    source = (OFFICIAL / 'Mooncake_Agent/model.py').read_text() + (OFFICIAL / 'Mooncake_Agent/deploy.py').read_text()
    assert 'task_name' not in source and '_random' not in source and 'bundle' not in source.lower()
    template = OFFICIAL.parents[1] / 'runtime/official-xpolicylab'
    staged = sorted(template.glob('stage-*/XPolicyLab/policy/demo_policy/deploy.py'))
    if staged:  # The deploy loop is upstream's template, byte for byte.
        assert (OFFICIAL / 'Mooncake_Agent/deploy.py').read_bytes() == staged[-1].read_bytes()


def test_images_round_trip_losslessly(endpoint):
    api_module = sys.modules['agent_api']
    model = sys.modules['mooncake_model']
    obs = observation()
    encoded = model.encode_images(obs, 'png')
    assert isinstance(encoded['vision']['cam_head']['color'], dict)
    assert isinstance(obs['vision']['cam_head']['color'], np.ndarray)  # The caller's obs is untouched.
    decoded = api_module.decode_images(encoded)
    for cam, camera in obs['vision'].items():
        assert np.array_equal(decoded['vision'][cam]['color'], camera['color'])


def test_episodes_are_routed_on_our_side(endpoint):
    api, answers, model = endpoint
    client = model.Model({})
    answers.append('make_toast_random')
    client.update_obs(observation())
    actions = client.get_action()
    assert np.array_equal(actions[0]['left_arm_joint_state'], np.arange(6))
    client.update_obs(observation(value=0.2))
    client.get_action()
    calls = [(url, name) for url, name, _ in FakeClient.log]
    assert calls == [('ws://w/toast-random', 'agentbundle_hello'), ('ws://w/toast-random', 'reset'),
                     ('ws://w/toast-random', 'update_obs'), ('ws://w/toast-random', 'get_action'),
                     ('ws://w/toast-random', 'update_obs'), ('ws://w/toast-random', 'get_action')]
    # A new episode is detected again and moves to the other task's worker.
    client.reset()
    answers.append('make_toast')
    client.update_obs(observation())
    client.get_action()
    assert [(u, n) for u, n, _ in FakeClient.log[6:8]] == [('ws://w/toast', 'agentbundle_hello'), ('ws://w/toast', 'reset')]
    assert api.pool.workers['make_toast_random'][0].owner is None  # Released for other clients.


def test_repeated_request_is_answered_once(endpoint):
    api, answers, _ = endpoint
    answers.append('make_toast')
    body = {'session_id': 's', 'episode_id': 'e', 'request_id': 'e:1', 'observation': observation()}
    first = api.act(dict(body, observation=observation()))
    again = api.act(dict(body, observation=observation()))
    assert again is first and [n for _, n, _ in FakeClient.log].count('get_action') == 1


def test_abandoned_lease_passes_to_the_next_client(endpoint):
    api, answers, _ = endpoint
    first = {'session_id': 'gone', 'episode_id': 'e1', 'request_id': 'e1:1', 'observation': observation()}
    answers.append('make_toast')
    api.act(first)
    worker = api.pool.workers['make_toast'][0]
    assert worker.owner == 'gone'
    answers.append('make_toast')
    second = {'session_id': 'next', 'episode_id': 'e2', 'request_id': 'e2:1', 'observation': observation()}
    assert set(api.act(second)['actions'][0]) != {'left_arm_joint_state'}  # Busy: held.
    worker.last_used -= api.pool.takeover_s + 1  # The first client went silent.
    answers.append('make_toast')
    third = dict(second, episode_id='e3', request_id='e3:1', observation=observation())
    assert set(api.act(third)['actions'][0]) == {'left_arm_joint_state'} and worker.owner == 'next'
    # If the silent client comes back, it re-leases once the worker is free again.
    api.pool.release(worker, 'next')
    back = dict(first, request_id='e1:2', observation=observation())
    assert set(api.act(back)['actions'][0]) == {'left_arm_joint_state'} and worker.owner == 'gone'


def test_closed_connection_frees_the_worker_at_once(endpoint):
    import time
    api, answers, model = endpoint
    gone = model.Model({})
    answers.append('make_toast')
    gone.update_obs(observation())
    gone.get_action()
    worker = api.pool.workers['make_toast'][0]
    gone.api.close()  # The evaluator's policy server exited.
    deadline = time.monotonic() + 5
    while worker.last_used != float('-inf') and time.monotonic() < deadline:
        time.sleep(0.01)
    following = model.Model({})
    answers.append('make_toast')
    following.update_obs(observation())
    assert set(following.get_action()[0]) == {'left_arm_joint_state'}
    assert worker.owner == following.session_id


def test_detection_falls_back_to_the_random_bundle(endpoint):
    api, answers, _ = endpoint
    decision = api.router.detect(observation('Make toast with the bread'))  # Gemini unavailable.
    assert decision == {'task': 'make_toast_random', 'method': 'fallback',
                        'word_candidates': ['make_toast', 'make_toast_random'],
                        'instruction': 'Make toast with the bread'}
    answers.append('not_a_task')
    assert api.router.detect(observation('Hang the mugs'))['task'] == 'hang_mugs'


def test_worker_failure_holds_instead_of_failing_the_trial(endpoint):
    api, answers, model = endpoint
    client = model.Model({})
    answers.append('hang_mugs')
    obs = observation(value=0.3)
    client.update_obs(obs)
    actions = client.get_action()
    assert set(actions[0]) == {'left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state',
                               'right_ee_joint_state'}
    assert np.allclose(actions[0]['right_arm_joint_state'], 0.3)


def test_endpoint_rejects_wrong_keys(endpoint, monkeypatch):
    _, _, model = endpoint
    monkeypatch.setenv('MOONCAKE_API_KEY', 'wrong')
    client = model.Model({})
    client.update_obs(observation())
    with pytest.raises(RuntimeError, match='401'):
        client.get_action()


def test_worker_serves_only_the_endpoint_for_its_task(xpl, monkeypatch, tmp_path):
    package = types.ModuleType('agentbundle_pkg')
    package.__path__ = [str(OFFICIAL / 'AgentBundle')]
    monkeypatch.setitem(sys.modules, 'agentbundle_pkg', package)
    for name in ('bundle_bridge', 'gemini_router', 'model'):
        load(f'agentbundle_pkg.{name}', OFFICIAL / f'AgentBundle/{name}.py', monkeypatch)
    model = sys.modules['agentbundle_pkg.model']
    checkpoint = tmp_path / 'ckpt'
    (checkpoint / 'make_toast').mkdir(parents=True)
    (checkpoint / 'make_toast' / 'controller.py').write_text('def main(ctx):\n    pass\n')
    cfg = {'task_name': 'make_toast', 'ckpt_name': str(checkpoint), 'action_type': 'joint',
           'env_cfg_type': 'arx_x5', 'warmup_planner': False}
    monkeypatch.setenv('AGENTBUNDLE_SERVER_TOKEN', 'shared-secret')
    worker = model.Model(cfg)
    for call in (lambda: worker.update_obs({}), worker.get_action, worker.reset):
        with pytest.raises(PermissionError):
            call()
    assert worker.agentbundle_hello({'task_name': 'make_kong', 'token': 'shared-secret'})['ok'] is False
    assert worker.agentbundle_hello({'task_name': 'make_toast', 'token': 'nope'})['ok'] is False
    assert worker.agentbundle_hello({'task_name': 'make_toast', 'token': 'shared-secret'})['ok'] is True
    worker.reset()
