"""Evaluation is idempotent only for the same trusted episode and step."""
from types import SimpleNamespace
import pytest

from services.controller.backend import NativeBackend
from services.mcp_contract import Contract


def test_evaluate_cache_is_episode_and_step_bound():
    backend = NativeBackend.__new__(NativeBackend)
    backend.contract = Contract(phase='isolated')
    backend._evaluation = None
    calls = []
    sim = SimpleNamespace(episode_id='first', step_id=0)
    def call(*_):
        identity = (sim.episode_id, sim.step_id)
        assert identity not in calls  # Legacy service rejects duplicate reviews.
        calls.append(identity)
        return {'human_evaluation_request': {'task_complete': len(calls) == 1}}, []
    backend.robot = SimpleNamespace(sim=sim, call=call)
    first = backend.evaluate()
    first['task_complete'] = False  # Caller must not mutate cached evidence.
    assert backend.evaluate()['task_complete'] is True
    assert len(calls) == 1
    sim.step_id = 1
    assert backend.evaluate()['task_complete'] is False
    sim.episode_id = 'second'
    assert backend.evaluate()['task_complete'] is False
    assert len(calls) == 3


def test_failed_evaluation_is_not_cached():
    backend = NativeBackend.__new__(NativeBackend)
    backend.contract = Contract(phase='isolated')
    backend._evaluation = None
    def fail(*_):
        raise RuntimeError('Evaluation failed')
    backend.robot = SimpleNamespace(sim=SimpleNamespace(episode_id='first', step_id=0), call=fail)
    with pytest.raises(RuntimeError):
        backend.evaluate()
    assert backend._evaluation is None


def test_failed_motion_invalidates_same_step_evaluation():
    backend = NativeBackend.__new__(NativeBackend)
    backend.contract = Contract(phase='isolated')
    backend._evaluation = (('first', 0), {'task_complete': True})
    def fail(*_):
        raise RuntimeError('Partial motion failure')
    backend.robot = SimpleNamespace(call=fail)
    with pytest.raises(RuntimeError):
        backend.call('robodojo_step', {'actions': [[0]*14]})
    assert backend._evaluation is None


def test_observation_preserves_unchanged_evaluation():
    backend = NativeBackend.__new__(NativeBackend)
    backend.contract = Contract(phase='isolated')
    expected = (('first', 0), {'task_complete': False})
    backend._evaluation = expected
    backend.robot = SimpleNamespace(call=lambda *_: ({'step_id': 0}, []))
    for name in ('robodojo_observe', 'robodojo_status'):
        backend.call(name, {})
    assert backend._evaluation is expected
