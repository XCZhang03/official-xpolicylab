"""No-op validation must not consume an episode; uncertain motion must fail closed."""
import json

import pytest

from services.controller.backend import NativeBackend
from services.controller.gateway import Gateway
from services.controller.errors import public_failure
from services.robodojo.action_validation import ActionValidationError, validate_motion_request


class Backend:
    validate = staticmethod(NativeBackend.validate)

    def __init__(self):
        self.calls = []
        self.error = None

    def call(self, name, arguments):
        self.calls.append(name)
        if self.error:
            raise self.error
        return {'step_id': len(self.calls)}, []


def gateway(**kwargs):
    backend = Backend()
    g = Gateway(backend, audit=lambda _: None, **kwargs)
    return g, g.acquire(), backend


def call(g, token, name, **args):
    return g.handle(token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                           'params': {'name': name, 'arguments': args}})


@pytest.mark.parametrize('name,field,row', [('robodojo_step', 'actions', [0]*14)])
def test_fifty_actions_allowed_and_fifty_one_rejected_without_poisoning(name, field, row):
    g, token, backend = gateway()
    failure = call(g, token, name, **{field: [row]*51})['error']
    assert '1..50' in failure['message'] and '51' in failure['message']
    assert failure['data']['no_action_executed'] and not failure['data']['control_uncertain']
    assert backend.calls == [] and not g.terminal
    assert 'result' in call(g, token, name, **{field: [row]*12})
    assert 'result' in call(g, token, name, **{field: [row]*50})
    assert len(backend.calls) == 2


def test_invalid_last_joint_row_rejects_entire_batch():
    g, token, backend = gateway()
    failure = call(g, token, 'robodojo_step', actions=[[0]*14]*49+[[0]*6+[2]+[0]*7])['error']
    assert 'Gripper openings' in failure['message'] and 'no action executed' in failure['message']
    assert backend.calls == [] and not g.terminal
    assert 'result' in call(g, token, 'robodojo_observe')


@pytest.mark.parametrize('stage', ['execution', 'publication', 'recording'])
def test_post_dispatch_failure_remains_uncertain_and_observation_is_available(stage):
    g, token, backend = gateway()
    def fail(_):
        raise RuntimeError('recording quota exhausted')
    if stage == 'execution':
        backend.error = TimeoutError('simulator response timed out')
    elif stage == 'publication':
        g.publish_sequence = fail
    else:
        g.audit = lambda event: fail(event) if event['event'] == 'mcp_result' else None
    failure = call(g, token, 'robodojo_step', actions=[[0]*14])['error']['data']
    assert failure['control_uncertain'] and not failure['no_action_executed']
    assert g.terminal and len(backend.calls) == 1
    failure = call(g, token, 'robodojo_step', actions=[[0]*14])['error']
    assert 'uncertain' in failure['message'] and len(backend.calls) == 1
    backend.error = None
    g.publish_sequence = lambda _: None
    g.audit = lambda _: None
    assert 'result' in call(g, token, 'robodojo_observe')


def test_audit_failure_before_execution_returns_reason_without_poisoning():
    g, token, backend = gateway()
    def fail(_):
        raise RuntimeError('audit quota exhausted')
    g.audit = fail
    error = call(g, token, 'robodojo_step', actions=[[0]*14])['error']['data']
    assert error['audit_recording_failed'] and error['no_action_executed']
    assert error['reason'] == 'audit quota exhausted' and backend.calls == []


def test_provider_errors_and_host_paths_do_not_leak():
    error = RuntimeError('API_KEY=private-key Bearer private-token /private/key https://private.example sk-secret')
    for stage in ('provider_request', 'validation'):
        public = json.dumps(public_failure(error, stage=stage))
        for secret in ('private-key', 'private-token', '/private/key', 'private.example', 'sk-secret'):
            assert secret not in public


@pytest.mark.parametrize('actions', [[[0]*13], [[float('nan')]*14], [[0]*6+[2]+[0]*7]])
def test_bad_joint_targets_are_explicit_preflight_failures(actions):
    with pytest.raises(ActionValidationError, match='no action executed'):
        validate_motion_request('robodojo_step', {'actions': actions})
