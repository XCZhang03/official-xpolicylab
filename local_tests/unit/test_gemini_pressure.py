"""Fault injection for gemini_generate: provider failures never touch robot control or the ledger's bounds.

No paid API: every provider response comes from a fake ``send``.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import random
import threading
import urllib.error

import pytest

from services.controller import formal_worker
from services.controller.billing import SESSION_LIMIT, nanodollars
from services.controller.gemini import GeminiRouter, MODEL
from services.controller.supervisor import Supervisor
from test_controller_backend import Backend, Runner, configuration, packet, source, value
from test_gemini_budget import frontend, response
import test_parallel_submit  # noqa: F401  (fixture_roots: worker roots under tmp_path)
from test_parallel_submit import fixture_roots, inline, rehearsed  # noqa: F401


def ok(payload):
    return response(.001)


def http_error(code):
    def send(payload):
        raise urllib.error.HTTPError('https://openrouter.ai/secret', code, 'provider body with sk-secret', None, None)
    return send


def raising(exc):
    def send(payload):
        raise exc
    return send


FAULTS = {
    'timeout': raising(TimeoutError('slow provider sk-secret-key')),
    'http_429': http_error(429),
    'http_500': http_error(500),
    'connection_reset': raising(ConnectionResetError('reset by peer')),
    'malformed_json': raising(json.JSONDecodeError('Expecting value', 'garbage', 0)),
    'not_a_dict': lambda payload: ['not', 'a', 'response'],
    'wrong_model': lambda payload: {**response(), 'model': 'openai/other'},
    'missing_usage': lambda payload: {k: v for k, v in response().items() if k != 'usage'},
    'excessive_usage': lambda payload: {**response(), 'usage': {'total_tokens': 10**9, 'cost': .001}},
    'no_choices': lambda payload: {**response(), 'choices': []},
}


def gemini_packet():
    return packet('gemini_generate', messages=[{'role': 'user', 'content': 'Hello'}], max_tokens=128)


@pytest.mark.parametrize('fault', sorted(FAULTS))
def test_provider_faults_leave_robot_control_and_steps_untouched(tmp_path, fault):
    s = Supervisor(configuration(tmp_path), Backend, GeminiRouter('host-key', send=FAULTS[fault]))
    try:
        gateway, token = s.start_interactive()
        before = s.backend.step
        reply = gateway.handle(token, gemini_packet())
        assert 'error' in reply, fault
        data = reply['error']['data']
        # A model API failure never moves the robot, never makes control uncertain.
        assert data['no_action_executed'] is True and data['control_uncertain'] is False
        assert not gateway.terminal and not gateway.control_uncertain
        assert s.backend.step == before
        serialized = json.dumps(reply)
        assert 'sk-secret' not in serialized and 'provider body' not in serialized
        spend = s.state['gemini_spend']
        # The reservation is kept (never refunded on an unverifiable outcome) and within the cap.
        assert 0 < spend['reserved'] <= SESSION_LIMIT and spend['reported'] == 0
        assert data['budget']['session_cost_reserved_usd'] > 0
        # The episode continues normally.
        assert 'result' in gateway.handle(token, packet('robodojo_step'))
        assert s.backend.step == before + 1
    finally:
        s.close()


def test_cost_above_reservation_closes_the_api_but_not_the_robot(tmp_path):
    s = Supervisor(configuration(tmp_path), Backend,
                   GeminiRouter('host-key', send=lambda payload: response(cost=5.0)))
    try:
        gateway, token = s.start_interactive()
        first = gateway.handle(token, gemini_packet())
        assert 'error' in first and 'exceeded reservation' in first['error']['message']
        assert s.state['gemini_spend']['blocked'] is True
        second = gateway.handle(token, gemini_packet())
        assert 'error' in second and 'budget exhausted' in second['error']['message']
        assert s.state['usage']['development']['calls'] == 1  # The blocked call reached no provider.
        assert 'result' in gateway.handle(token, packet('robodojo_step')) and not gateway.terminal
    finally:
        s.close()


def near_cap(s, remaining_usd):
    """Spend the session ledger down to ``remaining_usd`` through the normal ledger path."""
    s.charge(100, SESSION_LIMIT)
    s.refund(90, {'total_tokens': 10, 'cost': 10 - remaining_usd}, SESSION_LIMIT)


def test_budget_exhaustion_mid_episode_is_an_api_refusal(tmp_path):
    s = Supervisor(configuration(tmp_path), Backend,
                   GeminiRouter('host-key', send=lambda payload: response(cost=.01)))
    try:
        near_cap(s, .08)
        gateway, token = s.start_interactive()
        answered = refused = 0
        for _ in range(20):
            reply = gateway.handle(token, gemini_packet())
            if 'result' in reply:
                answered += 1
            else:
                assert 'budget exhausted' in reply['error']['message']
                assert reply['error']['data']['no_action_executed'] is True
                refused += 1
            assert 'result' in gateway.handle(token, packet('robodojo_step'))
        spend = s.state['gemini_spend']
        assert answered >= 1 and refused >= 1 and answered + refused == 20
        assert spend['reported'] + spend['reserved'] <= SESSION_LIMIT and spend['reserved'] == 0
        assert s.backend.step == 20 and not gateway.control_uncertain
    finally:
        s.close()


def test_concurrent_ledger_never_exceeds_cap_or_goes_negative(tmp_path):
    """Broker threads charge and refund concurrently, as parallel formal workers do."""
    s = Supervisor(configuration(tmp_path), Backend)
    rng = random.Random(0)
    plans = [(rng.randint(1, 4000), rng.choice([.001, .01, .05, .2]), rng.random() < .1) for _ in range(400)]
    violations = []
    lock = threading.Lock()

    def one(plan):
        tokens, cost, fail = plan
        reservation = nanodollars(.25)
        try:
            s.charge(tokens, reservation)
        except RuntimeError:
            return 'refused'
        spend = s.state['gemini_spend']
        with lock:
            if spend['reported'] + spend['reserved'] > SESSION_LIMIT or spend['reserved'] < 0:
                violations.append(dict(spend))
        if fail:
            return 'retained'  # Provider failure: the reservation stays charged.
        s.refund(tokens // 2, {'total_tokens': tokens // 2, 'cost': cost}, reservation)
        return 'settled'

    try:
        with ThreadPoolExecutor(16) as pool:
            outcomes = list(pool.map(one, plans))
        spend = s.state['gemini_spend']
        assert not violations
        assert spend['reported'] + spend['reserved'] <= SESSION_LIMIT and spend['reserved'] >= 0
        retained = outcomes.count('retained')
        assert spend['reserved'] == retained * nanodollars(.25)
        settled = [plan for plan, outcome in zip(plans, outcomes) if outcome == 'settled']
        assert spend['reported'] == sum(nanodollars(cost) for _, cost, _ in settled)
        assert 'refused' in outcomes  # The cap was actually reached under contention.
    finally:
        s.close()


def test_parallel_formal_workers_share_one_capped_ledger_under_faults(tmp_path):
    """Four concurrent workers, each calling Gemini repeatedly with injected faults."""
    lock = threading.Lock()
    calls = {'n': 0}

    def send(payload):
        with lock:
            calls['n'] += 1
            n = calls['n']
        if n % 5 == 0:
            raise TimeoutError('slow')
        if n % 7 == 0:
            raise urllib.error.HTTPError('', 429, 'rate limited', None, None)
        return response(cost=.01)

    s, backend, bundle = rehearsed(tmp_path, episodes=8, workers=4, gemini=GeminiRouter('host-key', send=send))
    near_cap(s, .5)  # Exhausts partway through the batch: admission must hold across workers.
    launched = []
    s.worker_factory = inline(s, backend, launched)
    seen = []
    try:
        def run(_, job):
            for _ in range(6):
                reply = job['gateway'].handle(job['token'], gemini_packet())
                if 'error' in reply:
                    data = reply['error']['data']
                    assert data['no_action_executed'] is True and not data['control_uncertain']
                with lock:
                    seen.append('error' not in reply)
                assert 'result' in job['gateway'].handle(job['token'], packet('robodojo_step'))
            return {'reason': 'exit', 'returncode': 0}
        Runner.hook = run
        result = s.run(bundle, formal=True)
        spend = s.state['gemini_spend']
        assert result['completed_episodes'] == 8 and result['success_count'] == 8
        assert spend['reported'] + spend['reserved'] <= SESSION_LIMIT and spend['reserved'] >= 0
        assert any(seen) and not all(seen)
        assert result['api_budget']['session_cost_remaining_usd'] >= 0
    finally:
        s.close()
