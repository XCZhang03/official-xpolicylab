"""Official-side gemini_generate: the AgentBundle bridge and the self-hosted server's budget."""
import json
from pathlib import Path
import sys
import threading

import pytest

ADAPTER = Path(__file__).resolve().parents[2] / 'official/xpolicylab/AgentBundle'
sys.path.insert(0, str(ADAPTER))
import gemini_router  # noqa: E402
from bundle_bridge import Bridge  # noqa: E402

REPLY = {'model': gemini_router.MODEL, 'usage': {'total_tokens': 40, 'cost': 0.001},
         'choices': [{'message': {'role': 'assistant', 'content': '{"ok": true}'}, 'finish_reason': 'stop'}]}
REQUEST = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 256}


def service(send, limit=1.0):
    return gemini_router.BudgetedGemini(gemini_router.GeminiRouter('key', send=send), limit_usd=limit)


def test_bridge_serves_configured_gemini_and_reports_failures_like_the_harness():
    bridge = Bridge(lambda ctx: None, gemini=service(lambda payload: REPLY))
    value = json.loads(bridge.handle('gemini_generate', REQUEST)['content'][0]['text'])
    assert value['message']['content'] == '{"ok": true}' and value['budget']['calls'] == 1
    with pytest.raises(RuntimeError, match='not configured'):
        Bridge(lambda ctx: None).handle('gemini_generate', REQUEST)
    def fail(payload):
        raise TimeoutError('provider timed out')
    with pytest.raises(RuntimeError) as caught:
        Bridge(lambda ctx: None, gemini=service(fail)).handle('gemini_generate', REQUEST)
    error = json.loads(str(caught.value).removeprefix('MCP tool failed: '))['error']
    assert error['type'] == 'TimeoutError' and error['no_action_executed'] is True
    with pytest.raises(RuntimeError, match='Unsupported Gemini parameter'):
        Bridge(lambda ctx: None, gemini=service(lambda p: REPLY)).handle('gemini_generate', {**REQUEST, 'model': 'x'})


@pytest.mark.parametrize('reply', [
    {**REPLY, 'model': 'other/model'},
    {**REPLY, 'usage': {'total_tokens': 10 ** 9, 'cost': 0.001}},
    {**REPLY, 'choices': []},
])
def test_invalid_provider_replies_fail_and_keep_the_worst_case_charge(reply):
    gemini = service(lambda payload: reply)
    with pytest.raises(ValueError):
        gemini(REQUEST)
    budget = gemini.budget()
    assert budget['session_cost_reserved_usd'] == 0 and budget['session_cost_reported_usd'] > 0.001


def test_concurrent_calls_never_exceed_the_server_budget():
    gemini = service(lambda payload: REPLY, limit=0.2)
    outcomes = []
    def worker():
        for _ in range(40):
            try:
                gemini(REQUEST)
                outcomes.append(True)
            except RuntimeError as error:
                assert 'exhausted' in str(error)
                outcomes.append(False)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    budget = gemini.budget()
    assert budget['session_cost_reported_usd'] <= 0.2 and budget['session_cost_reserved_usd'] == 0
    assert any(outcomes) and not all(outcomes)


def test_official_service_needs_a_key(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert gemini_router.official_service({}) is None
    monkeypatch.setenv('OPENROUTER_API_KEY', 'server-key')
    assert isinstance(gemini_router.official_service({'gemini_budget_usd': 2}), gemini_router.BudgetedGemini)
    assert gemini_router.official_service({'gemini_enabled': False}) is None


def test_rehearsal_forwards_bundle_gemini_calls_to_the_harness(tmp_path):
    sys.path.insert(0, str(ADAPTER.parents[2] / 'auto_research_agent/api'))
    from runtime import run_official
    (tmp_path / 'controller.py').write_text(
        "import json\n"
        "def main(ctx):\n"
        "    reply = ctx.call('gemini_generate', messages=[{'role': 'user', 'content': 'which cup?'}])\n"
        "    assert json.loads(reply['content'][0]['text'])['message']['content'] == 'left'\n"
        "    ctx.call('robodojo_step', actions=[[0.0] * 14])\n")

    class Context:
        output_dir = tmp_path / 'out'

        def __init__(self):
            self.calls, self.steps = [], 0

        def call(self, tool, **arguments):
            self.calls.append(tool)
            if tool == 'gemini_generate':
                return {'content': [{'type': 'text', 'text': json.dumps({'message': {'content': 'left'}})}]}
            self.steps += len(arguments.get('actions', []))
            meta = {'step_id': self.steps, 'states': [0.0] * 14, 'eef_positions': [[0, 0, 1]] * 2,
                    'eef_quaternions_wxyz': [[1, 0, 0, 0]] * 2, 'attachments': [],
                    'transition': {'steps': [{'episode_ended': self.steps >= 3}]}}
            return {'content': [{'type': 'text', 'text': json.dumps(meta)}]}

    context = Context()
    bridge = run_official(context, str(tmp_path), action_wait_s=5)
    assert bridge.error is None
    assert context.calls[:3] == ['robodojo_observe', 'gemini_generate', 'robodojo_step']
