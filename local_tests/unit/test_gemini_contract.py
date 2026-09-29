"""API-shaped Gemini requests, stateless tool loops and bounded accounting."""
import base64
from copy import deepcopy
import json

import pytest

from services.controller.gemini import GeminiRouter, MODEL, MAX_TEXT_BYTES
from services.controller.gateway import Gateway
from test_gemini_images import picture

QUESTION = [{'role': 'user', 'content': 'Add 2 and 3.'}]
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
          'required': ['a', 'b'], 'additionalProperties': False}
TOOLS = [{'type': 'function', 'function': {'name': 'add', 'parameters': SCHEMA}}]
ASSISTANT = {'role': 'assistant', 'content': None, 'refusal': None, 'reasoning': None,
    'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'opaque-signature', 'format': 'google-gemini-v1'}],
    'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a":2,"b":3}'}}]}


def prepare(**arguments):
    return GeminiRouter('not-exposed').prepare({'messages': QUESTION, **arguments})


@pytest.mark.parametrize('choice', ['auto', 'required', 'none', {'type': 'function', 'function': {'name': 'add'}}])
def test_function_choices_and_reasoning_are_preserved(choice):
    options = {'tools': TOOLS, 'tool_choice': choice, 'reasoning': {'effort': 'low'}}
    payload, _ = prepare(**options)
    assert {key: payload[key] for key in options} == options
    assert payload['model'] == MODEL and payload['provider']['require_parameters'] is True
    assert payload['provider']['allow_fallbacks'] is False
    assert payload['provider']['max_price'] == {'prompt': .75, 'completion': 3.75, 'request': 0, 'image': .00000075}
    assert payload['stream'] is False


def test_full_assistant_and_tool_history_preserved_and_charged():
    messages = QUESTION + [ASSISTANT, {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"answer":5}'}]
    payload, reservation = prepare(messages=messages, tools=TOOLS)
    assert payload['messages'] == messages
    extra = deepcopy(messages)
    extra[1]['reasoning_details'][0]['data'] += 'x'*10000
    _, larger = prepare(messages=extra, tools=TOOLS)
    assert larger-reservation == 10000
    _, without_tools = prepare(messages=messages)
    assert reservation > without_tools
    extra[1]['reasoning_details'][0]['data'] += 'x'*MAX_TEXT_BYTES
    with pytest.raises(ValueError):
        prepare(messages=extra)


@pytest.mark.parametrize('fmt', [{'type': 'text'}, {'type': 'json_object'},
    {'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True, 'schema': SCHEMA}},
    {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': {
        '$defs': {'value': {'type': 'integer'}}, '$ref': '#/$defs/value'}}}])
def test_response_format_pass_through(fmt):
    assert prepare(response_format=fmt)[0]['response_format'] == fmt


@pytest.mark.parametrize('options', [
    {'tools': [{'type': 'web_search'}]}, {'tools': [{'type': 'mcp', 'server_url': 'http://localhost'}]},
    {'tools': TOOLS*2}, {'tools': [{'type': 'function', 'function': {'name': 'bad name'}}]},
    {'tools': [{'type': 'function', 'function': {'name': 'x', 'strict': 'true'}}]},
    {'tools': [{'type': 'function', 'function': {'name': 'x', 'parameters': {'$ref': 'file:///etc/passwd'}}}]},
    {'tools': [{'type': 'function', 'function': {'name': 'x', 'parameters': {'description': 'x'*16385}}}]},
    {'tool_choice': 'required'}, {'tool_choice': 'auto'}, {'tool_choice': 'invalid'},
    {'tools': TOOLS, 'tool_choice': {'type': 'function', 'function': {'name': 'other'}}},
    {'response_format': {'type': 'json_schema'}}, {'response_format': {'type': 'json_object', 'schema': {}}},
    {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'$ref': 'https://example.com/schema'}}}},
    {'reasoning': {'effort': 'invalid'}}, {'reasoning': {'exclude': 'true'}},
    {'reasoning': {'max_tokens': 100000}}, {'reasoning': {'effort': 'low', 'max_tokens': 100}},
    {'reasoning': {'enabled': 1}}, {'reasoning': {'other': 'value'}},
    {'parallel_tool_calls': False}, {'stream': True}, {'plugins': [{'id': 'web'}]},
    {'provider': {'require_parameters': False}}, {'model': 'other'}, {'api_key': 'key'},
    {'temperature': float('nan')},
])
def test_unsupported_and_malformed_requests_fail_before_io(options):
    with pytest.raises(ValueError):
        prepare(**options)


@pytest.mark.parametrize('message', [
    {'role': 'tool', 'content': '5'}, {'role': 'tool', 'tool_call_id': 'x', 'content': {'answer': 5}},
    {'role': 'user', 'content': None}, {'role': 'user', 'content': 'x', 'tool_calls': []},
    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x', 'type': 'function',
         'function': {'name': 'add', 'arguments': {'a': 1}}}]},
    {'role': 'assistant', 'content': None, 'reasoning_details': 'not-array'},
])
def test_malformed_history_rejected(message):
    with pytest.raises(ValueError):
        prepare(messages=QUESTION+[message])


@pytest.mark.parametrize('url', ['https://example.com/a.png', 'file:///etc/passwd',
    'data:text/plain;base64,aGVsbG8=', 'data:image/png;base64,not base64'])
def test_image_url_never_fetches_external_resources(url):
    with pytest.raises(ValueError):
        prepare(messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}])


def test_mcp_image_and_openrouter_data_url_are_equivalent():
    data = base64.b64encode(picture()).decode()
    first = {'type': 'image', 'mimeType': 'image/png', 'data': data}
    second = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+data}}
    assert prepare(messages=[{'role': 'user', 'content': [first]}]) == prepare(messages=[{'role': 'user', 'content': [second]}])


def test_gateway_tool_roundtrip_returns_data_and_accounts_each_call():
    sent, charges, refunds = [], [], []
    class Backend:
        def tools(self):
            return []
        def call(self, *args):
            pytest.fail('Gemini tool calls must not execute robot calls')
    def send(payload):
        sent.append(deepcopy(payload))
        message = ASSISTANT if len(sent) == 1 else {'role': 'assistant', 'content': '{"answer":5}'}
        return {'model': MODEL, 'choices': [{'message': message, 'finish_reason': 'tool_calls' if len(sent) == 1 else 'stop'}],
                'usage': {'total_tokens': 10, 'cost': .001}}
    gateway = Gateway(Backend(), GeminiRouter('never-returned', send=send), audit=lambda _: None,
        charge=lambda amount, cost: charges.append(amount), refund=lambda amount, usage, cost: refunds.append(amount),
        budget=lambda: {'calls_used': len(charges), 'tokens_charged': sum(charges)-sum(refunds)})
    token = gateway.acquire()
    def call(messages):
        reply = gateway.handle(token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
            'params': {'name': 'gemini_generate', 'arguments': {'messages': messages, 'tools': TOOLS}}})
        return json.loads(reply['result']['content'][0]['text'])
    first = call(QUESTION)
    assert first['message'] == ASSISTANT and first['budget']['tokens_charged'] == 10
    history = QUESTION+[first['message'], {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"answer":5}'}]
    second = call(history)
    assert sent[1]['messages'] == history
    assert second['budget'] == {'calls_used': 2, 'tokens_charged': 20}
    assert second['cost_usd'] == .001 and 'never-returned' not in json.dumps(second)


def test_failed_provider_keeps_reservation_even_for_tool_calls():
    def send(_):
        raise TimeoutError('private provider diagnostic')
    charged = []
    class Backend:
        def tools(self):
            return []
    gateway = Gateway(Backend(), GeminiRouter('secret', send=send), audit=lambda _: None,
        charge=lambda amount, cost: charged.append(amount), refund=lambda *_: pytest.fail('Must retain failed-call reservation'), budget=lambda: {})
    token = gateway.acquire()
    reply = gateway.handle(token, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'gemini_generate', 'arguments': {'messages': QUESTION, 'tools': TOOLS}}})
    assert len(charged) == 1 and 'error' in reply and 'private' not in json.dumps(reply)


def test_skill_python_examples_follow_the_contract():
    from pathlib import Path
    import re
    reference = Path(__file__).resolve().parents[2]/'auto_research_agent/.agents/skills/gemini/references/api.md'
    seen = []
    class Context:
        def call(self, name, **arguments):
            assert name == 'gemini_generate'
            payload, _ = prepare(**arguments)
            seen.append(payload)
            if arguments.get('tool_choice') == 'required':
                message, reason = ASSISTANT, 'tool_calls'
            else:
                message = {'role': 'assistant', 'content': '{"label":"red block","visible":true}' if len(seen) == 1 else '5'}
                reason = 'stop'
            return {'content': [{'type': 'text', 'text': json.dumps({
                'message': message, 'finish_reason': reason, 'usage': {}, 'budget': {}})}]}
    scope = {'ctx': Context()}
    for snippet in re.findall(r'```python\n(.*?)```', reference.read_text(), flags=re.S):
        exec(compile(snippet, str(reference), 'exec'), scope)
    assert len(seen) == 3
    assert seen[-1]['messages'][1] == ASSISTANT
