"""Real local HTTP/Unix relay tests; no external API requests."""
import gzip
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest

from harness.codex_cli import model_relay
from harness.codex_cli.model_relay import BUDGET_EXHAUSTED, MAX_BODY, MODEL, ModelRelay


@pytest.fixture
def relay(tmp_path, request):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((self.path, dict(self.headers), body))
            metadata = body.get('metadata', {})
            status = 302 if metadata.get('redirect') else metadata.get('status', 200)
            if len(received) > metadata.get('fail_first', len(received)):
                status = 200
            self.send_response(status)
            if 'retry_after' in metadata:
                self.send_header('Retry-After', str(metadata['retry_after']))
            self.send_header('Location', 'http://127.0.0.1:1/private')
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            content = b'data: fixture-event\n\n' if status == 200 else b'private provider error body'
            self.wfile.write(b'x' * (4*1024*1024) if metadata.get('large') else content)
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    path = tmp_path/'relay.sock'
    router = ModelRelay(path, f'http://127.0.0.1:{upstream.server_port}/v1',
                        {'Authorization': 'Bearer host-secret'}, max_calls=2, client_write_timeout=5,
                        **getattr(request, 'param', {}))
    router.start()
    def request(body=None, *, route='/responses', method='POST', compressed=False):
        connection = http.client.HTTPConnection('localhost', timeout=5)
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.settimeout(5)
        connection.sock.connect(str(path))
        raw = json.dumps(body or {'model': MODEL, 'input': []}).encode()
        headers = {'Authorization': 'Bearer agent-spoof', 'X-Agent-Secret': 'do-not-forward'}
        if compressed:
            raw = gzip.compress(raw)
            headers['Content-Encoding'] = 'gzip'
        try:
            # Rejected routes/CONNECT need no body. Sending a body after the
            # server's immediate close races its 403 response with BrokenPipe.
            connection.request(method, route, raw if method == 'POST' and route == '/responses' else None, headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()
    try:
        yield request, received, router
    finally:
        router.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_stream_auth_budget_and_no_redirect(relay):
    request, received, router = relay
    status, body = request(compressed=True)
    assert status == 200 and body == b'data: fixture-event\n\n'
    path, headers, payload = received[0]
    assert path == '/v1/responses'
    assert headers['Authorization'] == 'Bearer host-secret'
    assert 'X-Agent-Secret' not in headers and b'host-secret' not in body
    assert payload['max_output_tokens'] == 32768
    assert request({'model': MODEL, 'metadata': {'redirect': True}}) == (502, b'')
    assert request() == (402, BUDGET_EXHAUSTED)
    assert len(received) == router.calls == 2


@pytest.mark.parametrize('relay', [{'additional_models': {'openai/gpt-6-sol': 'xhigh'},
                                  'model_aliases': {'gpt-6-sol': 'openai/gpt-6-sol'}}], indirect=True)
def test_teacher_and_selected_student_share_budget_with_model_allowlist(relay):
    request, received, router = relay
    assert request({'model': 'openai/gpt-6-sol', 'reasoning': {'effort': 'high'}})[0] == 403
    assert request({'model': 'openai/gpt-6-sol'})[0] == 403
    assert request({'model': 'gpt-6-sol', 'reasoning': {'effort': 'high'}})[0] == 403
    assert request({'model': 'another-model', 'reasoning': {'effort': 'xhigh'}})[0] == 403
    assert not received and router.calls == 0
    assert request({'model': 'gpt-6-sol', 'reasoning': {'effort': 'xhigh'}})[0] == 200
    assert request()[0] == 200
    assert [r[2]['model'] for r in received] == ['openai/gpt-6-sol', MODEL]
    assert request({'model': 'openai/gpt-6-sol', 'reasoning': {'effort': 'xhigh'}})[0] == 402
    assert router.calls == 2


@pytest.mark.parametrize('status', [400, 401, 403, 408, 413, 422, 500])
def test_provider_status_preserves_retry_classification_without_body_or_relay_retry(relay, status):
    request, received, router = relay
    assert request({'model': MODEL, 'metadata': {'status': status}}) == (status, b'')
    assert len(received) == router.calls == 1


@pytest.mark.parametrize('status', [429, 502, 503, 504])
def test_throttled_provider_is_retried_once_per_call_budget(relay, monkeypatch, status):
    delays = []
    monkeypatch.setattr(model_relay, 'retry_delay', lambda attempt, header: delays.append(header) or 0)
    request, received, router = relay
    body = {'model': MODEL, 'metadata': {'status': status, 'fail_first': 3, 'retry_after': 7}}
    assert request(body) == (200, b'data: fixture-event\n\n')
    assert len(received) == 4 and router.calls == 1
    assert delays == ['7', '7', '7']


@pytest.mark.parametrize('status', [429, 503])
def test_persistent_throttling_surfaces_status_after_retry_window(relay, monkeypatch, status):
    monkeypatch.setattr(model_relay, 'RETRY_WINDOW_SECONDS', .2)
    monkeypatch.setattr(model_relay, 'retry_delay', lambda attempt, header: .05)
    request, received, router = relay
    assert request({'model': MODEL, 'metadata': {'status': status}}) == (status, b'')
    assert 2 <= len(received) <= 5 and router.calls == 1


def test_retry_delay_honours_retry_after_within_cap():
    assert model_relay.retry_delay(0, '12') == 12
    assert model_relay.retry_delay(0, '3600') == model_relay.RETRY_MAX_DELAY_SECONDS
    assert all(2**n/2 <= model_relay.retry_delay(n, None) <= 2**n for n in range(5))
    assert model_relay.retry_delay(20, 'Wed, 21 Oct 2015 07:28:00 GMT') <= model_relay.RETRY_MAX_DELAY_SECONDS


@pytest.mark.parametrize('body', [
    {'model': 'another-model'},
    {'model': MODEL, 'tools': [{'type': 'web_search'}]},
    {'model': MODEL, 'tools': [{'type': 'namespace', 'tools': [{'type': 'mcp'}]}]},
    {'model': MODEL, 'additional_tools': [{'type': 'web_search'}]},
    {'model': MODEL, 'plugins': [{'id': 'web'}]},
    {'model': MODEL, 'input': [{'type': 'input_image', 'image_url': 'https://example.org/image'}]},
    {'model': MODEL, 'input': [{'type': 'input_file', 'file_id': 'private'}]},
    {'model': MODEL, 'previous_response_id': 'another-session'},
    {'model': MODEL, 'max_output_tokens': 32769},
])
def test_forbidden_routes_do_not_spend_budget(relay, body):
    request, received, router = relay
    assert request(body)[0] in {400, 403}
    assert request(route='/arbitrary')[0] == 403
    assert request(method='CONNECT')[0] == 403
    assert not received and router.calls == 0


def test_image_history_larger_than_old_64_mib_cap_is_forwarded(relay):
    request, received, router = relay
    # Synthetic inline image data: local fixture inspects transport, not decoding.
    image = 'data:image/png;base64,' + 'A' * (65 * 1024 * 1024)
    assert request({'model': MODEL, 'input': [{'type':'input_image', 'image_url':image}]})[0] == 200
    assert received[0][2]['input'][0]['image_url'] == image
    assert router.calls == 1


def test_declared_body_above_256_mib_is_rejected_without_upstream(relay):
    _, received, router = relay
    assert MAX_BODY == 256 * 1024 * 1024
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(router.server.server_address)
        connection.sendall(f'POST /responses HTTP/1.1\r\nHost: test\r\nContent-Length: {MAX_BODY+1}\r\n\r\n'.encode())
        assert b'413' in connection.recv(4096).split(b'\r\n')[0]
    assert not received and router.calls == 0


def test_decompressed_body_limit_is_still_enforced(relay, monkeypatch):
    from harness.codex_cli import model_relay
    monkeypatch.setattr(model_relay, 'MAX_BODY', 1024)
    request, received, router = relay
    assert request({'model':MODEL, 'input':'A'*2048}, compressed=True)[0] == 413
    assert not received and router.calls == 0


def test_paused_reader_does_not_truncate_response(relay, monkeypatch):
    # Scale the old shared read/write deadline down; the independent delivery
    # allowance must survive a full Unix socket buffer until the client resumes.
    monkeypatch.setattr('harness.codex_cli.model_relay.REQUEST_TIMEOUT_SECONDS', .05)
    _, received, router = relay
    connection = http.client.HTTPConnection('localhost', timeout=5)
    connection.sock = socket.socket(socket.AF_UNIX)
    connection.sock.settimeout(5)
    connection.sock.connect(router.server.server_address)
    try:
        connection.request('POST', '/responses', json.dumps({'model': MODEL, 'metadata': {'large': True}}))
        response = connection.getresponse()
        time.sleep(.3)  # No reads: the response exceeds the socket buffer.
        assert response.status == 200
        assert response.read() == b'x' * (4*1024*1024)
        assert len(received) == router.calls == 1
    finally:
        connection.close()


@pytest.mark.parametrize('relay', [{'additional_models': {'openai/gpt-6-sol': None},
                                  'model_aliases': {'gpt-6-sol': 'openai/gpt-6-sol'}}], indirect=True)
def test_cheaper_auditor_model_is_allowed_at_any_effort(relay):
    request, received, router = relay
    assert request({'model': 'gpt-6-sol', 'input': [], 'reasoning': {'effort': 'medium'}})[0] == 200
    assert received[-1][2]['model'] == 'openai/gpt-6-sol'
    assert request({'model': 'openai/gpt-6-luna', 'input': []})[0] == 403  # Still an allowlist.
