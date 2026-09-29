"""Host-only fixed Responses endpoint. Credentials never enter Docker."""
import gzip
from http.server import BaseHTTPRequestHandler
import http.client
import io
import json
import os
import random
import re
import socketserver
import threading
import time
from urllib.parse import urlsplit


MODEL = 'openai/gpt-6-astra'
# Image-heavy conversation history is serialized into every model request.
MAX_BODY = 256*1024*1024
REQUEST_TIMEOUT_SECONDS = 120
# Provider throttling/overload is retried here, before any response byte reaches
# Codex, so a partial stream is never replayed. Codex's own two retries back off
# for seconds; provider rate-limit windows last minutes.
RETRY_STATUSES = frozenset({429, 502, 503, 504})
RETRY_WINDOW_SECONDS = 600
RETRY_MAX_DELAY_SECONDS = 60
BUDGET_EXHAUSTED = b'Relay model-call budget exhausted; the operator must raise --max-model-calls.'


def retry_delay(attempt, retry_after):
    """Provider Retry-After seconds when given, else jittered exponential backoff."""
    try:
        delay = float(retry_after)
    except (TypeError, ValueError):
        delay = min(RETRY_MAX_DELAY_SECONDS, 2**attempt) * random.uniform(.5, 1)
    return max(0., min(delay, RETRY_MAX_DELAY_SECONDS))


def local_inputs(value):
    """Reject provider-side fetching; text may freely discuss URLs as ordinary text."""
    if isinstance(value, list):
        return all(local_inputs(item) for item in value)
    if not isinstance(value, dict):
        return True
    if value.get('type') in {'input_image', 'image_url'}:
        url = value.get('image_url', '')
        if isinstance(url, dict):
            url = url.get('url', '')
        if not isinstance(url, str) or not url.startswith('data:image/'):
            return False
    if value.get('type') == 'input_file':
        return False
    return all(local_inputs(item) for item in value.values())


def local_tools(items):
    return isinstance(items, list) and all(isinstance(item, dict) and
        item.get('type') in {'function', 'custom', 'namespace'} and
        local_tools(item.get('tools', [])) for item in items)


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class ModelRelay:
    def __init__(self, path, base_url, headers=None, *, max_calls=4000, client_write_timeout=120,
                 model=MODEL, reasoning_effort=None, additional_models=None, model_aliases=None):
        if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', model):
            raise ValueError('Invalid operator model identifier')
        allowed_models = {model: reasoning_effort}
        for name, effort in (additional_models or {}).items():
            if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', name)
                    or effort not in {None, 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'}):
                raise ValueError('Invalid operator subagent model/effort')
            if name != model:
                allowed_models[name] = effort
        aliases = dict(model_aliases or {})
        for alias, target in aliases.items():
            if (not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', alias)
                    or target not in allowed_models or (alias in allowed_models and alias != target)):
                raise ValueError('Invalid operator model alias')
        url = urlsplit(base_url)
        if url.scheme not in {'https', 'http'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid operator model endpoint')
        if url.scheme == 'http' and url.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Non-TLS endpoint is only allowed for local tests')
        self.url, self.headers, self.max_calls = url, dict(headers or {}), max_calls
        self.calls, self.lock = 0, threading.Lock()
        relay = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *_):
                pass  # Never log headers, prompts, provider failures or credentials.
            def reject(self, status, body=b''):
                self.send_response(status)
                if body:
                    self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
            def do_POST(self):
                upstream = None
                try:
                    self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)
                    if self.path != '/responses' or self.headers.get('Transfer-Encoding'):
                        self.reject(403)
                        return
                    length = int(self.headers.get('Content-Length', '-1'))
                    if not 0 < length <= MAX_BODY:
                        self.reject(413)
                        return
                    body = self.rfile.read(length)
                    if self.headers.get('Content-Encoding') == 'gzip':
                        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                            body = stream.read(MAX_BODY+1)
                    elif self.headers.get('Content-Encoding'):
                        self.reject(415)
                        return
                    if len(body) > MAX_BODY:
                        self.reject(413)
                        return
                    payload = json.loads(body)
                    if not isinstance(payload, dict) or not isinstance(payload.get('model'), str):
                        self.reject(403)
                        return
                    selected_model = aliases.get(payload['model'], payload['model'])
                    if selected_model not in allowed_models:
                        self.reject(403)
                        return
                    required_effort = allowed_models[selected_model]
                    if required_effort is not None:
                        reasoning = payload.get('reasoning', {})
                        if not isinstance(reasoning, dict) or reasoning.get('effort') != required_effort:
                            self.reject(403)
                            return
                    # This is an inference relay, never a browser/download/MCP proxy.
                    if (not local_tools(payload.get('tools', [])) or
                        not local_tools(payload.get('additional_tools', [])) or
                        payload.get('plugins') or payload.get('web_search_options') or
                        payload.get('previous_response_id') or not local_inputs(payload.get('input', []))):
                        self.reject(403)
                        return
                    maximum = payload.get('max_output_tokens', 32768)
                    if type(maximum) is not int or not 1 <= maximum <= 32768:
                        self.reject(400)
                        return
                    payload['max_output_tokens'] = maximum
                    payload['model'] = selected_model
                    with relay.lock:
                        if relay.calls >= relay.max_calls:
                            # Not 429: Codex would retry a cap that never resets
                            # and report it as provider rate limiting.
                            self.reject(402, BUDGET_EXHAUSTED)
                            return
                        relay.calls += 1
                    connection = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
                    headers = {**relay.headers, 'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                               'Accept-Encoding': 'identity'}
                    body = json.dumps(payload).encode()
                    # One logical model call: throttled/overloaded attempts are
                    # neither billed nor counted against the cap.
                    deadline, attempt = time.monotonic()+RETRY_WINDOW_SECONDS, 0
                    while True:
                        upstream = connection(url.hostname, url.port, timeout=600)
                        upstream.request('POST', url.path.rstrip('/')+'/responses', body, headers)
                        response = upstream.getresponse()
                        if response.status not in RETRY_STATUSES:
                            break
                        delay = retry_delay(attempt, response.getheader('Retry-After'))
                        if time.monotonic()+delay >= deadline:
                            break
                        upstream.close()
                        time.sleep(delay)
                        attempt += 1
                    if response.status != 200:
                        # Preserve retry classification, but never expose provider
                        # bodies/headers or follow redirects. Codex's bounded
                        # retries follow once the relay's retry window is spent.
                        self.reject(response.status if 400 <= response.status <= 599 else 502)
                        return
                    # Rehearsal/submission pauses the whole client container.
                    # Its full socket buffers must not truncate a valid model
                    # response. Request reads retain 120s; provider I/O 600s.
                    self.connection.settimeout(client_write_timeout)
                    self.send_response(200)
                    self.send_header('Content-Type', response.getheader('Content-Type', 'text/event-stream'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    total = 0
                    while block := response.read1(65536):
                        total += len(block)
                        if total > MAX_BODY:
                            break
                        self.wfile.write(block)
                        self.wfile.flush()
                except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
                    self.close_connection = True
                finally:
                    if upstream:
                        upstream.close()
            def do_GET(self):
                self.reject(403)
            do_CONNECT = do_GET
        self.server = UnixHTTPServer(str(path), Handler)
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
