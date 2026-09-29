"""Our hosted agent API: the endpoint the submitted Mooncake_Agent policy calls.

    POST /v1/act   msgpack {session_id, episode_id, request_id, observation}
                   -> {actions: [<official action dict>, ...]}
    GET  /v1/health

Everything task-specific stays here. On an episode's first request the router picks
the served task from the observation (task_router.py), and the episode is leased to
a warm AgentBundle worker for that task: a local-mode XPolicyLab policy server
(serve_remote.sh bound to 127.0.0.1) holding that task's frozen bundle, planner and
Gemini key. Requests are then forwarded to it over XPolicyLab's websocket protocol.

A worker failure answers with a hold step instead of an HTTP error, because an error
reaching the evaluator's client is fatal for its whole trial.
"""
import argparse
import hmac
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import msgpack
import msgpack_numpy
import numpy as np

MAX_BODY = 64 * 1024 * 1024
ARMS = ('left', 'right')


def decode_images(obs):
    """Undo the client's image encoding (encode_images in Mooncake_Agent/model.py)."""
    for camera in (obs.get('vision') or {}).values():
        if not isinstance(camera, dict):
            continue
        for key, value in list(camera.items()):
            if isinstance(value, dict) and '__image__' in value:
                image = cv2.imdecode(np.frombuffer(value['data'], np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Undecodable camera image {key}')
                camera[key] = image
    return obs


def hold_action(obs):
    state = obs['state']
    return {f'{arm}_{kind}': np.asarray(state[f'{arm}_{kind}'], dtype=np.float32).reshape(-1)
            for arm in ARMS for kind in ('arm_joint_state', 'ee_joint_state')}


class Worker:
    """One AgentBundle policy server for one task; serves one episode at a time."""

    def __init__(self, task, url, *, token, timeout_s=100.0, connect_s=30.0, client_factory=None):
        self.task, self.url, self.token = task, url, token
        self.timeout_s, self.connect_s = timeout_s, connect_s
        self.client_factory = client_factory
        self.client, self.owner, self.last_used = None, None, 0.0

    def _connect(self):
        if self.client is not None:
            return self.client
        factory = self.client_factory
        if factory is None:
            from XPolicyLab.client_server.ws.model_client import WsModelClient as factory
        client = factory(url=self.url, evaluation_id=f'endpoint-{self.task}', trial_id=f'{self.task}-endpoint',
                         action_case_id=f'{self.task}_case', request_timeout_s=self.timeout_s,
                         max_connect_seconds=self.connect_s)
        reply = client.call('agentbundle_hello', {'task_name': self.task, 'token': self.token})
        if not isinstance(reply, dict) or reply.get('ok') is not True:
            client.close()
            raise RuntimeError(f'Worker {self.url} refused task {self.task}: {reply!r}')
        self.client = client
        return client

    def call(self, name, payload=None):
        try:
            client = self._connect()
            return client.call(name) if payload is None else client.call(name, payload)
        except Exception:
            self.drop()
            raise

    def drop(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None


class WorkerPool:
    """Leases workers to sessions. The evaluator's client never says goodbye, so a lease
    whose owner sent nothing for takeover_s (far longer than any gap inside an episode:
    a chunk executing, a scene reset) passes to the next session that needs it."""

    def __init__(self, workers, *, lease_wait_s=60.0, takeover_s=180.0, idle_s=900.0):
        self.workers = workers  # {task: [Worker]}
        self.lease_wait_s, self.takeover_s, self.idle_s = lease_wait_s, takeover_s, idle_s
        self.cond = threading.Condition()

    def acquire(self, task, owner):
        if task not in self.workers:
            raise RuntimeError(f'No worker serves task {task}')
        deadline = time.monotonic() + self.lease_wait_s
        with self.cond:
            while True:
                now = time.monotonic()
                free = [w for w in self.workers[task] if w.owner is None]
                stale = [w for w in self.workers[task] if w.owner is not None and now - w.last_used > self.takeover_s]
                if free or stale:
                    worker = (free or sorted(stale, key=lambda w: w.last_used))[0]
                    worker.owner, worker.last_used = owner, now
                    return worker
                if now >= deadline:
                    raise RuntimeError(f'No free worker for task {task}')
                self.cond.wait(timeout=deadline - now)

    def release(self, worker, owner):
        with self.cond:
            if worker is not None and worker.owner == owner:
                worker.owner = None
                self.cond.notify_all()


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.episode_id, self.worker, self.task = None, None, None
        self.cached = None  # (request_id, reply)
        self.last_used = time.monotonic()
        self.connections = 0  # Open HTTP connections that carried this session.


class AgentAPI:
    def __init__(self, pool, router, *, log_path=None, log=print):
        self.pool, self.router, self.log = pool, router, log
        self.log_path = Path(log_path) if log_path else None
        self.sessions, self.lock = {}, threading.Lock()

    def record(self, event):
        event = {'time': time.time(), **event}
        self.log('[endpoint] ' + json.dumps(event, default=str), flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'a') as handle:
                handle.write(json.dumps(event, default=str) + '\n')

    def session(self, session_id):
        with self.lock:
            now = time.monotonic()
            for key, stale in list(self.sessions.items()):
                if now - stale.last_used > self.pool.idle_s and not stale.lock.locked():
                    self.pool.release(stale.worker, key)
                    del self.sessions[key]
            return self.sessions.setdefault(session_id, Session())

    def connected(self, session_id):
        session = self.session(session_id)
        with self.pool.cond:
            session.connections += 1

    def disconnected(self, session_id):
        """The client's keep-alive connection closed: when it was the last one, the
        evaluator's policy server has most likely exited, so another session may take
        the worker at once. If this client reconnects first, it keeps the worker."""
        session = self.sessions.get(session_id)
        if session is None:
            return
        with self.pool.cond:
            session.connections -= 1
            if session.connections <= 0 and session.worker is not None and session.worker.owner == session_id:
                session.worker.last_used = float('-inf')
                self.pool.cond.notify_all()

    def act(self, body):
        session_id, episode_id = str(body['session_id']), str(body['episode_id'])
        request_id, obs = str(body['request_id']), decode_images(body['observation'])
        session = self.session(session_id)
        with session.lock:
            session.last_used = time.monotonic()
            if session.cached and session.cached[0] == request_id:
                return session.cached[1]  # The client retried after a dropped reply.
            actions = self._act(session, session_id, episode_id, obs)
            reply = {'actions': actions}
            session.cached = (request_id, reply)
            session.last_used = time.monotonic()
            return reply

    def _lease(self, session, session_id, episode_id):
        try:
            session.worker = self.pool.acquire(session.task, session_id)
            session.worker.call('reset')
        except Exception as exc:
            self.record({'event': 'worker_error', 'episode': episode_id, 'task': session.task, 'error': repr(exc)})
            self.pool.release(session.worker, session_id)
            session.worker = None

    def _act(self, session, session_id, episode_id, obs):
        if episode_id != session.episode_id:
            self.pool.release(session.worker, session_id)
            session.episode_id, session.worker = episode_id, None
            decision = self.router.detect(obs)
            session.task = decision['task']
            self.record({'event': 'episode', 'session': session_id, 'episode': episode_id, **decision})
            self._lease(session, session_id, episode_id)
        elif session.worker is not None and session.worker.owner != session_id:
            # Taken over after a long silence: restart the bundle from the current state.
            self.record({'event': 'lease_lost', 'episode': episode_id, 'task': session.task})
            self._lease(session, session_id, episode_id)
        worker = session.worker
        if worker is None:
            return [hold_action(obs)]
        try:
            worker.last_used = time.monotonic()
            worker.call('update_obs', obs)
            actions = worker.call('get_action')
            worker.last_used = time.monotonic()
            return actions
        except Exception as exc:
            # The episode can no longer be served; hold until the evaluator ends it.
            self.record({'event': 'worker_error', 'episode': episode_id, 'task': session.task, 'error': repr(exc)})
            self.pool.release(worker, session_id)
            session.worker = None
            return [hold_action(obs)]


def make_handler(api, keys):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, payload, kind='application/msgpack'):
            data = payload if isinstance(payload, bytes) else msgpack.packb(
                payload, default=msgpack_numpy.encode, use_bin_type=True)
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/v1/health':
                return self._send(200, b'{"ok": true}', 'application/json')
            self._send(404, b'not found', 'text/plain')

        def handle(self):
            self.sessions_seen = set()
            try:
                super().handle()
            finally:
                for session_id in self.sessions_seen:
                    api.disconnected(session_id)

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            if self.path != '/v1/act':
                self.rfile.read(min(length, MAX_BODY))
                return self._send(404, b'not found', 'text/plain')
            header = self.headers.get('Authorization', '')
            supplied = header[7:] if header.startswith('Bearer ') else ''
            if keys and not any(hmac.compare_digest(supplied, key) for key in keys):
                self.rfile.read(min(length, MAX_BODY))
                return self._send(401, b'invalid api key', 'text/plain')
            if not 0 < length <= MAX_BODY:
                self.close_connection = True
                return self._send(413, b'bad request size', 'text/plain')
            try:
                body = msgpack.unpackb(self.rfile.read(length), object_hook=msgpack_numpy.decode, raw=False)
            except Exception:
                return self._send(400, b'invalid msgpack body', 'text/plain')
            if not isinstance(body, dict) or not {'session_id', 'episode_id', 'request_id', 'observation'} <= set(body):
                return self._send(400, b'missing fields', 'text/plain')
            try:
                session_id = str(body['session_id'])
                if session_id not in self.sessions_seen:
                    self.sessions_seen.add(session_id)
                    api.connected(session_id)
                reply = api.act(body)
            except Exception as exc:
                api.record({'event': 'request_error', 'error': repr(exc)})
                return self._send(500, f'agent error: {type(exc).__name__}'.encode(), 'text/plain')
            self._send(200, reply)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', required=True, help='JSON file {task: [ws://127.0.0.1:port, ...]}')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8600)
    parser.add_argument('--configs', help='RoboDojo task config directory (variant hints)')
    parser.add_argument('--log', help='JSONL of episode routing decisions')
    parser.add_argument('--keys-env', default='MOONCAKE_ENDPOINT_KEYS', help='comma-separated client keys')
    parser.add_argument('--worker-token-env', default='AGENTBUNDLE_SERVER_TOKEN')
    parser.add_argument('--tls-cert', help='serve https directly (keeps client disconnects visible)')
    parser.add_argument('--tls-key')
    parser.add_argument('--gemini-budget-usd', type=float, default=2.0)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent / 'AgentBundle'))
    from gemini_router import BudgetedGemini, GeminiRouter
    from task_router import TaskRouter

    keys = [k for k in os.environ.get(args.keys_env, '').split(',') if k]
    if not keys:
        raise SystemExit(f'Set {args.keys_env}: the endpoint refuses to run without client keys')
    token = os.environ.get(args.worker_token_env, '')
    routes = json.loads(Path(args.workers).read_text())
    workers = {task: [Worker(task, url, token=token) for url in urls] for task, urls in routes.items()}
    key = os.environ.get('OPENROUTER_API_KEY')
    gemini = BudgetedGemini(GeminiRouter(key), limit_usd=args.gemini_budget_usd) if key else None
    router = TaskRouter(list(workers), gemini=gemini, configs_dir=args.configs)
    api = AgentAPI(WorkerPool(workers), router, log_path=args.log)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(api, keys))
    scheme = 'http'
    if args.tls_cert:
        import ssl
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = 'https'
    print(f'[endpoint] serving {len(workers)} tasks on {scheme}://{args.bind}:{args.port}/v1/act '
          f'(task detection: {"gemini" if gemini else "instruction fallback only"})', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
