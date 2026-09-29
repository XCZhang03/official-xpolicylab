"""Loopback-only operator dashboard for official-interface auto-research sessions.

Privileged: it can launch paid sessions and read every artifact. Never give its
URL or token to an agent.
"""
from __future__ import annotations

import argparse
import hmac
from http.cookies import SimpleCookie
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import subprocess
import time
import urllib.parse
import webbrowser

from services.dashboard.sessions import Sessions

STATIC = Path(__file__).resolve().parent/'static'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'application/javascript; charset=utf-8')}


def installed_tasks(project):
    root = Path(project)/'RoboDojo/task/RoboDojo/config'
    return lambda: sorted(p.stem for p in root.glob('*.yml') if not p.stem.startswith('_'))


class Handler(BaseHTTPRequestHandler):
    sessions: Sessions
    token: str
    server_version = 'RoboDojoResearch/1.0'

    def log_message(self, format_string, *args):
        print(f'[dashboard] {self.address_string()} {format_string % args}')

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy',
            "default-src 'self'; img-src 'self' blob:; style-src 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()

    def _send(self, status, data, content_type):
        self._headers(status, content_type, len(data))
        self.wfile.write(data)

    def _json(self, status, value):
        self._send(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

    def _error(self, status, message):
        self._json(status, {'error': message})

    def _authorized(self):
        supplied = self.headers.get('X-Operator-Token', '')
        if not supplied:
            # <video> requests cannot send headers; /api/session sets an HttpOnly cookie.
            try:
                cookie = SimpleCookie(self.headers.get('Cookie', '')).get('dashboard_token')
                supplied = cookie.value if cookie else ''
            except Exception:
                supplied = ''
        if supplied and hmac.compare_digest(supplied, self.token):
            return True
        self._error(HTTPStatus.UNAUTHORIZED, 'Missing or invalid operator token')
        return False

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ASSETS:
            name, kind = ASSETS[parsed.path]
            self._send(HTTPStatus.OK, (STATIC/name).read_bytes(), kind)
            return
        if not self._authorized():
            return
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        history = query.get('history') == '1'
        try:
            if parsed.path == '/api/state':
                self._json(HTTPStatus.OK, self.sessions.status(query.get('run_id') or None, history=history))
            elif parsed.path == '/api/frame':
                data = self.sessions.frame(*(query.get(k, '') for k in ('run_id', 'camera', 'native_run', 'step_id')),
                                           history=history)
                self._send(HTTPStatus.OK, data, 'image/jpeg')
            elif parsed.path == '/api/artifact':
                data, kind = self.sessions.artifact(query.get('run_id', ''), query.get('path', ''), history=history)
                self._send(HTTPStatus.OK, data, kind)
            elif parsed.path == '/api/episodes':
                self._json(HTTPStatus.OK, self.sessions.episodes(query.get('run_id', ''), history=history))
            elif parsed.path == '/api/video':
                self._video(self.sessions.video(query.get('run_id', ''), query.get('episode', ''), history=history))
            else:
                self._error(HTTPStatus.NOT_FOUND, 'Not found')
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f'Unavailable: {type(exc).__name__}')

    def _video(self, path):
        """Stream an MP4 with byte ranges so the player can seek."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        header = self.headers.get('Range')
        if header:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start, end = int(left), min(int(right), size - 1) if right else size - 1
                else:
                    start = max(0, size - int(right))
                status = HTTPStatus.PARTIAL_CONTENT
            if not match or not any(match.groups()) or start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(status)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'private, no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as exc:
            raise ValueError('Invalid content length') from exc
        if not 0 < length <= 32768:
            raise ValueError('JSON body must be 1..32768 bytes')
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError('Invalid JSON') from exc
        if not isinstance(value, dict):
            raise ValueError('JSON body must be an object')
        return value

    def do_POST(self):
        if not self._authorized():
            return
        if self.path == '/api/session':
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header('Set-Cookie', f'dashboard_token={self.token}; HttpOnly; SameSite=Strict; Path=/')
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return
        try:
            body = self._body()
            if self.path == '/api/prepare':
                result = self.sessions.prepare(body)
            elif self.path in {'/api/launch', '/api/stop'}:
                if set(body) != {'run_id'}:
                    raise ValueError('Provide only run_id')
                action = self.sessions.launch if self.path == '/api/launch' else self.sessions.stop
                result = action(body['run_id'])
            else:
                self._error(HTTPStatus.NOT_FOUND, 'Not found')
                return
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except (OSError, subprocess.SubprocessError) as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, f'Operator action failed: {type(exc).__name__}')
            return
        self._json(HTTPStatus.OK, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--open', action='store_true', dest='open_browser')
    args = parser.parse_args()
    if args.host not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('the privileged dashboard may bind only to loopback')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1, 65535]')
    Handler.sessions = Sessions(args.project_root, installed_tasks(args.project_root), time.time())
    Handler.token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/#token={Handler.token}'
    print('RoboDojo auto-research dashboard (privileged; do not give this URL to the agent)')
    print(url, flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
