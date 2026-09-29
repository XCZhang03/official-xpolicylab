"""Container-side relays: no credentials, host filesystem or general proxy."""
import os
import selectors
import socket
import socketserver
import subprocess
import sys
import threading


def mcp():
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect('/run/agent-relay/mcp.sock')
        def upload():
            try:
                while block := os.read(0, 65536):
                    connection.sendall(block)
            finally:
                connection.shutdown(socket.SHUT_WR)
        threading.Thread(target=upload, daemon=True).start()
        while block := connection.recv(65536):
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()


class ModelConnection(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect('/run/agent-relay/model.sock')
            with selectors.DefaultSelector() as poll:
                poll.register(self.request, selectors.EVENT_READ, upstream)
                poll.register(upstream, selectors.EVENT_READ, self.request)
                while True:
                    for key, _ in poll.select():
                        block = key.fileobj.recv(65536)
                        if not block:
                            return
                        key.data.sendall(block)


class ModelServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if sys.argv[1:] == ['mcp']:
        mcp()
        return
    with ModelServer(('127.0.0.1', 17777), ModelConnection) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        # Inherited terminal/stdin is passed directly to the real agent CLI
        # (Codex by default; the Claude launcher sets AGENT_BINARY).
        result = subprocess.call([os.environ.get('AGENT_BINARY', '/opt/codex/codex'), *sys.argv[1:]])
        server.shutdown()
    raise SystemExit(result)


if __name__ == '__main__':
    main()
