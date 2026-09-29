"""Concurrent slot dispatch, with global lifecycle calls on the main thread.

Native alarm-based deadlines stay in workers/main-thread lifecycle code. Reader
threads never execute simulator calls. One request per connection is typical,
but JSON-RPC IDs also allow pipelined requests on a single Codex connection.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import select
import socket
import threading
import time

MAX_WIRE_BYTES = 32 * 1024 * 1024


def serve_socket(service, listener, *, slots):
    events = queue.Queue(maxsize=64)
    clients = set()
    clients_lock = threading.Lock()
    stopping = threading.Event()
    outstanding = {}
    eof = set()
    listener.settimeout(.25)

    def enqueue(event):
        while not stopping.is_set():
            try:
                events.put(event, timeout=.1)
                return
            except queue.Full:
                continue

    def reader(connection):
        try:
            with connection.makefile('rb') as stream:
                while not stopping.is_set():
                    line = stream.readline(MAX_WIRE_BYTES + 1)
                    if not line or len(line) > MAX_WIRE_BYTES:
                        return
                    try:
                        request = json.loads(line)
                    except ValueError:
                        request = None
                    enqueue(('request', connection, request))
        except OSError:
            pass
        finally:
            enqueue(('eof', connection, None))

    def accept():
        while not stopping.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with clients_lock:
                if len(clients) >= 64:
                    connection.close()
                    continue
                clients.add(connection)
            threading.Thread(target=reader, args=(connection,), daemon=True).start()

    def done(connection, future):
        try:
            response = future.result()
        except Exception as exc:
            response = {'jsonrpc': '2.0', 'id': None,
                        'error': {'code': -32000, 'message': str(exc)}}
        enqueue(('response', connection, response))

    def send(connection, response):
        if response is None:
            return
        try:
            data = memoryview(json.dumps(response, allow_nan=False).encode() + b'\n')
            deadline = time.monotonic() + 30
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [connection], [], remaining)[1]:
                    raise TimeoutError('MCP client stopped reading')
                try:
                    sent = connection.send(data[:65536], socket.MSG_DONTWAIT)
                except BlockingIOError:
                    continue
                if not sent:
                    raise ConnectionError('MCP client closed')
                data = data[sent:]
        except (OSError, ValueError):
            with clients_lock:
                clients.discard(connection)
            connection.close()

    threading.Thread(target=accept, daemon=True).start()
    executor = ThreadPoolExecutor(max_workers=slots + 2, thread_name_prefix='exploration')
    pending = 0
    def release(connection):
        if connection in eof and not outstanding.get(connection, 0):
            with clients_lock:
                clients.discard(connection)
            outstanding.pop(connection, None)
            eof.discard(connection)
            connection.close()
    try:
        while True:
            kind, connection, value = events.get()
            if kind == 'eof':
                eof.add(connection)
                release(connection)
            elif kind == 'response':
                pending -= 1
                outstanding[connection] -= 1
                send(connection, value)
                release(connection)
            elif service.parallel_request(value):
                if pending >= 32:
                    send(connection, {'jsonrpc': '2.0', 'id': (value or {}).get('id'),
                        'error': {'code': -32000, 'message': 'Exploration request queue full; wait for outstanding calls'}})
                    continue
                pending += 1
                outstanding[connection] = outstanding.get(connection, 0) + 1
                future = executor.submit(service.handle, value)
                future.add_done_callback(lambda f, c=connection: done(c, f))
            else:
                send(connection, service.handle(value))
    finally:
        stopping.set()
        service.interrupt()
        with clients_lock:
            for connection in clients:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        executor.shutdown(wait=True, cancel_futures=True)


def bridge(path):
    import os
    import sys
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(str(path))
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
