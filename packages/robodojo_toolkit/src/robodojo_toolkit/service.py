"""One warmed cuRobo planner shared by many short development scripts.

Building a Planner takes about 20 s (GPU setup and kernel loading). When
``ROBODOJO_TOOLKIT_PLANNER_SOCKET`` is set, ``shared_planner()`` connects to a
planner process listening on that Unix socket, starting it on first use, so every
later script plans immediately. Plans come from the same ``Planner`` code and give
the same result as an in-process planner. Without the variable (isolated
rehearsal, formal runs and the official policy server) the planner is built
in-process as before.

    python -m robodojo_toolkit.service --socket /tmp/robodojo-toolkit/planner.sock
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import threading
import time

import numpy as np

SOCKET_ENV = "ROBODOJO_TOOLKIT_PLANNER_SOCKET"
START_TIMEOUT_S = 300
MAX_LINE = 1 << 20


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def serve(path, config="official"):
    from .planning import Planner
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    planner = Planner(config=config)
    planner.warmup()
    lock = threading.Lock()  # cuRobo planners are not thread-safe.

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while line := self.rfile.readline(MAX_LINE):
                try:
                    request = json.loads(line)
                    if request.get("op") == "ping":
                        reply = {"config": config, "pid": os.getpid()}
                    elif request.get("op") == "plan":
                        with lock:
                            result = planner.plan(request["arm"], request["state"], request["target"],
                                                  request.get("gripper_opening"))
                        reply = {"result": _jsonable(result)}
                    else:
                        raise ValueError("Unknown planner service operation")
                except Exception as exc:  # Report to the client; keep serving others.
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    if path.exists():
        path.unlink()
    with Server(str(path), Handler) as server:
        os.chmod(path, 0o600)
        print(f"robodojo_toolkit planner service ({config}) ready on {path}", flush=True)
        server.serve_forever()


class RemotePlanner:
    """Same plan() interface as Planner, answered by the shared planner process."""

    def __init__(self, path, config="official"):
        from .kinematics import DualArm
        self.path, self.config = str(path), config
        self.kinematics = DualArm()
        self._lock = threading.Lock()
        self._socket = None

    def _request(self, value):
        with self._lock:
            for attempt in range(2):
                try:
                    if self._socket is None:
                        self._socket = socket.socket(socket.AF_UNIX)
                        self._socket.connect(self.path)
                        self._reader = self._socket.makefile("rb")
                    self._socket.sendall((json.dumps(_jsonable(value)) + "\n").encode())
                    line = self._reader.readline(MAX_LINE * 64)
                    if not line:
                        raise ConnectionError("Planner service closed the connection")
                    break
                except OSError:
                    self.close()
                    if attempt:
                        raise
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError("Planner service: " + reply["error"])
        return reply

    def ping(self):
        return self._request({"op": "ping"})

    def warmup(self):
        """The service warmed up when it started."""

    def plan(self, arm, state, target, gripper_opening=None):
        result = self._request({"op": "plan", "arm": arm, "state": state, "target": target,
                                "gripper_opening": gripper_opening})["result"]
        if "actions" in result:
            result["actions"] = np.asarray(result["actions"], dtype=np.float32).reshape(-1, 14)
        return result

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None


def connect(path, config="official", *, start=True, timeout_s=START_TIMEOUT_S):
    """A RemotePlanner for the service at ``path``, starting the service if needed."""
    import fcntl
    planner = RemotePlanner(path, config)
    try:
        served = planner.ping()["config"]
    except OSError:
        if not start:
            raise
        Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Concurrent first callers start exactly one service; the others wait for it.
        with open(Path(path).with_suffix(".lock"), "a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            try:
                served = planner.ping()["config"]
            except OSError:
                served = _start(planner, path, config, timeout_s)
    if served != config:
        raise RuntimeError(f"Planner service at {path} serves {served!r}, not {config!r}")
    return planner


def _start(planner, path, config, timeout_s):
    log = Path(path).with_suffix(".log")
    with open(log, "ab") as stream:
        process = subprocess.Popen([sys.executable, "-m", "robodojo_toolkit.service", "--socket", str(path),
                                    "--config", config], stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return planner.ping()["config"]
        except OSError:
            if process.poll() is not None:
                raise RuntimeError(f"Planner service exited ({process.returncode}); see {log}") from None
            if time.monotonic() > deadline:
                raise TimeoutError(f"Planner service did not start within {timeout_s} s; see {log}") from None
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default=os.environ.get(SOCKET_ENV))
    parser.add_argument("--config", default="official")
    args = parser.parse_args()
    if not args.socket:
        parser.error(f"--socket or {SOCKET_ENV} is required")
    serve(args.socket, args.config)


if __name__ == "__main__":
    main()
