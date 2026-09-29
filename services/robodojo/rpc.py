"""Single-owner blocking RPC server for a RoboDojo session."""

from __future__ import annotations

import json
import signal
import socket
import traceback

from .protocol import VERSION, receive_packet, send_packet


class EpisodeWallTimeout(BaseException):
    """Escape action handlers so an expired episode cannot accept further calls."""


def serve(session, port, *, episode_timeout_seconds=0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(1)
        bound_port = int(listener.getsockname()[1])
        print(
            json.dumps(
                {"event": "ready", "port": bound_port, "metadata": session.metadata}
            ),
            flush=True,
        )
        connection, _ = listener.accept()
        with connection:
            # Idle reasoning must not hit the old 15-minute socket timeout before
            # the episode deadline. Managed runs have their own host watchdog.
            connection.settimeout(None)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            previous_handler = None
            def expire(*_):
                raise EpisodeWallTimeout()
            try:
                while True:
                    request = receive_packet(connection)
                    response = {
                        "version": VERSION,
                        "request_id": request.get("request_id"),
                    }
                    try:
                        if request.get("version") != VERSION:
                            raise ValueError("Unsupported protocol version")
                        result = session.dispatch(
                            request["op"], request.get("args", {})
                        )
                        if request['op'] == 'reset' and episode_timeout_seconds:
                            previous_handler = signal.signal(signal.SIGALRM, expire)
                            signal.setitimer(signal.ITIMER_REAL, episode_timeout_seconds)
                        response.update(ok=True, result=result)
                    except Exception as exc:  # noqa: BLE001 - RPC must report simulator errors
                        session.poisoned = True
                        traceback.print_exc()
                        response.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    send_packet(connection, response)
            except EpisodeWallTimeout:
                # Preserve finalized native outcomes; an unfinished timed-out
                # attempt is not eligible for success. Closing the socket then
                # returning lets server.main close the simulator in its finally.
                if not (session.terminated or session.truncated):
                    session.poisoned = True
                    session.truncated = True
                session.finish_reason = 'episode_wall_timeout'
                session._write_summary('episode_wall_timeout')
                print(json.dumps({'event': 'episode_wall_timeout',
                                  'timeout_seconds': episode_timeout_seconds}), flush=True)
            except (EOFError, ConnectionError, TimeoutError):
                reason = (
                    "terminal"
                    if session.terminated or session.truncated
                    else "controller_disconnect"
                )
                session._write_summary(reason)
            finally:
                if previous_handler is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous_handler)
