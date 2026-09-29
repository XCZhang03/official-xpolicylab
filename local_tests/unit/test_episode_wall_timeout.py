"""Direct-control wall deadlines end the server, not merely the current RPC."""
from types import SimpleNamespace

import pytest

from services.robodojo import rpc


@pytest.mark.parametrize('phase', ['idle', 'action', 'completed'])
def test_episode_deadline_closes_rpc_and_preserves_finalized_outcome(monkeypatch, phase):
    handlers, timers, replies, reasons = [], [], [], []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def setsockopt(self, *_): pass
        def settimeout(self, value): assert value is None
        def bind(self, *_): pass
        def listen(self, *_): pass
        def getsockname(self): return ('127.0.0.1', 1)
        def accept(self): return self, None
    monkeypatch.setattr(rpc.socket, 'socket', lambda *_: Connection())
    def install(sig, handler):
        handlers.append(handler)
        return 'original-handler'
    monkeypatch.setattr(rpc.signal, 'signal', install)
    monkeypatch.setattr(rpc.signal, 'setitimer', lambda *args: timers.append(args))
    session = SimpleNamespace(metadata={}, terminated=phase=='completed', truncated=False,
                              poisoned=False, finish_reason=None, _write_summary=reasons.append)
    def dispatch(op, args):
        if op == 'step': handlers[0]()
        return {}
    session.dispatch = dispatch
    requests = iter(['reset', 'step'])
    def receive(_):
        op = next(requests)
        if op == 'step' and phase != 'action': handlers[0]()
        return {'version':rpc.VERSION,'request_id':op,'op':op}
    monkeypatch.setattr(rpc, 'receive_packet', receive)
    monkeypatch.setattr(rpc, 'send_packet', lambda conn, value: replies.append(value))
    rpc.serve(session, 0, episode_timeout_seconds=2400)
    assert len(replies) == 1 and replies[0]['ok']
    assert reasons == ['episode_wall_timeout']
    assert session.poisoned is (phase != 'completed')
    assert session.finish_reason == 'episode_wall_timeout'
    assert timers == [(rpc.signal.ITIMER_REAL,2400),(rpc.signal.ITIMER_REAL,0)]
    assert handlers[-1] == 'original-handler'
