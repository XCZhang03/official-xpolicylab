"""Dedicated auto-research dashboard: setup validation, auth and read-only session views."""
import http.client
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

from services.dashboard.server import Handler
from services.dashboard.sessions import DEFAULTS, Sessions

RUN = '20260928T120000Z_0123456789'


def sessions(tmp_path, *, started_at=None):
    project = tmp_path/'project'
    (project/'runtime/auto-research').mkdir(parents=True, exist_ok=True)
    return Sessions(project, lambda: ['make_kong', 'make_toast'], started_at)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def formal_session(model):
    root = model.root/RUN
    write(root/'operator-input.json', {'task': 'make_kong', 'exploration_seeds': [1, 2, 3],
                                      'demonstration_context': 'none', 'sim_gpu': 'GPU-a', 'training_gpu': 'GPU-b'})
    bundle = {'directory': 'abc', 'sha256': 'f'*64, 'workspace_path': 'code/reach'}
    rehearsal = {'mode': 'rehearsal', 'bundle': 'b1', 'sha256': 'f'*64, 'task_complete': True,
                 'reason': 'exit', 'returncode': 0}
    write(root/'state.json', {'exploration_started': 3, 'interactive_success': {'episode': 1},
        'formal_reserved': True, 'bundles': {'b1': bundle}, 'results': [rehearsal],
        'formal_batch': {'bundle': 'b1', 'status': 'completed', 'episode_count': 2,
                         'completed_episodes': 2, 'success_count': 1, 'success_rate': .5}})
    return root


def test_default_schedule_holds_out_formal_collection():
    explored = {int(c) for c in DEFAULTS['exploration_collections'].split(',')}
    assert explored == {0, 1, 2} and DEFAULTS['formal_eval_seed'] == 3


def test_settings_accept_only_official_cli_options(tmp_path):
    model = sessions(tmp_path)
    assert model.settings({})['image'] == DEFAULTS['image']
    for field in ('mode', 'observation_profile', 'student_model', 'formal_seconds'):
        with pytest.raises(ValueError, match='Unknown'):
            model.settings({field: 'x'})
    for bad, message in [({'task': 'unknown'}, 'installed task'), ({'research_gpu': '0'}, 'must differ'),
                         ({'episodes': 1}, 'episodes'), ({'image': 'bad image;rm'}, 'image'),
                         ({'exploration_collections': '0;1'}, 'exploration_collections'),
                         ({'demonstration_context': 'video'}, 'demonstration')]:
        with pytest.raises(ValueError, match=message):
            model.settings(bad)


def test_session_summary_release_and_scope(tmp_path):
    model = sessions(tmp_path)
    root = formal_session(model)
    detail = model.status()['selected']
    assert detail['run_id'] == RUN and detail['manual_success'] and detail['exploration_used'] == 3
    assert detail['stage'].startswith('Formal batch completed: 2/2')
    assert detail['gemini'] is None and 'student_usage' not in detail  # No ledger in this fixture.
    release = detail['release']
    assert release['submitted'] and release['bundle'] == 'b1'
    assert release['path'] == str(root/'bundles/abc')
    assert 'run_eval.sh sim make_kong' in release['official_check']
    assert f'make_kong={root}/bundles/abc' in release['build_checkpoint']
    for bad in ('../x', 'teacher-student', RUN+'x'):
        with pytest.raises(ValueError):
            model.directory(bad)
    with pytest.raises(ValueError, match='predates'):
        sessions(tmp_path, started_at=4e9).directory(RUN)


def test_http_requires_operator_token(tmp_path):
    model = sessions(tmp_path)
    formal_session(model)
    Handler.sessions, Handler.token = model, 'secret-token'
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, path, token=None, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        headers = {'X-Operator-Token': token} if token else {}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.getheader('Content-Type'), data
    try:
        status, kind, page = request('GET', '/')
        assert status == 200 and kind.startswith('text/html') and b'/app.js' in page
        assert request('GET', '/app.js')[0] == 200
        assert request('GET', '/api/state')[0] == 401
        assert request('GET', '/api/state', 'wrong')[0] == 401
        assert request('POST', '/api/launch', body=json.dumps({'run_id': RUN}))[0] == 401
        status, _, data = request('GET', '/api/state', 'secret-token')
        assert status == 200 and json.loads(data)['selected']['run_id'] == RUN
        assert request('GET', '/api/artifact?run_id='+RUN+'&path=../../etc/passwd', 'secret-token')[0] == 400
        assert request('POST', '/api/stop', 'secret-token', json.dumps({'run_id': RUN, 'x': 1}))[0] == 400
        assert request('POST', '/api/launch', 'secret-token', json.dumps({'run_id': RUN}))[0] == 400  # Already ran.
    finally:
        server.shutdown()
        server.server_close()


def test_episode_review_lists_videos_and_streams_ranges(tmp_path):
    import time
    model = sessions(tmp_path, started_at=time.time())
    root = formal_session(model)
    sims = {'exploration': root/'native/results/make_kong_exploration_20260928T120100Z_a/sim',
            'formal': root/'formal-workers/episode-002/native/results/make_kong_formal_20260928T130000Z_b/sim'}
    write(sims['exploration']/'operator_state.json', {'episode_id': 'e1', 'episode_mode': 'exploration', 'step_id': 7})
    write(sims['formal']/'summary.json', {'episode_id': 'f2', 'step_id': 600, 'success': True})
    write(sims['formal']/'operator_state.json', {'episode_id': 'f2', 'episode_mode': 'formal', 'episode_step_limit': 600})
    (sims['formal']/'sensors.mp4').write_bytes(bytes(range(256)) * 4)
    with pytest.raises(ValueError, match='predates'):
        model.episodes(RUN)  # The live view hides sessions older than the server.
    rows = model.episodes(RUN, history=True)
    assert [(r['mode'], r['live'], r['video'], r['formal_episode_index']) for r in rows] == [
        ('exploration', True, False, None), ('formal', False, True, 2)]
    assert model.status(history=True)['selected']['launchable'] is False
    for bad in ('../..', 'native/results/x/sim/../../..', str(sims['formal'])):
        with pytest.raises((ValueError, FileNotFoundError)):
            model.video(RUN, bad, history=True)

    Handler.sessions, Handler.token = model, 'secret-token'
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import urllib.parse
        path = '/api/video?' + urllib.parse.urlencode({'run_id': RUN, 'history': '1', 'episode': rows[1]['episode']})
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        connection.request('GET', path)
        assert connection.getresponse().status == 401
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        connection.request('POST', '/api/session', headers={'X-Operator-Token': 'secret-token'})
        response = connection.getresponse()
        cookie = response.getheader('Set-Cookie').split(';')[0]
        assert response.status == 204 and 'HttpOnly' in response.getheader('Set-Cookie')
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        connection.request('GET', path, headers={'Cookie': cookie, 'Range': 'bytes=10-19'})
        response = connection.getresponse()
        assert response.status == 206 and response.read() == bytes(range(10, 20))
        assert response.getheader('Content-Range') == 'bytes 10-19/1024'
    finally:
        server.shutdown()


def test_release_offers_combined_std_and_random_checkpoint(tmp_path):
    model = Sessions(tmp_path/'project', lambda: ['make_toast', 'make_toast_random', 'make_kong'])
    (model.root).mkdir(parents=True)
    bundle = {'directory': 'abc', 'sha256': 'f'*64, 'workspace_path': 'code/reach'}
    def session(run, task, bundle_id):
        rehearsal = {'mode': 'rehearsal', 'bundle': bundle_id, 'sha256': 'f'*64, 'task_complete': True,
                     'reason': 'exit', 'returncode': 0}
        write(model.root/run/'operator-input.json', {'task': task, 'exploration_seeds': [1]})
        write(model.root/run/'state.json', {'bundles': {bundle_id: bundle}, 'results': [rehearsal]})
    session('20260928T120000Z_0123456789', 'make_toast', 'std1')
    release = model.detail('20260928T120000Z_0123456789')['release']
    assert release['counterpart'] == {'task': 'make_toast_random', 'run_id': None} and 'build_combined' not in release
    assert ' sim make_toast ' in release['official_check']
    session('20260928T130000Z_abcdefabcd', 'make_toast_random', 'rnd1')
    release = model.detail('20260928T120000Z_0123456789')['release']
    assert release['counterpart']['run_id'] == '20260928T130000Z_abcdefabcd'
    assert 'make_toast=' in release['build_combined'] and 'make_toast_random=' in release['build_combined']
    session('20260928T140000Z_1111111111', 'make_kong', 'k1')
    assert 'counterpart' not in model.detail('20260928T140000Z_1111111111')['release']


def test_summary_reports_session_gemini_spend(tmp_path):
    model = sessions(tmp_path)
    root = formal_session(model)
    state = json.loads((root/'state.json').read_text())
    state.update(gemini_spend={'reported': 427_868_250, 'reserved': 0, 'blocked': False},
                 usage={'development': {'calls': 213}, 'formal': {'calls': 4}})
    write(root/'state.json', state)
    gemini = model.detail(RUN)['gemini']
    assert gemini['limit_usd'] == 10 and abs(gemini['reported_usd'] - 0.42786825) < 1e-9
    assert gemini['calls'] == {'development': 213, 'formal': 4} and not gemini['blocked']
