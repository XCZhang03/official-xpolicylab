"""Partial score survives controller failure without becoming binary success."""
import json
from types import SimpleNamespace

import pytest

from services.robodojo.scoring import episode_score, formal_scores
from services.robodojo.session import RoboDojoSession


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('ended,poisoned', [(False, False), (False, True), (True, False)])
def test_early_exit_and_controller_error_retain_native_score(tmp_path, ended, poisoned):
    manager = SimpleNamespace(get_score=lambda: [50.0])
    session = RoboDojoSession.__new__(RoboDojoSession)
    session.env = SimpleNamespace(reward_manager=manager, get_score=lambda: None,
                                  unstable_envs=set(), enforce_step_limit=True)
    session.output, session.episode_id, session.step_id = tmp_path, 'episode', 500
    session.video_writer, session.video_frames, session.finish_reason = None, 501, None
    session.metadata = {'control_dt': .04}
    session.native_step_limit = session.episode_step_limit = 1400
    session.success, session.terminated, session.truncated = False, False, ended
    session.poisoned = poisoned
    session._write_operator_state = lambda **kw: None
    session._write_summary('controller_error' if poisoned else 'controller_exit')
    result = json.loads((tmp_path/'evaluation_outcome.json').read_text())
    assert result['native_score'] == .5
    assert result['valid_for_score'] is True
    assert result['valid_for_success_rate'] is (ended and not poisoned)
    assert result['native_success'] is (False if ended and not poisoned else None)
    session.success = True
    session.terminated = True
    session.poisoned = False
    # Official run_eval overrides even a zero cached process score on success.
    manager.get_score = lambda: [0.0]
    session._write_summary('native_success')
    assert json.loads((tmp_path/'evaluation_outcome.json').read_text())['native_score'] == 1
    session.env.unstable_envs = {0}
    session._write_summary('invalid')
    result = json.loads((tmp_path/'evaluation_outcome.json').read_text())
    assert result['native_score'] is None and result['valid_for_score'] is False


def snapshot(root, number, percent, *, success=False, unstable=False, mode='formal'):
    sim = root/f'formal-workers/episode-{number:03d}/native/results/run/sim'
    write(sim/'operator_state.json', {'episode_id': f'episode{number}', 'step_id': 544,
        'episode_mode': mode, 'reward': {'native_score_percent': percent,
        'native_success': success, 'unstable_native_layout': unstable}})
    write(sim/'evaluation_outcome.json', {'native_score': None, 'complete': False, 'status': 'incomplete'})
    return sim


def test_legacy_native_fallback_uses_last_score_not_success_label(tmp_path):
    sim = snapshot(tmp_path, 1, 50)
    assert episode_score(sim, 'episode1')['score'] == .5
    assert episode_score(sim, 'other')['score'] is None
    sim = snapshot(tmp_path, 1, 0, success=True)
    assert episode_score(sim)['score'] == 1
    sim = snapshot(tmp_path, 1, 100, success=True, unstable=True)
    assert episode_score(sim)['score'] is None


@pytest.mark.parametrize('bad', [None, True, float('nan'), -5, 101, '50'])
def test_missing_or_invalid_score_not_fabricated(tmp_path, bad):
    assert episode_score(snapshot(tmp_path, 1, bad))['score'] is None


def test_batch_retains_partial_credit_on_error_and_fixed_denominator(tmp_path):
    rows = [{'formal_episode_index': n, 'episode_id': f'episode{n}',
             'formal_outcome': 'error' if n==2 else 'success'} for n in range(1, 5)]
    snapshot(tmp_path, 1, 0, success=True)
    snapshot(tmp_path, 2, 50)
    snapshot(tmp_path, 3, 100, unstable=True)
    batch = {'status': 'completed', 'episode_count': 4, 'episodes': rows}
    score = formal_scores(tmp_path, batch)
    assert score['average_score'] == .375
    assert score['average_score_percent'] == 37.5
    assert score['recorded_score_count'] == 2 and score['missing_score_count'] == 2
    assert not score['score_complete']
    batch['status'] = 'running'
    assert formal_scores(tmp_path, batch)['average_score'] is None


def test_direct_native_outcome_and_exploration_are_not_confused(tmp_path):
    snapshot(tmp_path, 1, 100, success=True, mode='exploration')
    batch = {'status':'completed', 'episode_count':1, 'episodes':[
        {'index':0, 'episode_id':'episode1', 'native_outcome':{
            'native_score':.25, 'status':'incomplete', 'valid_for_score':True}}]}
    assert formal_scores(tmp_path,batch)['average_score'] == .25


def test_dashboard_enriches_archive_without_changing_agent_state(tmp_path):
    from services.dashboard.sessions import Sessions
    root = tmp_path/'runtime/auto-research/20260927T235115Z_bb7cb7a096'
    write(root/'operator-input.json', {'task':'make_toast', 'mode':'auto-research'})
    write(root/'state.json', {'results':[{'mode':'formal','formal_episode_index':1,
        'episode_id':'episode1'}], 'formal_batch':{'status':'completed', 'episode_count':1,
        'completed_episodes':1, 'success_count':0, 'success_rate':0}})
    snapshot(root, 1, 50)
    before = (root/'state.json').read_bytes()
    result = Sessions(tmp_path, lambda: ['make_toast']).summary(root)
    assert result['formal_batch']['score_summary']['average_score_percent'] == 50
    assert (root/'state.json').read_bytes() == before
    assert 'score_summary' not in json.loads(before)['formal_batch']
